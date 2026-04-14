import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.functional.image import structural_similarity_index_measure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unet_testing"))
from model import UNet  # noqa: E402
from dataset import CFDDataset  # noqa: E402

OUTPUT_CHANNELS = [0, 1, 2, 3]
N_BLOCKS = 3
N_LAYERS = 2
TRAIN_SPLIT = 0.8
MSE_WEIGHT = 0.6
SSIM_WEIGHT = 0.4
EPOCHS = 5000
PATIENCE = 50
SEED = 42


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse = self.mse(pred, target)
        ssim_val = structural_similarity_index_measure(pred, target, data_range=1.0)
        return MSE_WEIGHT * mse + SSIM_WEIGHT * (1.0 - ssim_val)


class GPUMonitor:
    """Collects utilization, memory and optional power once per epoch."""

    def __init__(self, cuda_device_idx: int):
        self.idx = cuda_device_idx
        self._nvml_handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            target_uuid = str(torch.cuda.get_device_properties(cuda_device_idx).uuid)
            target_uuid_norm = target_uuid.lower().removeprefix("gpu-")
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                nvml_uuid = str(pynvml.nvmlDeviceGetUUID(h))
                nvml_uuid_norm = nvml_uuid.lower().removeprefix("gpu-")
                if nvml_uuid_norm == target_uuid_norm:
                    self._nvml_handle = h
                    break
        except Exception:
            pass

    def sample(self):
        mem_mb = torch.cuda.memory_allocated(self.idx) / (1024 ** 2)
        util_pct = None
        power_w = None
        if self._nvml_handle is not None:
            import pynvml

            util_pct = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle).gpu
            power_w = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
        return util_pct, mem_mb, power_w


def sync_avg(value: float, device: torch.device) -> float:
    t = torch.tensor(value, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / dist.get_world_size()


def sync_sum(value: float, device: torch.device) -> float:
    t = torch.tensor(value, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def parse_args():
    p = argparse.ArgumentParser(description="DDP UNet training with AMP.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=16, help="Per-GPU batch size")
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--data-dir", default="/PATH/TO/DATA")
    p.add_argument("--output-dir", default="/PATH/TO/OUTPUT")
    return p.parse_args()


def main():
    args = parse_args()

    for slurm_k, torch_k in [
        ("SLURM_PROCID", "RANK"),
        ("SLURM_LOCALID", "LOCAL_RANK"),
        ("SLURM_NTASKS", "WORLD_SIZE"),
    ]:
        if torch_k not in os.environ and slurm_k in os.environ:
            os.environ[torch_k] = os.environ[slurm_k]

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this job.")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but visible CUDA devices={torch.cuda.device_count()}."
        )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    # GradScaler is useful for fp16; for bf16 we keep it disabled.
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp_dtype == "fp16"))

    if rank == 0:
        print(
            f"[DDP-AMP] ranks={world_size} lr={args.lr} base_ch={args.base_channels} "
            f"batch/gpu={args.batch_size} global_batch={args.batch_size * world_size} "
            f"workers={args.num_workers} amp={args.amp_dtype}"
        )

    h5_files = sorted(Path(args.data_dir).glob("*.h5"))
    dataset = CFDDataset(h5_files, slice_axis="z")

    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
    )
    if rank == 0:
        print(f"[DDP-AMP] train={len(train_ds)} val={len(val_ds)}")

    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=SEED, drop_last=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)

    loader_kw = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, **loader_kw)

    sample_in, _ = dataset[0]
    model = UNet(
        n_channels=sample_in.shape[0],
        n_classes=len(OUTPUT_CHANNELS),
        base_channels=args.base_channels,
        n_blocks=N_BLOCKS,
        n_layers=N_LAYERS,
    ).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    monitor = GPUMonitor(local_rank)
    best_val = float("inf")
    patience = 0

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []
    gpu_utils, mem_mbs, power_ws, epoch_times = [], [], [], []

    train_loader_sps_hist, val_loader_sps_hist = [], []
    train_wait_s_hist, train_compute_s_hist, train_overlap_pct_hist = [], [], []
    val_wait_s_hist, val_compute_s_hist = [], []

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        ep_t0 = time.time()

        model.train()
        train_loss_sum = 0.0
        train_mae_sum = 0.0
        train_wait_local = 0.0
        train_compute_local = 0.0
        train_samples_local = 0

        train_iter = iter(train_loader)
        while True:
            wait_t0 = time.perf_counter()
            try:
                inp, tgt = next(train_iter)
            except StopIteration:
                break
            train_wait_local += time.perf_counter() - wait_t0
            train_samples_local += inp.shape[0]

            comp_t0 = time.perf_counter()
            inp = inp.to(device, non_blocking=True)
            tgt = tgt[:, OUTPUT_CHANNELS, :, :].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                out = model(inp)
                loss = criterion(out, tgt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_compute_local += time.perf_counter() - comp_t0

            train_loss_sum += loss.item()
            train_mae_sum += torch.mean(torch.abs(out.detach() - tgt)).item()

        train_loss = train_loss_sum / max(len(train_loader), 1)
        train_mae = train_mae_sum / max(len(train_loader), 1)

        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        val_wait_local = 0.0
        val_compute_local = 0.0
        val_samples_local = 0
        with torch.no_grad():
            val_iter = iter(val_loader)
            while True:
                wait_t0 = time.perf_counter()
                try:
                    inp, tgt = next(val_iter)
                except StopIteration:
                    break
                val_wait_local += time.perf_counter() - wait_t0
                val_samples_local += inp.shape[0]

                comp_t0 = time.perf_counter()
                inp = inp.to(device, non_blocking=True)
                tgt = tgt[:, OUTPUT_CHANNELS, :, :].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                    out = model(inp)
                    vloss = criterion(out, tgt)
                val_loss_sum += vloss.item()
                val_mae_sum += torch.mean(torch.abs(out - tgt)).item()
                val_compute_local += time.perf_counter() - comp_t0

        val_loss = val_loss_sum / max(len(val_loader), 1)
        val_mae = val_mae_sum / max(len(val_loader), 1)

        train_loss = sync_avg(train_loss, device)
        val_loss = sync_avg(val_loss, device)
        train_mae = sync_avg(train_mae, device)
        val_mae = sync_avg(val_mae, device)

        train_wait = sync_sum(train_wait_local, device)
        train_compute = sync_sum(train_compute_local, device)
        val_wait = sync_sum(val_wait_local, device)
        val_compute = sync_sum(val_compute_local, device)
        train_samples = sync_sum(float(train_samples_local), device)
        val_samples = sync_sum(float(val_samples_local), device)

        train_sps = train_samples / train_wait if train_wait > 0 else 0.0
        val_sps = val_samples / val_wait if val_wait > 0 else 0.0
        train_overlap_pct = (
            100.0 * train_compute / (train_compute + train_wait)
            if (train_compute + train_wait) > 0
            else 0.0
        )

        elapsed = time.time() - ep_t0
        epoch_times.append(elapsed)
        u, m, p = monitor.sample()
        if u is not None:
            gpu_utils.append(u)
        mem_mbs.append(m)
        if p is not None:
            power_ws.append(p)

        if rank == 0:
            train_loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)
            train_acc_hist.append(1.0 - train_mae)
            val_acc_hist.append(1.0 - val_mae)
            train_loader_sps_hist.append(train_sps)
            val_loader_sps_hist.append(val_sps)
            train_wait_s_hist.append(train_wait)
            train_compute_s_hist.append(train_compute)
            train_overlap_pct_hist.append(train_overlap_pct)
            val_wait_s_hist.append(val_wait)
            val_compute_s_hist.append(val_compute)
            print(
                f"[DDP-AMP] {epoch+1}/{EPOCHS} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} val_acc={1.0 - val_mae:.4f} {elapsed:.1f}s"
            )

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            if rank == 0:
                torch.save(model.module.state_dict(), output_dir / "best_unet_ddp_amp.pth")
        else:
            patience += 1
        if patience >= PATIENCE:
            if rank == 0:
                print("[DDP-AMP] Early stopping triggered.")
            break

    if rank == 0:
        np.savez(
            output_dir / "history_ddp_amp.npz",
            train_loss=train_loss_hist,
            val_loss=val_loss_hist,
            train_acc=train_acc_hist,
            val_acc=val_acc_hist,
            train_loader_sps=train_loader_sps_hist,
            val_loader_sps=val_loader_sps_hist,
            train_wait_s=train_wait_s_hist,
            train_compute_s=train_compute_s_hist,
            train_overlap_pct=train_overlap_pct_hist,
            val_wait_s=val_wait_s_hist,
            val_compute_s=val_compute_s_hist,
        )

        print("\n" + "=" * 56)
        print("  DDP-AMP Training Summary")
        print("=" * 56)
        if gpu_utils:
            print(f"  Mean GPU Use (%)       : {np.mean(gpu_utils):.1f}")
        else:
            print("  Mean GPU Use (%)       : N/A  (install pynvml)")
        print(f"  Mean Memory Use (MB)   : {np.mean(mem_mbs):.1f}")
        if power_ws:
            print(f"  Mean Power (W)         : {np.mean(power_ws):.1f}")
        else:
            print("  Mean Power (W)         : N/A  (install pynvml)")
        print(f"  Mean Time / Epoch (s)  : {np.mean(epoch_times):.2f}")
        if train_loader_sps_hist:
            print(
                "  DataLoader Train (samples/s) "
                f"mean/min/max: {np.mean(train_loader_sps_hist):.1f} / "
                f"{np.min(train_loader_sps_hist):.1f} / {np.max(train_loader_sps_hist):.1f}"
            )
        if train_wait_s_hist:
            print(
                "  Train Data Wait (s)    mean/min/max: "
                f"{np.mean(train_wait_s_hist):.2f} / {np.min(train_wait_s_hist):.2f} / "
                f"{np.max(train_wait_s_hist):.2f}"
            )
        if train_compute_s_hist:
            print(
                "  Train GPU Compute (s)  mean/min/max: "
                f"{np.mean(train_compute_s_hist):.2f} / {np.min(train_compute_s_hist):.2f} / "
                f"{np.max(train_compute_s_hist):.2f}"
            )
        if train_overlap_pct_hist:
            print(
                "  Train Overlap Proxy (%) mean/min/max: "
                f"{np.mean(train_overlap_pct_hist):.1f} / {np.min(train_overlap_pct_hist):.1f} / "
                f"{np.max(train_overlap_pct_hist):.1f}"
            )
        if val_loader_sps_hist:
            print(
                "  DataLoader Val   (samples/s) "
                f"mean/min/max: {np.mean(val_loader_sps_hist):.1f} / "
                f"{np.min(val_loader_sps_hist):.1f} / {np.max(val_loader_sps_hist):.1f}"
            )
        if val_wait_s_hist:
            print(
                "  Val Data Wait (s)      mean/min/max: "
                f"{np.mean(val_wait_s_hist):.2f} / {np.min(val_wait_s_hist):.2f} / "
                f"{np.max(val_wait_s_hist):.2f}"
            )
        if val_compute_s_hist:
            print(
                "  Val GPU Compute (s)    mean/min/max: "
                f"{np.mean(val_compute_s_hist):.2f} / {np.min(val_compute_s_hist):.2f} / "
                f"{np.max(val_compute_s_hist):.2f}"
            )
        if train_acc_hist:
            print(f"  Final Train Acc        : {train_acc_hist[-1]:.4f}")
            print(f"  Final Val Acc          : {val_acc_hist[-1]:.4f}")
        print("=" * 56)
        print(f"  Best model  ->  {output_dir / 'best_unet_ddp_amp.pth'}\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
