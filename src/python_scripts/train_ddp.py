import os  
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import sys, time, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
#from pytorch_msssim import ssim
from torchmetrics.functional.image import structural_similarity_index_measure

# ---------------------------------------------------------------------------
# Import model & dataset from the existing unet_testing module
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unet_testing"))
from model   import UNet          # noqa: E402
from dataset import CFDDataset    # noqa: E402

# ===========================================================================
# Fixed configuration  (only --lr / --base-channels / --batch-size are tunable)
# ===========================================================================
OUTPUT_CHANNELS = [0, 1, 2, 3]   # U_0, U_1, U_2, T
N_BLOCKS        = 3
N_LAYERS        = 2
TRAIN_SPLIT     = 0.8
MSE_WEIGHT      = 0.6
SSIM_WEIGHT     = 0.4
EPOCHS          = 5000
PATIENCE        = 50
SEED            = 42


# ===========================================================================
# Loss
# ===========================================================================
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse      = self.mse(pred, target)
        # Use stateless functional SSIM to avoid device/state issues across DDP ranks.
        ssim_val = structural_similarity_index_measure(pred, target, data_range=1.0)
        return MSE_WEIGHT * mse + SSIM_WEIGHT * (1.0 - ssim_val)


# ===========================================================================
# GPU monitoring   (pynvml is optional — power is skipped without it)
# ===========================================================================
class GPUMonitor:
    """Collects utilization, memory and (optionally) power once per epoch."""

    def __init__(self, cuda_device_idx: int):
        self.idx = cuda_device_idx
        self._nvml_handle = None
        try:                                    # nvidia-ml-py  (pip install pynvml)
            import pynvml
            pynvml.nvmlInit()
            # match by UUID so CUDA_VISIBLE_DEVICES remapping is respected
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
        """Returns (util_pct, mem_mb, power_w).  util/power are None without pynvml."""
        mem_mb   = torch.cuda.memory_allocated(self.idx) / (1024 ** 2)
        util_pct = None
        power_w  = None
        if self._nvml_handle is not None:
            import pynvml
            util_pct = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle).gpu
            power_w  = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
        return util_pct, mem_mb, power_w


# ===========================================================================
# Helpers
# ===========================================================================
def sync_scalar(value: float, device: torch.device) -> float:
    """All-reduce-average a scalar across every rank."""
    t = torch.tensor(value, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / dist.get_world_size()


def sync_sum(value: float, device: torch.device) -> float:
    """All-reduce-sum a scalar across every rank."""
    t = torch.tensor(value, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def _plot_history(train_losses, val_losses, path: Path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses,   label="Val")
    plt.yscale("log")
    plt.xlabel("Epoch"); plt.ylabel("Loss (MSE + SSIM)")
    plt.legend(); plt.title("DDP Training History")
    plt.savefig(path); plt.close()


# ===========================================================================
# CLI
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="DDP UNet training — 4-channel CFD")
    p.add_argument("--lr",            type=float, default=1e-4,  help="Learning rate")
    p.add_argument("--base-channels", type=int,   default=32,    help="Base filter count")
    p.add_argument("--batch-size",    type=int,   default=16,    help="Per-GPU batch size")
    p.add_argument("--data-dir",      default="/PATH/TO/DATA")
    p.add_argument("--output-dir",    default="/PATH/TO/OUTPUT")
    return p.parse_args()


# ===========================================================================
# Main
# ===========================================================================
def main():
    args = parse_args()

    # ── SLURM → torch env-var shim (fallback when not launched via torchrun) ──
    for _slurm, _torch in [("SLURM_PROCID",  "RANK"),
                            ("SLURM_LOCALID", "LOCAL_RANK"),
                            ("SLURM_NTASKS",  "WORLD_SIZE")]:
        if _torch not in os.environ and _slurm in os.environ:
            os.environ[_torch] = os.environ[_slurm]

    # ── distributed init ──────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this job. "
            "Check Slurm GPU allocation and launcher configuration."
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("No CUDA devices are visible to the process.")
    if local_rank >= gpu_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {gpu_count} CUDA device(s) are visible."
        )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"[DDP] ranks={world_size}  lr={args.lr}  "
              f"base_ch={args.base_channels}  batch/gpu={args.batch_size}  "
              f"global_batch={args.batch_size * world_size}")

    # ── dataset  (deterministic split: every rank builds the same subsets) ──
    h5_files = sorted(Path(args.data_dir).glob("*.h5"))
    dataset  = CFDDataset(h5_files, slice_axis="z")

    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    if rank == 0:
        print(f"[DDP] train={len(train_ds)}  val={len(val_ds)}")

    train_sampler = DistributedSampler(train_ds, shuffle=True,  seed=SEED, drop_last=True)
    val_sampler   = DistributedSampler(val_ds,   shuffle=False, drop_last=False)

    loader_kw = dict(num_workers=20, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, sampler=val_sampler, **loader_kw)

    # ── model ─────────────────────────────────────────────────────────
    sample_in, _ = dataset[0]
    model = UNet(
        n_channels=sample_in.shape[0],
        n_classes=len(OUTPUT_CHANNELS),
        base_channels=args.base_channels,
        n_blocks=N_BLOCKS,
        n_layers=N_LAYERS,
    ).to(device)
    model     = DDP(model, device_ids=[local_rank], output_device=local_rank)
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── output dir + GPU monitor ──────────────────────────────────────
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    monitor = GPUMonitor(local_rank)

    # ── training ──────────────────────────────────────────────────────
    best_val  = float("inf")
    patience  = 0

    # histories (populated on rank 0 only)
    train_loss_hist, val_loss_hist = [], []
    train_acc_hist,  val_acc_hist  = [], []
    train_loader_sps_hist, val_loader_sps_hist = [], []
    train_wait_s_hist, train_compute_s_hist, train_overlap_pct_hist = [], [], []
    val_wait_s_hist, val_compute_s_hist = [], []

    # per-epoch GPU metrics (every rank collects; rank 0 averages at the end)
    gpu_utils, mem_mbs, power_ws, epoch_times = [], [], [], []

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        t0 = time.time()

        # ── train pass ──
        model.train()
        train_loss_sum = 0.0
        train_mae_sum  = 0.0
        train_loader_wait_s_local = 0.0
        train_compute_s_local = 0.0
        train_samples_local = 0
        train_iter = iter(train_loader)
        while True:
            fetch_t0 = time.perf_counter()
            try:
                inp, tgt = next(train_iter)
            except StopIteration:
                break
            train_loader_wait_s_local += time.perf_counter() - fetch_t0
            train_samples_local += inp.shape[0]

            compute_t0 = time.perf_counter()
            inp = inp.to(device)
            tgt = tgt[:, OUTPUT_CHANNELS, :, :].to(device)
            optimizer.zero_grad()
            out  = model(inp)
            loss = criterion(out, tgt)
            loss.backward()
            optimizer.step()
            train_compute_s_local += time.perf_counter() - compute_t0
            train_loss_sum += loss.item()
            train_mae_sum  += torch.mean(torch.abs(out.detach() - tgt)).item()

        train_loss = train_loss_sum / max(len(train_loader), 1)
        train_mae  = train_mae_sum  / max(len(train_loader), 1)

        # ── val pass ──
        model.eval()
        val_loss_sum = 0.0
        val_mae_sum  = 0.0
        val_loader_wait_s_local = 0.0
        val_compute_s_local = 0.0
        val_samples_local = 0
        with torch.no_grad():
            val_iter = iter(val_loader)
            while True:
                fetch_t0 = time.perf_counter()
                try:
                    inp, tgt = next(val_iter)
                except StopIteration:
                    break
                val_loader_wait_s_local += time.perf_counter() - fetch_t0
                val_samples_local += inp.shape[0]

                compute_t0 = time.perf_counter()
                inp = inp.to(device)
                tgt = tgt[:, OUTPUT_CHANNELS, :, :].to(device)
                out  = model(inp)
                val_loss_sum += criterion(out, tgt).item()
                val_mae_sum  += torch.mean(torch.abs(out - tgt)).item()
                val_compute_s_local += time.perf_counter() - compute_t0

        val_loss = val_loss_sum / max(len(val_loader), 1)
        val_mae  = val_mae_sum  / max(len(val_loader), 1)

        # ── sync across ranks ──
        train_loss = sync_scalar(train_loss, device)
        val_loss   = sync_scalar(val_loss,   device)
        train_mae  = sync_scalar(train_mae,  device)
        val_mae    = sync_scalar(val_mae,    device)
        train_loader_wait_s = sync_sum(train_loader_wait_s_local, device)
        val_loader_wait_s = sync_sum(val_loader_wait_s_local, device)
        train_compute_s = sync_sum(train_compute_s_local, device)
        val_compute_s = sync_sum(val_compute_s_local, device)
        train_samples = sync_sum(float(train_samples_local), device)
        val_samples = sync_sum(float(val_samples_local), device)

        train_loader_sps = (
            train_samples / train_loader_wait_s if train_loader_wait_s > 0.0 else 0.0
        )
        val_loader_sps = val_samples / val_loader_wait_s if val_loader_wait_s > 0.0 else 0.0
        train_overlap_pct = (
            100.0 * (train_compute_s / (train_compute_s + train_loader_wait_s))
            if (train_compute_s + train_loader_wait_s) > 0.0
            else 0.0
        )

        # ── GPU sample ──
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        u, m, p = monitor.sample()
        if u is not None:
            gpu_utils.append(u)
        mem_mbs.append(m)
        if p is not None:
            power_ws.append(p)

        # ── log (rank 0) ──
        if rank == 0:
            train_loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)
            train_acc_hist.append(1.0 - train_mae)
            val_acc_hist.append(1.0 - val_mae)
            train_loader_sps_hist.append(train_loader_sps)
            val_loader_sps_hist.append(val_loader_sps)
            train_wait_s_hist.append(train_loader_wait_s)
            train_compute_s_hist.append(train_compute_s)
            train_overlap_pct_hist.append(train_overlap_pct)
            val_wait_s_hist.append(val_loader_wait_s)
            val_compute_s_hist.append(val_compute_s)
            print(f"[DDP] {epoch+1}/{EPOCHS}  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                  f"val_acc={1.0 - val_mae:.4f}  {elapsed:.1f}s")

        # ── checkpoint + early-stop  (all ranks see the same val_loss) ──
        if val_loss < best_val:
            best_val  = val_loss
            patience  = 0
            if rank == 0:
                torch.save(model.module.state_dict(),
                           output_dir / "best_unet_ddp.pth")
        else:
            patience += 1

        if patience >= PATIENCE:
            if rank == 0:
                print("[DDP] Early stopping triggered.")
            break

    # ── end-of-training summary (rank 0) ──────────────────────────────
    if rank == 0:
        np.savez(output_dir / "history_ddp.npz",
                 train_loss=train_loss_hist, val_loss=val_loss_hist,
                 train_acc=train_acc_hist,   val_acc=val_acc_hist,
                 train_loader_sps=train_loader_sps_hist, val_loader_sps=val_loader_sps_hist,
                 train_wait_s=train_wait_s_hist, train_compute_s=train_compute_s_hist,
                 train_overlap_pct=train_overlap_pct_hist,
                 val_wait_s=val_wait_s_hist, val_compute_s=val_compute_s_hist)
        _plot_history(train_loss_hist, val_loss_hist, output_dir / "history_ddp.png")

        print("\n" + "=" * 52)
        print("  DDP  Training Summary")
        print("=" * 52)
        if gpu_utils:
            print(f"  Mean GPU Use (%)       : {np.mean(gpu_utils):.1f}")
        else:
            print(f"  Mean GPU Use (%)       : N/A  (install pynvml)")
        print(f"  Mean Memory Use (MB)   : {np.mean(mem_mbs):.1f}")
        if power_ws:
            print(f"  Mean Power (W)         : {np.mean(power_ws):.1f}")
        else:
            print(f"  Mean Power (W)         : N/A  (install pynvml)")
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
        print("=" * 52)
        print(f"  Best model  →  {output_dir / 'best_unet_ddp.pth'}\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
