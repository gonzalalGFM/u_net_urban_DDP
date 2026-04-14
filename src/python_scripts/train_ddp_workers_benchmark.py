import argparse
import os
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

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unet_testing"))
from dataset import CFDDataset  # noqa: E402
from model import UNet  # noqa: E402

OUTPUT_CHANNELS = [0, 1, 2, 3]
N_BLOCKS = 3
N_LAYERS = 2
TRAIN_SPLIT = 0.8
MSE_WEIGHT = 0.6
SSIM_WEIGHT = 0.4
SEED = 42


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse = self.mse(pred, target)
        ssim_val = structural_similarity_index_measure(pred, target, data_range=1.0)
        return MSE_WEIGHT * mse + SSIM_WEIGHT * (1.0 - ssim_val)


def sync_scalar_avg(value: float, device: torch.device) -> float:
    t = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


def sync_scalar_sum(value: float, device: torch.device) -> float:
    t = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def parse_args():
    p = argparse.ArgumentParser(description="DDP benchmark over num_workers.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-epochs", type=int, default=100)
    p.add_argument("--num-workers-list", default="0,2,4,8,12,16")
    p.add_argument(
        "--data-dir",
        default="/PATH/TO/DATA",
    )
    p.add_argument(
        "--output-dir",
        default="/PATH/TO/OUTPUT",
    )
    return p.parse_args()


def format_table(rows):
    headers = [
        "workers",
        "epoch_s_mean",
        "train_sps(mean/min/max)",
        "val_sps(mean/min/max)",
        "train_wait_s_mean",
        "train_compute_s_mean",
        "overlap_%_mean",
        "final_val_acc",
    ]
    out = []
    out.append(" | ".join(headers))
    out.append("-|-|-|-|-|-|-|-")
    for r in rows:
        out.append(
            f"{r['workers']} | "
            f"{r['epoch_s_mean']:.2f} | "
            f"{r['train_sps_mean']:.1f}/{r['train_sps_min']:.1f}/{r['train_sps_max']:.1f} | "
            f"{r['val_sps_mean']:.1f}/{r['val_sps_min']:.1f}/{r['val_sps_max']:.1f} | "
            f"{r['train_wait_s_mean']:.2f} | "
            f"{r['train_compute_s_mean']:.2f} | "
            f"{r['train_overlap_pct_mean']:.1f} | "
            f"{r['final_val_acc']:.4f}"
        )
    return "\n".join(out)


def create_loaders(train_ds, val_ds, batch_size, num_workers, rank):
    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=SEED, drop_last=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
    loader_kw = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler, **loader_kw)
    if rank == 0:
        print(
            f"[BENCH] workers={num_workers} "
            f"train_batches={len(train_loader)} val_batches={len(val_loader)}"
        )
    return train_loader, val_loader, train_sampler


def main():
    args = parse_args()

    # Fallback for direct srun launch without torchrun.
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
            f"LOCAL_RANK={local_rank}, but visible CUDA devices={torch.cuda.device_count()}."
        )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    workers_list = [int(x.strip()) for x in args.num_workers_list.split(",") if x.strip()]
    if any(w < 0 for w in workers_list):
        raise ValueError("--num-workers-list must contain non-negative integers.")

    if rank == 0:
        print("[BENCH] DDP workers benchmark")
        print(f"[BENCH] ranks={world_size} batch/gpu={args.batch_size} epochs={args.num_epochs}")
        print(f"[BENCH] workers_list={workers_list}")

    h5_files = sorted(Path(args.data_dir).glob("*.h5"))
    dataset = CFDDataset(h5_files, slice_axis="z")
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
    )
    if rank == 0:
        print(f"[BENCH] train={len(train_ds)} val={len(val_ds)}")

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    sample_in, _ = dataset[0]
    results_rows = []

    for cfg_idx, workers in enumerate(workers_list):
        if rank == 0:
            print("\n" + "=" * 70)
            print(f"[BENCH] Starting config {cfg_idx + 1}/{len(workers_list)} | workers={workers}")
            print("=" * 70)

        train_loader, val_loader, train_sampler = create_loaders(
            train_ds, val_ds, args.batch_size, workers, rank
        )

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

        epoch_times = []
        train_loader_sps_hist, val_loader_sps_hist = [], []
        train_wait_s_hist, train_compute_s_hist, train_overlap_pct_hist = [], [], []
        val_wait_s_hist = []
        final_val_acc = 0.0

        for epoch in range(args.num_epochs):
            train_sampler.set_epoch(cfg_idx * 100000 + epoch)
            ep_t0 = time.time()

            model.train()
            train_loss_sum = 0.0
            train_mae_sum = 0.0
            train_wait_local = 0.0
            train_compute_local = 0.0
            train_samples_local = 0

            it = iter(train_loader)
            while True:
                wait_t0 = time.perf_counter()
                try:
                    inp, tgt = next(it)
                except StopIteration:
                    break
                train_wait_local += time.perf_counter() - wait_t0
                train_samples_local += inp.shape[0]

                comp_t0 = time.perf_counter()
                inp = inp.to(device)
                tgt = tgt[:, OUTPUT_CHANNELS, :, :].to(device)
                optimizer.zero_grad()
                out = model(inp)
                loss = criterion(out, tgt)
                loss.backward()
                optimizer.step()
                train_compute_local += time.perf_counter() - comp_t0

                train_loss_sum += loss.item()
                train_mae_sum += torch.mean(torch.abs(out.detach() - tgt)).item()

            train_loss = train_loss_sum / max(len(train_loader), 1)
            train_mae = train_mae_sum / max(len(train_loader), 1)

            model.eval()
            val_loss_sum = 0.0
            val_mae_sum = 0.0
            val_wait_local = 0.0
            val_samples_local = 0
            with torch.no_grad():
                it = iter(val_loader)
                while True:
                    wait_t0 = time.perf_counter()
                    try:
                        inp, tgt = next(it)
                    except StopIteration:
                        break
                    val_wait_local += time.perf_counter() - wait_t0
                    val_samples_local += inp.shape[0]

                    inp = inp.to(device)
                    tgt = tgt[:, OUTPUT_CHANNELS, :, :].to(device)
                    out = model(inp)
                    val_loss_sum += criterion(out, tgt).item()
                    val_mae_sum += torch.mean(torch.abs(out - tgt)).item()

            val_loss = val_loss_sum / max(len(val_loader), 1)
            val_mae = val_mae_sum / max(len(val_loader), 1)

            train_loss = sync_scalar_avg(train_loss, device)
            val_loss = sync_scalar_avg(val_loss, device)
            train_mae = sync_scalar_avg(train_mae, device)
            val_mae = sync_scalar_avg(val_mae, device)

            train_wait = sync_scalar_sum(train_wait_local, device)
            val_wait = sync_scalar_sum(val_wait_local, device)
            train_compute = sync_scalar_sum(train_compute_local, device)
            train_samples = sync_scalar_sum(float(train_samples_local), device)
            val_samples = sync_scalar_sum(float(val_samples_local), device)

            train_sps = train_samples / train_wait if train_wait > 0 else 0.0
            val_sps = val_samples / val_wait if val_wait > 0 else 0.0
            overlap_pct = (
                100.0 * train_compute / (train_compute + train_wait)
                if (train_compute + train_wait) > 0
                else 0.0
            )

            train_loader_sps_hist.append(train_sps)
            val_loader_sps_hist.append(val_sps)
            train_wait_s_hist.append(train_wait)
            train_compute_s_hist.append(train_compute)
            train_overlap_pct_hist.append(overlap_pct)
            val_wait_s_hist.append(val_wait)

            epoch_times.append(time.time() - ep_t0)
            final_val_acc = 1.0 - val_mae

            if rank == 0 and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(
                    f"[BENCH][w={workers}] {epoch+1}/{args.num_epochs} "
                    f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                    f"val_acc={final_val_acc:.4f} epoch_s={epoch_times[-1]:.2f} "
                    f"train_sps={train_sps:.1f}"
                )

        summary = {
            "workers": workers,
            "epoch_s_mean": float(np.mean(epoch_times)),
            "train_sps_mean": float(np.mean(train_loader_sps_hist)),
            "train_sps_min": float(np.min(train_loader_sps_hist)),
            "train_sps_max": float(np.max(train_loader_sps_hist)),
            "val_sps_mean": float(np.mean(val_loader_sps_hist)),
            "val_sps_min": float(np.min(val_loader_sps_hist)),
            "val_sps_max": float(np.max(val_loader_sps_hist)),
            "train_wait_s_mean": float(np.mean(train_wait_s_hist)),
            "train_compute_s_mean": float(np.mean(train_compute_s_hist)),
            "train_overlap_pct_mean": float(np.mean(train_overlap_pct_hist)),
            "val_wait_s_mean": float(np.mean(val_wait_s_hist)),
            "final_val_acc": float(final_val_acc),
        }
        results_rows.append(summary)

        if rank == 0:
            print(
                f"[BENCH] Completed workers={workers} | "
                f"epoch_s_mean={summary['epoch_s_mean']:.2f} "
                f"train_sps_mean={summary['train_sps_mean']:.1f} "
                f"overlap_mean={summary['train_overlap_pct_mean']:.1f}%"
            )

        del train_loader, val_loader, train_sampler, model, optimizer, criterion
        torch.cuda.empty_cache()
        dist.barrier()

    if rank == 0:
        table_text = format_table(results_rows)
        print("\n" + "=" * 90)
        print("DDP DataLoader Workers Benchmark (short training)")
        print("=" * 90)
        print(table_text)
        print("=" * 90)

        np.savez(output_dir / "workers_benchmark_summary.npz", rows=results_rows)
        with open(output_dir / "workers_benchmark_summary.txt", "w", encoding="utf-8") as f:
            f.write(table_text + "\n")
        print(f"[BENCH] Saved summary to {output_dir}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
