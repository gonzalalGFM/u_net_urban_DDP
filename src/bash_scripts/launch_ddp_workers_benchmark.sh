#!/bin/bash

# ── SBATCH ───────
#SBATCH --job-name=ddp_unet
#SBATCH --partition=acc
#SBATCH --nodes=1                 
#SBATCH --ntasks-per-node=1        
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=20         
#SBATCH --time=48:00:00
#SBATCH --output=logs/ddp_%j.log
#SBATCH --error=logs/ddp_%j.err

# ── Benchmark parameters (edit these) ─────────────────────────
LR="1e-4"
BASE_CHANNELS=32
BATCH_SIZE=16
NUM_EPOCHS=100
NUM_WORKERS_LIST="0,1,2,4,8,12,16,20,24,28,32,36"
OUTPUT_DIR="ddp_workers_benchmark_output"

# ── Environment ───────────────────────────────────────────────
module purge
module load nvidia-hpc-sdk/23.11-cuda11.8 miniforge
source activate torch_infer_env

export HDF5_USE_FILE_LOCKING=FALSE
set -euo pipefail
mkdir -p logs

# ── Rendezvous + Slurm compatibility ──────────────────────────
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
export MASTER_PORT=29510
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-40}"
NNODES="${SLURM_NNODES:-1}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-2}"
export OMP_NUM_THREADS="${CPUS_PER_TASK}"
export SRUN_CPUS_PER_TASK="${CPUS_PER_TASK}"

echo "=== DDP Workers Benchmark Launch ==="
echo "  Nodes        : $NNODES"
echo "  GPUs/node    : $GPUS_PER_NODE"
echo "  Batch/GPU    : $BATCH_SIZE"
echo "  Epochs       : $NUM_EPOCHS"
echo "  Workers list : $NUM_WORKERS_LIST"

srun --ntasks="${NNODES}" --ntasks-per-node=1 \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --gres="gpu:${GPUS_PER_NODE}" \
  torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --node_rank="${SLURM_NODEID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    train_ddp_workers_benchmark.py \
      --lr "${LR}" \
      --base-channels "${BASE_CHANNELS}" \
      --batch-size "${BATCH_SIZE}" \
      --num-epochs "${NUM_EPOCHS}" \
      --num-workers-list "${NUM_WORKERS_LIST}" \
      --output-dir "${OUTPUT_DIR}"
