#!/bin/bash
#SBATCH --job-name=transfer_fixalloc
#SBATCH --output=logs/slurm/transfer_fixalloc_%A_%a.out
#SBATCH --error=logs/slurm/transfer_fixalloc_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-7
# Per-array-task slurm script for the transfer-fixalloc queues.
# Each array worker pulls from $QUEUE_DIR and runs evaluate_reward_model.py.
#
# Usage (set QUEUE_DIR + optional overrides; do NOT call directly — go through
# scripts/launch_transfer_fixalloc.sh):
#
#   QUEUE_DIR=experiments/transfer_fixalloc_acrobot_v1 \
#   WANDB_PROJECT=transfer-fixalloc-acrobot \
#       sbatch --array=0-31 scripts/submit_euler_transfer_fixalloc.sh
set -e

echo "=============================================="
echo "Transfer-fixalloc worker"
echo "=============================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node:          $(hostname)"
echo "Date:          $(date)"
echo "Working dir:   $SLURM_SUBMIT_DIR"
echo "=============================================="

cd "$SLURM_SUBMIT_DIR"

# Modules (mirrors submit_euler.sh).
module load stack/2024-06
module load python/3.11.6 glew/2.2.0
module load eth_proxy

# Activate env if available.
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "../venv/bin/activate" ]; then
    source ../venv/bin/activate
else
    echo "Warning: no venv found, using system Python."
fi

export PYTHONNOUSERSITE=1
echo "Python: $(which python)  $(python --version)"

mkdir -p logs/slurm

QUEUE_DIR="${QUEUE_DIR:?QUEUE_DIR must be set}"

# wandb is configured per-experiment from the config dict (each cfg sets
# wandb_project + log_wandb=True). evaluation_worker takes no wandb flags.
echo "Queue: $QUEUE_DIR"
echo "=============================================="

python -m mavrl_experiments.evaluation_worker --queue-dir "$QUEUE_DIR"

echo "=============================================="
echo "Worker exited at $(date)"
echo "=============================================="
