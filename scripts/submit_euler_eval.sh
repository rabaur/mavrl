#!/bin/bash

#SBATCH --job-name=mavrl-eval
#SBATCH --output=logs/slurm/submit_%j/mavrl_eval_%j_%a.out
#SBATCH --error=logs/slurm/submit_%j/mavrl_eval_%j_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-32
#SBATCH --export=ALL

# =============================================================================
# MAVRL Evaluation Experiment Runner - Euler Submission Script
# =============================================================================
#
# Runs reward model evaluation experiments (MAVRL, average ensemble,
# ground-truth) from a file-based queue.  Each array task acts as an
# independent worker that claims and processes experiments until the queue
# is empty.
#
# Usage:
#   1. Populate the evaluation queue:
#      python -m mavrl_experiments.cli --queue-dir tasks_eval_baseline \
#          add-grid eval_lander_baseline --seeds 10
#
#   2. Check status:
#      python -m mavrl_experiments.cli --queue-dir tasks_eval_baseline status
#
#   3. Submit:
#      sbatch scripts/submit_euler_eval.sh
#
#   4. Override array size:
#      sbatch --array=0-15 scripts/submit_euler_eval.sh
#
#   5. Override queue directory:
#      QUEUE_DIR=tasks_eval_v2 sbatch scripts/submit_euler_eval.sh
#
# =============================================================================

set -e

echo "=============================================="
echo "MAVRL Evaluation Worker"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Working Directory: $SLURM_SUBMIT_DIR"
echo "=============================================="

cd $SLURM_SUBMIT_DIR

module load stack/2024-06
module load python/3.11.6
module load eth_proxy  # For wandb access

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "../venv/bin/activate" ]; then
    source ../venv/bin/activate
else
    echo "Warning: No virtual environment found. Using system Python."
fi

export PYTHONNOUSERSITE=1

echo "Python: $(which python)"
echo "Python version: $(python --version)"

mkdir -p logs/slurm

QUEUE_DIR="${QUEUE_DIR:-tasks_eval_baseline}"

echo "Queue directory: $QUEUE_DIR"
echo "=============================================="

python -m mavrl_experiments.evaluation_worker --queue-dir $QUEUE_DIR

echo "=============================================="
echo "Evaluation worker completed at $(date)"
echo "=============================================="
