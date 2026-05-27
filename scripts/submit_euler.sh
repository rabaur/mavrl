#!/bin/bash

#SBATCH --job-name=mavrl
#SBATCH --output=logs/slurm/mavrl_%j_%a.out
#SBATCH --error=logs/slurm/mavrl_%j_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-127
#SBATCH --export=ALL

# =============================================================================
# UMFAVI Distributed Experiment Runner - Euler Submission Script
# =============================================================================
#
# This script submits a job array to the Euler cluster where each task runs
# as an independent worker processing experiments from the file-based queue.
#
# Usage:
#   1. First, populate the queue with experiments:
#      python -m mavrl_experiments.cli --queue-dir tasks add-grid feedback_mix --seeds 5
#
#   2. Check the queue status:
#      python -m mavrl_experiments.cli --queue-dir tasks status
#
#   3. Submit to Euler:
#      sbatch scripts/submit_euler.sh
#
#   4. To run fewer workers, override the array:
#      sbatch --array=0-31 scripts/submit_euler.sh
#
#   5. Override wandb project (decouples wandb project from experiment grid config):
#      WANDB_PROJECT=my_new_project sbatch scripts/submit_euler.sh
#
#   6. Override both queue directory and wandb project:
#      QUEUE_DIR=tasks_v2 WANDB_PROJECT=my_new_project sbatch --array=0-127 scripts/submit_euler.sh
#
# Configuration:
#   - Adjust --time based on expected experiment duration
#   - Adjust --mem-per-cpu based on model size
#   - Adjust --cpus-per-task for environments that benefit from parallelism
#   - The array range (0-127) determines the number of parallel workers
#
# Environment Variables:
#   - QUEUE_DIR: Path to the task queue directory (default: tasks)
#   - WANDB_PROJECT: Override wandb project name (default: from experiment config)
#   - WANDB_LOG_DIR: Override wandb log directory (default: $SCRATCH/mavrl/wandb)
#
# Output Directories:
#   Models and policies are saved to $SCRATCH/mavrl/<wandb_project>_<date>/:
#   - models/<experiment_id>/best_model.pt     - Best reward model (PyTorch state dict)
#   - policies/<experiment_id>/best_policy.zip - Best estimated policy (SB3 PPO model)
#
# =============================================================================

# Exit on error
set -e

# Print job information
echo "=============================================="
echo "UMFAVI Experiment Worker"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Working Directory: $SLURM_SUBMIT_DIR"
echo "=============================================="

# Change to the project directory
cd $SLURM_SUBMIT_DIR

# Load required modules (adjust based on your Euler setup)
module load stack/2024-06
module load python/3.11.6 glew/2.2.0
module load eth_proxy  # For wandb access
# module load cuda/12.1.1  # If using GPU

# Activate virtual environment (adjust path as needed)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "../venv/bin/activate" ]; then
    source ../venv/bin/activate
else
    echo "Warning: No virtual environment found. Using system Python."
fi

export PYTHONNOUSERSITE=1

# Print Python info
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Ensure logs directory exists
mkdir -p logs/slurm

# Set environment variables
# export WANDB_MODE="${WANDB_MODE:-online}"

# Queue directory - use a shared location accessible to all nodes
# On Euler, this should be on the shared filesystem (e.g., $HOME or project space)
# The file-based queue uses atomic rename operations which are NFS-safe
QUEUE_DIR="${QUEUE_DIR:-tasks}"

# Wandb settings - can be overridden to decouple from experiment grid config
# This allows rerunning the same experiment configs with different wandb projects
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_LOG_DIR="${WANDB_LOG_DIR:-$SCRATCH/mavrl/wandb}"

echo "Queue directory: $QUEUE_DIR"
echo "Run directory: \$SCRATCH/mavrl/<wandb_project>_<date>/"
echo "  Models: \$SCRATCH/mavrl/<wandb_project>_<date>/models/<experiment_id>/best_model.pt"
echo "  Policies: \$SCRATCH/mavrl/<wandb_project>_<date>/policies/<experiment_id>/best_policy.zip"
echo "Wandb project: ${WANDB_PROJECT:-<from experiment config>}"
echo "Wandb log directory: $WANDB_LOG_DIR"
echo "=============================================="

# Build worker arguments
WORKER_ARGS="--queue-dir $QUEUE_DIR"
if [ -n "$WANDB_PROJECT" ]; then
    WORKER_ARGS="$WORKER_ARGS --wandb-project $WANDB_PROJECT"
fi
if [ -n "$WANDB_LOG_DIR" ]; then
    WORKER_ARGS="$WORKER_ARGS --wandb-log-dir $WANDB_LOG_DIR"
fi

# Run the worker
# The worker will:
# 1. Claim pending experiments from the queue (via atomic file rename)
# 2. Run each experiment with wandb logging
# 3. Save results to the queue
# 4. Exit when no more pending experiments
python -m mavrl_experiments.worker $WORKER_ARGS

echo "=============================================="
echo "Worker completed at $(date)"
echo "=============================================="

