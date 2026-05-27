#!/bin/bash

#SBATCH --job-name=optuna
#SBATCH --output=logs/slurm/optuna_%j_%a.out
#SBATCH --error=logs/slurm/optuna_%j_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-31
#SBATCH --export=ALL

# =============================================================================
# Optuna Budget-Matched Feedback Allocation Search - Cluster Submission
# =============================================================================
#
# Each array task runs as an independent Optuna worker. All workers share a
# single study via JournalFileStorage (append-only, NFS-safe).
#
# Top-3 trial model checkpoints (one per seed) are persisted under
# $STORAGE_PATH.models/trial_<n>/seed_<i>/best_model.pt.
#
# Required environment variables:
#   STUDY_NAME    - Optuna study name (e.g. grid_sparse_b64)
#   ENV_CONFIG    - Config module name (e.g. grid_sparse)
#   BUDGET        - Maximum total feedback samples
#   STORAGE_PATH  - Path to journal file (e.g. optuna_journal_b64.log)
#
# Optional environment variables:
#   N_SEEDS       - Seeds per trial (default: 3)
#   N_TRIALS      - Trials per worker (default: 20)
#   METRIC        - Metric to optimize (default: eval/regret)
#   DIRECTION     - minimize or maximize (default: minimize)
#   MODALITIES    - Comma-separated subset of modalities to allocate budget across
#                   (e.g. "pref,demo" or "pref"). Default: all four.
#   FIXED_ALLOCATION - If "1", use prescribed per-modality counts from the env
#                   config's FIXED_SAMPLE_COUNTS (skips Dirichlet). BUDGET is
#                   then derived and may be omitted.
#   WANDB_PROJECT - Log trials to wandb (default: disabled)
#
# Usage:
#   # Basic submission
#   STUDY_NAME=grid_sparse_b64 BUDGET=64 ENV_CONFIG=grid_sparse \
#       STORAGE_PATH=optuna_journal_b64.log \
#       sbatch scripts/submit_optuna.sh
#
#   # With fewer workers and custom settings
#   STUDY_NAME=grid_sparse_b128 BUDGET=128 ENV_CONFIG=grid_sparse \
#       STORAGE_PATH=optuna_journal_b128.log N_SEEDS=5 N_TRIALS=30 \
#       sbatch --array=0-15 scripts/submit_optuna.sh
#
#   # Inspect results after completion
#   python -m mavrl_experiments.optuna_search \
#       --study-name grid_sparse_b64 \
#       --storage optuna_journal_b64.log \
#       --show-results
# =============================================================================

set -e

echo "=============================================="
echo "Optuna Budget Search Worker"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Working Directory: $SLURM_SUBMIT_DIR"
echo "=============================================="

cd $SLURM_SUBMIT_DIR

# Load required modules
module load stack/2024-06
module load python/3.11.6 glew/2.2.0
module load eth_proxy

# Activate virtual environment
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

# Validate required variables. BUDGET is optional when FIXED_ALLOCATION=1
# (it's derived from the env config's FIXED_SAMPLE_COUNTS).
if [ -z "$STUDY_NAME" ] || [ -z "$ENV_CONFIG" ] || [ -z "$STORAGE_PATH" ]; then
    echo "Error: STUDY_NAME, ENV_CONFIG, and STORAGE_PATH must be set."
    exit 1
fi
if [ -z "$BUDGET" ] && [ "$FIXED_ALLOCATION" != "1" ]; then
    echo "Error: BUDGET must be set (or pass FIXED_ALLOCATION=1 to derive it)."
    exit 1
fi

# Defaults for optional variables
N_SEEDS="${N_SEEDS:-3}"
N_TRIALS="${N_TRIALS:-20}"
METRIC="${METRIC:-eval/regret}"
DIRECTION="${DIRECTION:-minimize}"

echo "Study: $STUDY_NAME"
echo "Env config: $ENV_CONFIG"
echo "Budget: ${BUDGET:-(derived from FIXED_SAMPLE_COUNTS)}"
echo "Storage: $STORAGE_PATH"
echo "Seeds/trial: $N_SEEDS, Trials/worker: $N_TRIALS"
echo "Metric: $METRIC ($DIRECTION)"
if [ -n "$MODALITIES" ]; then
    echo "Modalities: $MODALITIES"
fi
if [ "$FIXED_ALLOCATION" = "1" ]; then
    echo "Fixed allocation: enabled (FIXED_SAMPLE_COUNTS from env config)"
fi
echo "=============================================="

WORKER_ARGS="--study-name $STUDY_NAME --storage $STORAGE_PATH"
WORKER_ARGS="$WORKER_ARGS --env-config $ENV_CONFIG"
if [ -n "$BUDGET" ]; then
    WORKER_ARGS="$WORKER_ARGS --budget $BUDGET"
fi
WORKER_ARGS="$WORKER_ARGS --n-seeds $N_SEEDS --n-trials $N_TRIALS"
WORKER_ARGS="$WORKER_ARGS --metric $METRIC --direction $DIRECTION"

if [ -n "$MODALITIES" ]; then
    WORKER_ARGS="$WORKER_ARGS --modalities $MODALITIES"
fi

if [ "$FIXED_ALLOCATION" = "1" ]; then
    WORKER_ARGS="$WORKER_ARGS --fixed-allocation"
fi

if [ -n "$WANDB_PROJECT" ]; then
    WORKER_ARGS="$WORKER_ARGS --wandb-project $WANDB_PROJECT"
fi

if [ -n "$DATASET_CACHE_DIR" ]; then
    WORKER_ARGS="$WORKER_ARGS --dataset-cache-dir $DATASET_CACHE_DIR"
fi

python -m mavrl_experiments.optuna_search $WORKER_ARGS

echo "=============================================="
echo "Worker completed at $(date)"
echo "=============================================="
