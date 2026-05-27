#!/bin/bash
#SBATCH --job-name=grid-mcmc
#SBATCH --output=logs/slurm/grid_mcmc_%j_%a.out
#SBATCH --error=logs/slurm/grid_mcmc_%j_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-14
#SBATCH --export=ALL
#
# One array task = one (reward_type × feedback_type) cell from run_grid_experiments.jl.
# After all tasks finish, merge shards:
#   julia --project=FeedbackInformativeness FeedbackInformativeness/scripts/merge_grid_mcmc_results.jl \
#     FeedbackInformativeness/results/grid_mcmc_shards/*.json -o FeedbackInformativeness/results/grid_mcmc_merged.json
#
# Override array range if you change REWARD_TYPES × FEEDBACK_TYPES in the Julia script.
# Use NUM_CHAINS=1 per task to avoid nested Distributed workers inside each Slurm task.
#
# Usage (from repo root):
#   mkdir -p logs/slurm
#   sbatch FeedbackInformativeness/scripts/submit_grid_mcmc_euler.sh
#
# Optional env:
#   RESULTS_SUBDIR=grid_mcmc_shards  — shard directory under FeedbackInformativeness/results/
#   JULIA_PROJECT=FeedbackInformativeness  — project path (default below)
#   JULIA_FLAGS  — extra Julia CLI flags (default: -O3). Example: JULIA_FLAGS="-O3 -t4"

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-.}"
cd "${REPO_ROOT}"
FI_ROOT="${REPO_ROOT}/FeedbackInformativeness"

TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Run under Slurm or set SLURM_ARRAY_TASK_ID manually}"
OUT_SUBDIR="${RESULTS_SUBDIR:-grid_mcmc_shards}"
SHARD_DIR="${FI_ROOT}/results_combined/${OUT_SUBDIR}"
mkdir -p "${SHARD_DIR}" logs/slurm

JULIA_PROJ="${JULIA_PROJECT:-${FI_ROOT}}"
OUT_FILE="${SHARD_DIR}/shard_task_${TASK_INDEX}.json"

echo "task_index=${TASK_INDEX}  out=${OUT_FILE}  host=$(hostname)  date=$(date)"

module load stack/2024-06 2>/dev/null || true
module load julia 2>/dev/null || module load julia/1.10.0 2>/dev/null || true

export JULIA_PROJECT="${JULIA_PROJ}"
# Headless plotting backends sometimes needed on compute nodes:
export GKSwstype="${GKSwstype:-100}"

julia -O3 "${FI_ROOT}/scripts/run_grid_experiments.jl" \
  --task_index "${TASK_INDEX}" \
  --num_chains "${NUM_CHAINS:-4}" \
  --samples_per_chain "${SAMPLES_PER_CHAIN:-2500}" \
  --output "${OUT_FILE}"

echo "Done: ${OUT_FILE}"
