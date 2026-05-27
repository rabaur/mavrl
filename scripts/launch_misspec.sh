#!/bin/bash
# Run / submit all misspecification experiments enqueued by
# scripts/enqueue_misspec.py. Three queues, one per grid env.
#
# Modes:
#   bash scripts/launch_misspec.sh local           # one worker per queue in background
#   bash scripts/launch_misspec.sh local-serial    # one worker, all queues sequentially
#   bash scripts/launch_misspec.sh euler [N]       # sbatch array of N workers per queue (default 4)
#   bash scripts/launch_misspec.sh status          # print queue status for each env
#
# Env vars:
#   ENVS         override which grids to process  (default: cliff sparse trap)
#   WANDB_PROJECT  overrides the wandb project written into configs
#   EXTRA_WORKER_ARGS  appended to the python -m mavrl_experiments.worker invocation
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_DIR"

ENVS=${ENVS:-"grid_cliff grid_sparse grid_trap"}
MODE=${1:-status}
ARRAY_WIDTH=${2:-4}
WANDB_PROJECT=${WANDB_PROJECT:-mavrl-misspec}
EXTRA_WORKER_ARGS=${EXTRA_WORKER_ARGS:-}

queue_for() { echo "experiments/misspec_$1"; }

run_worker_local() {
    local env="$1"
    local q
    q="$(queue_for "$env")"
    echo "[local] worker on $q"
    python -m mavrl_experiments.worker \
        --queue-dir "$q" \
        --wandb-project "$WANDB_PROJECT" \
        $EXTRA_WORKER_ARGS
}

submit_euler() {
    local env="$1"
    local q
    q="$(queue_for "$env")"
    local upper=$((ARRAY_WIDTH - 1))
    echo "[euler] sbatch --array=0-$upper QUEUE_DIR=$q  WANDB_PROJECT=$WANDB_PROJECT"
    QUEUE_DIR="$q" WANDB_PROJECT="$WANDB_PROJECT" \
        sbatch --array=0-"$upper" \
            --job-name="misspec_$env" \
            -o "logs/slurm/misspec_${env}_%A_%a.out" \
            scripts/submit_euler.sh
}

print_status() {
    local env="$1"
    local q
    q="$(queue_for "$env")"
    if [ ! -d "$q" ]; then
        echo "$env: queue $q does not exist (run enqueue_misspec.py first)"
        return
    fi
    python -m mavrl_experiments.cli --queue-dir "$q" status || true
}

case "$MODE" in
    local)
        mkdir -p logs/misspec
        for env in $ENVS; do
            log="logs/misspec/${env}.log"
            echo "[local] launching $env worker → $log"
            ( run_worker_local "$env" ) >"$log" 2>&1 &
        done
        wait
        echo "All local workers exited."
        ;;
    local-serial)
        for env in $ENVS; do run_worker_local "$env"; done
        ;;
    euler)
        mkdir -p logs/slurm
        for env in $ENVS; do submit_euler "$env"; done
        ;;
    status)
        for env in $ENVS; do
            echo "=== $env ==="
            print_status "$env"
        done
        ;;
    *)
        echo "Usage: $0 {local|local-serial|euler [N]|status}"
        exit 2
        ;;
esac
