#!/bin/bash
# Top-level launcher for the transfer-fixalloc queues.
#
# Modes:
#   bash scripts/launch_transfer_fixalloc.sh status               # queue depths
#   bash scripts/launch_transfer_fixalloc.sh local                # one worker per env, in background
#   bash scripts/launch_transfer_fixalloc.sh local-serial         # all envs sequentially, one worker
#   bash scripts/launch_transfer_fixalloc.sh euler [N]            # sbatch array per env (default N=8)
#
# Env vars:
#   ENVS         override which envs to process (default: all four)
#   ARRAY_TIME   slurm --time per array task (default 24:00:00)
#   ARRAY_MEM    slurm --mem-per-cpu (default 4G)
#   ARRAY_CPUS   slurm --cpus-per-task (default 4)
#
# Recommended per-env array widths (override the 2nd arg or set in script):
#   grid_cliff  ~8    (tabular, fast)
#   grid_trap   ~8
#   acrobot_v1  ~32   (PPO ~10-15 min/run)
#   lander      ~64   (PPO ~20-30 min/run)
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_DIR"

ENVS=${ENVS:-"grid_cliff grid_trap acrobot_v1 lunar_lander_v3"}
MODE=${1:-status}
ARRAY_WIDTH_DEFAULT=${2:-8}
ARRAY_TIME=${ARRAY_TIME:-24:00:00}
ARRAY_MEM=${ARRAY_MEM:-4G}
ARRAY_CPUS=${ARRAY_CPUS:-4}

per_env_width() {
    case "$1" in
        grid_cliff)       echo 8 ;;
        grid_trap)        echo 8 ;;
        acrobot_v1)       echo 32 ;;
        lunar_lander_v3)  echo 64 ;;
        *)                echo "$ARRAY_WIDTH_DEFAULT" ;;
    esac
}

queue_for() { echo "experiments/transfer_fixalloc_$1"; }

run_worker_local() {
    local env="$1"
    local q
    q="$(queue_for "$env")"
    echo "[local] $env  →  $q"
    python -m mavrl_experiments.evaluation_worker --queue-dir "$q"
}

submit_euler() {
    local env="$1"
    local q
    q="$(queue_for "$env")"
    local width
    width="$(per_env_width "$env")"
    local upper=$((width - 1))
    echo "[euler] $env  width=$width  $q"
    QUEUE_DIR="$q" sbatch \
        --array=0-"$upper" \
        --job-name="transfer_fixalloc_$env" \
        --time="$ARRAY_TIME" \
        --cpus-per-task="$ARRAY_CPUS" \
        --mem-per-cpu="$ARRAY_MEM" \
        -o "logs/slurm/transfer_fixalloc_${env}_%A_%a.out" \
        -e "logs/slurm/transfer_fixalloc_${env}_%A_%a.err" \
        scripts/submit_euler_transfer_fixalloc.sh
}

print_status() {
    local env="$1"
    local q
    q="$(queue_for "$env")"
    if [ ! -d "$q" ]; then
        echo "$env: queue $q does not exist (run enqueue_transfer.py first)"
        return
    fi
    python -m mavrl_experiments.cli --queue-dir "$q" status || true
}

case "$MODE" in
    local)
        mkdir -p logs/transfer_fixalloc
        for env in $ENVS; do
            log="logs/transfer_fixalloc/${env}.log"
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
