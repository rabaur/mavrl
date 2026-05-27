#!/bin/bash

# =============================================================================
# Launch all Optuna studies for the equal-budget table.
#
# Mirrors the 11 columns of the original (unequal-budget) main table, but at
# a single per-environment cumulative feedback budget. The Dirichlet allocation
# in optuna_search.py distributes that budget across the active modalities.
#
# Layout:
#   - 6 environments × 11 modality subsets = 66 studies
#   - Each study uses one job array of N workers sharing a JournalFileStorage
#
# Subsets (study suffix → --modalities):
#   pref         → pref
#   demo         → demo
#   rating       → rating
#   stop         → stop
#   demo_pref    → demo,pref
#   pref_rating  → pref,rating
#   pref_stop    → pref,stop
#   demo_rating  → demo,rating
#   demo_stop    → demo,stop
#   rating_stop  → rating,stop
#   pdrs         → pref,demo,rating,stop
#
# Per-environment budget / array width / metric (defaults):
#   grid_cliff       :  64,  2 workers, eval/discounted_value
#   grid_sparse      :  64,  2 workers, eval/discounted_value
#   grid_trap        :  64,  2 workers, eval/discounted_value
#   acrobot_v1       :  64, 16 workers, eval/mean_rew
#   cartpole_v1      :  64, 16 workers, eval/mean_rew
#   lunar_lander_v3  : 256, 30 workers, eval/mean_rew
#
# Override budgets per env group (the metric and array width stay fixed):
#   BUDGET_GRID=128    bash scripts/launch_equal_budget_table.sh   # grid_*
#   BUDGET_CONTROL=256 bash scripts/launch_equal_budget_table.sh   # acrobot/cartpole
#   BUDGET_LANDER=512  bash scripts/launch_equal_budget_table.sh   # lunar_lander_v3
#
# NOTE: Larger budgets need pre-generated dataset caches sized to match.
# See README "Pre-generate datasets" — re-run pregenerate_datasets.py with
# --gen_samples >= the new budget for the affected envs (and --gen_samples_demo
# >= the demo cap from configs/optuna/<env>.py:MODALITY_MAX_SAMPLES).
#
# Filtering / dry-run:
#   ENVS="grid_trap acrobot_v1"  bash scripts/launch_equal_budget_table.sh
#   SUBSETS="pdrs pref"          bash scripts/launch_equal_budget_table.sh
#   DRY_RUN=1                    bash scripts/launch_equal_budget_table.sh
# =============================================================================

set -e

# ─── Defaults shared across all envs ─────────────────────────────────────────
N_SEEDS=${N_SEEDS:-10}
DIRECTION=${DIRECTION:-"maximize"}
SLURM_TIME=${SLURM_TIME:-"24:00:00"}
SLURM_CPUS=${SLURM_CPUS:-4}
STORAGE_ROOT=${STORAGE_ROOT:-${SCRATCH}/mavrl/optuna_studies}
DRY_RUN=${DRY_RUN:-0}

# Per-env-group budgets (override via env var; defaults match the original
# equal-budget table). Study names embed the budget (e.g. grid_cliff_pref_b128),
# so different budgets coexist as separate studies under the same env subdir.
BUDGET_GRID=${BUDGET_GRID:-64}        # grid_cliff, grid_sparse, grid_trap
BUDGET_CONTROL=${BUDGET_CONTROL:-64} # acrobot_v1, cartpole_v1
BUDGET_LANDER=${BUDGET_LANDER:-256}   # lunar_lander_v3

ALL_ENVS=(grid_cliff grid_sparse grid_trap acrobot_v1 cartpole_v1 lunar_lander_v3)
ALL_SUBSETS=(pref demo rating stop demo_pref pref_rating pref_stop demo_rating demo_stop rating_stop pdrs)

# ─── Subset suffix → comma-separated --modalities argument ───────────────────
modalities_for() {
    case "$1" in
        pref)         echo "pref" ;;
        demo)         echo "demo" ;;
        rating)       echo "rating" ;;
        stop)         echo "stop" ;;
        demo_pref)    echo "demo,pref" ;;
        pref_rating)  echo "pref,rating" ;;
        pref_stop)    echo "pref,stop" ;;
        demo_rating)  echo "demo,rating" ;;
        demo_stop)    echo "demo,stop" ;;
        rating_stop)  echo "rating,stop" ;;
        pdrs)         echo "pref,demo,rating,stop" ;;
        *)            echo "" ;;
    esac
}

# ─── Per-environment settings ────────────────────────────────────────────────
# Echoes: BUDGET WORKERS METRIC MEM N_TRIALS DATASET_CACHE_DIR
env_settings() {
    local cache_dir="${HOME}/mavrl/dataset_cache/$1"
    case "$1" in
        grid_cliff|grid_sparse|grid_trap)
            echo "${BUDGET_GRID} 2 eval/discounted_value 2G 125 ${cache_dir}"
            ;;
        acrobot_v1|cartpole_v1)
            echo "${BUDGET_CONTROL} 16 eval/mean_rew 4G 200 ${cache_dir}"
            ;;
        lunar_lander_v3)
            echo "${BUDGET_LANDER} 30 eval/mean_rew 4G 200 ${cache_dir}"
            ;;
        *)
            echo ""
            ;;
    esac
}

# ─── Resolve filters ─────────────────────────────────────────────────────────
if [ -n "$ENVS" ]; then
    read -r -a SELECTED_ENVS <<< "$ENVS"
else
    SELECTED_ENVS=("${ALL_ENVS[@]}")
fi

if [ -n "$SUBSETS" ]; then
    read -r -a SELECTED_SUBSETS <<< "$SUBSETS"
else
    SELECTED_SUBSETS=("${ALL_SUBSETS[@]}")
fi

echo "=============================================="
echo "Equal-budget table launcher"
echo "=============================================="
echo "Envs    : ${SELECTED_ENVS[*]}"
echo "Subsets : ${SELECTED_SUBSETS[*]}"
echo "Seeds   : $N_SEEDS"
echo "Budgets : grid=${BUDGET_GRID}, control=${BUDGET_CONTROL}, lander=${BUDGET_LANDER}"
echo "Storage : $STORAGE_ROOT"
echo "DRY_RUN : $DRY_RUN"
echo "=============================================="
echo ""

mkdir -p logs/slurm

COUNT=0

submit_study() {
    local env_config="$1"
    local study_name="$2"
    local budget="$3"
    local modalities="$4"
    local workers="$5"
    local metric="$6"
    local mem="$7"
    local n_trials="$8"
    local dataset_cache_dir="$9"

    local storage_dir="${STORAGE_ROOT}/${env_config}"
    local storage_path="${storage_dir}/${study_name}.log"
    mkdir -p "$storage_dir"

    local array_flag="--array=0-$((workers - 1))"

    # MODALITIES contains commas for multi-modality subsets, which can't be
    # passed inside sbatch's `--export=KEY=VAL,KEY=VAL,...` (commas separate
    # entries). Instead, set env vars in this shell and use --export=ALL so
    # sbatch propagates them to the worker.
    echo "  [${env_config}] ${study_name} (budget=${budget}, modalities=${modalities}, workers=${workers})"

    if [ "$DRY_RUN" = "1" ]; then
        echo "    DRY_RUN: STUDY_NAME=$study_name ENV_CONFIG=$env_config BUDGET=$budget MODALITIES=$modalities STORAGE_PATH=$storage_path N_SEEDS=$N_SEEDS N_TRIALS=$n_trials METRIC=$metric DIRECTION=$DIRECTION DATASET_CACHE_DIR=${dataset_cache_dir} sbatch --job-name=opt_${study_name} --time=$SLURM_TIME --mem-per-cpu=$mem --cpus-per-task=$SLURM_CPUS $array_flag --export=ALL scripts/submit_optuna.sh"
    else
        STUDY_NAME="$study_name" \
        ENV_CONFIG="$env_config" \
        BUDGET="$budget" \
        MODALITIES="$modalities" \
        STORAGE_PATH="$storage_path" \
        N_SEEDS="$N_SEEDS" \
        N_TRIALS="$n_trials" \
        METRIC="$metric" \
        DIRECTION="$DIRECTION" \
        DATASET_CACHE_DIR="$dataset_cache_dir" \
            sbatch \
                --job-name="opt_${study_name}" \
                --time="$SLURM_TIME" \
                --mem-per-cpu="$mem" \
                --cpus-per-task="$SLURM_CPUS" \
                "$array_flag" \
                --export=ALL \
                scripts/submit_optuna.sh
    fi

    COUNT=$((COUNT + 1))
}

for env_config in "${SELECTED_ENVS[@]}"; do
    settings=$(env_settings "$env_config")
    if [ -z "$settings" ]; then
        echo "WARNING: unknown env '$env_config', skipping"
        continue
    fi
    # shellcheck disable=SC2086
    set -- $settings
    budget="$1"; workers="$2"; metric="$3"; mem="$4"; n_trials="$5"; dataset_cache_dir="${6:-}"

    echo "Environment: $env_config (budget=$budget, workers=$workers, metric=$metric)"

    for subset in "${SELECTED_SUBSETS[@]}"; do
        modalities=$(modalities_for "$subset")
        if [ -z "$modalities" ]; then
            echo "  WARNING: unknown subset '$subset', skipping"
            continue
        fi
        study_name="${env_config}_${subset}_b${budget}"
        submit_study "$env_config" "$study_name" "$budget" "$modalities" \
                     "$workers" "$metric" "$mem" "$n_trials" "$dataset_cache_dir"
    done
    echo ""
done

echo "=============================================="
echo "Submitted $COUNT studies total."
echo "Storage root: $STORAGE_ROOT"
echo "=============================================="
