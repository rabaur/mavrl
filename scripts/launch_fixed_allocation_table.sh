#!/bin/bash

# =============================================================================
# Launch Optuna studies for the fixed-allocation table.
#
# Companion to ``launch_equal_budget_table.sh``. The two cover complementary
# stories:
#   - launch_equal_budget_table.sh    : equal-budget Dirichlet allocation
#   - launch_fixed_allocation_table.sh: prescribed (offline) per-modality counts
#
# Each (env, modality-subset) study uses the corresponding ``<env>_fixed.py``
# Optuna config: per-modality sample counts come from FIXED_SAMPLE_COUNTS,
# and Optuna searches td_error_weight, kl_weight, encoder_hidden_sizes,
# use_importance_weights, lr, batch_size (and the PPO retraining hparams for
# non-tabular envs).
#
# Default layout: 6 environments × 11 modality subsets = 66 studies.
# All cells (singletons, pairs, PDRS) are launched against the same env config
# and storage dir; the table builder reads every cell from the same root with
# study suffix <subset>_fixed[_<variant>].
#
# Default subsets (study suffix → --modalities):
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
# Filtering / dry-run:
#   ENVS="grid_trap acrobot_v1"  bash scripts/launch_fixed_allocation_table.sh
#   SUBSETS="pdrs pref"          bash scripts/launch_fixed_allocation_table.sh
#   DRY_RUN=1                    bash scripts/launch_fixed_allocation_table.sh
#
# Allocation variant:
#   Uses configs/optuna/<env>_fixed_paper.py (the paper's main-text per-env
#   budgets); study names follow the suffix `_fixed_paper`. Set VARIANT=<name>
#   to point at configs/optuna/<env>_fixed_<name>.py for a custom run.
# =============================================================================

set -e

# ─── Defaults shared across all envs ─────────────────────────────────────────
N_SEEDS=${N_SEEDS:-10}
DIRECTION=${DIRECTION:-"maximize"}
SLURM_TIME=${SLURM_TIME:-"24:00:00"}
SLURM_CPUS=${SLURM_CPUS:-4}
STORAGE_ROOT=${STORAGE_ROOT:-${SCRATCH}/mavrl/optuna_studies}
DRY_RUN=${DRY_RUN:-0}
VARIANT=${VARIANT:-paper}
VARIANT_SFX="_$VARIANT"

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
# Echoes: WORKERS METRIC MEM N_TRIALS DATASET_CACHE_DIR
# (Same shape as launch_equal_budget_table.sh, minus BUDGET — derived per-trial
# from each env's FIXED_SAMPLE_COUNTS.)
env_settings() {
    local cache_dir="${HOME}/mavrl/dataset_cache/$1"
    case "$1" in
        grid_cliff|grid_sparse|grid_trap)
            echo "2 eval/discounted_value 2G 125 ${cache_dir}"
            ;;
        acrobot_v1|cartpole_v1)
            echo "16 eval/mean_rew 4G 200 ${cache_dir}"
            ;;
        lunar_lander_v3)
            echo "30 eval/mean_rew 4G 200 ${cache_dir}"
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
echo "Fixed-allocation table launcher"
echo "=============================================="
echo "Envs    : ${SELECTED_ENVS[*]}"
echo "Subsets : ${SELECTED_SUBSETS[*]}"
echo "Seeds   : $N_SEEDS"
echo "Storage : $STORAGE_ROOT"
echo "Variant : ${VARIANT:-<default>} (config/study suffix: _fixed${VARIANT_SFX})"
echo "DRY_RUN : $DRY_RUN"
echo "=============================================="
echo ""

mkdir -p logs/slurm

COUNT=0

submit_study() {
    local env_config="$1"          # e.g. grid_trap_fixed
    local env_storage_dir="$2"     # storage dir keyed by base env (grid_trap)
    local study_name="$3"
    local modalities="$4"
    local workers="$5"
    local metric="$6"
    local mem="$7"
    local n_trials="$8"
    local dataset_cache_dir="$9"

    local storage_path="${env_storage_dir}/${study_name}.log"
    mkdir -p "$env_storage_dir"

    local array_flag="--array=0-$((workers - 1))"

    echo "  [${env_config}] ${study_name} (modalities=${modalities}, workers=${workers})"

    if [ "$DRY_RUN" = "1" ]; then
        echo "    DRY_RUN: STUDY_NAME=$study_name ENV_CONFIG=$env_config FIXED_ALLOCATION=1 MODALITIES=$modalities STORAGE_PATH=$storage_path N_SEEDS=$N_SEEDS N_TRIALS=$n_trials METRIC=$metric DIRECTION=$DIRECTION DATASET_CACHE_DIR=${dataset_cache_dir} sbatch --job-name=opt_${study_name} --time=$SLURM_TIME --mem-per-cpu=$mem --cpus-per-task=$SLURM_CPUS $array_flag --export=ALL scripts/submit_optuna.sh"
    else
        STUDY_NAME="$study_name" \
        ENV_CONFIG="$env_config" \
        FIXED_ALLOCATION=1 \
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
    workers="$1"; metric="$2"; mem="$3"; n_trials="$4"; dataset_cache_dir="${5:-}"

    fixed_env_config="${env_config}_fixed${VARIANT_SFX}"
    env_storage_dir="${STORAGE_ROOT}/${env_config}"

    echo "Environment: $env_config (config=$fixed_env_config, workers=$workers, metric=$metric)"

    for subset in "${SELECTED_SUBSETS[@]}"; do
        modalities=$(modalities_for "$subset")
        if [ -z "$modalities" ]; then
            echo "  WARNING: unknown subset '$subset', skipping"
            continue
        fi
        # The variant-aware suffix in the study name disambiguates from both
        # equal-budget studies and other fixed-allocation variants under the
        # same env storage dir.
        study_name="${env_config}_${subset}_fixed${VARIANT_SFX}"
        submit_study "$fixed_env_config" "$env_storage_dir" "$study_name" \
                     "$modalities" "$workers" "$metric" "$mem" "$n_trials" \
                     "$dataset_cache_dir"
    done
    echo ""
done

echo "=============================================="
echo "Submitted $COUNT studies total."
echo "Storage root: $STORAGE_ROOT"
echo "=============================================="
