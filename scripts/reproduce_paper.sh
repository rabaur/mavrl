#!/usr/bin/env bash
# Single-entrypoint regeneration of every table and figure in the paper.
#
# Reads from the in-repo `results/` artifact bundle; does not
# require Optuna journals or any cluster access. Each step is guarded by an
# existence check so the script is idempotent: delete a target file to force
# its step to re-run.
#
# Usage:
#   conda activate mavrl-env
#   bash scripts/reproduce_paper.sh
#
# Flags:
#   --force       re-run all steps even if outputs exist
#
# Outputs land under:
#   results/tables/*.tex   — rendered LaTeX tables
#   figures/*.{png,pdf}    — rendered figures
set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

CR_ROOT="results"
TABLES_DIR="$CR_ROOT/tables"
FIGURES_DIR="$CR_ROOT/figures"
mkdir -p "$TABLES_DIR" "$FIGURES_DIR"

step() {
    local target="$1"; shift
    if [ "$FORCE" = "1" ] || [ ! -e "$target" ]; then
        echo "▶ $* → $target"
        "$@"
    else
        echo "✓ $target (exists; pass --force to rebuild)"
    fi
}

# ─── Tables ──────────────────────────────────────────────────────────────────

step "$TABLES_DIR/fixed_allocation.tex" \
    python -m mavrl_experiments.fixed_allocation_table \
        --camera-ready-root "$CR_ROOT" \
        --variant paper \
        --combined \
        --epic-json "$CR_ROOT/epic/epic_fixalloc_paper.json" \
        --latex --latex-out "$TABLES_DIR/fixed_allocation.tex" \
        --color gray

step "$TABLES_DIR/equal_budget.tex" \
    python -m mavrl_experiments.equal_budget_table \
        --camera-ready-root "$CR_ROOT" \
        --combined --stderr \
        --epic-json "$CR_ROOT/epic/epic_eqbudget_paper.json" \
        --latex --latex-out "$TABLES_DIR/equal_budget.tex" \
        --color gray

step "$TABLES_DIR/misspec.tex" \
    python scripts/summarize_misspec.py \
        --tex "$TABLES_DIR/misspec.tex"

# Post-Hoc Avg row of tab:baselines: prints to stdout (parse + splice
# into the composite tab:baselines LaTeX by hand for now; full glue
# script pending — see plan A4).
echo "▶ tab:baselines Post-Hoc Avg row (stdout only)"
python scripts/baseline_comparison_table.py

# MCMC row of tab:baselines: numbers live under results/mcmc/
# (see FeedbackInformativeness/scripts/merge_grid_mcmc_results.jl). The
# headline numbers are stable across reruns; auto-aggregation pending —
# read directly from results/mcmc/*_combined_*.json.
echo "▶ tab:baselines MCMC row → results/mcmc/*_combined_*.json (manual)"

# ─── Figures ─────────────────────────────────────────────────────────────────

step "$FIGURES_DIR/transfer_combined.pdf" \
    python scripts/plot_combined_transfer.py

step "$FIGURES_DIR/transfer_lander_full.pdf" \
    python scripts/plot_individual_transfer.py

# Qualitative reward heatmaps (appendix). grid_trap regenerates from the
# in-tree checkpoint bundle. grid_cliff and grid_sparse are pending
# Yannick's bundle and ship as committed PNGs under
# $FIGURES_DIR/appendix_grids/.
QUAL_CFG_DIR="$CR_ROOT/qualitative_configs"

step "$FIGURES_DIR/appendix_grids/final-fig_grid-trap.png" \
    python scripts/visualize_checkpoint_comparison.py \
        --config "$QUAL_CFG_DIR/grid_trap.json" \
        --no_minigrid_background \
        --output "$FIGURES_DIR/appendix_grids/final-fig_grid-trap.png"

for env in grid_cliff grid_sparse; do
    cfg="$QUAL_CFG_DIR/${env}.json"
    if [ -f "$cfg" ]; then
        step "$FIGURES_DIR/appendix_grids/final-fig_${env//_/-}.png" \
            python scripts/visualize_checkpoint_comparison.py \
                --config "$cfg" \
                --no_minigrid_background \
                --output "$FIGURES_DIR/appendix_grids/final-fig_${env//_/-}.png"
    else
        echo "⏸  qualitative $env: pending Yannick's bundle (config $cfg missing)"
    fi
done

echo ""
echo "✓ Done. Inspect:"
echo "    $TABLES_DIR/*.tex"
echo "    $FIGURES_DIR/transfer_*.{png,pdf}"
echo "    $FIGURES_DIR/appendix_grids/*.png"
