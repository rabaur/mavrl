<p align="center">
  <img src="results/figures/mavrl_color.png" alt="MAVRL" width="600">
</p>

# MAVRL — Multi-feedback Amortized Variational Reward Learning

Code release for the ICML 2026 camera-ready of *"MAVRL: Learning Reward
Functions from Multiple Feedback Types with Amortized Variational
Inference"*. Implements an amortized variational reward learner that
combines preference, demonstration, rating and stop feedback in a single
ELBO.

## Repository layout

- **`mavrl/`** — the algorithm: encoders, feedback likelihoods, losses,
  retraining utilities. `import mavrl`.
- **`mavrl_experiments/`** — experiment infrastructure: Optuna search,
  file queues, table printers, CLI. Depends on `mavrl`.
- **`scripts/`** — entrypoints. `reproduce_paper.sh`, `reproduce_run.py`,
  per-figure/table renderers, and SLURM launchers for re-running on a
  cluster.
- **`results/`** — the canonical artifact bundle: model
  checkpoints, hyperparameter logs, MCMC results, normalization
  constants, rendered tables, paper source, published figures.
- **`FeedbackInformativeness/`** — the Julia MCMC pipeline for the
  baseline-comparison.
- **`expert_policies/`** — pre-trained external policies used as
  Boltzmann-rational simulators (inputs, not our outputs; see the
  README inside).
- **`tests/`** — pytest suite for the algorithm + feedback models.

Top-level entry-point scripts (`train.py`,
`evaluate_reward_model.py`) live at the repo root.

## Installation

The algorithm package is available on PyPI:

```bash
pip install mavrl
```

To set up the full repo (experiments, scripts, tests):

```bash
conda env create -f environment.yml
conda activate mavrl-env
pip install -e .
```

## Reproducing the paper

All tables and figures regenerate from `results/`. No
Optuna journals, cluster access, or retraining required.

```bash
conda activate mavrl-env
bash scripts/reproduce_paper.sh
```

Each step is skipped if its output exists; pass `--force` to rebuild
everything.

| Paper element | Renderer | Output | Runtime |
|---|---|---|---|
| `tab:fixed_allocation_combined_paper` | `python -m mavrl_experiments.fixed_allocation_table --camera-ready-root results` | `results/tables/fixed_allocation.tex` | ~5 s |
| `tab:equal_budget_combined` | `python -m mavrl_experiments.equal_budget_table --camera-ready-root results` | `results/tables/equal_budget.tex` | ~5 s |
| `tab:misspec-excerpt`, `tab:app-full-misspec` | `python scripts/summarize_misspec.py --tex …` | `results/tables/misspec.tex` | ~1 s |
| `tab:baselines` (Post-Hoc Avg row) | `python scripts/baseline_comparison_table.py` | stdout (manual splice) | ~1 s |
| `tab:baselines` (MCMC row) | inspect `results/mcmc/*_combined_*.json` | manual | — |
| `fig:robustness-ood` | `python scripts/plot_combined_transfer.py` | `results/figures/transfer_combined.{png,pdf}` | ~10 s |
| `fig:appendix-transfer-*` | `python scripts/plot_individual_transfer.py` | `results/figures/transfer_*_full.{png,pdf}` | ~30 s |
| `fig:qualitative-comparison-app-grid-trap` | `python scripts/visualize_checkpoint_comparison.py --config results/qualitative_configs/grid_trap.json …` | `results/figures/appendix_grids/final-fig_grid-trap.png` | ~5 s |
| `fig:qualitative-comparison-app-grid-{cliff,sparse}` | committed PNG (regeneration deferred to a follow-up checkpoint bundle) | — | — |

## Reproducing a single training run

Each cell under
`results/models/{equal_budget,fixed_allocation}/<env>/<subset>/`
ships per-seed checkpoints plus a `hparams.json` with the full merged
training config. To re-train one seed:

```bash
# Grid env (~15 s):
python scripts/reproduce_run.py \
    --cell results/models/fixed_allocation/grid_trap/pdrs --seed 0

# Control env (~15 min):
python scripts/reproduce_run.py \
    --cell results/models/fixed_allocation/cartpole_v1/pdrs --seed 0

# Lunar Lander (~15 min):
python scripts/reproduce_run.py \
    --cell results/models/fixed_allocation/lunar_lander_v3/pdrs --seed 0
```

The script prints the produced final eval metric next to the published
`_meta.per_seed_values[seed]` so you can eyeball the match. Swap
`--cell` and `--seed` to cover any other cell.

### Bit-exact reproduction vs. from-scratch retraining

Loading a published `r_model_seed{i}.pt` and re-evaluating it
reproduces the per-seed metric **bit-identically on any machine** —
that's the canonical reproducibility path.

Re-training the same config from scratch (`reproduce_run.py`) produces
metrics in the same distribution but may bounce between basins
per-seed. This is the well-known floor of PyTorch non-determinism
across hardware/BLAS backends (Linux x86 + MKL vs macOS arm64 +
Accelerate). Deterministic algorithms only fix intra-platform
non-determinism, not cross-platform. The aggregate means in the paper
tables average this out; individual seeds bounce.

## Re-running the full pipeline on a SLURM cluster

For readers who want to redo the experiments end-to-end (not just
verify the published numbers), the cluster-side entrypoints are:

- `scripts/launch_fixed_allocation_table.sh` — 66 Optuna studies for
  the fixed-allocation table (uses `configs/optuna/<env>_fixed_paper.py`).
- `scripts/launch_equal_budget_table.sh` — 66 Optuna studies for the
  equal-budget appendix table.
- `scripts/launch_transfer_fixalloc.sh` — transfer/perturbation
  experiments (consumes the published encoders from
  `results/models/fixed_allocation/`).
- `scripts/launch_misspec.sh` — misspecification robustness sweeps.
- `scripts/pregenerate_datasets.py` — populates the dataset cache
  before launching the Optuna sweeps.
- `FeedbackInformativeness/scripts/submit_grid_mcmc_euler.sh` — Julia
  MCMC baseline.

All launchers read `--help`-style docstrings at the top of each file.
They default to writing into `$SCRATCH/mavrl/optuna_studies/`.

## Testing

```bash
pytest tests/
```

## Citation

```bibtex
@inproceedings{baur2026mavrl,
  title={MAVRL: Learning Reward Functions from Multiple Feedback Types with Amortized Variational Inference},
  author={Baur, Rapha\"el and Metz, Yannick and Gkoulta, Maria and El-Assady, Mennatallah and Ramponi, Giorgia and Kleine Buening, Thomas},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026},
}
```
