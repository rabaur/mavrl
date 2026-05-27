"""Generate a budget table from completed Optuna studies.

Scans a storage directory for journal files matching the naming convention
from launch_budget_table.sh and assembles performance into a table:

    Rows    = feedback budgets (N)
    Columns = Preferences | Demos | Ratings | Stops | Combined | Budget allocation

Usage:
    python -m mavrl_experiments.optuna_budget_table \
        --storage-dir optuna_studies/grid_trap \
        --env-config grid_trap

    # LaTeX output
    python -m mavrl_experiments.optuna_budget_table \
        --storage-dir optuna_studies/grid_trap \
        --env-config grid_trap --latex
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

NORM_VALUES_PATH = Path(__file__).resolve().parent.parent / "results" / "normalization_values.json"

METRIC_TO_NORM_KEYS = {
    "eval/regret": ("optimal_regret", "uniform_regret"),
    "eval/discounted_value": ("optimal_discounted_value", "uniform_discounted_value"),
    "eval/mean_rew": ("optimal_mean_return", "uniform_mean_return"),
}

MODALITIES = ["pref", "demo", "rating", "stop"]
MODALITY_LABELS = {"pref": "Preferences", "demo": "Demos", "rating": "Ratings", "stop": "Stops"}
BUDGETS_STANDARD = [8, 16, 32, 64, 128, 256]
BUDGETS_DEMO = [1, 2, 4, 8, 16]

_STUDY_BUDGET_RE = re.compile(
    r"^(?P<env>.+)_(?P<kind>pref|demo|rating|stop|combined|pdrs)_b(?P<budget>\d+)\.log$"
)


def load_normalization(env_id: str, metric: str) -> tuple[float, float] | None:
    if metric not in METRIC_TO_NORM_KEYS or not NORM_VALUES_PATH.exists():
        return None
    with open(NORM_VALUES_PATH) as f:
        all_norms = json.load(f)
    env_norms = all_norms.get(env_id)
    if env_norms is None:
        return None
    opt_key, uni_key = METRIC_TO_NORM_KEYS[metric]
    if opt_key not in env_norms or uni_key not in env_norms:
        return None
    return env_norms[opt_key], env_norms[uni_key]


def normalize_value(raw: float, optimal: float, uniform: float) -> float:
    return 100.0 * (raw - uniform) / (optimal - uniform)


def discover_budgets(storage_dir: Path, env_config: str) -> tuple[list[int], list[int]]:
    """Infer budget scales from journal filenames in *storage_dir*.

    Returns:
        (standard_budgets, demo_budgets)
    """
    standard_budgets: set[int] = set()
    demo_budgets: set[int] = set()

    for journal_path in storage_dir.glob(f"{env_config}_*.log"):
        m = _STUDY_BUDGET_RE.match(journal_path.name)
        if m is None or m.group("env") != env_config:
            continue
        budget = int(m.group("budget"))
        kind = m.group("kind")
        if kind == "demo":
            demo_budgets.add(budget)
        else:
            standard_budgets.add(budget)

    if not standard_budgets:
        standard_budgets = set(BUDGETS_STANDARD)
    if not demo_budgets:
        demo_budgets = set(BUDGETS_DEMO)

    return sorted(standard_budgets), sorted(demo_budgets)


def load_study_best(storage_dir: Path, env_config: str, study_suffix: str, direction: str):
    """Load a study and return (best_value, best_trial) or (None, None) if unavailable."""
    study_name = f"{env_config}_{study_suffix}"
    journal_path = storage_dir / f"{study_name}.log"
    if not journal_path.exists():
        return None, None

    try:
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(str(journal_path)),
        )
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception as e:
        print(f"  Warning: could not load study '{study_name}': {e}", file=sys.stderr)
        return None, None

    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value not in (float("inf"), float("-inf"))
    ]
    if not completed:
        return None, None

    ascending = direction == "minimize"
    completed.sort(key=lambda t: t.value, reverse=not ascending)
    best = completed[0]
    return best.value, best


def extract_numeric(
    val: float | None,
    norm: tuple[float, float] | None,
    trial=None,
    use_stderr: bool = False,
) -> tuple[float | None, float | None]:
    """Numeric counterpart to ``format_value``: returns ``(mean, spread)``.

    Mean is normalized to a 0-100 scale when ``norm`` is given, otherwise raw.
    Spread mirrors ``format_value``: sample std by default, standard error of
    the mean when ``use_stderr=True``, and ``None`` if no per-seed data is
    available on the trial.
    """
    if val is None:
        return None, None
    mean = normalize_value(val, *norm) if norm else val
    if trial is None:
        return mean, None
    per_seed = trial.user_attrs.get("per_seed_values")
    if not per_seed or len(per_seed) <= 1:
        return mean, None
    if use_stderr:
        spread = float(np.std(per_seed, ddof=1) / np.sqrt(len(per_seed)))
    else:
        spread = float(np.std(per_seed))
    if norm:
        scale = 100.0 / abs(norm[0] - norm[1])
        spread *= scale
    return mean, spread


def format_value(
    val: float | None,
    norm: tuple[float, float] | None,
    show_std: bool,
    trial=None,
    use_stderr: bool = False,
) -> str:
    """Format a cell value, optionally with normalized percentage and dispersion.

    When ``show_std`` is True, the spread of ``trial.user_attrs["per_seed_values"]``
    is appended as ``"value ± spread"``. ``use_stderr=True`` switches the spread
    from sample std to standard error of the mean (std / sqrt(n), n>1, ddof=1).
    """
    if val is None:
        return "-"
    parts = []
    if norm:
        nv = normalize_value(val, *norm)
        parts.append(f"{nv:.1f}%")
    else:
        parts.append(f"{val:.4f}")

    if show_std and trial is not None:
        per_seed = trial.user_attrs.get("per_seed_values")
        if per_seed and len(per_seed) > 1:
            if use_stderr:
                spread = float(np.std(per_seed, ddof=1) / np.sqrt(len(per_seed)))
            else:
                spread = float(np.std(per_seed))
            if norm:
                scale = 100.0 / abs(norm[0] - norm[1])
                parts[-1] = f"{normalize_value(val, *norm):.1f}±{spread * scale:.1f}%"
            else:
                parts[-1] = f"{val:.4f}±{spread:.4f}"
    return parts[0]


def format_allocation(trial) -> str:
    """Format the budget allocation of a combined trial as (np=x, nd=y, nr=z, ns=d).

    Sample counts are stored as a user_attr by optuna_search.objective (the raw
    `n_*_samples` keys are not Optuna-suggested params, so they don't appear in
    `trial.params`).
    """
    if trial is None:
        return "-"
    sc = trial.user_attrs.get("sample_counts") or {}
    p = sc.get("n_pref_samples", 0)
    d = sc.get("n_demo_samples", 0)
    r = sc.get("n_rating_samples", 0)
    s = sc.get("n_stop_samples", 0)
    return f"(np={p}, nd={d}, nr={r}, ns={s})"


def format_hyperparams(trial) -> str:
    """Format td_error_weight and kl_weight from a trial."""
    if trial is None:
        return "-"
    td = trial.params.get("td_error_weight")
    kl = trial.params.get("kl_weight")
    parts = []
    if td is not None:
        parts.append(f"td={td:.3f}")
    if kl is not None:
        parts.append(f"kl={kl:.3f}")
    return ", ".join(parts) if parts else "-"


def build_table(storage_dir: Path, env_config: str, metric: str, direction: str, show_std: bool,
                combined_suffix: str = "combined"):
    """Build the budget table data structure."""
    env_id = env_config.replace("_", " ").split()[0]
    try:
        from mavrl_experiments.optuna_search import load_env_config
        cfg = load_env_config(env_config)
        env_id = cfg["base_config"].get("env_id", env_config)
    except Exception:
        env_id = env_config

    norm = load_normalization(env_id, metric)
    budgets_standard, budgets_demo = discover_budgets(storage_dir, env_config)

    rows = []
    for budget in budgets_standard:
        row = {"budget": budget}

        for mod in MODALITIES:
            budgets = budgets_demo if mod == "demo" else budgets_standard
            if budget not in budgets:
                row[mod] = ("-", None)
                continue
            suffix = f"{mod}_b{budget}"
            val, trial = load_study_best(storage_dir, env_config, suffix, direction)
            row[mod] = (format_value(val, norm, show_std, trial), trial)

        suffix = f"{combined_suffix}_b{budget}"
        val, trial = load_study_best(storage_dir, env_config, suffix, direction)
        row["combined"] = (format_value(val, norm, show_std, trial), trial)
        row["allocation"] = format_allocation(trial)

        rows.append(row)

    # Demo has its own budget scale; fill in rows that only exist for demo
    demo_only_budgets = sorted(set(budgets_demo) - set(budgets_standard))
    for budget in demo_only_budgets:
        row = {"budget": budget}
        for mod in MODALITIES:
            if mod == "demo":
                suffix = f"demo_b{budget}"
                val, trial = load_study_best(storage_dir, env_config, suffix, direction)
                row[mod] = (format_value(val, norm, show_std, trial), trial)
            else:
                row[mod] = ("-", None)
        row["combined"] = ("-", None)
        row["allocation"] = "-"
        rows.append(row)

    rows.sort(key=lambda r: r["budget"])
    return rows, norm


def print_table(rows, norm, metric: str, env_config: str, direction: str):
    """Print the budget table to stdout."""
    print(f"\nBudget Table: {env_config}")
    print(f"Metric: {metric} ({direction})")
    if norm:
        print(f"Normalization: uniform={norm[1]:.4f} (0%), optimal={norm[0]:.4f} (100%)")
    print()

    col_widths = {
        "budget": 8,
        "pref": 16,
        "demo": 16,
        "rating": 16,
        "stop": 16,
        "combined": 20,
        "allocation": 32,
    }

    header = (
        f"{'N':<{col_widths['budget']}}"
        f"{'Preferences':<{col_widths['pref']}}"
        f"{'Demos':<{col_widths['demo']}}"
        f"{'Ratings':<{col_widths['rating']}}"
        f"{'Stops':<{col_widths['stop']}}"
        f"{'Combined':<{col_widths['combined']}}"
        f"{'Budget Allocation':<{col_widths['allocation']}}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        line = (
            f"{row['budget']:<{col_widths['budget']}}"
            f"{row['pref'][0]:<{col_widths['pref']}}"
            f"{row['demo'][0]:<{col_widths['demo']}}"
            f"{row['rating'][0]:<{col_widths['rating']}}"
            f"{row['stop'][0]:<{col_widths['stop']}}"
            f"{row['combined'][0]:<{col_widths['combined']}}"
            f"{row['allocation']:<{col_widths['allocation']}}"
        )
        print(line)
    print()


def print_latex(rows, norm, metric: str, env_config: str, direction: str):
    """Print the budget table as a LaTeX tabular."""
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Performance under equal feedback budgets (" + env_config.replace("_", r"\_") + r").}")
    print(r"\label{tab:budget_" + env_config + r"}")
    print(r"\begin{tabular}{r|cccc|c|l}")
    print(r"\toprule")
    print(r"$N$ & Pref & Demo & Rating & Stop & Combined & Budget Allocation \\")
    print(r"\midrule")

    for row in rows:
        cells = [str(row["budget"])]
        for mod in MODALITIES:
            cells.append(row[mod][0])
        cells.append(row["combined"][0])
        alloc = row["allocation"].replace("_", r"\_") if row["allocation"] != "-" else "-"
        cells.append(f"\\scriptsize{{{alloc}}}")
        print(" & ".join(cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def print_detailed(rows, storage_dir: Path, env_config: str, direction: str,
                   combined_suffix: str = "combined"):
    """Print per-study details (hyperparams, number of trials)."""
    print(f"\nDetailed per-study information:")
    print(f"{'Study':<35} {'Trials':<8} {'Best Value':<14} {'Hyperparams':<25}")
    print("-" * 82)

    budgets_standard, budgets_demo = discover_budgets(storage_dir, env_config)
    all_budgets = sorted(set(budgets_standard + budgets_demo))
    for budget in all_budgets:
        for mod in MODALITIES:
            budgets = budgets_demo if mod == "demo" else budgets_standard
            if budget not in budgets:
                continue
            suffix = f"{mod}_b{budget}"
            study_name = f"{env_config}_{suffix}"
            journal_path = storage_dir / f"{study_name}.log"
            if not journal_path.exists():
                continue
            val, trial = load_study_best(storage_dir, env_config, suffix, direction)
            n_trials = "?"
            try:
                storage = optuna.storages.JournalStorage(
                    optuna.storages.journal.JournalFileBackend(str(journal_path)),
                )
                study = optuna.load_study(study_name=study_name, storage=storage)
                n_trials = len(study.trials)
            except Exception:
                pass
            val_str = f"{val:.4f}" if val is not None else "-"
            hp_str = format_hyperparams(trial) if trial else "-"
            print(f"{study_name:<35} {n_trials:<8} {val_str:<14} {hp_str:<25}")

        suffix = f"{combined_suffix}_b{budget}"
        if budget in budgets_standard:
            study_name = f"{env_config}_{suffix}"
            journal_path = storage_dir / f"{study_name}.log"
            if journal_path.exists():
                val, trial = load_study_best(storage_dir, env_config, suffix, direction)
                n_trials = "?"
                try:
                    storage = optuna.storages.JournalStorage(
                        optuna.storages.journal.JournalFileBackend(str(journal_path)),
                    )
                    study = optuna.load_study(study_name=study_name, storage=storage)
                    n_trials = len(study.trials)
                except Exception:
                    pass
                val_str = f"{val:.4f}" if val is not None else "-"
                hp_str = format_hyperparams(trial) if trial else "-"
                alloc_str = format_allocation(trial)
                print(f"{study_name:<35} {n_trials:<8} {val_str:<14} {hp_str:<25} {alloc_str}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate a budget table from completed Optuna studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--storage-dir", type=str, required=True,
                        help="Directory containing journal files (e.g. optuna_studies/grid_trap)")
    parser.add_argument("--env-config", type=str, required=True,
                        help="Config module name (e.g. grid_trap)")
    parser.add_argument("--metric", type=str, default="eval/discounted_value",
                        help="Metric that was optimized (default: eval/discounted_value)")
    parser.add_argument("--direction", type=str, default="maximize",
                        choices=["minimize", "maximize"],
                        help="Optimization direction (default: maximize)")
    parser.add_argument("--std", action="store_true",
                        help="Show standard deviation across seeds")
    parser.add_argument("--latex", action="store_true",
                        help="Output as LaTeX tabular")
    parser.add_argument("--detailed", action="store_true",
                        help="Show per-study details (hyperparams, trial counts)")
    parser.add_argument("--combined-suffix", type=str, default="combined",
                        help="Study suffix used for the joint multi-modality column "
                             "(default: combined; use 'pdrs' for the equal-budget table)")
    args = parser.parse_args()

    storage_dir = Path(args.storage_dir)
    if not storage_dir.exists():
        print(f"Error: storage directory '{storage_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    rows, norm = build_table(storage_dir, args.env_config, args.metric, args.direction, args.std,
                             combined_suffix=args.combined_suffix)

    if args.latex:
        print_latex(rows, norm, args.metric, args.env_config, args.direction)
    else:
        print_table(rows, norm, args.metric, args.env_config, args.direction)

    if args.detailed:
        print_detailed(rows, storage_dir, args.env_config, args.direction,
                       combined_suffix=args.combined_suffix)


if __name__ == "__main__":
    main()
