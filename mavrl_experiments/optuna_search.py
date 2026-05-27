"""Bayesian optimization for feedback budget allocation using Optuna.

Searches for multi-feedback configurations that match or outperform
single-modality baselines under a total feedback budget constraint.

Each invocation acts as an independent worker. Multiple workers can connect
to the same study via JournalFileStorage (NFS-safe, append-only) for
distributed optimization on a cluster.

Usage:
    # Run locally (single worker)
    python -m mavrl_experiments.optuna_search \\
        --study-name grid_sparse_b64 \\
        --storage optuna_journal.log \\
        --env-config grid_sparse \\
        --budget 64 --n-seeds 3 --n-trials 10

    # Submit to cluster (32 parallel workers sharing the same study)
    STUDY_NAME=grid_sparse_b64 BUDGET=64 ENV_CONFIG=grid_sparse \\
        N_SEEDS=3 N_TRIALS=20 STORAGE_PATH=optuna_journal_b64.log \\
        sbatch scripts/submit_optuna.sh

    # Inspect results after completion
    python -m mavrl_experiments.optuna_search \\
        --study-name grid_sparse_b64 \\
        --storage optuna_journal_b64.log \\
        --show-results
"""

import argparse
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import optuna

from train import run_experiment, get_default_args
from mavrl_experiments.config_loader import load_optuna_config, list_configs


SAMPLE_COUNT_KEYS = {
    "n_pref_samples": "pref",
    "n_demo_samples": "demo",
    "n_rating_samples": "rating",
    "n_stop_samples": "stop",
}

# train.py parses these as strings (they may be "lin_<float>" schedules).
# Optuna suggests them as floats, so we cast back to str before applying.
STRING_SCHEDULE_KEYS = {
    "retrain_learning_rate",
    "retrain_clip_range",
    "retrain_clip_range_vf",
}

# Virtual search param: not a real train.py arg. If suggested, we resolve
# retrain_batch_size = (retrain_n_envs * retrain_n_steps) // retrain_n_minibatches
# so the minibatch count is what's actually being searched (PPO requires
# batch_size to divide n_envs * n_steps).
DERIVED_BATCH_SIZE_KEY = "retrain_n_minibatches"

# Optuna's categorical distribution only accepts hashable scalars
# (None | bool | int | float | str). For list-of-int args we encode each
# choice as a comma-separated string in the config and decode here before
# attaching to the args namespace.
LIST_OF_INTS_KEYS = {"encoder_hidden_sizes"}

# Number of best trials whose model checkpoints are persisted to disk.
TOP_K_MODELS_TO_KEEP = 3


def _is_currently_top_k(study: optuna.Study, candidate_value: float, k: int) -> bool:
    """Whether candidate_value would currently rank in the study's top-k.

    Reads the study's already-committed COMPLETE trials. The current trial
    (still inside objective()) is not yet COMPLETE, so it isn't double-counted.
    """
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and t.value not in (float("inf"), float("-inf"))
    ]
    if len(completed) < k:
        return True
    minimize = study.direction == optuna.study.StudyDirection.MINIMIZE
    completed.sort(key=lambda t: t.value, reverse=not minimize)
    kth_best = completed[k - 1].value
    return candidate_value <= kth_best if minimize else candidate_value >= kth_best


def _decode_value(key, value):
    if key in LIST_OF_INTS_KEYS and isinstance(value, str):
        return [int(x) for x in value.split(",") if x]
    if key in STRING_SCHEDULE_KEYS and not isinstance(value, str):
        return str(value)
    return value


def sample_budget_allocation(
    trial: optuna.Trial,
    modality_keys: list[str],
    budget: int,
    caps: dict[str, int] | None = None,
) -> dict[str, int]:
    """Sample a flat-Dirichlet allocation of `budget` samples across modalities.

    Suggests one float per modality on (0, 1], maps to a uniform-simplex
    proportion via -log, then converts to integers summing exactly to `budget`
    using largest-remainder rounding.

    If `caps` is provided (mapping full sample-count keys -> integer cap),
    over-cap proportions are clipped to `cap_i / budget` and the excess mass
    is redistributed proportionally among the uncapped modalities. Integer
    rounding is bounded by the caps so no modality exceeds its cap.

    Caller must guarantee `sum(min(caps[i], budget)) >= budget` over the
    active modality subset (otherwise the budget is unreachable).
    """
    exp_samples = np.array([
        -np.log(trial.suggest_float(f"x_{key}", 1e-10, 1.0))
        for key in modality_keys
    ])
    props = exp_samples / exp_samples.sum()

    cap_int = np.array(
        [min((caps or {}).get(k, budget), budget) for k in modality_keys],
        dtype=int,
    )
    cap_prop = cap_int / budget

    # Clip over-cap proportions and redistribute excess to uncapped modalities.
    # Loop in case a redistribution pushes another modality above its own cap.
    while True:
        over = props > cap_prop + 1e-12
        if not over.any():
            break
        excess = float((props[over] - cap_prop[over]).sum())
        props[over] = cap_prop[over]
        under = (~over) & (props > 0)
        s = float(props[under].sum())
        if s == 0 or excess == 0:
            break
        props[under] += excess * props[under] / s

    raw = props * budget
    floors = np.minimum(np.floor(raw).astype(int), cap_int)
    remainders = raw - floors
    deficit = budget - int(floors.sum())
    deficit = min(deficit, int((cap_int - floors).sum()))
    if deficit > 0:
        order = np.argsort(-remainders)
        added = 0
        for idx in order:
            if added >= deficit:
                break
            if floors[idx] < cap_int[idx]:
                floors[idx] += 1
                added += 1

    return {key: int(floors[i]) for i, key in enumerate(modality_keys)}


def load_env_config(env_config_name: str) -> dict[str, Any]:
    """Import and return BASE_CONFIG, MODALITY_PARAMS, optional HYPERPARAM_SEARCH_SPACE,
    optional MODALITY_MAX_SAMPLES (short-name -> integer cap), and optional
    FIXED_SAMPLE_COUNTS (short-name -> integer count, for --fixed-allocation mode)."""
    module = load_optuna_config(env_config_name)
    return {
        "base_config": module.BASE_CONFIG,
        "modality_params": module.MODALITY_PARAMS,
        "hyperparam_search_space": getattr(module, "HYPERPARAM_SEARCH_SPACE", {}),
        "modality_max_samples": getattr(module, "MODALITY_MAX_SAMPLES", {}),
        "fixed_sample_counts": getattr(module, "FIXED_SAMPLE_COUNTS", {}),
    }


def build_training_args(
    base_config: dict,
    modality_params: dict[str, dict],
    sample_counts: dict[str, int],
    seed: int,
    log_wandb: bool = False,
    wandb_project: str | None = None,
) -> argparse.Namespace:
    """Build a train.py-compatible Namespace from config + trial suggestions."""
    args = get_default_args()
    known = set(vars(args))

    def _set(key: str, value, source: str) -> None:
        if key not in known:
            raise KeyError(
                f"Unknown training arg {key!r} (from {source}). "
                f"Check the spelling against train.py's CLI flags."
            )
        setattr(args, key, value)

    for key, value in base_config.items():
        _set(key, _decode_value(key, value), "BASE_CONFIG")

    for key, value in sample_counts.items():
        _set(key, value, "sample_counts")

    # Apply per-modality conditional hyperparameters
    for param_key, modality_name in SAMPLE_COUNT_KEYS.items():
        if sample_counts.get(param_key, 0) > 0 and modality_name in modality_params:
            for k, v in modality_params[modality_name].items():
                _set(k, v, f"MODALITY_PARAMS[{modality_name!r}]")

    args.seed = seed
    args.log_wandb = log_wandb
    if log_wandb and wandb_project:
        args.wandb_project = wandb_project
    args.retrain_verbose = 0
    args.retrain_pbar = False

    return args


def make_objective(
    base_config: dict,
    modality_params: dict[str, dict],
    modalities: list[str],
    hyperparam_search_space: dict[str, tuple[float, float, bool]],
    budget: int,
    n_seeds: int,
    metric: str,
    direction: str,
    model_root: Path,
    log_wandb: bool = False,
    wandb_project: str | None = None,
    caps: dict[str, int] | None = None,
    fixed_sample_counts: dict[str, int] | None = None,
):
    """Return a closure suitable for study.optimize().

    `modalities` is a list of full sample-count keys (e.g. "n_pref_samples")
    indicating which modalities the budget is allocated across:
      - len 1: full budget (capped) goes to that modality (no Dirichlet sampling).
      - len >= 2: Dirichlet allocation over the listed modalities.
    Any of the four canonical sample-count keys not in `modalities` is
    explicitly set to 0 so BASE_CONFIG defaults can't leak through.

    `caps` maps full sample-count keys to their integer maximum allocation.
    Modalities not in `caps` are uncapped (effectively capped at `budget`).

    `fixed_sample_counts`, if provided, bypasses Dirichlet allocation entirely:
    the trial uses exactly these per-modality counts (full sample-count keys).
    `caps` and `budget` are then only consulted for logging.
    """
    worst = float("inf") if direction == "minimize" else float("-inf")
    modality_keys = list(modalities)
    if not modality_keys:
        raise ValueError("modalities must be a non-empty list of sample-count keys")

    caps = caps or {}

    # Pre-compute single-modality fixed counts (no Dirichlet) when only one
    # modality is active and the caller didn't explicitly prescribe counts.
    if fixed_sample_counts is None and len(modality_keys) == 1:
        sole = modality_keys[0]
        sole_alloc = min(budget, caps.get(sole, budget))
        fixed_sample_counts = {k: (sole_alloc if k == sole else 0) for k in SAMPLE_COUNT_KEYS}

    def objective(trial: optuna.Trial) -> float:
        if fixed_sample_counts is not None:
            sample_counts = {k: fixed_sample_counts.get(k, 0) for k in SAMPLE_COUNT_KEYS}
        else:
            allocated = sample_budget_allocation(trial, modality_keys, budget, caps=caps)
            # Fill in 0 for inactive modalities so the full 4-key dict is set
            # in args (otherwise BASE_CONFIG / train.py defaults would apply).
            sample_counts = {k: allocated.get(k, 0) for k in SAMPLE_COUNT_KEYS}

        # Suggest additional hyperparameters (categorical list or continuous range)
        hyperparam_overrides = {}
        for param_name, spec in hyperparam_search_space.items():
            if isinstance(spec, list):
                hyperparam_overrides[param_name] = trial.suggest_categorical(param_name, spec)
            else:
                lo, hi, log = spec
                hyperparam_overrides[param_name] = trial.suggest_float(param_name, lo, hi, log=log)

        # Resolve derived retrain_batch_size from minibatch count (if searched).
        if DERIVED_BATCH_SIZE_KEY in hyperparam_overrides:
            n_minibatches = int(hyperparam_overrides.pop(DERIVED_BATCH_SIZE_KEY))
            n_envs = hyperparam_overrides.get(
                "retrain_n_envs", base_config.get("retrain_n_envs", 8)
            )
            n_steps = hyperparam_overrides.get(
                "retrain_n_steps", base_config.get("retrain_n_steps", 2048)
            )
            rollout = int(n_envs) * int(n_steps)
            if rollout % n_minibatches != 0:
                # Round down to the nearest divisor; PPO requires divisibility.
                while rollout % n_minibatches != 0 and n_minibatches > 1:
                    n_minibatches -= 1
            hyperparam_overrides["retrain_batch_size"] = rollout // n_minibatches
            trial.set_user_attr("retrain_n_minibatches", n_minibatches)
            trial.set_user_attr("retrain_batch_size", hyperparam_overrides["retrain_batch_size"])

        total = sum(sample_counts.values())
        num_active = sum(1 for v in sample_counts.values() if v > 0)

        trial.set_user_attr("total_samples", total)
        trial.set_user_attr("num_active_modalities", num_active)
        trial.set_user_attr("sample_counts", sample_counts)
        if caps:
            active_caps = {SAMPLE_COUNT_KEYS[k]: caps[k] for k in modality_keys if k in caps}
            if active_caps:
                trial.set_user_attr("caps", active_caps)
            trial.set_user_attr("effective_budget", total)

        # Merge base config with hyperparam overrides for this trial
        trial_base_config = {**base_config, **hyperparam_overrides}

        # One outer tempdir holds per-seed subdirs. Keeping it open until
        # after the top-k decision lets us copy best_model.pt out before
        # the directory is cleaned up.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmproot = Path(tmpdir)
            values = []
            seed_best_paths: list[tuple[int, str]] = []
            for seed in range(n_seeds):
                args = build_training_args(
                    trial_base_config, modality_params, sample_counts, seed,
                    log_wandb=log_wandb, wandb_project=wandb_project,
                )
                args.model_save_dir = str(tmproot / f"seed_{seed}")
                args.save_behavior = "best"
                try:
                    result = run_experiment(args)
                except Exception:
                    traceback.print_exc()
                    continue
                val = result["final_metrics"].get(metric)
                if val is not None:
                    values.append(val)
                best_path = result.get("best_model_path")
                if best_path and Path(best_path).exists():
                    seed_best_paths.append((seed, best_path))

            if not values:
                return worst

            mean_val = float(np.mean(values))
            trial.set_user_attr("per_seed_values", values)

            if seed_best_paths and _is_currently_top_k(
                trial.study, mean_val, k=TOP_K_MODELS_TO_KEEP
            ):
                trial_root = model_root / f"trial_{trial.number}"
                persisted = []
                for seed, src in seed_best_paths:
                    dst_dir = trial_root / f"seed_{seed}"
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    dst = dst_dir / "best_model.pt"
                    shutil.copy2(src, dst)
                    persisted.append(str(dst))
                trial.set_user_attr("model_paths", persisted)
                trial.set_user_attr("model_dir", str(trial_root))

            return mean_val

    return objective


METRIC_TO_NORM_KEYS = {
    "eval/regret": ("optimal_regret", "uniform_regret"),
    "eval/discounted_value": ("optimal_discounted_value", "uniform_discounted_value"),
    "eval/mean_rew": ("optimal_mean_return", "uniform_mean_return"),
}

NORM_VALUES_PATH = Path(__file__).resolve().parent.parent / "results" / "normalization_values.json"


def load_normalization(env_id: str, metric: str) -> tuple[float, float] | None:
    """Load (optimal, uniform) normalization constants for a metric, or None if unavailable."""
    if metric not in METRIC_TO_NORM_KEYS:
        return None
    if not NORM_VALUES_PATH.exists():
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
    """Normalize to 0-100 scale where uniform=0% and optimal=100%."""
    return 100.0 * (raw - uniform) / (optimal - uniform)


def show_results(
    study: optuna.Study,
    top_k: int = 20,
    env_id: str | None = None,
    metric: str | None = None,
) -> None:
    """Print a summary of the best trials from a completed study."""
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value not in (float("inf"), float("-inf"))]
    if not completed:
        print("No completed feasible trials found.")
        return

    norm = None
    if env_id and metric:
        norm = load_normalization(env_id, metric)

    ascending = study.direction == optuna.study.StudyDirection.MINIMIZE
    completed.sort(key=lambda t: t.value, reverse=not ascending)
    best = completed[:top_k]

    direction_label = "minimize" if ascending else "maximize"
    print(f"\nStudy: {study.study_name} (direction: {direction_label})")
    print(f"Total trials: {len(study.trials)}")
    print(f"Completed (feasible): {len(completed)}")
    if norm:
        opt_val, uni_val = norm
        print(f"Normalization: uniform={uni_val:.4f} (0%), optimal={opt_val:.4f} (100%)")
    print(f"Best {min(top_k, len(best))} trials:\n")

    if norm:
        header = f"{'Rank':<5} {'Value':<12} {'Norm%':<8} {'Total':<7} {'#Mod':<5} {'Pref':<6} {'Demo':<6} {'Rate':<6} {'Stop':<6}"
    else:
        header = f"{'Rank':<5} {'Value':<12} {'Total':<7} {'#Mod':<5} {'Pref':<6} {'Demo':<6} {'Rate':<6} {'Stop':<6}"
    print(header)
    print("-" * len(header))

    for i, t in enumerate(best):
        total = t.user_attrs.get("total_samples", "?")
        n_mod = t.user_attrs.get("num_active_modalities", "?")
        sc = t.user_attrs.get("sample_counts", {})
        p = sc.get("n_pref_samples", 0)
        d = sc.get("n_demo_samples", 0)
        r = sc.get("n_rating_samples", 0)
        s = sc.get("n_stop_samples", 0)
        if norm:
            nv = normalize_value(t.value, *norm)
            print(f"{i+1:<5} {t.value:<12.4f} {nv:<8.1f} {total:<7} {n_mod:<5} {p:<6} {d:<6} {r:<6} {s:<6}")
        else:
            print(f"{i+1:<5} {t.value:<12.4f} {total:<7} {n_mod:<5} {p:<6} {d:<6} {r:<6} {s:<6}")

    best_trial = completed[0]
    print(f"\nBest trial (#{best_trial.number}):")
    print(f"  Allocation: {best_trial.user_attrs.get('sample_counts', {})}")
    print(f"  Params: {best_trial.params}")
    print(f"  Value:  {best_trial.value:.6f}")
    if norm:
        print(f"  Norm%:  {normalize_value(best_trial.value, *norm):.2f}%")
    per_seed = best_trial.user_attrs.get("per_seed_values")
    if per_seed:
        if norm:
            norm_seeds = [f'{normalize_value(v, *norm):.2f}%' for v in per_seed]
            print(f"  Per-seed: {[f'{v:.4f}' for v in per_seed]} (norm: {norm_seeds})")
        else:
            print(f"  Per-seed: {[f'{v:.4f}' for v in per_seed]}")


def parse_modalities_arg(value: str | None) -> list[str]:
    """Parse `--modalities pref,demo` into a list of full sample-count keys.

    None / unset -> all four (combined mode).
    """
    valid_short = list(SAMPLE_COUNT_KEYS.values())  # ["pref","demo","rating","stop"]
    if value is None:
        return [f"n_{m}_samples" for m in valid_short]
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("--modalities cannot be empty")
    seen: set[str] = set()
    for p in parts:
        if p not in valid_short:
            raise ValueError(
                f"--modalities: unknown modality {p!r}; "
                f"must be one of {valid_short}"
            )
        if p in seen:
            raise ValueError(f"--modalities: duplicate modality {p!r}")
        seen.add(p)
    return [f"n_{m}_samples" for m in parts]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna budget-constrained feedback allocation search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--study-name", type=str, required=True,
                        help="Shared Optuna study name")
    parser.add_argument("--storage", type=str, required=True,
                        help="Path to journal file for NFS-safe storage")
    parser.add_argument("--env-config", type=str, default=None,
                        help="Config name under configs/optuna/ (e.g. grid_sparse). "
                             "Override root via $MAVRL_CONFIG_ROOT.")
    parser.add_argument("--budget", type=int, default=None,
                        help="Maximum total feedback samples across all modalities")
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Number of seeds per trial (default: 3)")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of trials this worker evaluates (default: 50)")
    parser.add_argument("--metric", type=str, default="eval/regret",
                        help="Metric to optimize (default: eval/regret)")
    parser.add_argument("--direction", type=str, default="minimize",
                        choices=["minimize", "maximize"],
                        help="Optimization direction (default: minimize)")
    parser.add_argument("--modalities", type=str, default=None,
                        help="Comma-separated subset of modalities to allocate budget across "
                             "(choices: pref, demo, rating, stop). Default: all four. "
                             "A single element is equivalent to fixing the full budget to "
                             "that modality (no Dirichlet sampling).")
    parser.add_argument("--fixed-allocation", action="store_true",
                        help="Use prescribed per-modality sample counts from the env config's "
                             "FIXED_SAMPLE_COUNTS dict instead of sampling a Dirichlet "
                             "allocation over --budget. --budget is then derived (sum of "
                             "fixed counts for active modalities) and only used for logging.")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="Log individual trial runs to this wandb project")
    parser.add_argument("--dataset-cache-dir", type=str, default=None,
                        help="Directory for cached datasets (avoids regenerating data each trial)")
    parser.add_argument("--show-results", action="store_true",
                        help="Show results from an existing study and exit")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of top trials to show (default: 20)")
    return parser.parse_args()


def main():
    args = parse_args()

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(args.storage),
    )

    # Resolve env config and env_id for normalization (if env-config is provided)
    env_cfg = None
    env_id = None
    if args.env_config:
        env_cfg = load_env_config(args.env_config)
        env_id = env_cfg["base_config"].get("env_id")

    if args.show_results:
        study = optuna.load_study(study_name=args.study_name, storage=storage)
        show_results(study, top_k=args.top_k, env_id=env_id, metric=args.metric)
        return

    if args.env_config is None:
        print("Error: --env-config is required when running optimization.", file=sys.stderr)
        sys.exit(1)
    if args.budget is None and not args.fixed_allocation:
        print("Error: --budget is required (or pass --fixed-allocation to derive it from FIXED_SAMPLE_COUNTS).", file=sys.stderr)
        sys.exit(1)

    if env_cfg is None:
        env_cfg = load_env_config(args.env_config)
        env_id = env_cfg["base_config"].get("env_id")

    if args.dataset_cache_dir:
        env_cfg["base_config"]["dataset_cache_dir"] = args.dataset_cache_dir

    try:
        modalities = parse_modalities_arg(args.modalities)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Convert MODALITY_MAX_SAMPLES (short-name keyed) to full sample-count keys.
    raw_caps = env_cfg.get("modality_max_samples") or {}
    valid_short = list(SAMPLE_COUNT_KEYS.values())
    for short in raw_caps:
        if short not in valid_short:
            print(
                f"Error: MODALITY_MAX_SAMPLES contains unknown modality "
                f"{short!r}; must be one of {valid_short}",
                file=sys.stderr,
            )
            sys.exit(1)
    caps = {f"n_{short}_samples": int(cap) for short, cap in raw_caps.items()}

    # Resolve fixed (prescribed) per-modality counts when --fixed-allocation is on.
    fixed_sample_counts: dict[str, int] | None = None
    if args.fixed_allocation:
        raw_fixed = env_cfg.get("fixed_sample_counts") or {}
        if not raw_fixed:
            print(
                f"Error: --fixed-allocation requires FIXED_SAMPLE_COUNTS in "
                f"{args.env_config!r}'s config (short-name keyed: pref/demo/rating/stop).",
                file=sys.stderr,
            )
            sys.exit(1)
        for short in raw_fixed:
            if short not in valid_short:
                print(
                    f"Error: FIXED_SAMPLE_COUNTS contains unknown modality "
                    f"{short!r}; must be one of {valid_short}",
                    file=sys.stderr,
                )
                sys.exit(1)
        # Active modalities pull from FIXED_SAMPLE_COUNTS; inactive go to 0.
        active_short = [SAMPLE_COUNT_KEYS[k] for k in modalities]
        missing = [m for m in active_short if m not in raw_fixed]
        if missing:
            print(
                f"Error: FIXED_SAMPLE_COUNTS in {args.env_config!r} is missing "
                f"counts for active modalities {missing}.",
                file=sys.stderr,
            )
            sys.exit(1)
        fixed_sample_counts = {
            f"n_{short}_samples": (int(raw_fixed[short]) if short in active_short else 0)
            for short in valid_short
        }
        derived_budget = sum(fixed_sample_counts.values())
        if args.budget is None:
            args.budget = derived_budget
        elif args.budget != derived_budget:
            print(
                f"Note: --budget {args.budget} ignored in --fixed-allocation mode; "
                f"using derived budget {derived_budget} (sum of FIXED_SAMPLE_COUNTS "
                f"for active modalities).",
                file=sys.stderr,
            )
            args.budget = derived_budget

    # Sanity-check: the active subset must be able to absorb the whole budget.
    # Skipped in --fixed-allocation mode (budget is derived, allocation is prescribed).
    if not args.fixed_allocation and len(modalities) >= 2:
        max_reachable = sum(min(caps.get(k, args.budget), args.budget) for k in modalities)
        if max_reachable < args.budget:
            print(
                f"Error: caps over active modalities sum to {max_reachable} "
                f"< budget {args.budget}; raise caps or reduce --budget.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Top-k trial checkpoints are persisted alongside the journal file.
    model_root = Path(args.storage + ".models")

    tpe_sampler = optuna.samplers.TPESampler(
        multivariate=True,   # we expect correlations between the parameters
        group=False,  # we have a static parameter space, no grouping needed
        constant_liar=True,  # recommended for concurrent workers
        n_startup_trials=20  # higher number than default (10) for larger search space
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction=args.direction,
        load_if_exists=True,
        sampler=tpe_sampler
    )

    objective = make_objective(
        base_config=env_cfg["base_config"],
        modality_params=env_cfg["modality_params"],
        modalities=modalities,
        hyperparam_search_space=env_cfg["hyperparam_search_space"],
        budget=args.budget,
        n_seeds=args.n_seeds,
        metric=args.metric,
        direction=args.direction,
        model_root=model_root,
        log_wandb=args.wandb_project is not None,
        wandb_project=args.wandb_project,
        caps=caps,
        fixed_sample_counts=fixed_sample_counts,
    )

    short_names = [SAMPLE_COUNT_KEYS[k] for k in modalities]
    print(f"Starting Optuna worker for study '{args.study_name}'")
    print(f"  Budget: {args.budget}, Seeds/trial: {args.n_seeds}, Trials: {args.n_trials}")
    print(f"  Metric: {args.metric} ({args.direction})")
    print(f"  Top-{TOP_K_MODELS_TO_KEEP} model checkpoints persisted under: {model_root}")
    if args.fixed_allocation:
        active_counts = {
            SAMPLE_COUNT_KEYS[k]: fixed_sample_counts[k]
            for k in modalities
        }
        print(f"  Mode: fixed-allocation {active_counts} (no Dirichlet sampling)")
    elif len(short_names) == 1:
        sole_full = modalities[0]
        sole_short = short_names[0]
        sole_alloc = min(args.budget, caps.get(sole_full, args.budget))
        print(f"  Mode: single-modality ({sole_short}, samples={sole_alloc})")
    else:
        print(f"  Mode: Dirichlet over {short_names} (|S|={len(short_names)})")
    if caps and not args.fixed_allocation:
        active_caps = {SAMPLE_COUNT_KEYS[k]: v for k, v in caps.items() if k in modalities}
        if active_caps:
            print(f"  Caps: {active_caps}")

    study.optimize(objective, n_trials=args.n_trials)

    show_results(study, env_id=env_id, metric=args.metric)


if __name__ == "__main__":
    main()
