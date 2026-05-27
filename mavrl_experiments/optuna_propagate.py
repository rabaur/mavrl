"""Propagate completed trials from smaller-budget studies into larger-budget studies.

A trial that is feasible under budget N (total_samples <= N) is also feasible
under any budget M > N.  However, because each budget level is a separate
Optuna study, larger-budget studies may miss good configurations discovered at
smaller scales.  This tool copies those trials upward so that every study has
access to all configurations that fit within its budget.

Usage:
    # Preview what would be copied (safe, read-only)
    python -m mavrl_experiments.optuna_propagate \
        --storage-dir optuna_studies/lunar_lander_v3 \
        --env-config lunar_lander_v3 \
        --direction maximize --dry-run

    # Actually propagate
    python -m mavrl_experiments.optuna_propagate \
        --storage-dir optuna_studies/lunar_lander_v3 \
        --env-config lunar_lander_v3 \
        --direction maximize
"""

import argparse
import re
import sys
from pathlib import Path

import optuna

from mavrl_experiments.optuna_search import compute_constraints

optuna.logging.set_verbosity(optuna.logging.WARNING)

_STUDY_BUDGET_RE = re.compile(
    r"^(?P<env>.+)_(?P<kind>pref|demo|rating|stop|combined)_b(?P<budget>\d+)\.log$"
)


def discover_combined_budgets(storage_dir: Path, env_config: str) -> list[int]:
    """Return sorted list of budget values that have a combined journal file."""
    budgets: set[int] = set()
    for journal_path in storage_dir.glob(f"{env_config}_combined_b*.log"):
        m = _STUDY_BUDGET_RE.match(journal_path.name)
        if m is None or m.group("env") != env_config or m.group("kind") != "combined":
            continue
        budgets.add(int(m.group("budget")))
    return sorted(budgets)


def load_study(storage_dir: Path, env_config: str, budget: int) -> optuna.Study | None:
    study_name = f"{env_config}_combined_b{budget}"
    journal_path = storage_dir / f"{study_name}.log"
    if not journal_path.exists():
        return None
    try:
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(str(journal_path)),
        )
        return optuna.load_study(study_name=study_name, storage=storage)
    except Exception as e:
        print(f"  Warning: could not load {study_name}: {e}", file=sys.stderr)
        return None


def native_feasible_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    """Return completed trials with finite values that originated in this study.

    Trials that were themselves propagated from another study (have a
    ``propagated_from`` user attr) are excluded to prevent transitive
    duplication: b128 -> b256 -> b512 would otherwise create duplicates
    of trials that b128 already propagated directly to b512.
    """
    return [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and t.value not in (float("inf"), float("-inf"))
        and "propagated_from" not in t.user_attrs
    ]


def existing_propagation_keys(study: optuna.Study) -> set[str]:
    """Collect all `propagated_from` markers already present in a study."""
    keys: set[str] = set()
    for t in study.trials:
        marker = t.user_attrs.get("propagated_from")
        if marker is not None:
            keys.add(marker)
    return keys


def make_propagation_key(source_study_name: str, trial_number: int) -> str:
    return f"{source_study_name}:{trial_number}"


def create_trial_copy(
    source_trial: optuna.trial.FrozenTrial,
    source_study_name: str,
    target_budget: int,
    min_active_modalities: int,
) -> optuna.trial.FrozenTrial:
    """Build a FrozenTrial suitable for add_trial() in a different study.

    Recomputes the ``"constraints"`` system attribute relative to the
    *target* budget so the constrained TPESampler sees correct values.
    """
    user_attrs = dict(source_trial.user_attrs)
    user_attrs["propagated_from"] = make_propagation_key(
        source_study_name, source_trial.number
    )

    return optuna.trial.create_trial(
        params=source_trial.params,
        distributions=source_trial.distributions,
        values=[source_trial.value],
        user_attrs=user_attrs,
        system_attrs={
            "constraints": compute_constraints(
                source_trial.params, target_budget, min_active_modalities
            ),
        },
    )


def propagate(
    storage_dir: Path,
    env_config: str,
    min_active_modalities: int = 2,
    dry_run: bool = False,
) -> int:
    """Propagate trials from smaller to larger combined-budget studies.

    Returns the total number of trials propagated.
    """
    budgets = discover_combined_budgets(storage_dir, env_config)
    if len(budgets) < 2:
        print("Need at least 2 combined budget levels to propagate.")
        return 0

    print(f"Propagating trials: {env_config} (combined)")
    print(f"Budget levels: {budgets}")
    if dry_run:
        print("(dry run — no writes)\n")
    else:
        print()

    source_trials_by_budget: dict[int, list[tuple[str, optuna.trial.FrozenTrial]]] = {}
    for budget in budgets:
        study = load_study(storage_dir, env_config, budget)
        if study is None:
            source_trials_by_budget[budget] = []
            continue
        source_trials_by_budget[budget] = [
            (study.study_name, t) for t in native_feasible_trials(study)
        ]

    total_propagated = 0
    for i, target_budget in enumerate(budgets):
        if i == 0:
            continue

        target_study = load_study(storage_dir, env_config, target_budget)
        if target_study is None:
            continue

        existing_keys = existing_propagation_keys(target_study)

        trials_to_add: list[optuna.trial.FrozenTrial] = []
        for source_budget in budgets[:i]:
            new_from_source = 0
            skipped = 0
            for source_name, trial in source_trials_by_budget[source_budget]:
                key = make_propagation_key(source_name, trial.number)
                if key in existing_keys:
                    skipped += 1
                    continue
                trials_to_add.append(create_trial_copy(
                    trial, source_name, target_budget, min_active_modalities
                ))
                existing_keys.add(key)
                new_from_source += 1

            skip_msg = f" ({skipped} already present)" if skipped else ""
            print(
                f"  combined_b{source_budget} -> combined_b{target_budget}: "
                f"{new_from_source} new trials{skip_msg}"
            )

        if trials_to_add and not dry_run:
            target_study.add_trials(trials_to_add)

        total_propagated += len(trials_to_add)

    label = "would propagate" if dry_run else "propagated"
    print(f"\nTotal: {total_propagated} trials {label}")
    return total_propagated


def main():
    parser = argparse.ArgumentParser(
        description="Propagate Optuna trials from smaller to larger budget studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--storage-dir", type=str, required=True,
        help="Directory containing journal .log files",
    )
    parser.add_argument(
        "--env-config", type=str, required=True,
        help="Config module name (e.g. lunar_lander_v3)",
    )
    parser.add_argument(
        "--direction", type=str, default="maximize",
        choices=["minimize", "maximize"],
        help="Optimization direction (default: maximize)",
    )
    parser.add_argument(
        "--min-active-modalities", type=int, default=2,
        help="Minimum active modalities for constraint (default: 2)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be propagated without writing",
    )
    args = parser.parse_args()

    storage_dir = Path(args.storage_dir)
    if not storage_dir.exists():
        print(f"Error: storage directory '{storage_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    propagate(
        storage_dir,
        args.env_config,
        min_active_modalities=args.min_active_modalities,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
