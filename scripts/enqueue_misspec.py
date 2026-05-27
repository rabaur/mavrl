"""Enqueue the 270 misspecification runs into per-grid file-based queues.

Layout per grid env:
    4 single-modality misspec conditions  (1 cell each)  ×  10 seeds  = 40
    5 PDRS conditions                    (all-misspec + 4 single)  ×  10 seeds  = 50
    --------------------------------------------------------------------------
    90 runs / env  ×  3 envs  =  270 runs total.

The well-specified ("all correct") baselines are NOT re-enqueued here — those
are taken from the existing main-table sweeps.

Misspec axes (data side vs. model side):
    pref:    pref_data_rationality = 0.01   (model stays at MODALITY_PARAMS value)
    demo:    demo_rationality      = 0.1    +  demo_model_rationality = 10.0
    stop:    stop_c                = 0.1    +  stop_model_c           = 2.0
    rating:  rating_noise_std      = 2.0    (no model param)

Hyperparameters (td_error_weight, kl_weight, lr, batch_size,
encoder_hidden_sizes, use_importance_weights) and per-modality sample counts
come from `scripts/misspec_hparams.json` (best fixed-allocation Optuna trial
per (env, modality_subset)).

Run:
    python scripts/enqueue_misspec.py
    python scripts/enqueue_misspec.py --seeds 10 --wandb-project mavrl-misspec
    python scripts/enqueue_misspec.py --dry-run
"""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mavrl_experiments.config_loader import load_optuna_config
from mavrl_experiments.file_queue import FileTaskQueue

ENVS = ["grid_cliff", "grid_sparse", "grid_trap"]

# Single-modality misspec specifications.
# Each entry: (condition_label, modalities_active, overrides applied on top of
# the optuna BASE_CONFIG + MODALITY_PARAMS for the active modalities.)
SINGLE_MOD = {
    "pref":   {"pref_data_rationality": 0.01},
    "demo":   {"demo_rationality": 0.1, "demo_model_rationality": 10.0},
    "stop":   {"stop_c": 0.1, "stop_model_c": 2.0},
    "rating": {"rating_noise_std": 2.0},
}

# PDRS conditions: keys are condition_labels, values are the misspec overrides
# applied on top of the all-correct PDRS base.
PDRS_CONDITIONS = {
    "all_misspec": {
        "pref_data_rationality": 0.01,
        "demo_rationality": 0.1,
        "demo_model_rationality": 10.0,
        "stop_c": 0.1,
        "stop_model_c": 2.0,
        "rating_noise_std": 2.0,
    },
    "pref_misspec":   {"pref_data_rationality": 0.01},
    "demo_misspec":   {"demo_rationality": 0.1, "demo_model_rationality": 10.0},
    "stop_misspec":   {"stop_c": 0.1, "stop_model_c": 2.0},
    "rating_misspec": {"rating_noise_std": 2.0},
}

# Modality short names → train.py sample-count keys.
SAMPLE_KEYS = {
    "pref": "n_pref_samples",
    "demo": "n_demo_samples",
    "rating": "n_rating_samples",
    "stop": "n_stop_samples",
}


def build_config(
    *,
    env: str,
    active_modalities: list[str],
    hparams_entry: dict,
    misspec_overrides: dict,
    condition_label: str,
    wandb_project: str | None,
) -> dict:
    """Assemble one experiment config dict."""
    opt_cfg = load_optuna_config(f"{env}_fixed_paper")
    cfg = deepcopy(opt_cfg.BASE_CONFIG)

    # Well-specified modality parameters for every active modality.
    for mod in active_modalities:
        cfg.update(opt_cfg.MODALITY_PARAMS[mod])

    # Sample counts (active ones from optuna best trial; inactive set to 0).
    for mod, key in SAMPLE_KEYS.items():
        if mod in active_modalities:
            cfg[key] = hparams_entry["sample_counts"][key]
        else:
            cfg[key] = 0

    # Tuned hyperparameters.
    cfg["td_error_weight"] = hparams_entry["td_error_weight"]
    cfg["kl_weight"] = hparams_entry["kl_weight"]
    cfg["lr"] = hparams_entry["lr"]
    cfg["batch_size"] = hparams_entry["batch_size"]
    cfg["encoder_hidden_sizes"] = list(hparams_entry["encoder_hidden_sizes"])
    cfg["use_importance_weights"] = hparams_entry["use_importance_weights"]

    # Training schedule (num_epochs, val_every_n_epochs, final_eval_model
    # default, patience default) is inherited verbatim from the
    # `<env>_fixed_paper` optuna BASE_CONFIG so the well-specified baseline
    # matches the optuna best-trial result.

    # Misspecification overrides (final word).
    cfg.update(misspec_overrides)

    # Logging + analysis tagging. condition_label is unused by train.py but
    # the worker filters unknown args via hasattr(), so it is safe to keep
    # and is picked up by the queue/summary tooling.
    cfg["log_wandb"] = True
    if wandb_project is not None:
        cfg["wandb_project"] = wandb_project
    cfg["condition_label"] = condition_label
    cfg["misspec_axis"] = condition_label  # alias for analysis filters

    return cfg


def enqueue_env(
    env: str,
    hparams: dict,
    queue_dir: Path,
    seeds: list[int],
    wandb_project: str | None,
    dry_run: bool,
) -> tuple[int, int]:
    """Enqueue all 90 runs for one grid env. Returns (added, skipped)."""
    queue = None if dry_run else FileTaskQueue(queue_dir)
    added = 0
    skipped = 0

    # ---- Single-modality misspec conditions ----
    for mod, overrides in SINGLE_MOD.items():
        entry = hparams[env][mod]
        if entry is None:
            print(f"  SKIP {env}/{mod}: no hparams entry", file=sys.stderr)
            continue
        cfg = build_config(
            env=env,
            active_modalities=[mod],
            hparams_entry=entry,
            misspec_overrides=overrides,
            condition_label=f"{mod}_only_misspec",
            wandb_project=wandb_project,
        )
        for s in seeds:
            if dry_run:
                added += 1
            else:
                exp_id = queue.insert_experiment(cfg, seed=s, skip_existing=True)
                if exp_id is None:
                    skipped += 1
                else:
                    added += 1

    # ---- PDRS conditions ----
    entry = hparams[env]["pdrs"]
    if entry is None:
        print(f"  SKIP {env}/pdrs: no hparams entry", file=sys.stderr)
        return added, skipped

    for cond_label, overrides in PDRS_CONDITIONS.items():
        cfg = build_config(
            env=env,
            active_modalities=["pref", "demo", "rating", "stop"],
            hparams_entry=entry,
            misspec_overrides=overrides,
            condition_label=f"pdrs_{cond_label}",
            wandb_project=wandb_project,
        )
        for s in seeds:
            if dry_run:
                added += 1
            else:
                exp_id = queue.insert_experiment(cfg, seed=s, skip_existing=True)
                if exp_id is None:
                    skipped += 1
                else:
                    added += 1

    return added, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hparams", default=str(REPO / "scripts" / "misspec_hparams.json"))
    ap.add_argument("--out-root", default=str(REPO / "experiments"),
                    help="Parent dir; one queue dir 'misspec_<env>' is created per grid.")
    ap.add_argument("--seeds", type=int, default=10, help="Number of seeds per condition (uses range(seeds)).")
    ap.add_argument("--wandb-project", default="mavrl-misspec",
                    help="W&B project name written into each config. Pass '' to leave unset.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't write anything; just print counts.")
    args = ap.parse_args()

    hparams = json.loads(Path(args.hparams).read_text())
    seeds = list(range(args.seeds))
    wandb_project = args.wandb_project or None

    total_added = 0
    total_skipped = 0
    for env in ENVS:
        queue_dir = Path(args.out_root) / f"misspec_{env}"
        mode = "[DRY-RUN] " if args.dry_run else ""
        print(f"\n=== {mode}{env}  →  {queue_dir} ===")
        added, skipped = enqueue_env(env, hparams, queue_dir, seeds, wandb_project, args.dry_run)
        total_added += added
        total_skipped += skipped
        print(f"  added={added}  skipped(existing)={skipped}")

    print(f"\nTotal: added={total_added}, skipped={total_skipped}")


if __name__ == "__main__":
    main()
