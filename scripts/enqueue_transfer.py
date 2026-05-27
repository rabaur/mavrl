"""Enqueue transfer evaluations directly from results/ models.

Replaces the two-stage extract_transfer_ckpts.py → enqueue_transfer.py pipeline
with a single pass that reads per-cell hparams.json + r_model_seed*.pt files
from ``results/models/fixed_allocation/<env>/<subset>/``.

Run:
    python scripts/enqueue_transfer.py --dry-run
    python scripts/enqueue_transfer.py --envs acrobot_v1 --seeds 10
    python scripts/enqueue_transfer.py --envs acrobot_v1,lunar_lander_v3 --seeds 10 --dry-run
"""
import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ----------------------------------------------------------------------------
# Per-env constants (cribbed from the old transfer_*.py configs).
# ----------------------------------------------------------------------------

# Common evaluation-side knobs.
BASE_BY_ENV = {
    "grid_cliff": {
        "env_id": "grid_cliff",
        "obs_transform": "one_hot",
        "act_transform": "one_hot",
        "reward_domain": "s",
        "gamma": 0.95,
        "wandb_project": "transfer-fixalloc-grid-cliff",
    },
    "grid_trap": {
        "env_id": "grid_trap",
        "obs_transform": "one_hot",
        "act_transform": "one_hot",
        "reward_domain": "s",
        "gamma": 0.95,
        "wandb_project": "transfer-fixalloc-grid-trap",
    },
    "acrobot_v1": {
        "env_id": "Acrobot-v1",
        "act_transform": "one_hot",
        "reward_domain": "sa",
        "gamma": 0.99,
        "num_samples": 1000,
        "max_num_steps": 1000,
        "no_progress_bar": True,
        "wandb_project": "transfer-fixalloc-acrobot",
    },
    "lunar_lander_v3": {
        "env_id": "LunarLander-v3",
        "act_transform": "one_hot",
        "reward_domain": "sa",
        "gamma": 0.999,
        "num_samples": 1000,
        "max_num_steps": 1000,
        "no_progress_bar": True,
        "wandb_project": "transfer-fixalloc-lander",
    },
}

# Perturbation grids. Each entry is a list of dicts of ``env_params.*`` overrides
# that get spread onto the config one cell at a time.
def _grid_cliff_perturbs() -> list[dict]:
    return [
        {
            "env_params.grid_size": 10,
            "env_params.gamma": 0.95,
            "env_params.p_rand": p,
        }
        for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ]


def _grid_trap_perturbs() -> list[dict]:
    # Same axis as grid_cliff.
    return [
        {
            "env_params.grid_size": 10,
            "env_params.gamma": 0.95,
            "env_params.p_rand": p,
        }
        for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ]


# Acrobot: per-asymmetry optimal policy on the cluster. Note the /dqn/ segment:
# create_policy() detects the policy type by substring-matching /ppo/ or /dqn/
# in the path, so the segment is load-bearing. (The old transfer_acrobot.py
# config dropped the /dqn/ — those runs probably never actually evaluated.)
ACROBOT_OPT_BASE = "/cluster/home/rabaur/mavrl/expert_policies/dqn/expert_models_acrobot_asymmetry"
ACROBOT_ASYMMETRY_TO_POLICY = {
    a: f"{ACROBOT_OPT_BASE}/Acrobot-v1_1_asymmetry_{a}/best_model.zip"
    for a in (1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0)
}


def _acrobot_perturbs() -> list[dict]:
    return [
        {"env_params.asymmetry": a, "optimal_policy_path": ACROBOT_ASYMMETRY_TO_POLICY[a]}
        for a in ACROBOT_ASYMMETRY_TO_POLICY
    ]


# Lander: cartesian wind_power × gravity, with optimal_policy_path keyed on wind_power.
LANDER_OPT_BY_WIND = {
    0.0: "~/mavrl/expert_policies/ppo/LunarLander-v3_2/best_model.zip",
    5.0: "~/mavrl/expert_policies/ppo/LunarLander-v3/wind_power_5.0/ppo/LunarLander-v3_2/best_model.zip",
    10.0: "~/mavrl/expert_policies/ppo/LunarLander-v3/wind_power_10.0/ppo/LunarLander-v3_1/best_model.zip",
    15.0: "~/mavrl/expert_policies/ppo/LunarLander-v3/wind_power_15.0/ppo/LunarLander-v3_1/best_model.zip",
    20.0: "~/mavrl/expert_policies/ppo/LunarLander-v3/wind_power_20.0/ppo/LunarLander-v3_1/best_model.zip",
    25.0: "~/mavrl/expert_policies/ppo/LunarLander-v3/wind_power_25.0/ppo/LunarLander-v3_1/best_model.zip",
}


def _lander_perturbs() -> list[dict]:
    out = []
    for wp in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]:
        for g in [-10.0, -11.0, -11.9]:
            out.append({
                "env_params.enable_wind": True,
                "env_params.wind_power": wp,
                "env_params.gravity": g,
                "optimal_policy_path": LANDER_OPT_BY_WIND[wp],
            })
    return out


PERTURBS = {
    "grid_cliff":      _grid_cliff_perturbs,
    "grid_trap":       _grid_trap_perturbs,
    "acrobot_v1":      _acrobot_perturbs,
    "lunar_lander_v3": _lander_perturbs,
}

from mavrl_experiments.file_queue import FileTaskQueue

SUBSET_TO_COMBO = {
    "pref":        "pref_only",
    "demo":        "demo_only",
    "rating":      "rating_only",
    "stop":        "stop_only",
    "demo_pref":   "demo+pref",
    "demo_rating": "demo+rating",
    "demo_stop":   "demo+stop",
    "pref_rating": "pref+rating",
    "pref_stop":   "pref+stop",
    "rating_stop": "rating+stop",
    "pdrs":        "demo+pref+rating+stop",
}

SEED_RE = re.compile(r"r_model_seed(\d+)\.pt$")


def discover_seeds(cell_dir: Path, max_seed: int | None) -> list[tuple[int, Path]]:
    """Return sorted (seed_index, path) pairs for all checkpoint files in cell_dir."""
    found = []
    for f in cell_dir.iterdir():
        m = SEED_RE.match(f.name)
        if m:
            s = int(m.group(1))
            if max_seed is None or s < max_seed:
                found.append((s, f))
    found.sort()
    return found


def enqueue_env(
    env: str,
    models_root: Path,
    queue_dir: Path,
    max_seed: int | None,
    wandb_project: str | None,
    dry_run: bool,
) -> tuple[int, int, list[str]]:
    """Returns (added, skipped, missing_cells)."""
    queue = None if dry_run else FileTaskQueue(queue_dir)
    perturbs = PERTURBS[env]()
    env_dir = models_root / env

    added = 0
    skipped = 0
    missing: list[str] = []
    sample_cfg = None

    for subset_name in sorted(SUBSET_TO_COMBO):
        cell_dir = env_dir / subset_name
        if not cell_dir.is_dir():
            missing.append(subset_name)
            continue

        hp_path = cell_dir / "hparams.json"
        if not hp_path.exists():
            print(f"  WARN {env}/{subset_name}: no hparams.json", file=sys.stderr)
            missing.append(subset_name)
            continue

        hp = json.loads(hp_path.read_text())
        seeds = discover_seeds(cell_dir, max_seed)
        if not seeds:
            print(f"  WARN {env}/{subset_name}: no seed checkpoints", file=sys.stderr)
            missing.append(subset_name)
            continue

        combo = SUBSET_TO_COMBO[subset_name]

        for perturb in perturbs:
            for seed_idx, ckpt_path in seeds:
                cfg = deepcopy(BASE_BY_ENV[env])
                if wandb_project is not None:
                    cfg["wandb_project"] = wandb_project
                cfg.update(perturb)
                cfg["feedback_combo"] = combo
                cfg["fb_model_path"] = str(ckpt_path.resolve())
                cfg["mode"] = "mavrl"
                cfg["encoder_hidden_sizes"] = list(hp["encoder_hidden_sizes"])
                for k, v in hp.items():
                    if k.startswith("retrain_") or k == "n_regret_samples":
                        cfg[k] = v
                cfg["log_wandb"] = True

                if dry_run:
                    added += 1
                    if sample_cfg is None:
                        sample_cfg = cfg
                    continue
                exp_id = queue.insert_experiment(cfg, seed=seed_idx, skip_existing=True)
                if exp_id is None:
                    skipped += 1
                else:
                    added += 1

    return added, skipped, missing, sample_cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Enqueue transfer evals from results models.")
    ap.add_argument("--models-root",
                    default=str(REPO / "results" / "models" / "fixed_allocation"))
    ap.add_argument("--out-root", default=str(REPO / "experiments"))
    ap.add_argument("--envs", default=",".join(sorted(PERTURBS)),
                    help="Comma-separated env keys.")
    ap.add_argument("--seeds", type=int, default=None,
                    help="Max seed index to enqueue (default: all available).")
    ap.add_argument("--wandb-project", default=None,
                    help="Override wandb_project (else per-env default).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    envs = [e for e in args.envs.replace(",", " ").split() if e]
    unknown = [e for e in envs if e not in PERTURBS]
    if unknown:
        print(f"Unknown env(s): {unknown}. Valid: {sorted(PERTURBS)}", file=sys.stderr)
        sys.exit(2)

    models_root = Path(args.models_root)
    if not models_root.is_dir():
        print(f"Models root not found: {models_root}", file=sys.stderr)
        sys.exit(2)

    total_added = 0
    total_skipped = 0
    all_missing: list[tuple[str, str]] = []

    for env in envs:
        queue_dir = Path(args.out_root) / f"transfer_fixalloc_{env}"
        mode = "[DRY-RUN] " if args.dry_run else ""
        print(f"\n=== {mode}{env}  →  {queue_dir} ===")

        added, skipped, missing, sample_cfg = enqueue_env(
            env, models_root, queue_dir, args.seeds, args.wandb_project, args.dry_run,
        )
        for m in missing:
            all_missing.append((env, m))
        total_added += added
        total_skipped += skipped
        print(f"  added={added}  skipped(existing)={skipped}")

        if sample_cfg is not None and args.dry_run:
            print(f"\n  Sample config ({env}):")
            print(json.dumps(sample_cfg, indent=2))

    print(f"\nTotal: added={total_added}, skipped={total_skipped}")
    if all_missing:
        print("\nWARN: missing cells (no directory, hparams.json, or seed files):")
        for env, subset in all_missing:
            print(f"  {env}/{subset}")


if __name__ == "__main__":
    main()
