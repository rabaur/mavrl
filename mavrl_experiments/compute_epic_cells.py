"""Post-hoc EPIC distance per (env, subset) cell.

For each (env, subset) Optuna study under the given storage root, loads the
best completed trial, reads the per-seed checkpoint paths from
``trial.user_attrs['model_paths']``, computes EPIC distance per checkpoint,
and writes a JSON sidecar:

    {
      "<env>": {
        "<subset>": {
          "mean": float, "stderr": float, "std": float,
          "n_seeds": int, "per_seed": [..], "trial": int
        }, ...
      }, ...
    }

The output JSON is consumed by ``equal_budget_table`` / ``fixed_allocation_table``
via their ``--epic-json`` flag to fill the EPIC group of the combined table.

Usage:
    # fixed-allocation paper variant
    python -m mavrl_experiments.compute_epic_cells \\
        --storage-root /local/copy/optuna_fixalloc_gridseed20 \\
        --variant paper \\
        --out epic_fixalloc_paper.json

    # equal-budget at b=64
    python -m mavrl_experiments.compute_epic_cells \\
        --storage-root /local/copy/optuna_eqbudget_gridseed20 \\
        --budget 64 \\
        --out epic_eqbudget_b64.json

Path relocation: if the absolute ``model_paths`` recorded on the cluster
don't exist locally, the script falls back to resolving relative to the
study log file's ``.log.models/`` sibling. So a vanilla ``scp -r`` of the
storage root just works.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import optuna
import torch

optuna.logging.set_verbosity(optuna.logging.WARNING)

from mavrl.encoder.averaged_reward_encoder import reconstruct_reward_encoder
from mavrl.envs.make_env import make_env
from mavrl.evaluation.epic import evaluate_epic_distance
from mavrl.utils.feature_transforms import get_act_transform, get_obs_transform
from mavrl_experiments.camera_ready import list_model_paths
from mavrl_experiments.optuna_search import load_env_config
from train import get_default_args


GRID_ENVS = ["grid_cliff", "grid_sparse", "grid_trap"]
SUBSETS = [
    "pref", "demo", "rating", "stop",
    "demo_pref", "pref_rating", "pref_stop",
    "demo_rating", "demo_stop", "rating_stop", "pdrs",
]


def _relocate_path(stored_path: str, study_log: Path) -> Path | None:
    """Resolve a checkpoint path that may have an absolute cluster prefix.

    Tries the literal path first; falls back to ``<study_log>.models/...`` so
    the script works after copying the storage root locally.
    """
    p = Path(stored_path).expanduser()
    if p.exists():
        return p
    parts = stored_path.split(".log.models/")
    if len(parts) == 2:
        candidate = Path(str(study_log) + ".models") / parts[1]
        if candidate.exists():
            return candidate
    return None


def _best_trial_paths(study_log: Path) -> tuple[list[Path], int | None]:
    """Return (resolved model paths, trial number) for the best COMPLETE trial."""
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(study_log)),
    )
    study = optuna.load_study(study_name=study_log.stem, storage=storage)
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value not in (float("inf"), float("-inf"))
    ]
    if not completed:
        return [], None
    best = max(completed, key=lambda t: t.value)  # studies are all 'maximize'
    raw_paths = best.user_attrs.get("model_paths", []) or []
    resolved: list[Path] = []
    for rp in raw_paths:
        p = _relocate_path(rp, study_log)
        if p is None:
            print(f"  WARN: cannot resolve {rp}", file=sys.stderr)
            continue
        resolved.append(p)
    return resolved, best.number


def _build_env_and_transforms(env: str, base_config: dict):
    """Construct (base_env, act_transform, obs_transform) the same way the
    trainer does: train defaults (e.g. ``p_rand=0.0``) overridden by the env's
    BASE_CONFIG (e.g. ``grid_size=10``). GridEnv ignores extra kwargs.
    """
    env_kwargs = {**vars(get_default_args()), **base_config}
    env_kwargs.setdefault("env_id", env)
    env_kwargs.setdefault("seed", 0)
    base_env = make_env(**env_kwargs)
    args_ns = types.SimpleNamespace(
        act_transform=base_config.get("act_transform"),
        obs_transform=base_config.get("obs_transform"),
    )
    act_transform = get_act_transform(args_ns, base_env)
    obs_transform = get_obs_transform(args_ns, base_env)
    return base_env, act_transform, obs_transform


def _epic_for_paths(
    env: str, paths: list[Path], base_config: dict, gamma_override: float | None,
) -> list[float]:
    base_env, act_transform, obs_transform = _build_env_and_transforms(
        env, base_config,
    )
    gamma = gamma_override if gamma_override is not None else base_config.get("gamma", 0.99)
    epics: list[float] = []
    for p in paths:
        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
        encoder = reconstruct_reward_encoder(
            ckpt, base_env,
            act_transform=act_transform, obs_transform=obs_transform,
            device="cpu",
        )
        encoder.eval()
        epics.append(float(evaluate_epic_distance(base_env, encoder, gamma)))
    return epics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--storage-root", type=str, default=None)
    source.add_argument(
        "--camera-ready-root", type=str, default=None,
        help="Camera-ready results directory. Mutually exclusive with "
             "--storage-root. Requires --setting to choose fixed_allocation "
             "or equal_budget.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--variant", type=str, default=None,
        help="Fixed-allocation variant (e.g. 'paper'). Study suffix: "
             "<subset>_fixed_<variant> (empty variant → <subset>_fixed). "
             "Required when using --storage-root.",
    )
    mode.add_argument(
        "--budget", type=int, default=None,
        help="Equal-budget mode. Study suffix: <subset>_b<budget>. "
             "Required when using --storage-root.",
    )
    parser.add_argument(
        "--setting", type=str, default=None,
        choices=["fixed_allocation", "equal_budget"],
        help="Setting name (required with --camera-ready-root).",
    )
    parser.add_argument("--envs", nargs="+", default=GRID_ENVS)
    parser.add_argument(
        "--gamma", type=float, default=None,
        help="Override gamma. Default: read each env's gamma from its "
             "BASE_CONFIG (grids use 0.95).",
    )
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    use_camera_ready = args.camera_ready_root is not None
    if use_camera_ready:
        cr_root = Path(args.camera_ready_root).expanduser()
        if not cr_root.exists():
            parser.error(f"camera-ready root does not exist: {cr_root}")
        if args.setting is None:
            parser.error("--setting is required with --camera-ready-root")
        setting = args.setting
    else:
        storage_root = Path(args.storage_root).expanduser()
        if not storage_root.exists():
            parser.error(f"storage root does not exist: {storage_root}")
        if args.variant is None and args.budget is None:
            parser.error("--variant or --budget is required with --storage-root")
        if args.variant is not None:
            suffix_template = (
                f"{{subset}}_fixed_{args.variant}" if args.variant else "{subset}_fixed"
            )
        else:
            suffix_template = f"{{subset}}_b{args.budget}"

    out: dict = {}
    for env in args.envs:
        try:
            base_config = load_env_config(env)["base_config"]
        except Exception as e:
            print(f"  err loading config for {env}: {e}", file=sys.stderr)
            continue
        out[env] = {}
        print(f"[{env}] (gamma={base_config.get('gamma')}, "
              f"grid_size={base_config.get('grid_size', '-')})")
        for subset in SUBSETS:
            if use_camera_ready:
                paths = list_model_paths(cr_root, setting, env, subset)
                trial_num = None
            else:
                env_dir = storage_root / env
                if not env_dir.is_dir():
                    print(f"miss env dir: {env_dir}", file=sys.stderr)
                    break
                suffix = suffix_template.format(subset=subset)
                study_log = env_dir / f"{env}_{suffix}.log"
                if not study_log.exists():
                    print(f"  miss: {study_log.name}", file=sys.stderr)
                    continue
                try:
                    paths, trial_num = _best_trial_paths(study_log)
                except Exception as e:
                    print(f"  err loading {study_log.name}: {e}", file=sys.stderr)
                    continue
            if not paths:
                label = f"{env}/{subset}" if use_camera_ready else study_log.name
                print(f"  no model_paths: {label}", file=sys.stderr)
                continue
            try:
                epics = _epic_for_paths(env, paths, base_config, args.gamma)
            except Exception as e:
                label = f"{env}/{subset}" if use_camera_ready else study_log.name
                print(f"  EPIC fail {label}: {e}", file=sys.stderr)
                continue
            arr = np.asarray(epics, dtype=float)
            n = len(arr)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if n > 1 else 0.0
            stderr = std / np.sqrt(n) if n > 1 else 0.0
            out[env][subset] = {
                "mean": mean,
                "stderr": stderr,
                "std": std,
                "n_seeds": n,
                "per_seed": [float(x) for x in arr],
                "trial": trial_num,
            }
            print(f"  {subset:<13} mean={mean:.4f}  stderr={stderr:.4f}  n={n}")

    Path(args.out).expanduser().write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
