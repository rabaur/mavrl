#!/usr/bin/env python
"""Pre-generate and cache datasets for all seeds and modalities.

Supports two config sources:
  --config  Optuna config name (seeds × modalities loop)
  --grid    ExperimentGrid module path (deduplicates by cache key)

Usage (Optuna):
    python scripts/pregenerate_datasets.py \\
        --config lunar_lander_v3 \\
        --seeds 10 \\
        --cache_dir /scratch/mavrl/dataset_cache \\
        --gen_samples 4096 \\
        --gen_samples_demo 32

Usage (ExperimentGrid):
    python scripts/pregenerate_datasets.py \\
        --grid sweep_cartpole_full \\
        --cache_dir /scratch/mavrl/dataset_cache/cartpole_full
"""

import argparse
import sys
import time
from pathlib import Path

from mavrl_experiments.config_loader import load_experiment_config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import get_default_args
from mavrl.data.cache import compute_cache_key, _FEEDBACK_PREFIX
from mavrl.data.make_dataset import make_dataset
from mavrl.envs.make_env import make_env
from mavrl.types import FeedbackType
from mavrl.utils.reproducibility import seed_everything
from mavrl.utils.train_utils import _create_policies, validate_args
from mavrl.utils.feature_transforms import get_act_transform, get_obs_transform
from mavrl.utils.policies import TabularQValueModel, DQNQValueModel
from mavrl.envs.env_types import TabularEnv
import stable_baselines3 as sb3


MODALITY_TO_SAMPLE_KEY = {
    "pref": "n_pref_samples",
    "demo": "n_demo_samples",
    "rating": "n_rating_samples",
    "ranking": "n_ranking_samples",
    "stop": "n_stop_samples",
}

FEEDBACK_SAMPLE_KEYS = {
    FeedbackType.PREF: "n_pref_samples",
    FeedbackType.DEMO: "n_demo_samples",
    FeedbackType.RATE: "n_rating_samples",
    FeedbackType.STOP: "n_stop_samples",
}


# ── Shared generation logic ─────────────────────────────────────────────────

def generate_datasets(training_args, active, device="cpu"):
    """Generate and cache train + val datasets for one (seed, modality) combo."""
    def make_env_fn():
        return make_env(**vars(training_args))

    env = make_env_fn()
    act_transform = get_act_transform(training_args, env)
    obs_transform = get_obs_transform(training_args, env)

    q_true = None
    if FeedbackType.STOP in active:
        is_tabular = isinstance(env.unwrapped, TabularEnv)
        if is_tabular:
            q_true = TabularQValueModel(env.unwrapped, gamma=training_args.gamma)
        else:
            stop_q_path = str(Path(training_args.stop_q_value_model).expanduser())
            q_true = DQNQValueModel(sb3.DQN.load(stop_q_path, env=env, device=device))

    for split_name in ["train", "val"]:
        policies = _create_policies(training_args, env, active, split_name)
        make_dataset(
            active, training_args, make_env_fn, policies, device,
            obs_transform, act_transform, name=split_name, q_true=q_true,
        )


def generate_single_split(training_args, active, split, device="cpu"):
    """Generate and cache a single split for one (seed, modality) combo."""
    def make_env_fn():
        return make_env(**vars(training_args))

    env = make_env_fn()
    policies = _create_policies(training_args, env, active, split)
    act_transform = get_act_transform(training_args, env)
    obs_transform = get_obs_transform(training_args, env)

    q_true = None
    if FeedbackType.STOP in active:
        is_tabular = isinstance(env.unwrapped, TabularEnv)
        if is_tabular:
            q_true = TabularQValueModel(env.unwrapped, gamma=training_args.gamma)
        else:
            stop_q_path = str(Path(training_args.stop_q_value_model).expanduser())
            q_true = DQNQValueModel(sb3.DQN.load(stop_q_path, env=env, device=device))

    make_dataset(
        active, training_args, make_env_fn, policies, device,
        obs_transform, act_transform, name=split, q_true=q_true,
    )


# ── Optuna config path ──────────────────────────────────────────────────────

def run_optuna_mode(args):
    from mavrl_experiments.optuna_search import load_env_config, build_training_args, SAMPLE_COUNT_KEYS

    env_cfg = load_env_config(args.config)
    base_config = env_cfg["base_config"]
    modality_params = env_cfg["modality_params"]

    available_modalities = list(modality_params.keys())
    modalities = args.modalities or available_modalities

    print(f"Config:      {args.config}")
    print(f"Seeds:       0..{args.seeds - 1}")
    print(f"Cache dir:   {args.cache_dir}")
    print(f"Modalities:  {modalities}")
    print(f"Gen samples: {args.gen_samples} (demo: {args.gen_samples_demo})")
    print()

    total_t0 = time.time()

    for seed in range(args.seeds):
        for modality in modalities:
            sample_key = MODALITY_TO_SAMPLE_KEY[modality]
            gen_n = args.gen_samples_demo if modality == "demo" else args.gen_samples

            sample_counts = {k: (gen_n if k == sample_key else 0) for k in SAMPLE_COUNT_KEYS}
            training_args = build_training_args(base_config, modality_params, sample_counts, seed)
            training_args.dataset_cache_dir = args.cache_dir
            training_args.dataset_cache_gen_samples = None

            print(f"{'='*60}")
            print(f"Seed {seed} | Modality {modality} | {gen_n} samples")
            print(f"{'='*60}")

            seed_everything(training_args.seed)

            feedback_config = {
                FeedbackType.PREF: training_args.n_pref_samples,
                FeedbackType.DEMO: training_args.n_demo_samples,
                FeedbackType.RATE: training_args.n_rating_samples,
                FeedbackType.STOP: training_args.n_stop_samples,
            }
            active = validate_args(training_args, feedback_config)

            t0 = time.time()
            generate_datasets(training_args, active)
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s\n")

    total_elapsed = time.time() - total_t0
    print(f"All done in {total_elapsed:.1f}s")


# ── ExperimentGrid path ─────────────────────────────────────────────────────

def _config_to_args(config: dict, seed: int) -> argparse.Namespace:
    """Convert a grid config dict + seed into a train.py-compatible Namespace."""
    training_args = get_default_args()
    for key, value in config.items():
        if hasattr(training_args, key):
            setattr(training_args, key, value)
    training_args.seed = seed
    training_args.log_wandb = False
    return training_args


def _collect_unique_jobs(grid, cache_dir: str):
    """Scan every grid config and return the minimal set of generation jobs.

    Deduplicates by cache key so configs that only differ in training-only
    hyperparameters (td_error_weight, kl_weight, etc.) share one cache entry.
    """
    configs = grid.generate_configs()
    print(f"Grid has {len(configs)} total (config, seed) combinations.")

    jobs: dict[tuple[FeedbackType, str, str], dict] = {}

    for config, seed in configs:
        training_args = _config_to_args(config, seed)

        feedback_config = {
            FeedbackType.PREF: training_args.n_pref_samples,
            FeedbackType.DEMO: training_args.n_demo_samples,
            FeedbackType.RATE: training_args.n_rating_samples,
            FeedbackType.RANKING: training_args.n_ranking_samples,
            FeedbackType.STOP: training_args.n_stop_samples,
        }

        for fb_type, n_samples in feedback_config.items():
            if n_samples <= 0:
                continue

            for split in ["train", "val"]:
                key_hash = compute_cache_key(fb_type, training_args, split)
                job_key = (fb_type, key_hash, split)

                if job_key not in jobs:
                    args_copy = _config_to_args(config, seed)
                    args_copy.dataset_cache_dir = cache_dir
                    args_copy.dataset_cache_gen_samples = None
                    jobs[job_key] = {
                        "args": args_copy,
                        "feedback_type": fb_type,
                        "split": split,
                        "max_samples": n_samples,
                        "seed": seed,
                        "cache_key": key_hash,
                    }
                else:
                    jobs[job_key]["max_samples"] = max(
                        jobs[job_key]["max_samples"], n_samples
                    )

    return list(jobs.values())


def run_grid_mode(args):
    mod = load_experiment_config(args.grid)
    grid = mod.grid

    modality_filter = None
    if args.modalities:
        name_to_fb = {v: k for k, v in _FEEDBACK_PREFIX.items()}
        modality_filter = {name_to_fb[m] for m in args.modalities}

    print(f"Grid module: {args.grid}")
    print(f"Cache dir:   {args.cache_dir}")
    print()

    jobs = _collect_unique_jobs(grid, args.cache_dir)

    if modality_filter:
        jobs = [j for j in jobs if j["feedback_type"] in modality_filter]

    seeds = sorted({j["seed"] for j in jobs})
    modalities_found = sorted({_FEEDBACK_PREFIX[j["feedback_type"]] for j in jobs})
    print(f"Unique generation jobs: {len(jobs)}")
    print(f"Seeds:      {seeds}")
    print(f"Modalities: {modalities_found}")
    print()

    total_t0 = time.time()

    for i, job in enumerate(jobs, 1):
        fb_name = _FEEDBACK_PREFIX[job["feedback_type"]]
        training_args = job["args"]
        fb_type = job["feedback_type"]
        n_samples = job["max_samples"]

        sample_key = FEEDBACK_SAMPLE_KEYS[fb_type]
        for key in FEEDBACK_SAMPLE_KEYS.values():
            setattr(training_args, key, n_samples if key == sample_key else 0)

        print(f"{'='*60}")
        print(
            f"[{i}/{len(jobs)}] seed={job['seed']} | {fb_name} | "
            f"split={job['split']} | {n_samples} samples | "
            f"key={job['cache_key']}"
        )
        print(f"{'='*60}")

        seed_everything(training_args.seed)

        feedback_config = {
            FeedbackType.PREF: training_args.n_pref_samples,
            FeedbackType.DEMO: training_args.n_demo_samples,
            FeedbackType.RATE: training_args.n_rating_samples,
            FeedbackType.RANKING: training_args.n_ranking_samples,
            FeedbackType.STOP: training_args.n_stop_samples,
        }
        active = validate_args(training_args, feedback_config)

        t0 = time.time()
        generate_single_split(training_args, active, job["split"])
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s\n")

    total_elapsed = time.time() - total_t0
    print(f"All done in {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate cached datasets for experiments"
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config", type=str,
        help="Optuna config name (e.g. lunar_lander_v3)",
    )
    source.add_argument(
        "--grid", type=str,
        help="Experiment config name under configs/experiments/ "
             "(e.g. sweep_cartpole_full). Override root via $MAVRL_CONFIG_ROOT.",
    )

    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Directory to store cached datasets")
    parser.add_argument("--modalities", type=str, nargs="+", default=None,
                        help="Subset of modalities to generate (pref, demo, rating, ranking, stop)")

    # Optuna-specific options
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of seeds, 0..seeds-1 (--config mode only)")
    parser.add_argument("--gen_samples", type=int, default=4096,
                        help="Samples per non-demo modality (--config mode only)")
    parser.add_argument("--gen_samples_demo", type=int, default=32,
                        help="Samples for demonstration modality (--config mode only)")

    args = parser.parse_args()

    if args.config:
        run_optuna_mode(args)
    else:
        run_grid_mode(args)


if __name__ == "__main__":
    main()
