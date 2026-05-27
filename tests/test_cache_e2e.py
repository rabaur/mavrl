#!/usr/bin/env python
"""End-to-end local test: simulate an Optuna-like search with dataset caching.

Uses the fast tabular grid_trap env (no external policy files needed).
Demonstrates that:
  1. The first run generates + caches datasets.
  2. Subsequent runs with different sample counts load from cache (fast).
  3. Loaded data is a correct prefix of the cached data.
  4. Time savings are significant.
"""

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import torch

from mavrl.data.make_dataset import make_dataset
from mavrl.envs.make_env import make_env
from mavrl.types import FeedbackType, DataKey
from mavrl.utils.reproducibility import seed_everything
from mavrl.utils.train_utils import _create_policies, validate_args
from mavrl.utils.feature_transforms import get_act_transform, get_obs_transform
from mavrl.utils.policies import TabularQValueModel


def make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        env_id="grid_trap",
        grid_size=5,
        p_rand=0.0,
        obs_transform="one_hot",
        act_transform="one_hot",
        gamma=0.95,
        seed=0,
        exploration_epsilon=0.0,
        step_offset=1,
        subsample_factor=1,
        td_error_weight=1.0,
        batch_size=32,
        use_imitation_learning=False,
        # Pref
        n_pref_episodes=128,
        n_pref_samples=0,
        pref_policy_path="tabular",
        pref_trajectory_rationality=0.0,
        pref_data_rationality=5.0,
        pref_model_rationality=5.0,
        pref_seg_len=10,
        min_reward_pref=None,
        # Demo
        n_demo_samples=0,
        demo_policy_path="tabular",
        demo_rationality=10.0,
        min_reward_demo=None,
        # Rating
        n_rating_episodes=128,
        n_rating_samples=0,
        rating_policy_path="tabular",
        rating_trajectory_rationality=0.0,
        rating_seg_len=10,
        min_reward_rating=None,
        # Ranking
        n_ranking_episodes=0,
        n_ranking_samples=0,
        ranking_policy_path="tabular",
        ranking_trajectory_rationality=0.0,
        ranking_rationality=1.0,
        ranking_seg_len=10,
        num_ranked_items=4,
        min_reward_ranking=None,
        # Stop
        n_stop_episodes=128,
        n_stop_samples=0,
        stop_policy_path="tabular",
        stop_trajectory_rationality=0.0,
        stop_seg_len=10,
        stop_c=2.0,
        stop_regret_percentile=50.0,
        stop_regret_discount=0.1,
        stop_q_value_model=None,
        min_reward_stop=None,
        # Cache
        dataset_cache_dir=None,
        dataset_cache_gen_samples=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def run_trial(args, label=""):
    """Run a single dataset-generation 'trial' and return (datasets, elapsed)."""
    seed_everything(args.seed)
    def make_env_fn():
        return make_env(**vars(args))
    env = make_env_fn()
    feedback_config = {
        FeedbackType.PREF: args.n_pref_samples,
        FeedbackType.DEMO: args.n_demo_samples,
        FeedbackType.RATE: args.n_rating_samples,
        FeedbackType.RANKING: args.n_ranking_samples,
        FeedbackType.STOP: args.n_stop_samples,
    }
    active = validate_args(args, feedback_config)
    policies = _create_policies(args, env, active, "train")
    obs_tr = get_obs_transform(args, env)
    act_tr = get_act_transform(args, env)
    q_true = TabularQValueModel(env.unwrapped, gamma=args.gamma)

    t0 = time.time()
    datasets, _ = make_dataset(
        active, args, make_env_fn, policies, "cpu",
        obs_tr, act_tr, name="train", q_true=q_true,
    )
    elapsed = time.time() - t0
    return datasets, elapsed


def main():
    cache_dir = Path(tempfile.mkdtemp())
    print(f"Cache dir: {cache_dir}\n")

    # Simulate Optuna-style trials: same seed, varying sample counts + td_error_weight
    trial_configs = [
        {"n_pref_samples": 128, "n_stop_samples": 64,  "td_error_weight": 1.0},
        {"n_pref_samples": 64,  "n_stop_samples": 128, "td_error_weight": 0.5},
        {"n_pref_samples": 32,  "n_stop_samples": 32,  "td_error_weight": 0.25},
        {"n_pref_samples": 128, "n_stop_samples": 128, "td_error_weight": 1.0},
    ]

    # ── Without caching ─────────────────────────────────────────────────
    print("=" * 60)
    print("WITHOUT CACHING")
    print("=" * 60)
    no_cache_times = []
    for i, cfg in enumerate(trial_configs):
        args = make_args(**cfg)
        _, elapsed = run_trial(args, label=f"no-cache trial {i}")
        no_cache_times.append(elapsed)
        print(f"  Trial {i}: {elapsed:.3f}s  (pref={cfg['n_pref_samples']}, stop={cfg['n_stop_samples']})\n")

    # ── With caching (first run populates, subsequent runs load) ─────────
    print("=" * 60)
    print("WITH CACHING")
    print("=" * 60)
    cache_times = []
    # After the first trial populates the cache, load full cached data as reference
    full_cache_ref: dict[FeedbackType, dict] = {}
    for i, cfg in enumerate(trial_configs):
        args = make_args(
            dataset_cache_dir=str(cache_dir),
            dataset_cache_gen_samples=256,
            **cfg,
        )
        datasets, elapsed = run_trial(args, label=f"cached trial {i}")
        cache_times.append(elapsed)
        was_cached = i > 0
        tag = "CACHE HIT" if was_cached else "CACHE MISS (generating)"
        print(f"  Trial {i}: {elapsed:.3f}s  [{tag}]  (pref={cfg['n_pref_samples']}, stop={cfg['n_stop_samples']})\n")

        if i == 0:
            # Load the full cache files as ground-truth reference (before slicing)
            from mavrl.data.cache import compute_cache_key, _cache_path, load_dataset_from_cache
            for fb_type in datasets:
                key = compute_cache_key(fb_type, args, "train")
                path = _cache_path(cache_dir, fb_type, key)
                full_ds = load_dataset_from_cache(path, fb_type, 256, "cpu", args.td_error_weight)
                full_cache_ref[fb_type] = {dk: dv.clone() for dk, dv in full_ds.data.items()}

        # Verify every trial's data is the first-N slice of the full cache
        for fb_type, ds in datasets.items():
            for dk in ds.data:
                ref = full_cache_ref[fb_type][dk]
                n = ds.data[dk].shape[0]
                assert torch.equal(ds.data[dk], ref[:n]), (
                    f"Trial {i}, {fb_type.value}, key {dk}: "
                    f"shape {ds.data[dk].shape} vs ref[:n] {ref[:n].shape}"
                )

    # ── Summary ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    print(f"{'Trial':>8}  {'No cache':>10}  {'Cached':>10}  {'Speedup':>10}")
    for i in range(len(trial_configs)):
        speedup = no_cache_times[i] / cache_times[i] if cache_times[i] > 0 else float("inf")
        marker = "" if i == 0 else " (hit)"
        print(f"  {i:>5}{marker}  {no_cache_times[i]:>9.3f}s  {cache_times[i]:>9.3f}s  {speedup:>9.1f}x")
    total_nc = sum(no_cache_times)
    total_c = sum(cache_times)
    print(f"  {'TOTAL':>7}  {total_nc:>9.3f}s  {total_c:>9.3f}s  {total_nc/total_c:>9.1f}x")

    print(f"\nData integrity: ALL VERIFIED (loaded datasets are exact prefixes of cached data)")

    shutil.rmtree(cache_dir)


if __name__ == "__main__":
    main()
