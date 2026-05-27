"""Diagnostic CLI: run a tabular experiment twice and report the first
layer where the two runs diverge.

Designed for human debugging when a future commit accidentally introduces
non-determinism. Reuses the same equality predicates as the pytest suite
(``tests/repro/_repro_helpers.py``) so the divergence reported here is the same
one any failing test would see.

Usage:

    python tests/repro/repro_diff.py [--seed 0] [--env grid_trap]

Output (representative):

    [primitives] OK
    [datasets]   OK (PREF, DEMO, RATE, STOP)
    [model init] OK
    [train_step]
        epoch 0 step 0: OK
        epoch 0 step 1: DIVERGE
            encoder.mean_head.weight max abs diff = 1.2e-08
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import torch

# Repo root on path: ``train``, ``mavrl``, and ``tests.*`` imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.repro._repro_helpers import (  # noqa: E402
    dataset_equal,
    first_param_diff,
    metrics_equal,
    set_pinned_threading,
    state_dict_equal,
)


# ── Plumbing ────────────────────────────────────────────────────────────────


def _build_args(seed: int, env: str) -> argparse.Namespace:
    """Minimal grid_trap args that exercise all 4 modalities."""
    from train import get_default_args

    args = get_default_args()
    overrides = dict(
        seed=seed,
        env_id=env,
        grid_size=4,
        p_rand=0.0,
        gamma=0.95,
        obs_transform="one_hot",
        act_transform="one_hot",
        reward_domain="s",
        td_error_weight=1.0,
        kl_weight=1.0,
        num_epochs=2,
        batch_size=8,
        lr=1e-3,
        encoder_hidden_sizes=[16, 16],
        val_every_n_epochs=1,
        skip_first_val_epoch=False,
        vis_every_n_epochs=None,
        early_stop_patience=None,
        log_wandb=False,
        log_every_n_steps=1,
        model_save_dir=None,
        save_behavior="best",
        skip_final_eval=False,
        n_regret_samples=4,
        retrain_verbose=0,
        retrain_pbar=False,
        n_pref_episodes=8, n_pref_samples=8, pref_policy_path="tabular",
        pref_trajectory_rationality=0.5, pref_data_rationality=2.0,
        pref_model_rationality=2.0, pref_seg_len=4, min_reward_pref=None,
        n_demo_samples=4, demo_policy_path="tabular", demo_rationality=2.0,
        demo_model_rationality=2.0, min_reward_demo=None,
        n_rating_episodes=8, n_rating_samples=8, rating_policy_path="tabular",
        rating_trajectory_rationality=2.0, rating_seg_len=4,
        min_reward_rating=None, rating_noise_std=0.0,
        n_stop_episodes=8, n_stop_samples=8, stop_policy_path="tabular",
        stop_trajectory_rationality=0.5, stop_seg_len=4, stop_c=1.0,
        stop_model_c=None, stop_regret_percentile=50.0,
        stop_regret_discount=0.5, stop_q_value_model=None, min_reward_stop=None,
        exploration_epsilon=0.0, step_offset=1, subsample_factor=1,
        use_imitation_learning=False, use_importance_weights=False,
        dataset_cache_dir=None, dataset_cache_gen_samples=None,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _build_components(args):
    """Mirror the prefix of ``run_experiment`` up to the training loop.

    Returns ``(fb_model, train_dataloaders, val_dataloaders, datasets,
    active_feedback_types, optimizer)``.
    """
    from mavrl.data.make_dataset import make_dataset
    from mavrl.envs.make_env import make_env
    from mavrl.types import FeedbackType
    from mavrl.utils.feature_transforms import get_act_transform, get_obs_transform
    from mavrl.utils.gym import get_act_dim, get_obs_dim
    from mavrl.utils.policies import TabularQValueModel
    from mavrl.utils.reproducibility import seed_everything
    from mavrl.utils.train_utils import (
        _create_model, _create_policies, validate_args,
    )

    seed_everything(args.seed, quiet=True)

    def make_env_fn():
        return make_env(**vars(args))

    env = make_env_fn()
    feedback_config = {
        FeedbackType.PREF: args.n_pref_samples,
        FeedbackType.DEMO: args.n_demo_samples,
        FeedbackType.RATE: args.n_rating_samples,
        FeedbackType.STOP: args.n_stop_samples,
    }
    active = validate_args(args, feedback_config)
    policies_train = _create_policies(args, env, active, "train")
    policies_val = _create_policies(args, env, active, "val")
    obs_tr = get_obs_transform(args, env)
    act_tr = get_act_transform(args, env)
    q_true = TabularQValueModel(env.unwrapped, gamma=args.gamma)

    train_datasets, train_dls = make_dataset(
        active, args, make_env_fn, policies_train, "cpu",
        obs_tr, act_tr, name="train", q_true=q_true,
    )
    _, val_dls = make_dataset(
        active, args, make_env_fn, policies_val, "cpu",
        obs_tr, act_tr, name="val", q_true=q_true,
    )

    obs_dim = get_obs_dim(env, obs_tr)
    act_dim = get_act_dim(env, act_tr)
    fb_model, _ = _create_model(args, env, obs_dim, act_dim, active, "cpu")
    optimizer = torch.optim.AdamW(lr=args.lr, params=fb_model.parameters())

    return fb_model, train_dls, val_dls, train_datasets, active, optimizer


# ── Diagnostic stages ────────────────────────────────────────────────────────


def _check_primitives() -> tuple[bool, str]:
    import numpy as np
    import random as pyrand
    from mavrl.utils.reproducibility import seed_everything

    seed_everything(0, quiet=True)
    a = (torch.randn(8).tolist(), np.random.rand(8).tolist(),
         [pyrand.random() for _ in range(8)])
    seed_everything(0, quiet=True)
    b = (torch.randn(8).tolist(), np.random.rand(8).tolist(),
         [pyrand.random() for _ in range(8)])
    if a != b:
        which = ["torch", "numpy", "random"]
        for i, (av, bv) in enumerate(zip(a, b)):
            if av != bv:
                return False, (
                    f"seed_everything not idempotent for {which[i]} "
                    f"(first val: {av[0]:.6f} vs {bv[0]:.6f})"
                )
    return True, "ok"


def _check_datasets(args) -> tuple[bool, str]:
    """Build all-modality datasets twice; report which modality differs first."""
    from mavrl.data.make_dataset import make_dataset
    from mavrl.envs.make_env import make_env
    from mavrl.types import FeedbackType
    from mavrl.utils.feature_transforms import get_act_transform, get_obs_transform
    from mavrl.utils.policies import TabularQValueModel
    from mavrl.utils.reproducibility import seed_everything
    from mavrl.utils.train_utils import _create_policies, validate_args

    def build_all():
        seed_everything(args.seed, quiet=True)

        def make_env_fn():
            return make_env(**vars(args))

        env = make_env_fn()
        feedback_config = {
            FeedbackType.PREF: args.n_pref_samples,
            FeedbackType.DEMO: args.n_demo_samples,
            FeedbackType.RATE: args.n_rating_samples,
            FeedbackType.STOP: args.n_stop_samples,
        }
        active = validate_args(args, feedback_config)
        policies = _create_policies(args, env, active, "train")
        obs_tr = get_obs_transform(args, env)
        act_tr = get_act_transform(args, env)
        q_true = TabularQValueModel(env.unwrapped, gamma=args.gamma)
        ds, _ = make_dataset(
            active, args, make_env_fn, policies, "cpu",
            obs_tr, act_tr, name="train", q_true=q_true,
        )
        return ds

    ds_a = build_all()
    ds_b = build_all()

    for fb_type in ds_a:
        ok, reason = dataset_equal(ds_a[fb_type], ds_b[fb_type])
        if not ok:
            return False, f"{fb_type.value}: {reason}"
    return True, "ok (all modalities)"


def _check_model_init(args) -> tuple[bool, str]:
    fb_a, *_ = _build_components(args)
    fb_b, *_ = _build_components(args)
    return state_dict_equal(fb_a.state_dict(), fb_b.state_dict())


def _check_train_step_by_step(args, num_epochs: int = 2) -> tuple[bool, str]:
    """Run two trainings step-by-step, reporting the first divergent step."""
    from mavrl.utils.train_utils import _compute_importance_weights, train_step

    fb_a, train_dls_a, _, ds_a, active_a, opt_a = _build_components(args)
    fb_b, train_dls_b, _, ds_b, active_b, opt_b = _build_components(args)

    weights_a = _compute_importance_weights(args, ds_a, active_a)
    weights_b = _compute_importance_weights(args, ds_b, active_b)

    iters_a = {k: iter(train_dls_a[k]) for k in active_a}
    iters_b = {k: iter(train_dls_b[k]) for k in active_b}

    steps_per_epoch = max(len(dl) for dl in train_dls_a.values())

    # Sanity: pre-step weights must agree (otherwise model_init is the
    # actual culprit, but the caller already checked that).
    ok, reason = state_dict_equal(fb_a.state_dict(), fb_b.state_dict())
    if not ok:
        return False, f"pre-train: {reason}"

    for epoch in range(num_epochs):
        for step in range(steps_per_epoch):
            train_step(fb_a, opt_a, iters_a, train_dls_a, active_a, args, weights_a)
            train_step(fb_b, opt_b, iters_b, train_dls_b, active_b, args, weights_b)
            ok, reason = state_dict_equal(fb_a.state_dict(), fb_b.state_dict())
            if not ok:
                hint = first_param_diff(fb_a.state_dict(), fb_b.state_dict())
                return False, f"epoch {epoch} step {step}: {hint}"
    return True, f"ok ({num_epochs} epochs, {steps_per_epoch} steps each)"


def _check_validation(args) -> tuple[bool, str]:
    from mavrl.envs.env_types import TabularEnv
    from mavrl.envs.make_env import make_env
    from mavrl.utils.train_utils import run_validation

    fb_a, _, val_a, _, active_a, _ = _build_components(args)
    fb_b, _, val_b, _, active_b, _ = _build_components(args)

    env = make_env(**vars(args))
    is_tabular = isinstance(env.unwrapped, TabularEnv)

    metrics_a = run_validation(
        fb_a, val_a, active_a, is_tabular, env,
        lambda: make_env(**vars(args)), optimal_policy=None, args=args,
    )
    metrics_b = run_validation(
        fb_b, val_b, active_b, is_tabular, env,
        lambda: make_env(**vars(args)), optimal_policy=None, args=args,
    )
    return metrics_equal(metrics_a, metrics_b)


def _check_e2e(args) -> tuple[bool, str]:
    """Final layer: ``run_experiment`` twice."""
    from train import run_experiment

    cb_a: list = []
    cb_b: list = []
    res_a = run_experiment(args, db_callback=lambda e, m: cb_a.append((e, dict(m))))
    res_b = run_experiment(args, db_callback=lambda e, m: cb_b.append((e, dict(m))))

    if len(cb_a) != len(cb_b):
        return False, f"different #callbacks: {len(cb_a)} vs {len(cb_b)}"
    for (ea, ma), (eb, mb) in zip(cb_a, cb_b):
        if ea != eb:
            return False, f"callback epoch mismatch: {ea} vs {eb}"
        ok, reason = metrics_equal(ma, mb)
        if not ok:
            return False, f"epoch {ea} callback: {reason}"

    ok, reason = metrics_equal(res_a["final_metrics"], res_b["final_metrics"])
    if not ok:
        return False, f"final_metrics: {reason}"
    return True, "ok"


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", type=str, default="grid_trap",
                        choices=["grid_trap", "grid_sparse", "grid_cliff", "chain_4"])
    parser.add_argument(
        "--skip-e2e", action="store_true",
        help="Skip the slow end-to-end stage. Useful when train_step has "
        "already isolated the divergence.",
    )
    cli = parser.parse_args()

    set_pinned_threading()
    args = _build_args(cli.seed, cli.env)

    stages = [
        ("primitives", lambda: _check_primitives()),
        ("datasets", lambda: _check_datasets(args)),
        ("model init", lambda: _check_model_init(args)),
        ("train_step", lambda: _check_train_step_by_step(args)),
        ("validation", lambda: _check_validation(args)),
    ]
    if not cli.skip_e2e:
        stages.append(("e2e", lambda: _check_e2e(args)))

    print(f"\nRunning reproducibility diagnostic (seed={cli.seed}, env={cli.env}).\n")
    first_diverge = None
    for name, fn in stages:
        try:
            ok, reason = fn()
        except Exception as exc:
            print(f"  [{name:11s}] ERROR  {type(exc).__name__}: {exc}")
            first_diverge = first_diverge or name
            break
        marker = "OK   " if ok else "DIVERGE"
        print(f"  [{name:11s}] {marker} {reason}")
        if not ok and first_diverge is None:
            first_diverge = name
            break  # first divergence: don't keep noisy-checking downstream layers

    print()
    if first_diverge is None:
        print("All stages reproducible.")
        return 0
    print(f"First divergent stage: {first_diverge}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
