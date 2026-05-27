"""Layer 2: validation reproducibility.

Pins ``run_validation`` on the tabular path: same model + same val
dataloaders -> identical metrics dict (NLL, KL, TD, EPIC, regret,
spearman, ...). For tabular envs this is fully analytical (no MC) so the
test should hold to bitwise precision.

This also serves as a smoke test for the (otherwise unguarded) computation
of ``compute_regret`` on the tabular branch, which is reached from
``run_validation`` and uses ``regret_tabular`` ([regret.py:126](mavrl/evaluation/regret.py))
under the hood.
"""

from __future__ import annotations

import argparse
from copy import deepcopy

import torch

from mavrl.data.make_dataset import make_dataset
from mavrl.envs.env_types import TabularEnv
from mavrl.envs.make_env import make_env
from mavrl.types import FeedbackType
from mavrl.utils.feature_transforms import get_act_transform, get_obs_transform
from mavrl.utils.gym import get_act_dim, get_obs_dim
from mavrl.utils.policies import TabularQValueModel
from mavrl.utils.reproducibility import seed_everything
from mavrl.utils.train_utils import (
    _create_model,
    _create_policies,
    run_validation,
    validate_args,
)

from tests.repro._repro_helpers import metrics_equal


def _build_validation_inputs(args: argparse.Namespace):
    """Return ``(fb_model, val_dataloaders, active, env, make_env_fn)``."""
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

    _, val_dataloaders = make_dataset(
        active, args, make_env_fn, policies, "cpu",
        obs_tr, act_tr, name="val", q_true=q_true,
    )

    obs_dim = get_obs_dim(env, obs_tr)
    act_dim = get_act_dim(env, act_tr)
    fb_model, _ = _create_model(args, env, obs_dim, act_dim, active, "cpu")

    return fb_model, val_dataloaders, active, env, make_env_fn


def test_run_validation_metrics_bitwise_equal(tabular_args):
    """Same model + same val data -> identical eval metrics dict."""
    fb_a, val_a, active_a, env_a, make_env_fn_a = _build_validation_inputs(tabular_args)
    is_tabular_a = isinstance(env_a.unwrapped, TabularEnv)
    metrics_a = run_validation(
        fb_a, val_a, active_a, is_tabular_a, env_a,
        make_env_fn_a, optimal_policy=None, args=tabular_args,
    )

    fb_b, val_b, active_b, env_b, make_env_fn_b = _build_validation_inputs(tabular_args)
    is_tabular_b = isinstance(env_b.unwrapped, TabularEnv)
    metrics_b = run_validation(
        fb_b, val_b, active_b, is_tabular_b, env_b,
        make_env_fn_b, optimal_policy=None, args=tabular_args,
    )

    ok, reason = metrics_equal(metrics_a, metrics_b)
    assert ok, reason


def test_run_validation_emits_expected_keys(tabular_args):
    """Sanity-check the metric-key contract so a future rename doesn't go
    silently un-pinned by ``test_run_validation_metrics_bitwise_equal``.
    """
    fb, val, active, env, make_env_fn = _build_validation_inputs(tabular_args)
    is_tabular = isinstance(env.unwrapped, TabularEnv)
    metrics = run_validation(
        fb, val, active, is_tabular, env,
        make_env_fn, optimal_policy=None, args=tabular_args,
    )
    expected = {
        "eval/regret",
        "eval/mean_rew",
        "eval/discounted_value",
        "eval/epic_distance",
    }
    missing = expected - set(metrics.keys())
    assert not missing, f"validation metrics missing keys: {missing}"

    # Per-modality NLL/KL/TD and spearman.
    for fb_type in active:
        v = fb_type.value
        for prefix in ("eval/negative_log_likelihood_",
                       "eval/kl_divergence_",
                       "eval/td_error_",
                       "eval/spearman_"):
            assert f"{prefix}{v}" in metrics, f"missing key {prefix}{v}"


def test_validation_does_not_mutate_model_weights(tabular_args):
    """``run_validation`` must not silently update parameters."""
    fb, val, active, env, make_env_fn = _build_validation_inputs(tabular_args)
    is_tabular = isinstance(env.unwrapped, TabularEnv)
    sd_before = deepcopy(fb.state_dict())
    run_validation(
        fb, val, active, is_tabular, env,
        make_env_fn, optimal_policy=None, args=tabular_args,
    )
    sd_after = fb.state_dict()
    for k in sd_before:
        if isinstance(sd_before[k], torch.Tensor):
            assert torch.equal(sd_before[k], sd_after[k]), (
                f"run_validation mutated parameter {k}"
            )
