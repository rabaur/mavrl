"""Layer 2: training-step reproducibility.

Pins ``train_step`` and ``train_epoch``: with the same seed, the same model
+ dataloaders, after a fixed number of optimizer steps, the model weights
must be bitwise equal.

This test is the canary for the (currently unguarded) reparameterization in
``MultiFeedbackTypeModel.forward`` ([multi_fb_model.py:40](mavrl/multi_fb_model.py)),
which calls ``self.encoder.sample(r_mu, r_log_var)`` *without* an explicit
generator. If anyone introduces a stray ``torch.randn`` call upstream of
that line, this test starts failing.
"""

from __future__ import annotations

import argparse
from copy import deepcopy

import torch

from mavrl.data.make_dataset import make_dataset
from mavrl.envs.make_env import make_env
from mavrl.types import FeedbackType
from mavrl.utils.feature_transforms import get_act_transform, get_obs_transform
from mavrl.utils.gym import get_act_dim, get_obs_dim
from mavrl.utils.policies import TabularQValueModel
from mavrl.utils.reproducibility import seed_everything
from mavrl.utils.train_utils import (
    _compute_importance_weights,
    _create_model,
    _create_policies,
    train_epoch,
    train_step,
    validate_args,
)

from tests.repro._repro_helpers import state_dict_equal


def _build_training_state(args: argparse.Namespace):
    """Build (model, optimizer, dataloaders, dloader_iters, active, weights)."""
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

    train_datasets, train_dataloaders = make_dataset(
        active, args, make_env_fn, policies, "cpu",
        obs_tr, act_tr, name="train", q_true=q_true,
    )

    obs_dim = get_obs_dim(env, obs_tr)
    act_dim = get_act_dim(env, act_tr)
    fb_model, _ = _create_model(args, env, obs_dim, act_dim, active, "cpu")
    optimizer = torch.optim.AdamW(lr=args.lr, params=fb_model.parameters())

    importance_weights = _compute_importance_weights(args, train_datasets, active)
    dloader_iters = {k: iter(train_dataloaders[k]) for k in active}

    return (
        fb_model, optimizer, train_dataloaders, dloader_iters,
        active, importance_weights,
    )


def test_train_step_bitwise_equal(tabular_args):
    """One ``train_step`` -> identical weights run-to-run."""
    fb_a, opt_a, dls_a, iters_a, active_a, weights_a = _build_training_state(tabular_args)
    sd_before_a = deepcopy(fb_a.state_dict())
    train_step(fb_a, opt_a, iters_a, dls_a, active_a, tabular_args, weights_a)

    fb_b, opt_b, dls_b, iters_b, active_b, weights_b = _build_training_state(tabular_args)
    sd_before_b = deepcopy(fb_b.state_dict())
    train_step(fb_b, opt_b, iters_b, dls_b, active_b, tabular_args, weights_b)

    # Sanity: pre-step weights match.
    ok, reason = state_dict_equal(sd_before_a, sd_before_b)
    assert ok, f"pre-step weights differ: {reason}"

    # Sanity: weights actually changed (otherwise the test is vacuous).
    same, _ = state_dict_equal(sd_before_a, fb_a.state_dict())
    assert not same, "train_step did not update any parameter"

    ok, reason = state_dict_equal(fb_a.state_dict(), fb_b.state_dict())
    assert ok, f"post-step weights differ: {reason}"


def test_train_epoch_bitwise_equal(tabular_args):
    """One full ``train_epoch`` -> identical weights run-to-run.

    Exercises (a) reparameterization on every batch, (b) DataLoader
    shuffling via the shared ``torch.Generator``, (c) full optimizer state
    accumulation across multiple steps.
    """
    fb_a, opt_a, dls_a, iters_a, active_a, weights_a = _build_training_state(tabular_args)
    steps_per_epoch = max(len(dl) for dl in dls_a.values())
    train_epoch(fb_a, steps_per_epoch, opt_a, dls_a, active_a, weights_a,
                tabular_args, 0, iters_a)

    fb_b, opt_b, dls_b, iters_b, active_b, weights_b = _build_training_state(tabular_args)
    steps_per_epoch_b = max(len(dl) for dl in dls_b.values())
    assert steps_per_epoch == steps_per_epoch_b
    train_epoch(fb_b, steps_per_epoch_b, opt_b, dls_b, active_b, weights_b,
                tabular_args, 0, iters_b)

    ok, reason = state_dict_equal(fb_a.state_dict(), fb_b.state_dict())
    assert ok, f"post-epoch weights differ: {reason}"


def test_two_epochs_bitwise_equal(tabular_args):
    """Pin the cumulative drift case: two epochs of training agree."""
    fb_a, opt_a, dls_a, iters_a, active_a, weights_a = _build_training_state(tabular_args)
    steps = max(len(dl) for dl in dls_a.values())
    for epoch in range(2):
        train_epoch(fb_a, steps, opt_a, dls_a, active_a, weights_a,
                    tabular_args, epoch, iters_a)

    fb_b, opt_b, dls_b, iters_b, active_b, weights_b = _build_training_state(tabular_args)
    steps_b = max(len(dl) for dl in dls_b.values())
    for epoch in range(2):
        train_epoch(fb_b, steps_b, opt_b, dls_b, active_b, weights_b,
                    tabular_args, epoch, iters_b)

    ok, reason = state_dict_equal(fb_a.state_dict(), fb_b.state_dict())
    assert ok, f"weights differ after 2 epochs: {reason}"
