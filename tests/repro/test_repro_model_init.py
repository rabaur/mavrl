"""Layer 2: model-initialization reproducibility.

Pins ``_create_model`` and the ``MultiFeedbackTypeModel`` constructor: with
the same seed, the same architecture must produce a bitwise-identical
``state_dict``. Failure here means torch's default initializer changed,
``encoder_hidden_sizes`` lost its ordering, or someone introduced a stray
``torch.randn`` call between ``seed_everything`` and the linear layer
constructors.
"""

from __future__ import annotations

import argparse

import torch

from mavrl.envs.make_env import make_env
from mavrl.types import FeedbackType
from mavrl.utils.feature_transforms import get_act_transform, get_obs_transform
from mavrl.utils.gym import get_act_dim, get_obs_dim
from mavrl.utils.reproducibility import seed_everything
from mavrl.utils.train_utils import _create_model

from tests.repro._repro_helpers import state_dict_equal


def _build_model(args: argparse.Namespace):
    """Build ``(fb_model, reward_encoder)`` after reseeding globals."""
    seed_everything(args.seed, quiet=True)

    env = make_env(**vars(args))
    obs_tr = get_obs_transform(args, env)
    act_tr = get_act_transform(args, env)
    obs_dim = get_obs_dim(env, obs_tr)
    act_dim = get_act_dim(env, act_tr)

    active = {
        FeedbackType.PREF: args.n_pref_samples,
        FeedbackType.DEMO: args.n_demo_samples,
        FeedbackType.RATE: args.n_rating_samples,
        FeedbackType.STOP: args.n_stop_samples,
    }
    active = {k: v for k, v in active.items() if v > 0}
    return _create_model(args, env, obs_dim, act_dim, active, "cpu")


def test_create_model_state_dict_bitwise_equal(tabular_args):
    """Same seed -> identical model weights."""
    fb_model_a, _ = _build_model(tabular_args)
    fb_model_b, _ = _build_model(tabular_args)
    ok, reason = state_dict_equal(fb_model_a.state_dict(), fb_model_b.state_dict())
    assert ok, reason


def test_create_model_different_seed_differs(tabular_args):
    """Sanity: different seed -> different weights."""
    fb_model_a, _ = _build_model(tabular_args)

    other = argparse.Namespace(**vars(tabular_args))
    other.seed = tabular_args.seed + 1
    fb_model_b, _ = _build_model(other)

    ok, _ = state_dict_equal(fb_model_a.state_dict(), fb_model_b.state_dict())
    assert not ok, "different seeds produced identical weights"


def test_optimizer_initial_state_equal(tabular_args):
    """Sanity: a freshly-constructed AdamW has empty optimizer state, so two
    runs always agree pre-step. Pin this so the next test has a baseline.
    """
    fb_model_a, _ = _build_model(tabular_args)
    opt_a = torch.optim.AdamW(lr=tabular_args.lr, params=fb_model_a.parameters())

    fb_model_b, _ = _build_model(tabular_args)
    opt_b = torch.optim.AdamW(lr=tabular_args.lr, params=fb_model_b.parameters())

    # state_dict of a fresh AdamW: 'state' is empty, 'param_groups' should match.
    sd_a, sd_b = opt_a.state_dict(), opt_b.state_dict()
    assert sd_a["state"] == {}
    assert sd_b["state"] == {}
    # param_groups: ignore the ``params`` field (parameter ids differ between
    # runs because the tensors are distinct objects).
    pg_a = [{k: v for k, v in g.items() if k != "params"} for g in sd_a["param_groups"]]
    pg_b = [{k: v for k, v in g.items() if k != "params"} for g in sd_b["param_groups"]]
    assert pg_a == pg_b
