"""Layer 2: SB3 PPO + non-tabular regret reproducibility.

Marked ``slow`` because PPO learn (even at the small budget used here) takes
~10 s per call. Excluded from the default pytest run via ``pytest.ini``;
opt in with ``pytest -m slow``.

Pins two things:

1. **SB3 PPO + ``DummyVecEnv`` determinism.** Two ``sb3.PPO(seed=...)``
   ``learn(total_timesteps=...)`` calls with the same seed and env factory
   produce a bitwise-identical ``policy.state_dict()``. This is the
   foundation that ``compute_regret`` non-tabular relies on.

2. **``compute_regret`` non-tabular determinism.** Same reward model + same
   seed -> identical ``regret``, ``mean_rew``. Exercises the
   ``joblib.Parallel(backend="loky")`` MC rollout path
   ([regret.py:294](mavrl/evaluation/regret.py)).

We monkeypatch ``train_ppo`` in the regret test with a stub that uses tiny
budgets so the test finishes quickly without running full SB3 training.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import stable_baselines3 as sb3
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from mavrl.utils.reproducibility import seed_everything

from tests.repro._repro_helpers import state_dict_equal


pytestmark = pytest.mark.slow


# ── 1. SB3 + DummyVecEnv determinism ─────────────────────────────────────────


def _train_tiny_ppo(seed: int, n_timesteps: int = 1024, n_envs: int = 2):
    """Train a tiny PPO on CartPole. Returns ``policy.state_dict()``."""
    seed_everything(seed, quiet=True)

    def make_env():
        env = gym.make("CartPole-v1")
        env.reset(seed=seed)
        return env

    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
    model = sb3.PPO(
        env=vec_env,
        policy="MlpPolicy",
        seed=seed,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        verbose=0,
    )
    model.learn(total_timesteps=n_timesteps, progress_bar=False)
    sd = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    vec_env.close()
    return sd


def test_sb3_ppo_dummyvecenv_bitwise_equal():
    """Two PPO trainings with the same seed produce identical policy weights.

    This pins the foundation: if SB3 itself is non-deterministic on our
    setup, no downstream non-tabular reproducibility test can work.
    """
    sd_a = _train_tiny_ppo(seed=0)
    sd_b = _train_tiny_ppo(seed=0)
    ok, reason = state_dict_equal(sd_a, sd_b)
    assert ok, f"PPO policy weights differ: {reason}"


def test_sb3_ppo_different_seed_differs():
    """Sanity: different seed -> different policy weights."""
    sd_a = _train_tiny_ppo(seed=0)
    sd_b = _train_tiny_ppo(seed=1)
    ok, _ = state_dict_equal(sd_a, sd_b)
    assert not ok, "PPO produced identical weights for different seeds"


# ── 2. compute_regret non-tabular determinism ────────────────────────────────


def _stub_train_ppo(make_train_env_fn, make_eval_env_fn, seed: int, **kwargs):
    """Drop-in replacement for ``mavrl.utils.sb3.train_ppo`` for tests.

    Uses tiny budgets (n_envs=1, n_timesteps=512) and skips the
    ``EvalCallback`` / ``best_model`` machinery so the call finishes in a
    few seconds. Returns just an ``sb3.PPO`` (no eval-history), matching
    the default ``return_training_stats=False`` contract.
    """
    train_env = DummyVecEnv([lambda: make_train_env_fn(seed)])
    model = sb3.PPO(
        env=train_env,
        policy="MlpPolicy",
        seed=seed,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        verbose=0,
    )
    model.learn(total_timesteps=512, progress_bar=False)
    train_env.close()
    return model


class _ConstantActionPolicy:
    """Always returns action 0. Fully deterministic, no RNG.

    Used as ``true_optimal_policy`` in the regret test so that any
    reproducibility failure is attributable to the *retraining* path
    (PPO + joblib MC), not to gymnasium's lazily-seeded ``action_space``.
    """

    def predict(self, observation, deterministic: bool = False):
        return 0


def test_compute_regret_nontabular_bitwise_equal(monkeypatch, tmp_path):
    """Two ``compute_regret`` calls with the same seed produce identical
    regret and mean_rew. Pins the joblib + MC rollout path.
    """
    from mavrl.evaluation import regret as regret_module
    from mavrl.utils.feature_transforms import get_act_transform, get_obs_transform
    from mavrl.utils.train_utils import _create_model
    from mavrl.utils.gym import get_act_dim, get_obs_dim
    from mavrl.types import FeedbackType
    from mavrl.envs.make_env import make_env
    from mavrl.learned_reward_wrapper import LearnedRewardWrapper

    monkeypatch.setattr(regret_module, "train_ppo", _stub_train_ppo)

    args = argparse.Namespace(
        seed=0, env_id="CartPole-v1", grid_size=None, p_rand=None,
        gamma=0.99, obs_transform=None, act_transform="one_hot",
        reward_domain="sa", encoder_hidden_sizes=[16, 16],
        n_pref_samples=1, n_demo_samples=0, n_rating_samples=0, n_stop_samples=0,
        use_imitation_learning=False,
        retrain_reward_thresh=None, retrain_verbose=0, retrain_pbar=False,
        n_regret_samples=2,
    )

    seed_everything(args.seed, quiet=True)

    def make_env_fn():
        return make_env(env_id=args.env_id, seed=args.seed)

    env = make_env_fn()
    obs_tr = get_obs_transform(args, env)
    act_tr = get_act_transform(args, env)
    obs_dim = get_obs_dim(env, obs_tr)
    act_dim = get_act_dim(env, act_tr)
    active = {FeedbackType.PREF: args.n_pref_samples}
    fb_model, _ = _create_model(args, env, obs_dim, act_dim, active, "cpu")

    def make_train_env(seed: int):
        e = make_env_fn()
        e.reset(seed=seed)
        return LearnedRewardWrapper(
            e, fb_model.encoder, seed=seed,
            act_transform=act_tr, obs_transform=obs_tr,
        )

    true_optimal = _ConstantActionPolicy()

    def call():
        seed_everything(args.seed, quiet=True)
        regret, mean_rew, dv, _, _ = regret_module.compute_regret(
            ppo_seed=args.seed,
            true_optimal_policy=true_optimal,
            train_env_fn=make_train_env,
            eval_env_fn=make_env_fn,
            is_tabular=False,
            is_imitation=False,
            fb_model=fb_model,
            gamma=args.gamma,
            verbose=0,
            progress_bar=False,
            n_regret_samples=args.n_regret_samples,
            seed_fn=lambda i: args.seed * args.n_regret_samples + i,
        )
        return regret, mean_rew

    regret_a, rew_a = call()
    regret_b, rew_b = call()

    assert regret_a == regret_b, f"regret differs: {regret_a} vs {regret_b}"
    assert rew_a == rew_b, f"mean_rew differs: {rew_a} vs {rew_b}"
