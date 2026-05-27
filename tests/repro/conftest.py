"""Shared fixtures for the reproducibility test suite.

The single most important fixture is ``_pinned_threading``, which is autouse
and session-scoped. Without it, run-to-run float equality on CPU is not
guaranteed across torch reductions.

Argument fixtures (``tabular_args``, ``chain_args``, ``lunarlander_args``)
return a fully-populated ``argparse.Namespace`` built from
``train.get_default_args()`` so that any new arg added to the parser
automatically gets a default rather than tripping over ``AttributeError`` in
the middle of a test run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from train import get_default_args
from tests.repro._repro_helpers import set_pinned_threading


# ── Threading / determinism (autouse) ────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def _pinned_threading():
    """Pin CPU thread count for the whole test session.

    Required for run-to-run float equality on CPU. Idempotent.
    """
    set_pinned_threading()
    yield


# ── Argument fixtures ────────────────────────────────────────────────────────


def _override(args: argparse.Namespace, **kwargs) -> argparse.Namespace:
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args


@pytest.fixture
def tabular_args(tmp_path: Path) -> argparse.Namespace:
    """`argparse.Namespace` for grid_trap with all 4 modalities, tiny sizes.

    Designed so a full ``run_experiment`` finishes in well under a second.
    All four feedback modalities are active so that the test exercises the
    full reparameterization / dataloader-iteration code path.
    """
    args = get_default_args()
    return _override(
        args,
        seed=0,
        env_id="grid_trap",
        grid_size=4,
        p_rand=0.0,
        gamma=0.95,
        obs_transform="one_hot",
        act_transform="one_hot",
        reward_domain="s",
        td_error_weight=1.0,
        kl_weight=1.0,
        # Training
        num_epochs=2,
        batch_size=8,
        lr=1e-3,
        encoder_hidden_sizes=[16, 16],
        val_every_n_epochs=1,
        skip_first_val_epoch=False,
        vis_every_n_epochs=None,
        early_stop_patience=None,
        # Logging / saving
        log_wandb=False,
        log_every_n_steps=1,
        model_save_dir=str(tmp_path / "models"),
        save_behavior="best",
        skip_final_eval=False,
        # Eval
        n_regret_samples=4,
        retrain_verbose=0,
        retrain_pbar=False,
        # ── Preference ───────────────────────────────
        n_pref_episodes=8,
        n_pref_samples=8,
        pref_policy_path="tabular",
        pref_trajectory_rationality=0.5,
        pref_data_rationality=2.0,
        pref_model_rationality=2.0,
        pref_seg_len=4,
        min_reward_pref=None,
        # ── Demonstration ────────────────────────────
        n_demo_samples=4,
        demo_policy_path="tabular",
        demo_rationality=2.0,
        demo_model_rationality=2.0,
        min_reward_demo=None,
        # ── Rating ───────────────────────────────────
        n_rating_episodes=8,
        n_rating_samples=8,
        rating_policy_path="tabular",
        rating_trajectory_rationality=2.0,
        rating_seg_len=4,
        min_reward_rating=None,
        rating_noise_std=0.0,
        # ── Stop ─────────────────────────────────────
        n_stop_episodes=8,
        n_stop_samples=8,
        stop_policy_path="tabular",
        stop_trajectory_rationality=0.5,
        stop_seg_len=4,
        stop_c=1.0,
        stop_model_c=None,
        stop_regret_percentile=50.0,
        stop_regret_discount=0.5,
        stop_q_value_model=None,
        min_reward_stop=None,
        # Misc
        exploration_epsilon=0.0,
        step_offset=1,
        subsample_factor=1,
        use_imitation_learning=False,
        use_importance_weights=False,
        dataset_cache_dir=None,
        dataset_cache_gen_samples=None,
    )


@pytest.fixture
def chain_args(tabular_args: argparse.Namespace) -> argparse.Namespace:
    """`argparse.Namespace` for chain env (different action space)."""
    return _override(
        tabular_args,
        env_id="chain_4",
    )


@pytest.fixture
def lunarlander_args(tmp_path: Path) -> argparse.Namespace:
    """`argparse.Namespace` for non-tabular LunarLander runs.

    Sized for the slow tier: only one feedback modality (preference, smallest
    that exercises the full pipeline), one epoch, one validation pass, and a
    minimal regret-sample count. PPO retraining is *not* shrunk here -- tests
    that use this fixture are expected to monkeypatch ``train_ppo`` (or pass
    ``skip_final_eval=True``).
    """
    args = get_default_args()
    expert = Path("expert_policies").resolve()
    pref_policy = expert / "dqn" / "LunarLander-v3_1" / "best_model.zip"
    optimal_policy = expert / "ppo" / "LunarLander-v3_2" / "best_model.zip"

    return _override(
        args,
        seed=0,
        env_id="LunarLander-v3",
        gamma=0.999,
        obs_transform=None,
        act_transform="one_hot",
        reward_domain="sa",
        td_error_weight=1.0,
        kl_weight=1.0,
        # Training
        num_epochs=1,
        batch_size=8,
        lr=1e-4,
        encoder_hidden_sizes=[32, 32],
        val_every_n_epochs=None,  # skip validation in slow tests; saves time
        skip_first_val_epoch=False,
        vis_every_n_epochs=None,
        early_stop_patience=None,
        # Logging / saving
        log_wandb=False,
        log_every_n_steps=1,
        model_save_dir=str(tmp_path / "models"),
        save_behavior="best",
        skip_final_eval=False,
        # Eval (kept tiny)
        n_regret_samples=2,
        retrain_verbose=0,
        retrain_pbar=False,
        optimal_policy_path=str(optimal_policy),
        # Single modality: preference
        n_pref_episodes=2,
        n_pref_samples=4,
        pref_policy_path=str(pref_policy),
        pref_trajectory_rationality=5.0,
        pref_data_rationality=5.0,
        pref_model_rationality=5.0,
        pref_seg_len=8,
        min_reward_pref=None,
        # All other modalities disabled
        n_demo_samples=0,
        n_rating_samples=0,
        n_stop_samples=0,
        # Misc
        exploration_epsilon=0.0,
        step_offset=1,
        subsample_factor=1,
        use_imitation_learning=False,
        use_importance_weights=False,
        dataset_cache_dir=None,
        dataset_cache_gen_samples=None,
    )
