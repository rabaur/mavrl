"""Layer 3: end-to-end reproducibility on non-tabular (LunarLander).

Marked ``slow``. Even with a tiny PPO budget, excluded from the default run;
opt in with ``pytest -m slow``.

Two ``run_experiment`` invocations with the same seed should produce:

- **Training** that matches (bitwise model checkpoint when saved).
- **Final metrics** consistent within **Monte Carlo / floating-point tolerance**
  (``metrics_allclose`` with ``rtol=1e-3``), *not* bitwise equality: the path
  chains Box2D, SB3 ``PPO``, and joblib MC regret in
  ``regret_non_tabular``, which does not admit exact run-to-run reproducibility
  even with fixed seeds on one machine.

Exact equality is still required for ``best_epoch`` / early-stop bookkeeping.

If expert policy files are missing, the test skips.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import stable_baselines3 as sb3
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from train import run_experiment

from tests.repro._repro_helpers import metrics_allclose, state_dict_equal


pytestmark = pytest.mark.slow


# ── PPO stub: tiny budget, no EvalCallback ───────────────────────────────────


def _stub_train_ppo(make_train_env_fn, make_eval_env_fn, seed: int, **kwargs):
    """Minimal ``train_ppo`` replacement (1 env, 1024 timesteps, no callbacks)."""
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


# ── Fixture availability ─────────────────────────────────────────────────────


def _expert_files_present(args: argparse.Namespace) -> bool:
    pref = Path(args.pref_policy_path).expanduser()
    optimal = Path(args.optimal_policy_path).expanduser()
    return pref.exists() and optimal.exists()


def _env_constructible(env_id: str) -> bool:
    """Return True iff ``gym.make(env_id)`` succeeds.

    LunarLander needs ``gymnasium[box2d]`` which isn't a hard dependency of
    mavrl. Skip rather than fail when Box2D is missing.
    """
    import gymnasium as gym
    try:
        env = gym.make(env_id)
        env.close()
        return True
    except Exception:
        return False


# ── Tests ────────────────────────────────────────────────────────────────────


def test_e2e_lunarlander_reproducible(monkeypatch, lunarlander_args):
    """Same args + seed -> same training artifacts; ``final_metrics`` within rtol.

    Floating final metrics allow ``rtol=1e-3`` (see ``metrics_allclose``) for the
    non-tabular eval stack; checkpoints remain bitwise-equal when saved.
    """
    if not _env_constructible(lunarlander_args.env_id):
        pytest.skip(
            f"{lunarlander_args.env_id} cannot be constructed (likely missing "
            "gymnasium[box2d] dependency); skipping non-tabular e2e test."
        )
    if not _expert_files_present(lunarlander_args):
        pytest.skip(
            f"Expert policy not found at {lunarlander_args.pref_policy_path} "
            "or optimal policy not found; skipping non-tabular e2e test."
        )

    from mavrl.evaluation import regret as regret_module

    monkeypatch.setattr(regret_module, "train_ppo", _stub_train_ppo)

    result_a = run_experiment(lunarlander_args)
    result_b = run_experiment(lunarlander_args)

    # Final metrics: approximate (MC + simulator + joblib regret; see module doc).
    ok, reason = metrics_allclose(result_a["final_metrics"], result_b["final_metrics"])
    assert ok, f"final_metrics beyond rtol/atol tolerance: {reason}"

    # Best epoch / early stop bookkeeping.
    for k in ("best_epoch", "early_stop_epoch", "early_stop_reason"):
        assert result_a[k] == result_b[k], (
            f"{k} differs: {result_a[k]!r} vs {result_b[k]!r}"
        )

    # Saved best-model state dict (if any).
    if result_a["best_model_path"] is not None:
        assert result_b["best_model_path"] is not None
        ckpt_a = torch.load(result_a["best_model_path"], map_location="cpu",
                            weights_only=False)
        ckpt_b = torch.load(result_b["best_model_path"], map_location="cpu",
                            weights_only=False)
        ok, reason = state_dict_equal(
            ckpt_a["model_state_dict"], ckpt_b["model_state_dict"],
        )
        assert ok, f"best_model state_dict differs: {reason}"
