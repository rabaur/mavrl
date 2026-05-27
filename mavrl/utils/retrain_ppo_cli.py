"""CLI defaults and packing for downstream PPO (regret retraining).

All hyperparameters come from argparse (e.g. :mod:`train`) — not from ``hyperparams/ppo.yml`` —
so joint optimization with other MAVRL settings uses a single :class:`argparse.Namespace`.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from mavrl.utils.sb3 import parse_schedule_string


def add_retrain_ppo_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``downstream PPO (regret retraining)`` argument group."""
    g = parser.add_argument_group("downstream PPO (regret retraining)")
    g.add_argument(
        "--retrain_verbose",
        type=int,
        default=1,
        help="Stable-Baselines3 verbosity during regret retraining (0=quiet, 1=info).",
    )
    g.add_argument(
        "--retrain_pbar",
        action="store_true",
        dest="retrain_pbar",
        default=True,
        help="Show a progress bar during PPO learning (default: on).",
    )
    g.add_argument(
        "--no-retrain_pbar",
        action="store_false",
        dest="retrain_pbar",
        help="Disable the PPO progress bar.",
    )
    g.add_argument(
        "--retrain_reward_thresh",
        type=lambda x: None if x.lower() == "none" else float(x),
        default=None,
        help="Early-stop PPO if rolling mean true reward falls below this (None = disabled).",
    )
    g.add_argument("--retrain_n_envs", type=int, default=8, help="Number of parallel envs (VecEnv)")
    g.add_argument(
        "--retrain_n_timesteps",
        type=int,
        default=1_000_000,
        help="Total PPO training timesteps",
    )
    g.add_argument(
        "--retrain_eval_freq",
        type=int,
        default=1000,
        help="Evaluate on the true-reward env every this many timesteps **per env**",
    )
    g.add_argument(
        "--retrain_n_eval_episodes",
        type=int,
        default=20,
        help="Episodes per evaluation callback",
    )
    g.add_argument(
        "--retrain_policy",
        type=str,
        default="MlpPolicy",
        help="SB3 policy class name (e.g. MlpPolicy, CnnPolicy)",
    )
    g.add_argument(
        "--retrain_policy_kwargs",
        type=str,
        default=None,
        help='Optional JSON object for ``policy_kwargs`` (e.g. ``"{}"``)',
    )
    g.add_argument("--retrain_n_steps", type=int, default=2048, help="PPO rollout length per env")
    g.add_argument("--retrain_batch_size", type=int, default=64, help="PPO minibatch size")
    g.add_argument("--retrain_n_epochs", type=int, default=10, help="PPO optimization epochs per rollout")
    g.add_argument("--retrain_gamma", type=float, default=0.99, help="PPO discount factor")
    g.add_argument("--retrain_gae_lambda", type=float, default=0.95, help="PPO GAE lambda")
    g.add_argument(
        "--retrain_learning_rate",
        type=str,
        default="3e-4",
        help="Learning rate: float string (e.g. 3e-4) or linear schedule lin_<float>",
    )
    g.add_argument(
        "--retrain_clip_range",
        type=str,
        default="0.2",
        help="PPO clip range: float string or lin_<float>",
    )
    g.add_argument(
        "--retrain_clip_range_vf",
        type=lambda x: None if x.lower() == "none" else str(x),
        default=None,
        help="VF clip coefficient: None for SB3 default; else float string or lin_<float>",
    )
    g.add_argument("--retrain_ent_coef", type=float, default=0.01, help="PPO entropy coefficient")
    g.add_argument("--retrain_vf_coef", type=float, default=0.5, help="PPO value-function loss coefficient")
    g.add_argument("--retrain_max_grad_norm", type=float, default=0.5, help="PPO max gradient norm")
    g.add_argument(
        "--retrain_normalize_advantage",
        action="store_true",
        dest="retrain_normalize_advantage",
        default=True,
        help="Normalize advantages in PPO (default: on).",
    )
    g.add_argument(
        "--no-retrain_normalize_advantage",
        action="store_false",
        dest="retrain_normalize_advantage",
        help="Disable advantage normalization in PPO.",
    )


def _coerce_numeric_schedule(value: Any) -> Any:
    """After :func:`parse_schedule_string`, turn a plain numeric string into ``float``."""
    if isinstance(value, str):
        return float(value)
    return value


def retrain_ppo_settings_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build the settings dict for :func:`mavrl.utils.sb3.train_ppo` / :func:`compute_regret`."""
    merge_retrain_ppo_defaults_into_args(args)
    policy_kwargs: dict[str, Any] | None = None
    raw_pk = getattr(args, "retrain_policy_kwargs", None)
    if raw_pk is not None:
        policy_kwargs = json.loads(raw_pk)

    lr = _coerce_numeric_schedule(parse_schedule_string(args.retrain_learning_rate))
    cr = _coerce_numeric_schedule(parse_schedule_string(args.retrain_clip_range))

    clip_range_vf_parsed: Any = None
    crvf = getattr(args, "retrain_clip_range_vf", None)
    if crvf is not None:
        clip_range_vf_parsed = _coerce_numeric_schedule(parse_schedule_string(crvf))

    return {
        "n_envs": args.retrain_n_envs,
        "n_timesteps": args.retrain_n_timesteps,
        "eval_freq": args.retrain_eval_freq,
        "n_eval_episodes": args.retrain_n_eval_episodes,
        "policy": args.retrain_policy,
        "policy_kwargs": policy_kwargs,
        "learning_rate": lr,
        "n_steps": args.retrain_n_steps,
        "batch_size": args.retrain_batch_size,
        "n_epochs": args.retrain_n_epochs,
        "gamma": args.retrain_gamma,
        "gae_lambda": args.retrain_gae_lambda,
        "clip_range": cr,
        "clip_range_vf": clip_range_vf_parsed,
        "ent_coef": args.retrain_ent_coef,
        "vf_coef": args.retrain_vf_coef,
        "max_grad_norm": args.retrain_max_grad_norm,
        "normalize_advantage": args.retrain_normalize_advantage,
    }


def merge_retrain_ppo_defaults_into_args(args: Any) -> None:
    """Fill missing ``retrain_*`` attributes on ``args`` from :func:`parse_retrain_ppo_defaults`."""
    for key, value in vars(parse_retrain_ppo_defaults()).items():
        if not hasattr(args, key):
            setattr(args, key, value)


def parse_retrain_ppo_defaults() -> argparse.Namespace:
    """Namespace with default retrain-PPO values (same as ``parse_args([])`` on that group)."""
    p = argparse.ArgumentParser(add_help=False)
    add_retrain_ppo_arguments(p)
    return p.parse_args([])


def default_retrain_ppo_settings() -> dict[str, Any]:
    """Processed defaults when no full experiment ``args`` is available (e.g. tests)."""
    return retrain_ppo_settings_from_args(parse_retrain_ppo_defaults())
