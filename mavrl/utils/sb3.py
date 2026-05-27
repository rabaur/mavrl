import tempfile
from pathlib import Path
from typing import Any, Callable
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from mavrl.true_reward_callback import TrueRewardCallback


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Create a linear learning rate schedule.
    
    Args:
        initial_value: Initial value (at progress_remaining=1.0)
        
    Returns:
        A function that takes the remaining progress (1.0 -> 0.0) and returns the scheduled value.
    """
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


def parse_schedule_string(value: Any) -> Any:
    """
    Parse schedule strings like 'lin_0.001' into callable schedules.
    
    Supports:
        - 'lin_<value>': Linear schedule from <value> to 0
        
    If the value is not a schedule string, returns it unchanged.
    """
    if not isinstance(value, str):
        return value
    
    if value.startswith("lin_"):
        try:
            initial_value = float(value[4:])
            return linear_schedule(initial_value)
        except ValueError:
            raise ValueError(f"Invalid linear schedule string: {value}")
    
    # Not a schedule string, return as-is
    return value


# Parameters that can be schedule strings (lin_, const_, etc.)
SCHEDULE_PARAMS = {"learning_rate", "clip_range", "clip_range_vf"}


def train_ppo(
    make_train_env_fn: Callable,
    make_eval_env_fn: Callable,
    seed: int,
    retrain_ppo_settings: dict[str, Any],
    true_reward_threshold: float = None,
    verbose: int = 1,
    progress_bar: bool = True,
    return_training_stats: bool = False,
):
    """
    Train a PPO model on the wrapped environment. Hyperparameters must be supplied
    explicitly (e.g. from ``retrain_ppo_settings_from_args``); YAML is not used.

    Returns the best model based on true reward performance.

    Args:
        make_train_env_fn: Factory for one training env (often learned-reward wrapped).
        make_eval_env_fn: Factory for one evaluation env (true reward).
        seed: Random seed for the PPO model and env instances.
        retrain_ppo_settings: Dict with ``n_envs``, ``n_timesteps``, ``eval_freq``,
            ``n_eval_episodes``, ``policy``, optional ``policy_kwargs``, and remaining
            keys passed to ``stable_baselines3.PPO`` (after schedule parsing).
        true_reward_threshold: If set, stop training early if mean true reward drops below it.
        verbose: SB3 PPO verbosity (0=none, 1=info).
        progress_bar: Whether to show the SB3 progress bar during ``learn``.
        return_training_stats: If True, return ``(model, eval_history, eval_summary)``.
    """
    import numpy as np
    from stable_baselines3.common.vec_env import DummyVecEnv

    settings = dict(retrain_ppo_settings)
    n_envs = settings.pop("n_envs")
    n_timesteps = settings.pop("n_timesteps")
    eval_freq = settings.pop("eval_freq")
    n_eval_episodes = settings.pop("n_eval_episodes")
    policy = settings.pop("policy", "MlpPolicy")
    policy_kwargs = settings.pop("policy_kwargs", None)

    for param in SCHEDULE_PARAMS:
        if param in settings and settings[param] is not None:
            v = parse_schedule_string(settings[param])
            if isinstance(v, str):
                v = float(v)
            settings[param] = v

    ppo_kwargs = {k: v for k, v in settings.items() if v is not None}
    if policy_kwargs is not None:
        ppo_kwargs["policy_kwargs"] = policy_kwargs

    train_env = DummyVecEnv([lambda i=i: make_train_env_fn(seed + i) for i in range(n_envs)])

    with tempfile.TemporaryDirectory() as tmpdir:

        true_reward_cb = TrueRewardCallback(
            window_size=100,
            true_reward_threshold=true_reward_threshold,
        )

        eval_env = Monitor(make_eval_env_fn())

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=tmpdir,
            log_path=str(Path(tmpdir) / "eval_history"),
            eval_freq=max(eval_freq, 1000),
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            verbose=0
        )
        callbacks = CallbackList([true_reward_cb, eval_cb])

        ppo_model = sb3.PPO(
            policy=policy,
            env=train_env,
            verbose=verbose,
            seed=seed,
            **ppo_kwargs,
        )
        print(f"Training PPO for {n_timesteps:,} timesteps with {n_envs} parallel envs")
        ppo_model.learn(total_timesteps=n_timesteps, callback=callbacks, progress_bar=progress_bar)
        
        train_env.close()
        
        # Build eval history from EvalCallback: deterministic rollouts on the
        # true-reward evaluation env.  Each entry is (timestep, [ep_rewards]).
        eval_history = list(zip(
            eval_cb.evaluations_timesteps,
            [list(map(float, eps)) for eps in eval_cb.evaluations_results],
        ))

        # Also build a compact summary: (timestep, mean_reward)
        eval_summary = [
            (ts, float(np.mean(ep_rews)))
            for ts, ep_rews in eval_history
        ]

        # Load and return the best model based on TRUE reward
        best_eval_reward_path = Path(tmpdir) / "best_model.zip"
        if best_eval_reward_path.exists():
            model = sb3.PPO.load(best_eval_reward_path, env=make_eval_env_fn())
            return (model, eval_history, eval_summary) if return_training_stats else model

        print("Warning: No best model saved, returning final model")
        return (ppo_model, eval_history, eval_summary) if return_training_stats else ppo_model