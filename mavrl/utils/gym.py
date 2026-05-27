import gymnasium as gym
from typing import Callable, Optional
import numpy as np
from mavrl.utils.policies import Expert
from mavrl.types import DataKey, Trajectory
from mavrl.envs.env_types import TabularEnv


def rollout(
    env: gym.Env,
    policy: Expert,
    num_steps: Optional[int] = None,
    seed: int = None,
    deterministic: bool = True
) -> Trajectory:
    """Single-episode rollout with optional seed for reproducible starting state.
    
    Args:
        env: The environment to roll out in.
        policy: The policy to use for action selection.
        num_steps: Maximum number of steps (None for no limit).
        seed: Random seed for environment reset.
        deterministic: If True, use deterministic action selection. Default True for reproducibility.
    """
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()
    
    traj = {
        DataKey.OBS: [],
        DataKey.ACTS: [],
        DataKey.REWS: [],
        DataKey.TERMINAL: [],
        DataKey.VALID: [],
    }
    
    done = False
    step = 0
    while not done and (num_steps is None or step < num_steps):
        traj[DataKey.OBS].append(np.atleast_1d(obs))
        action = policy.predict(obs, deterministic=deterministic)
        traj[DataKey.ACTS].append(np.atleast_1d(action))
        obs, reward, terminated, truncated, info = env.step(action)
        traj[DataKey.REWS].append(reward)
        traj[DataKey.TERMINAL].append(terminated or truncated)
        traj[DataKey.VALID].append(True)
        done = terminated or truncated
        step += 1
    
    # Convert to numpy
    return {k: np.array(v) for k, v in traj.items()}

def get_undiscounted_return(trajectory: Trajectory):
    return np.nansum(trajectory[DataKey.REWS])

def get_discounted_return(trajectory: Trajectory, gamma: float):
    return np.nansum(trajectory[DataKey.REWS] * gamma ** np.arange(len(trajectory[DataKey.REWS])))

def is_registered_gym_env(env_name: str) -> bool:
    """
    Check if a Gym environment is registered.
    """
    return env_name in gym.envs.registry.keys()


def get_obs_dim(env: gym.Env, observation_transform: Callable = None) -> int:
    """
    Get the dimensionality of the observation representation by sampling a random observation and applying the observation-transform.
    """
    rand_obs = env.observation_space.sample()
    if observation_transform is not None:
        rand_obs = observation_transform(rand_obs)
    if isinstance(rand_obs, np.ndarray):
        return rand_obs.shape[0]
    elif isinstance(rand_obs, int) or isinstance(rand_obs, float) or isinstance(rand_obs, np.integer):
        return 1
    else:
        raise ValueError(f"Invalid observation type: {type(rand_obs)}")


def get_act_dim(env: gym.Env, action_transform: Callable = None) -> int:
    """
    Get the dimensionality of the action representation by sampling a random action and applying the action-transform.
    """
    rand_action = env.action_space.sample()
    if action_transform is not None:
        rand_action = action_transform(rand_action)
    if isinstance(rand_action, np.ndarray):
        return rand_action.shape[0]
    elif isinstance(rand_action, int) or isinstance(rand_action, float) or isinstance(rand_action, np.integer):
        return 1
    else:
        raise ValueError(f"Invalid action type: {type(rand_action)}")

def get_env_name(env: gym.Env):
    if isinstance(env.unwrapped, TabularEnv):
        return env.unwrapped.id
    else:
        return env.unwrapped.spec.id