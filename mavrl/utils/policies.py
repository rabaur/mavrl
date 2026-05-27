import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from typing import Any, Optional
from abc import ABC, abstractmethod
from mavrl.utils.tabular import q_opt
from mavrl.utils.math import softmax
import stable_baselines3 as sb3
from mavrl.envs.env_types import TabularEnv
from mavrl.utils.torch_utils import get_model_device, to_numpy, to_torch
from stable_baselines3.common.utils import obs_as_tensor
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def create_policy(
    policy_path: str,
    beta: float,
    env: gym.Env,
    gamma: float = 0.99,
    *,
    rng_seed: Optional[int] = None,
):
    """
    Create a policy from a saved model path.
    
    Args:
        policy_path: Path to the saved policy model.
        beta: Rationality parameter for Q-value based policies.
        env: The environment to create the policy for.
        gamma: Discount factor (used for tabular environments).
        
    Returns:
        A policy object that can be used for trajectory generation.
    """
    # Expand ~ to home directory if present
    load_path = str(Path(policy_path).expanduser())
    
    # Check for policy type in path by looking for directory names "/ppo/" or "/dqn/"
    policy_path_lower = load_path.lower()
    is_tabular_env = isinstance(env.unwrapped, TabularEnv)
    if is_tabular_env:
        q_model = TabularQValueModel(env.unwrapped, gamma=gamma)
        rng = (
            np.random.default_rng(rng_seed) if rng_seed is not None else None
        )
        return QValueExpert(q_model, beta=beta, numpy_rng=rng)
    else:
        # Remove .zip extension if present (sb3.load expects path without extension)
        if "/ppo/" in policy_path_lower:
            print(f"Detected PPO policy from path. PPO does not support rationality parameter, using default value of 1.0")
            return PPOExpert(sb3.PPO.load(load_path))
        elif "/dqn/" in policy_path_lower:
            print(f"Detected DQN policy from path")
            dqn_model = DQNQValueModel(sb3.DQN.load(load_path))
            rng = (
                np.random.default_rng(rng_seed) if rng_seed is not None else None
            )
            return QValueExpert(dqn_model, beta=beta, numpy_rng=rng)
        else:
            raise ValueError(f"Could not determine policy type from path.")


class QValueModel(ABC):
    """
    Abstract base class for Q-value models.
    
    Q-value models provide the expected return Q(s,a) for state-action pairs.
    """
    
    @abstractmethod
    def q_values(self, observation) -> NDArray:
        """
        Get Q-values for all actions given an observation.
        
        Args:
            observation: Environment observation
            
        Returns:
            Array of Q-values for each action
        """
        pass


class TabularQValueModel(QValueModel):
    """
    Q-value model for tabular environments.
    
    Computes optimal Q-values from transition dynamics P and rewards R.
    """
    
    def __init__(self, env: TabularEnv, gamma: float = 0.99):
        """
        Initialize tabular Q-value model.
        
        Args:
            env: Tabular environment with P and R attributes
            gamma: Discount factor for Q-value computation
        """
        R = env.get_reward_matrix()
        P = env.get_transition_matrix()
        self.Q_optimal = q_opt(P, R, gamma)

    @classmethod
    def from_reward_matrix(
        cls, R: NDArray, P: NDArray, gamma: float,
    ) -> "TabularQValueModel":
        """Build a TabularQValueModel from explicit (R, P, gamma).

        Useful when ``R`` is an arbitrary reward (e.g., a posterior sample
        during IG estimation) rather than the env's true reward matrix.
        """
        instance = cls.__new__(cls)
        instance.Q_optimal = q_opt(P, R, gamma)
        return instance

    def q_values(self, observation) -> NDArray:
        """Get Q-values for an observation."""
        # Obs is the state index  
        return self.Q_optimal[observation]


class DQNQValueModel(QValueModel):
    """
    Q-value model using Deep Q-Networks.
    
    Wraps a trained DQN model from stable-baselines3.
    """
    
    def __init__(self, dqn_model: sb3.DQN):
        """
        Initialize DQN Q-value model.
        
        Args:
            dqn_model: Trained stable-baselines3 DQN model
        """
        self.model = dqn_model
        self.action_dim = dqn_model.action_space.n
        self.gamma = float(dqn_model.gamma)
    
    def q_values(self, obs: Any) -> NDArray:
        """Get Q-values for an observation."""
        obs_t = obs_as_tensor(obs, self.model.device)
        # Add batch dimension if missing (q_net expects batch dimension)
        single_obs = obs_t.ndim == 1
        if single_obs:
            obs_t = obs_t.unsqueeze(0)
        q_values = self.model.q_net(obs_t)
        if single_obs:
            q_values = q_values.squeeze(0)
        return to_numpy(q_values)


class NeuralQValueModel(QValueModel):
    """Wrapper to use a trained nn.Module as a QValueModel.

    q_net takes only state and outputs Q-values for all actions,
    shape (batch, num_actions).
    """

    def __init__(self, q_net: nn.Module):
        self.q_net = q_net
        self.device = get_model_device(q_net)

    def q_values(self, obs: Any, action: Any = None) -> NDArray:
        """Get Q-values for an observation.

        Args:
            obs: Environment observation

        Returns:
            Array of Q-values for each action, shape (num_actions,) or (batch, num_actions)
        """
        with torch.no_grad():
            obs_t = to_torch(obs, self.device)
            single_obs = obs_t.ndim == 1
            if single_obs:
                obs_t = obs_t.unsqueeze(0)

            q_vals = self.q_net(obs_t)

            if single_obs:
                q_vals = q_vals.squeeze(0)
            return to_numpy(q_vals)


class Expert(ABC):
    """
    Abstract base class for expert policies.
    """

    def __init__(self, model: Any, beta: float = 1.0):
        """
        Initialize expert policy.
        
        Args:
            q_model: Q-value model to use for action selection
            rationality: Rationality parameter (β) for softmax policy
        """
        self.model = model
        self.beta = beta
    
    @abstractmethod
    def predict(self, obs: Any, deterministic: bool = False):
        pass

    @abstractmethod
    def predict_proba(self, obs: Any) -> NDArray:
        pass


class QValueExpert(Expert):
    """
    Expert policy for Q-value models.
    """

    def __init__(
        self,
        model: QValueModel,
        beta: float = 1.0,
        numpy_rng: Optional[np.random.Generator] = None,
    ):
        """
        Initialize tabular expert policy.
        
        Args:
            q_model: TabularQValueModel for Q-value computation
            rationality: Rationality parameter (β) for softmax policy
            numpy_rng:
                Dedicated NumPy RNG for stochastic policy draws. When ``None``
                (default), falls back to the global ``np.random`` legacy API for
                backward compatibility.

                Pass ``numpy_rng=np.random.default_rng(derived_seed)`` so that
                train vs val dataset generation can diverge deterministically,
                despite ``seed_everything`` resetting global state equally for
                each split.
        """
        super().__init__(model, beta)
        self._numpy_rng = numpy_rng

    def _rng_rand(self):
        """Return the RNG used for stochastic decisions."""
        return self._numpy_rng if self._numpy_rng is not None else np.random
    
    def predict(self, observation: Any, deterministic: bool = False):
        rng = self._rng_rand()
        if deterministic or self.beta == float('inf'):
            q_values = self.model.q_values(observation)
            # Handle both single observation and batch
            # When multiple actions have equal max Q-value, sample uniformly among them
            if q_values.ndim == 1:
                max_q = np.max(q_values)
                max_actions = np.flatnonzero(q_values == max_q)
                action = rng.choice(max_actions)
            else:
                max_q = np.max(q_values, axis=-1, keepdims=True)
                is_max = (q_values == max_q)
                action = np.array([rng.choice(np.flatnonzero(row)) for row in is_max])
        else:
            probs = self.predict_proba(observation)
            # Handle both single observation and batch
            if probs.ndim == 1:
                # Single observation case
                action = rng.choice(len(probs), p=probs)
            else:
                # Batch case: sample an action for each observation
                # Use cumulative probabilities for efficient vectorized sampling
                cumsum_probs = np.cumsum(probs, axis=-1)
                # Generate random values for each observation
                rand_vals = rng.random(probs.shape[0])
                # Find the first index where cumulative prob >= random value
                action = np.argmax(cumsum_probs >= rand_vals[:, np.newaxis], axis=-1)
        return action

    def predict_proba(self, observation: Any) -> NDArray:
        q_values = self.model.q_values(observation)
        return softmax(self.beta * q_values, dims=-1)


class EpsilonGreedyWrapper(Expert):
    """Wraps any Expert policy with epsilon-greedy exploration.

    With probability ``epsilon`` a uniform-random action is sampled from
    ``action_space``; otherwise the wrapped policy's action is used.
    """

    def __init__(
        self,
        policy: Expert,
        action_space: gym.Space,
        epsilon: float,
        exploration_rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(model=policy, beta=getattr(policy, "beta", 1.0))
        self.policy = policy
        self.action_space = action_space
        self.epsilon = epsilon
        self._exploration_rng = exploration_rng

    def _sample_random_action(self) -> Any:
        """Sample uniformly from discrete action spaces using the exploration RNG."""
        space = self.action_space
        if hasattr(space, "n"):
            n = int(space.n)
            if self._exploration_rng is not None:
                return int(self._exploration_rng.integers(0, n))
            return int(np.random.randint(0, n))

        if hasattr(space, "shape") and getattr(space, "shape", ()) == ():
            return space.sample()

        raise NotImplementedError(
            f"EpsilonGreedy exploration for {type(space).__name__} is unsupported; "
            f"use a Discrete action space or supply an explicit exploration_rng."
        )

    def predict(self, obs: Any, deterministic: bool = False):
        actions = self.policy.predict(obs, deterministic=deterministic)
        if self.epsilon <= 0 or deterministic:
            return actions

        obs_arr = np.asarray(obs)
        is_batch = obs_arr.ndim >= 2
        rng = self._exploration_rng
        if is_batch:
            if rng is not None:
                mask = rng.random(obs_arr.shape[0]) < self.epsilon
            else:
                mask = np.random.random(obs_arr.shape[0]) < self.epsilon
            for i in np.where(mask)[0]:
                actions[i] = self._sample_random_action()
        else:
            do_explore = (
                rng.random() < self.epsilon
                if rng is not None
                else np.random.random() < self.epsilon
            )
            if do_explore:
                actions = self._sample_random_action()
        return actions

    def predict_proba(self, obs: Any) -> NDArray:
        return self.policy.predict_proba(obs)


class PPOExpert(Expert):
    """
    Expert policy for PPO models.
    """

    def __init__(self, model: sb3.PPO, beta: float = 1.0):
        """
        Initialize PPO expert policy.
        """
        super().__init__(model, beta)

    def predict(self, observation: Any, deterministic: bool = False):
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def predict_proba(self, observation: Any) -> NDArray:
        probs = self.model.predict_proba(observation)
        return probs