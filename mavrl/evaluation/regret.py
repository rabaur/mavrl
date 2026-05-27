import numpy as np
import torch
import gymnasium as gym
from typing import Any, Callable, Dict, Optional
from mavrl.multi_fb_model import MultiFeedbackTypeModel
from mavrl.utils.tabular import q_opt
from mavrl.utils.policies import PPOExpert, NeuralQValueModel, QValueExpert
from mavrl.envs.env_types import TabularEnv
from mavrl.encoder.reward_encoder import RewardEncoder
from mavrl.utils.feature_transforms import get_feature_combinations
from mavrl.utils.gym import rollout, get_discounted_return, get_undiscounted_return
from mavrl.utils.torch_utils import to_numpy
from mavrl.utils.sb3 import train_ppo
from mavrl.utils.retrain_ppo_cli import default_retrain_ppo_settings
from joblib import Parallel, delayed


class UniformRandomPolicy:
    """
    A uniform random policy that samples actions uniformly from the action space.
    
    Compatible with the Expert interface for use in rollouts.
    """
    
    def __init__(self, action_space: gym.spaces.Space):
        self.action_space = action_space
    
    def predict(self, observation, deterministic: bool = False):
        """Sample a random action from the action space."""
        return self.action_space.sample()
    
    def predict_proba(self, observation):
        """Return uniform probabilities for discrete action spaces."""
        if isinstance(self.action_space, gym.spaces.Discrete):
            return np.ones(self.action_space.n) / self.action_space.n
        else:
            raise NotImplementedError("predict_proba not supported for continuous action spaces")


def compute_single_return_sample(
    seed: int,
    env_fn: Callable[[], gym.Env],
    policy: Callable,
    max_num_steps: int,
) -> float:
    """Compute undiscounted return for a single episode (used for parallel execution)."""
    env = env_fn()
    traj = rollout(env, policy, num_steps=max_num_steps, seed=seed)
    return get_undiscounted_return(traj)


def mean_return_non_tabular(
    policy: Callable,
    env_fn: Callable[[], gym.Env],
    num_samples: int = 1000,
    max_num_steps: int = 1000,
    seed_fn: Callable[[int], int] = lambda x: x,
    n_jobs: int = -1,
) -> tuple[float, float]:
    """
    Monte Carlo estimate of the mean (undiscounted) return for a policy.
    
    Args:
        policy: Policy to evaluate (must have a predict method).
        env_fn: Function that creates a new environment instance.
        num_samples: Number of episodes to sample.
        max_num_steps: Maximum number of steps per episode.
        seed_fn: Function to generate seeds from sample index.
        n_jobs: Number of parallel jobs (-1 for all CPUs).
    
    Returns:
        tuple: (mean_return, std_return)
    """
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(compute_single_return_sample)(
            seed=seed_fn(i),
            env_fn=env_fn,
            policy=policy,
            max_num_steps=max_num_steps,
        )
        for i in range(num_samples)
    )
    returns = np.array(results)
    return float(np.mean(returns)), float(np.std(returns))


def value_under_policy(P, R_true, gamma, pi):
    S, A, Sp = P.shape
    assert Sp == S
    # r_pi[s] = E[R(s, pi[s], S')]
    if R_true.ndim == 2:      # R(s,a)
        r_pi = R_true[np.arange(S), pi]
    else:                     # R(s,a,s')
        r_sa = R_true[np.arange(S), pi, :]          # (S, S)
        r_pi = np.sum(P[np.arange(S), pi, :] * r_sa, axis=1)

    # P_pi[s, s'] = P(s'|s, pi[s])
    P_pi = P[np.arange(S), pi, :]                   # (S, S)

    # Solve (I - gamma P_pi) V = r_pi
    I = np.eye(S)
    V = np.linalg.solve(I - gamma * P_pi, r_pi)
    return V  # shape (S,)


def value_under_stochastic_policy(P, R_true, gamma, pi_probs):
    """Evaluate V^pi for a stochastic policy pi_probs (S, A) under true reward.

    Solves (I - gamma P_pi) V = r_pi  exactly, where expectations are taken
    over the action distribution pi_probs(a|s).
    """
    S, A, Sp = P.shape
    assert Sp == S

    if R_true.ndim == 2:  # R(s,a)
        r_pi = np.sum(pi_probs * R_true, axis=1)
    else:  # R(s,a,s')
        r_sa = np.sum(P * R_true, axis=2)            # (S, A)
        r_pi = np.sum(pi_probs * r_sa, axis=1)       # (S,)

    P_pi = np.einsum('sa,sap->sp', pi_probs, P)      # (S, S)

    V = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)
    return V  # shape (S,)


def regret_tabular(
    env: TabularEnv,
    encoder: RewardEncoder,
    gamma: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
    softmax_beta: float | None = None,
) -> tuple[float, float]:
    """Computes expected regret by deriving a policy from the estimated reward.

    Args:
        softmax_beta: If *None* (default) derive a deterministic argmax policy.
            If a positive float, derive a Boltzmann policy
            ``pi(a|s) = softmax(beta * Q(s,a))`` instead.  Higher beta is
            closer to argmax; lower beta is more stochastic.

    Returns:
        (regret, discounted_value) weighted by the initial-state distribution.
    """

    R_true = env.unwrapped.get_reward_matrix()
    P = env.unwrapped.get_transition_matrix()

    num_states = P.shape[0]
    num_actions = P.shape[1]

    device = next(encoder.parameters()).device
    all_obs_features = torch.eye(num_states, device=device)
    all_act_features = torch.eye(num_actions, device=device)

    Q_true_opt = q_opt(P, R_true, gamma, max_iter=max_iter, tol=tol)
    V_true_star = np.max(Q_true_opt, axis=1)

    reward_domain = encoder.features.reward_domain

    expanded_s_feats, expanded_a_feats, expanded_sp_feats = \
        get_feature_combinations(reward_domain, all_obs_features, all_act_features)

    with torch.no_grad():
        R_est_mean, _ = encoder.forward(expanded_s_feats, expanded_a_feats, expanded_sp_feats)

    R_est_mean = to_numpy(R_est_mean).squeeze()

    if reward_domain == 's':
        R_est = np.broadcast_to(R_est_mean[:, None, None], (num_states, num_actions, num_states))
    elif reward_domain == 'sa':
        R_est = np.reshape(R_est_mean, (num_states, num_actions))
        R_est = np.broadcast_to(R_est[:, :, None], (num_states, num_actions, num_states))
    else:
        R_est = np.reshape(R_est_mean, (num_states, num_actions, num_states))

    Q_est_opt = q_opt(P, R_est, gamma, max_iter=max_iter, tol=tol)

    if softmax_beta is not None:
        logits = softmax_beta * Q_est_opt
        logits -= logits.max(axis=1, keepdims=True)
        pi_probs = np.exp(logits)
        pi_probs /= pi_probs.sum(axis=1, keepdims=True)
        V_est_pi = value_under_stochastic_policy(P, R_true, gamma, pi_probs)
    else:
        pi_est = np.argmax(Q_est_opt, axis=1)
        V_est_pi = value_under_policy(P, R_true, gamma, pi_est)

    init_dist = env.unwrapped.get_init_state_dist()
    regret = float(np.average(V_true_star - V_est_pi, weights=init_dist))
    discounted_value = float(np.average(V_est_pi, weights=init_dist))
    return regret, discounted_value


def regret_tabular_imitation(
    env: TabularEnv,
    q_model: torch.nn.Module,
    gamma: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """
    Computes expected regret over states for imitation learning.
    
    Instead of deriving a policy from estimated rewards, this function
    directly uses the learned Q-value model to derive a greedy policy.
    
    Regret is weighted by the initial state distribution.
    
    Args:
        env: Tabular environment with transition and reward matrices.
        q_model: Neural network Q-value model that takes one-hot state features
                 and outputs Q-values for all actions.
        gamma: Discount factor.
        max_iter: Maximum iterations for Q-value computation on true reward.
        tol: Convergence tolerance for Q-value iteration.
        
    Returns:
        tuple: (regret, discounted_value) where discounted_value is the average
               discounted value of the estimated policy at initial states.
    """
    R_true = env.unwrapped.get_reward_matrix()
    P = env.unwrapped.get_transition_matrix()

    num_states = P.shape[0]
    
    # Get device from q_model
    device = next(q_model.parameters()).device
    
    # Create one-hot observation features for all states (matching training transform)
    all_obs_features = torch.eye(num_states, device=device)

    # Compute optimal Q-values for the true reward
    Q_true_opt = q_opt(P, R_true, gamma, max_iter=max_iter, tol=tol)
    V_true_star = np.max(Q_true_opt, axis=1)  # (S,)
    
    # Get Q-values from the learned q_model for all states
    with torch.no_grad():
        Q_est = q_model(all_obs_features)  # (S, A)
    Q_est = to_numpy(Q_est)
    
    # Derive greedy policy from learned Q-values
    pi_est = np.argmax(Q_est, axis=1)  # (S,)
    
    # Compute value of the estimated policy under true reward
    V_est_pi = value_under_policy(P, R_true, gamma, pi_est)

    init_dist = env.unwrapped.get_init_state_dist()
    regret = float(np.average(V_true_star - V_est_pi, weights=init_dist))
    discounted_value = float(np.average(V_est_pi, weights=init_dist))
    return regret, discounted_value


def compute_single_regret_sample(
    seed: int,
    env_fn: Callable[[], gym.Env],
    true_expert_policy: Callable,
    est_optimal_policy: Callable,
    gamma: float,
    max_num_steps: int,
) -> tuple[float, float]:
    """Compute regret for a single sample (used for parallel execution)."""
    env = env_fn()

    traj_expert = rollout(env, true_expert_policy, num_steps=max_num_steps, seed=seed)
    ret_expert = get_discounted_return(traj_expert, gamma)
    
    traj_est = rollout(env, est_optimal_policy, num_steps=max_num_steps, seed=seed)
    ret_est = get_discounted_return(traj_est, gamma)
    cum_rew = get_undiscounted_return(traj_est)
    
    return ret_expert - ret_est, cum_rew


def regret_non_tabular(
    true_optimal_policy: Callable,
    est_optimal_policy: Callable,
    eval_env_fn: Callable[[], gym.Env],
    gamma: float,
    num_samples: int = 1000,
    max_num_steps: int = 100,
    seed_fn: Callable[[int], int] = lambda x: x,
) -> tuple[float, float, Callable]:
    """
    MC estimate of the expected regret and the mean return of the estimated expert policy.
    
    Returns:
        tuple: (regret, mean_reward, estimated_expert_policy)
    """
    
    # ``backend="loky"`` (like ``mean_return_non_tabular``): each worker owns an
    # unpickled policy copy. ``backend="threading"`` is unsafe here because SB3 /
    # PyTorch ``predict`` is not thread-safe on a shared model, which breaks
    # run-to-run reproducibility for MC regret.
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(compute_single_regret_sample)(
            seed=seed_fn(i),
            env_fn=eval_env_fn,
            true_expert_policy=true_optimal_policy,
            est_optimal_policy=est_optimal_policy,
            gamma=gamma,
            max_num_steps=max_num_steps,
        )
        for i in range(num_samples)
    )
    regrets, rewards = zip(*results)
    return np.mean(regrets), np.mean(rewards)


# Compute expected regret
def compute_regret(
    true_optimal_policy: Callable,
    train_env_fn: Callable[[], gym.Env],
    eval_env_fn: Callable[[], gym.Env],
    is_tabular: bool,
    is_imitation: bool,
    fb_model: MultiFeedbackTypeModel,
    gamma,
    ppo_seed: int,
    n_regret_samples: int = 200,
    max_num_steps: int = 1000,
    true_reward_threshold: float = None,
    verbose: int = 1,
    progress_bar: bool = True,
    seed_fn: Callable[[int], int] = lambda x: x,
    retrain_ppo_settings: Optional[Dict[str, Any]] = None,
):
    reference_env = None
    reference_env_created = False
    if is_tabular:
        reference_env = train_env_fn(seed=0)
        reference_env_created = True
    regret, mean_rew, discounted_value, est_policy = None, None, None, None
    ppo_best_step: int | None = None
    if is_tabular:
        if is_imitation:
            # For tabular imitation learning, use Q-values directly from the q_model
            regret, discounted_value = regret_tabular_imitation(reference_env, fb_model.q_model, gamma)
        else:
            # For tabular reward learning, derive policy from estimated reward
            regret, discounted_value = regret_tabular(reference_env, fb_model.encoder, gamma)
    else:
        # get the estimated policy
        if is_imitation:
            q_model = NeuralQValueModel(fb_model.q_model)
            est_policy = QValueExpert(q_model, beta=float("inf"))
        else:
            # retrain a policy on the reward model
            settings = (
                retrain_ppo_settings
                if retrain_ppo_settings is not None
                else default_retrain_ppo_settings()
            )
            est_ppo_model, _eval_history, eval_summary = train_ppo(
                train_env_fn,
                eval_env_fn,
                seed=ppo_seed,
                retrain_ppo_settings=settings,
                true_reward_threshold=true_reward_threshold,
                verbose=verbose,
                progress_bar=progress_bar,
                return_training_stats=True,
            )
            est_policy = PPOExpert(est_ppo_model)
            if eval_summary:
                ppo_best_step = int(max(eval_summary, key=lambda tr: tr[1])[0])

        regret, mean_rew = regret_non_tabular(
            true_optimal_policy=true_optimal_policy,
            est_optimal_policy=est_policy,
            eval_env_fn=eval_env_fn,
            gamma=gamma,
            num_samples=n_regret_samples,
            max_num_steps=max_num_steps,
            seed_fn=seed_fn,
        )
        # For non-tabular, discounted_value remains None (mean_rew is undiscounted)
    return regret, mean_rew, discounted_value, est_policy, ppo_best_step