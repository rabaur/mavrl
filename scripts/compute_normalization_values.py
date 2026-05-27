#!/usr/bin/env python3
"""
Compute normalization values (optimal and uniform policy mean returns)
for tabular grid environments and non-tabular Gymnasium environments.

This script computes:
- Mean return of the optimal policy (our reference policy)
- Mean return of the uniform random policy

Results are saved to the results folder for use in normalizing final tables.
"""

import numpy as np
import json
import gymnasium as gym
from pathlib import Path
from mavrl.envs.grid_env.env import GridEnv
from mavrl.utils.tabular import q_opt
from mavrl.evaluation.regret import (
    value_under_policy,
    mean_return_non_tabular,
    UniformRandomPolicy,
)
from mavrl.utils.policies import create_policy


def compute_uniform_policy_value(P: np.ndarray, R: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute the value function for a uniform random policy.
    
    Under a uniform policy, pi(a|s) = 1/|A| for all s, a.
    """
    n_states, n_actions, _ = P.shape
    
    # For uniform policy, we average over all actions
    # P_uniform[s, s'] = (1/|A|) * sum_a P[s, a, s']
    P_uniform = np.mean(P, axis=1)  # (S, S)
    
    # Expected reward under uniform policy
    # r_uniform[s] = (1/|A|) * sum_a E[R(s, a, s')] = (1/|A|) * sum_a sum_s' P(s'|s,a) * R(s,a,s')
    expected_reward_per_action = np.sum(P * R, axis=2)  # (S, A)
    r_uniform = np.mean(expected_reward_per_action, axis=1)  # (S,)
    
    # Solve (I - gamma * P_uniform) V = r_uniform
    I = np.eye(n_states)
    V_uniform = np.linalg.solve(I - gamma * P_uniform, r_uniform)
    
    return V_uniform


def compute_normalization_values(
    env_id: str,
    grid_size: int,
    gamma: float,
    p_rand: float,
) -> dict:
    """
    Compute optimal and uniform policy values for a grid environment.
    
    Returns:
        dict with keys:
            - optimal_regret: regret of optimal policy (should be 0)
            - optimal_discounted_value: V*(s0)
            - uniform_regret: regret of uniform policy
            - uniform_discounted_value: V_uniform(s0)
    """
    # Parse reward type from env_id (e.g., "grid_cliff" -> "cliff")
    reward_type = env_id.split("_")[1]
    
    # Create environment
    env = GridEnv(
        grid_size=grid_size,
        reward_type=reward_type,
        p_rand=p_rand,
        gamma=gamma,
        seed=0,
    )
    
    P = env.get_transition_matrix()
    R = env.get_reward_matrix()
    init_dist = env.get_init_state_dist()
    
    # Compute optimal Q-values and policy
    Q_opt = q_opt(P, R, gamma)
    V_opt = np.max(Q_opt, axis=1)  # (S,)
    pi_opt = np.argmax(Q_opt, axis=1)  # (S,)
    
    # Compute value of optimal policy under true reward (should equal V_opt)
    V_pi_opt = value_under_policy(P, R, gamma, pi_opt)
    
    # Compute value of uniform policy
    V_uniform = compute_uniform_policy_value(P, R, gamma)
    
    # Compute regrets weighted by initial state distribution
    optimal_regret = float(np.average(V_opt - V_pi_opt, weights=init_dist))
    optimal_discounted_value = float(np.average(V_pi_opt, weights=init_dist))
    
    uniform_regret = float(np.average(V_opt - V_uniform, weights=init_dist))
    uniform_discounted_value = float(np.average(V_uniform, weights=init_dist))
    
    return {
        "env_id": env_id,
        "grid_size": grid_size,
        "gamma": gamma,
        "p_rand": p_rand,
        "optimal_regret": optimal_regret,
        "optimal_discounted_value": optimal_discounted_value,
        "uniform_regret": uniform_regret,
        "uniform_discounted_value": uniform_discounted_value,
    }


def compute_normalization_values_non_tabular(
    env_id: str,
    optimal_policy_path: str,
    num_samples: int = 1000,
    max_num_steps: int = 1000,
    beta: float = float("inf"),
) -> dict:
    """
    Compute optimal and uniform policy mean returns for a non-tabular Gymnasium environment.
    
    Uses Monte Carlo estimation to compute returns.
    
    Args:
        env_id: Gymnasium environment ID (e.g., "CartPole-v1", "LunarLander-v3").
        optimal_policy_path: Path to the saved optimal policy.
        num_samples: Number of episodes to sample for MC estimation.
        max_num_steps: Maximum number of steps per episode.
        beta: Rationality parameter for the optimal policy (inf = deterministic).
    
    Returns:
        dict with keys:
            - optimal_mean_return: mean return of optimal policy
            - optimal_std_return: std of optimal policy returns
            - uniform_mean_return: mean return of uniform policy
            - uniform_std_return: std of uniform policy returns
    """
    # Create environment factory
    def env_fn():
        return gym.make(env_id)
    
    # Create a reference environment to load the policy
    ref_env = env_fn()
    
    # Load the optimal policy
    optimal_policy = create_policy(optimal_policy_path, beta=beta, env=ref_env)
    
    # Create uniform random policy
    uniform_policy = UniformRandomPolicy(ref_env.action_space)
    
    print(f"  Computing optimal policy returns ({num_samples} samples)...")
    optimal_mean, optimal_std = mean_return_non_tabular(
        policy=optimal_policy,
        env_fn=env_fn,
        num_samples=num_samples,
        max_num_steps=max_num_steps,
    )
    
    print(f"  Computing uniform policy returns ({num_samples} samples)...")
    uniform_mean, uniform_std = mean_return_non_tabular(
        policy=uniform_policy,
        env_fn=env_fn,
        num_samples=num_samples,
        max_num_steps=max_num_steps,
    )
    
    ref_env.close()
    
    return {
        "env_id": env_id,
        "optimal_policy_path": optimal_policy_path,
        "num_samples": num_samples,
        "max_num_steps": max_num_steps,
        "optimal_mean_return": optimal_mean,
        "optimal_std_return": optimal_std,
        "uniform_mean_return": uniform_mean,
        "uniform_std_return": uniform_std,
    }


def main():
    # Configuration for tabular grid environments
    tabular_environments = [
        {"env_id": "grid_cliff", "grid_size": 10, "gamma": 0.95, "p_rand": 0.0},
        {"env_id": "grid_sparse", "grid_size": 10, "gamma": 0.95, "p_rand": 0.0},
        {"env_id": "grid_trap", "grid_size": 10, "gamma": 0.95, "p_rand": 0.0},
    ]
    
    # Configuration for non-tabular Gymnasium environments
    # Paths match those in the sweep files
    non_tabular_environments = [
        {
            "env_id": "CartPole-v1",
            "optimal_policy_path": "expert_policies/dqn/CartPole-v1_1/best_model.zip",
            "num_samples": 1000,
            "max_num_steps": 500,  # CartPole episode limit
        },
        {
            "env_id": "Acrobot-v1",
            "optimal_policy_path": "expert_policies/dqn/Acrobot-v1_1/best_model.zip",
            "num_samples": 1000,
            "max_num_steps": 500,  # Acrobot episode limit
        },
        {
            "env_id": "LunarLander-v3",
            "optimal_policy_path": "expert_policies/dqn/LunarLander-v3_1/best_model.zip",
            "num_samples": 1000,
            "max_num_steps": 1000,  # LunarLander episode limit
        },
    ]
    
    results = {}
    
    # =========================================================================
    # Compute normalization values for tabular environments
    # =========================================================================
    print("=" * 70)
    print("Computing normalization values for TABULAR grid environments")
    print("=" * 70)
    
    for env_config in tabular_environments:
        env_id = env_config["env_id"]
        print(f"\n{env_id}:")
        print("-" * 50)
        
        values = compute_normalization_values(**env_config)
        results[env_id] = values
        
        print(f"  Optimal policy:")
        print(f"    Regret:           {values['optimal_regret']:.6f}")
        print(f"    Discounted value: {values['optimal_discounted_value']:.6f}")
        print(f"  Uniform policy:")
        print(f"    Regret:           {values['uniform_regret']:.6f}")
        print(f"    Discounted value: {values['uniform_discounted_value']:.6f}")
    
    # =========================================================================
    # Compute normalization values for non-tabular environments
    # =========================================================================
    print("\n" + "=" * 70)
    print("Computing normalization values for NON-TABULAR Gymnasium environments")
    print("=" * 70)
    
    for env_config in non_tabular_environments:
        env_id = env_config["env_id"]
        print(f"\n{env_id}:")
        print("-" * 50)
        
        values = compute_normalization_values_non_tabular(**env_config)
        results[env_id] = values
        
        print(f"  Optimal policy:")
        print(f"    Mean return: {values['optimal_mean_return']:.2f} ± {values['optimal_std_return']:.2f}")
        print(f"  Uniform policy:")
        print(f"    Mean return: {values['uniform_mean_return']:.2f} ± {values['uniform_std_return']:.2f}")
    
    # Save results to file
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / "normalization_values.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_file}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
