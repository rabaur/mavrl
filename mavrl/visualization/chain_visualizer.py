"""
Visualization for Chain MDP Environment

Provides functions to visualize:
- Ground truth rewards vs estimated rewards
- Q-values across all states
- Comparison of learned vs optimal values
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mavrl.envs.chain_env.env import ChainEnv
from mavrl.utils.math import log_var_to_std
from mavrl.utils.torch_utils import to_numpy
from mavrl.utils.feature_transforms import get_feature_combinations
from mavrl.multi_fb_model import MultiFeedbackTypeModel


def vis_chain_env(
    env: ChainEnv,
    fb_model: MultiFeedbackTypeModel
) -> plt.Figure:
    """
    Visualizes the estimated rewards and Q-values for a chain environment.
    
    Creates a figure with:
    - Row 1: Ground truth rewards, estimated reward mean, estimated reward std
    - Row 2: Q-values (learned), Q-values (optimal), comparison
    
    Args:
        env: The ChainEnv environment
        fb_model: The trained multi-feedback model
        
    Returns:
        fig: matplotlib figure object that can be logged to wandb
    """
    encoder = fb_model.encoder
    n_states = env.n_states
    n_actions = env.n_actions
    device = next(encoder.parameters()).device
    reward_domain = encoder.features.reward_domain
    
    # Get ground truth values
    gt_rewards = env.get_optimal_rewards()  # Shape: (n_states, n_actions)
    gt_q_values = env.get_optimal_q_values()  # Shape: (n_states, n_actions)
    
    # Construct one-hot features for all states
    all_obs_features = torch.eye(n_states, device=device)
    all_act_features = torch.eye(n_actions, device=device)
    
    # Get feature combinations based on reward domain
    batch_state_features, batch_action_features, batch_next_state_features = \
        get_feature_combinations(reward_domain, all_obs_features, all_act_features)
    
    # Predict reward mean and logvar
    with torch.no_grad():
        mean, log_var = encoder(batch_state_features, batch_action_features, batch_next_state_features)
        q_values = fb_model.q_model(all_obs_features)  # Shape: (n_states, n_actions)
    
    mean = to_numpy(mean).squeeze()
    std = to_numpy(log_var_to_std(log_var)).squeeze()
    q_values = to_numpy(q_values)
    
    # Reshape based on reward domain
    if reward_domain == 's':
        # Shape: (n_states,)
        est_rewards = mean
        est_std = std
        gt_rewards_plot = gt_rewards[:, 0]  # State-based, same for all actions
    elif reward_domain == 'sa':
        # Shape: (n_states * n_actions,) -> (n_states, n_actions)
        est_rewards = mean.reshape(n_states, n_actions)[:, 0]  # Take action 0 (forward)
        est_std = std.reshape(n_states, n_actions)[:, 0]
        gt_rewards_plot = gt_rewards[:, 0]
    else:
        raise NotImplementedError(f"Visualization for reward domain {reward_domain} not implemented")
    
    # Q-values for both actions
    q_forward = q_values[:, 0]  # Action 0 = forward (optimal)
    q_stay = q_values[:, 1] if n_actions > 1 else q_values[:, 0]  # Action 1 = stay
    gt_q_forward = gt_q_values[:, 0]
    gt_q_stay = gt_q_values[:, 1] if n_actions > 1 else gt_q_values[:, 0]
    
    # Create figure
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
    
    states = np.arange(n_states)
    bar_width = 0.6
    
    # ===== Row 1: Rewards =====
    
    # Ground truth rewards
    ax = axs[0, 0]
    bars = ax.bar(states, gt_rewards_plot, width=bar_width, color='forestgreen', edgecolor='black')
    ax.set_xlabel('State')
    ax.set_ylabel('Reward')
    ax.set_title('Ground Truth R(s, a)')
    ax.set_xticks(states)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    _highlight_terminal(ax, n_states, gt_rewards_plot)
    
    # Estimated reward mean
    ax = axs[0, 1]
    bars = ax.bar(states, est_rewards, width=bar_width, color='steelblue', edgecolor='black')
    ax.set_xlabel('State')
    ax.set_ylabel('Reward')
    ax.set_title(r'Estimated $\mu_R(s)$')
    ax.set_xticks(states)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    _highlight_terminal(ax, n_states, est_rewards)
    
    # Estimated reward std
    ax = axs[0, 2]
    bars = ax.bar(states, est_std, width=bar_width, color='coral', edgecolor='black')
    ax.set_xlabel('State')
    ax.set_ylabel('Std Dev')
    ax.set_title(r'Estimated $\sigma_R(s)$')
    ax.set_xticks(states)
    _highlight_terminal(ax, n_states, est_std)
    
    # ===== Row 2: Q-values =====
    
    # Learned Q-values (both actions)
    ax = axs[1, 0]
    width = 0.35
    ax.bar(states - width/2, q_forward, width, label='Q(forward)', color='steelblue', edgecolor='black')
    ax.bar(states + width/2, q_stay, width, label='Q(stay)', color='lightblue', edgecolor='black')
    ax.set_xlabel('State')
    ax.set_ylabel('Q-value')
    ax.set_title('Learned Q(s, a)')
    ax.set_xticks(states)
    ax.legend(fontsize=8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Optimal Q-values (both actions)
    ax = axs[1, 1]
    ax.bar(states - width/2, gt_q_forward, width, label='Q*(forward)', color='forestgreen', edgecolor='black')
    ax.bar(states + width/2, gt_q_stay, width, label='Q*(stay)', color='lightgreen', edgecolor='black')
    ax.set_xlabel('State')
    ax.set_ylabel('Q-value')
    ax.set_title('Optimal Q*(s, a)')
    ax.set_xticks(states)
    ax.legend(fontsize=8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Comparison: Learned vs Optimal for forward action
    ax = axs[1, 2]
    ax.bar(states - width/2, gt_q_forward, width, label='Optimal Q*(fwd)', color='forestgreen', edgecolor='black')
    ax.bar(states + width/2, q_forward, width, label='Learned Q(fwd)', color='steelblue', edgecolor='black')
    ax.set_xlabel('State')
    ax.set_ylabel('Q-value')
    ax.set_title('Q(forward) Comparison')
    ax.set_xticks(states)
    ax.legend(fontsize=8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add state labels
    for ax_row in axs:
        for ax in ax_row:
            labels = [f's{i}' if i < n_states - 1 else f's{i}\n(goal)' for i in states]
            ax.set_xticklabels(labels)
    
    plt.tight_layout()
    return fig


def _highlight_terminal(ax, n_states: int, values: np.ndarray):
    """Add visual highlight to the terminal state bar."""
    # Add a small marker or annotation for the terminal state
    terminal_idx = n_states - 1
    if len(values) > terminal_idx:
        ax.annotate('★', xy=(terminal_idx, values[terminal_idx]), 
                   xytext=(terminal_idx, values[terminal_idx] + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
                   ha='center', fontsize=12, color='gold')


def vis_chain_detailed(
    env: ChainEnv,
    fb_model: MultiFeedbackTypeModel,
    show_td_errors: bool = True
) -> plt.Figure:
    """
    Detailed visualization showing TD error analysis for debugging.
    
    Creates a figure showing:
    - Row 1: R(s), Q(s), Q(s') for each state
    - Row 2: TD error breakdown: Q(s) - γQ(s') vs R(s)
    
    Args:
        env: The ChainEnv environment
        fb_model: The trained multi-feedback model
        show_td_errors: Whether to show TD error analysis
        
    Returns:
        fig: matplotlib figure object
    """
    encoder = fb_model.encoder
    n_states = env.n_states
    n_actions = env.n_actions
    gamma = env.gamma
    device = next(encoder.parameters()).device
    reward_domain = encoder.features.reward_domain
    
    # Construct features
    all_obs_features = torch.eye(n_states, device=device)
    all_act_features = torch.eye(n_actions, device=device)
    
    batch_state_features, batch_action_features, batch_next_state_features = \
        get_feature_combinations(reward_domain, all_obs_features, all_act_features)
    
    with torch.no_grad():
        mean, log_var = encoder(batch_state_features, batch_action_features, batch_next_state_features)
        q_values = fb_model.q_model(all_obs_features)
    
    mean = to_numpy(mean).squeeze()
    q_values = to_numpy(q_values)[:, 0]  # Take action 0
    
    # Reshape rewards
    if reward_domain == 's':
        est_rewards = mean
    elif reward_domain == 'sa':
        est_rewards = mean.reshape(n_states, n_actions)[:, 0]
    else:
        est_rewards = mean
    
    # Compute TD targets: Q(s) - gamma * Q(s')
    # For terminal state, Q(s') = 0
    td_targets = np.zeros(n_states)
    for s in range(n_states):
        if s == n_states - 1:
            # Terminal state: no next state
            td_targets[s] = q_values[s]
        else:
            td_targets[s] = q_values[s] - gamma * q_values[s + 1]
    
    # Ground truth
    gt_rewards = env.get_optimal_rewards()[:, 0]
    gt_q_values = env.get_optimal_q_values()[:, 0]
    
    states = np.arange(n_states)
    
    if show_td_errors:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    else:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
        axs = axs.reshape(1, -1)
    
    # ===== Q-values comparison =====
    ax = axs[0, 0]
    width = 0.35
    ax.bar(states - width/2, gt_q_values, width, label='Optimal Q*', color='forestgreen', edgecolor='black', alpha=0.8)
    ax.bar(states + width/2, q_values, width, label='Learned Q', color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_xlabel('State')
    ax.set_ylabel('Q-value')
    ax.set_title('Q-values: Optimal vs Learned')
    ax.set_xticks(states)
    ax.set_xticklabels([f's{i}' for i in states])
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # ===== Rewards comparison =====
    ax = axs[0, 1]
    ax.bar(states - width/2, gt_rewards, width, label='Ground Truth R', color='forestgreen', edgecolor='black', alpha=0.8)
    ax.bar(states + width/2, est_rewards, width, label='Estimated R', color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_xlabel('State')
    ax.set_ylabel('Reward')
    ax.set_title('Rewards: Ground Truth vs Estimated')
    ax.set_xticks(states)
    ax.set_xticklabels([f's{i}' for i in states])
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    if show_td_errors:
        # ===== TD Error Analysis =====
        ax = axs[1, 0]
        ax.bar(states - width/2, td_targets, width, label='TD Target: Q(s)-γQ(s\')', color='purple', edgecolor='black', alpha=0.8)
        ax.bar(states + width/2, est_rewards, width, label='Estimated R(s)', color='steelblue', edgecolor='black', alpha=0.8)
        ax.set_xlabel('State')
        ax.set_ylabel('Value')
        ax.set_title('TD Error Constraint: R(s) ≈ Q(s) - γQ(s\')')
        ax.set_xticks(states)
        ax.set_xticklabels([f's{i}' for i in states])
        ax.legend()
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # ===== TD Error residuals =====
        ax = axs[1, 1]
        td_residuals = est_rewards - td_targets
        colors = ['red' if r < 0 else 'green' for r in td_residuals]
        ax.bar(states, td_residuals, width=0.6, color=colors, edgecolor='black', alpha=0.8)
        ax.set_xlabel('State')
        ax.set_ylabel('Residual')
        ax.set_title('TD Error Residual: R(s) - [Q(s) - γQ(s\')]')
        ax.set_xticks(states)
        ax.set_xticklabels([f's{i}' for i in states])
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    
    # Add text summary
    print("\n" + "="*60)
    print("CHAIN ENVIRONMENT DEBUG SUMMARY")
    print("="*60)
    print(f"Number of states: {n_states}")
    print(f"Discount factor (gamma): {gamma}")
    print(f"Reward domain: {reward_domain}")
    print("-"*60)
    print(f"{'State':<8} {'GT R':<10} {'Est R':<10} {'GT Q*':<10} {'Est Q':<10} {'TD Target':<10}")
    print("-"*60)
    for s in range(n_states):
        term_marker = " (term)" if s == n_states - 1 else ""
        print(f"s{s}{term_marker:<6} {gt_rewards[s]:<10.3f} {est_rewards[s]:<10.3f} "
              f"{gt_q_values[s]:<10.3f} {q_values[s]:<10.3f} {td_targets[s]:<10.3f}")
    print("="*60)
    
    return fig


def print_chain_analysis(env: ChainEnv, fb_model: MultiFeedbackTypeModel):
    """
    Print a detailed text analysis of the chain environment learning.
    
    Useful for debugging without visualization.
    """
    encoder = fb_model.encoder
    n_states = env.n_states
    n_actions = env.n_actions
    gamma = env.gamma
    device = next(encoder.parameters()).device
    reward_domain = encoder.features.reward_domain
    
    # Construct features
    all_obs_features = torch.eye(n_states, device=device)
    all_act_features = torch.eye(n_actions, device=device)
    
    batch_state_features, batch_action_features, batch_next_state_features = \
        get_feature_combinations(reward_domain, all_obs_features, all_act_features)
    
    with torch.no_grad():
        mean, log_var = encoder(batch_state_features, batch_action_features, batch_next_state_features)
        q_values = fb_model.q_model(all_obs_features)
    
    mean = to_numpy(mean).squeeze()
    std = to_numpy(log_var_to_std(log_var)).squeeze()
    q_values = to_numpy(q_values)[:, 0]
    
    if reward_domain == 's':
        est_rewards = mean
        est_std = std
    elif reward_domain == 'sa':
        est_rewards = mean.reshape(n_states, n_actions)[:, 0]
        est_std = std.reshape(n_states, n_actions)[:, 0]
    else:
        est_rewards = mean
        est_std = std
    
    gt_rewards = env.get_optimal_rewards()[:, 0]
    gt_q_values = env.get_optimal_q_values()[:, 0]
    
    print("\n" + "="*80)
    print("CHAIN ENVIRONMENT ANALYSIS")
    print("="*80)
    print(f"States: {n_states} | Actions: {n_actions} | Gamma: {gamma} | Reward Domain: {reward_domain}")
    print("-"*80)
    
    print("\n### Ground Truth ###")
    print(f"Terminal reward: {env.terminal_reward}")
    print(f"Step reward: {env.step_reward}")
    
    print("\n### Per-State Analysis ###")
    print(f"{'State':<10} {'GT_R':<12} {'Est_R':<12} {'Est_σ':<12} {'GT_Q*':<12} {'Est_Q':<12}")
    print("-"*80)
    
    max_r_state = np.argmax(est_rewards)
    max_q_state = np.argmax(q_values)
    
    for s in range(n_states):
        term = " [TERM]" if s == n_states - 1 else ""
        max_r = " ← MAX R" if s == max_r_state else ""
        max_q = " ← MAX Q" if s == max_q_state else ""
        
        print(f"s{s}{term:<7} {gt_rewards[s]:<12.4f} {est_rewards[s]:<12.4f} "
              f"{est_std[s]:<12.4f} {gt_q_values[s]:<12.4f} {q_values[s]:<12.4f}{max_r}{max_q}")
    
    print("\n### TD Error Analysis ###")
    print("TD Constraint: R(s) = Q(s,a) - γ·Q(s',a')")
    print("-"*80)
    
    for s in range(n_states - 1):
        q_curr = q_values[s]
        q_next = q_values[s + 1]
        td_target = q_curr - gamma * q_next
        td_error = est_rewards[s] - td_target
        print(f"s{s} → s{s+1}: Q(s{s})={q_curr:.4f} - {gamma}·Q(s{s+1})={q_next:.4f} = {td_target:.4f}")
        print(f"         Est R(s{s})={est_rewards[s]:.4f}, TD Error={td_error:.4f}")
    
    # Terminal state
    s = n_states - 1
    print(f"s{s} [TERM]: Q(s{s})={q_values[s]:.4f} - γ·0 = {q_values[s]:.4f}")
    print(f"         Est R(s{s})={est_rewards[s]:.4f}, TD Error={est_rewards[s] - q_values[s]:.4f}")
    
    print("\n### Key Observations ###")
    print(f"- Highest estimated reward at: s{max_r_state} (R={est_rewards[max_r_state]:.4f})")
    print(f"- Highest Q-value at: s{max_q_state} (Q={q_values[max_q_state]:.4f})")
    print(f"- Expected highest reward at: s{n_states-2} (transition to terminal)")
    
    if max_r_state != n_states - 2:
        print(f"⚠️  WARNING: Expected max reward at s{n_states-2}, but found at s{max_r_state}")
    
    print("="*80)
