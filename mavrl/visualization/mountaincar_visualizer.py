import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from mavrl.multi_fb_model import MultiFeedbackTypeModel
from mavrl.utils.math import log_var_to_std
from mavrl.utils.torch_utils import get_model_device, to_numpy
from tqdm import tqdm


def vis_mountaincar(
    env: gym.Env,
    fb_model: MultiFeedbackTypeModel,
    resolution: int = 64,
    batch_size: int = 1024
):
    """
    Visualize learned rewards for MountainCar-v0 environment.
    
    Creates a figure with 2 rows x 3 columns:
    - Top row: Mean reward for each action (push left, no push, push right)
    - Bottom row: Uncertainty (std) for each action
    
    Each subplot shows a heatmap with position on x-axis and velocity on y-axis,
    with the MountainCar terrain sinusoid overlaid in red.
    
    Args:
        env: MountainCar environment
        fb_model: Multi-feedback model
        resolution: Resolution of the visualization grid
        batch_size: Batch size for predictions
    """
    action_space = env.action_space
    num_actions = action_space.n
    
    # MountainCar state space bounds
    pos_range = np.linspace(-1.2, 0.6, resolution)
    vel_range = np.linspace(-0.07, 0.07, resolution)
    
    pos_grid, vel_grid = np.meshgrid(pos_range, vel_range)
    
    pos_flat = pos_grid.flatten()
    vel_flat = vel_grid.flatten()
    num_data = pos_flat.shape[0]
    
    # Create observation array: [position, velocity]
    all_feats = np.stack([pos_flat, vel_flat], axis=-1)
    
    model_device = get_model_device(fb_model)
    all_feats_torch = torch.tensor(all_feats, device=model_device, dtype=torch.float32)
    
    # Get reward domain from the encoder's feature module
    reward_domain = fb_model.encoder.features.reward_domain
    
    if reward_domain == 's':
        fig = _vis_state_only_reward(
            all_feats_torch, resolution, num_data, batch_size,
            fb_model, num_actions, pos_range, vel_range
        )
    elif reward_domain == 'sa':
        fig = _vis_state_action_reward(
            all_feats_torch, resolution, num_data, batch_size,
            fb_model, num_actions, model_device, pos_range, vel_range
        )
    elif reward_domain == 'sas':
        fig = _vis_state_action_nextstate_reward(
            all_feats_torch, resolution, num_data, batch_size,
            fb_model, num_actions, model_device, pos_range, vel_range
        )
    else:
        raise ValueError(f"Unsupported reward domain: {reward_domain}")
    
    return fig


def _add_terrain_overlay(ax, pos_range: np.ndarray, vel_range: np.ndarray):
    """
    Add the MountainCar terrain sinusoid as a visual reference.
    
    The terrain height is: sin(3*x)*0.45 + 0.55
    We scale it to fit within the velocity range for visual reference.
    """
    # Compute terrain height (from MountainCar source)
    height = np.sin(3 * pos_range) * 0.45 + 0.55
    
    # Normalize to velocity range
    height_min, height_max = height.min(), height.max()
    vel_min, vel_max = vel_range[0], vel_range[-1]
    scaled_height = vel_min + (height - height_min) / (height_max - height_min) * (vel_max - vel_min)
    
    ax.plot(pos_range, scaled_height, 'r-', linewidth=2, label='Terrain')


def _add_goal_marker(ax, pos_range: np.ndarray, vel_range: np.ndarray):
    """Add a marker for the goal position (x >= 0.5)."""
    # Goal is reached when position >= 0.5
    if 0.5 <= pos_range[-1]:
        ax.axvline(x=0.5, color='lime', linestyle='--', linewidth=2, label='Goal')


def _vis_state_only_reward(
    all_feats_torch: torch.Tensor,
    resolution: int,
    num_data: int,
    batch_size: int,
    fb_model: MultiFeedbackTypeModel,
    num_actions: int,
    pos_range: np.ndarray,
    vel_range: np.ndarray
):
    """Visualize state-only reward R(s) - single plot since reward doesn't depend on action."""
    fig, axs = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(12, 5))
    
    # Predict rewards (state-only)
    est_rewards = np.empty(num_data)
    est_std = np.empty(num_data)
    for i in tqdm(range(0, num_data, batch_size), desc="Predicting rewards"):
        batch = all_feats_torch[i:i+batch_size]
        r_batch, log_var_batch = fb_model.encoder(obs=batch, acts=None, next_obs=None)
        r_batch = to_numpy(r_batch).squeeze()
        std_batch = to_numpy(log_var_to_std(log_var_batch)).squeeze()
        est_rewards[i:i+batch_size] = r_batch
        est_std[i:i+batch_size] = std_batch
    
    est_rewards_resh = np.reshape(est_rewards, (resolution, resolution))
    est_std_resh = np.reshape(est_std, (resolution, resolution))
    
    # Plot reward
    im_r = axs[0].imshow(
        est_rewards_resh,
        extent=[pos_range[0], pos_range[-1], vel_range[0], vel_range[-1]],
        origin='lower',
        aspect='auto'
    )
    _add_terrain_overlay(axs[0], pos_range, vel_range)
    _add_goal_marker(axs[0], pos_range, vel_range)
    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Velocity")
    axs[0].set_title("R(s)")
    plt.colorbar(im_r, ax=axs[0])
    
    # Plot uncertainty
    im_std = axs[1].imshow(
        est_std_resh,
        extent=[pos_range[0], pos_range[-1], vel_range[0], vel_range[-1]],
        origin='lower',
        aspect='auto'
    )
    _add_terrain_overlay(axs[1], pos_range, vel_range)
    _add_goal_marker(axs[1], pos_range, vel_range)
    axs[1].set_xlabel("Position")
    axs[1].set_ylabel("Velocity")
    axs[1].set_title("std(s)")
    plt.colorbar(im_std, ax=axs[1])
    
    return fig


def _vis_state_action_reward(
    all_feats_torch: torch.Tensor,
    resolution: int,
    num_data: int,
    batch_size: int,
    fb_model: MultiFeedbackTypeModel,
    num_actions: int,
    device: torch.device,
    pos_range: np.ndarray,
    vel_range: np.ndarray
):
    """Visualize state-action reward R(s, a) - 2 rows x 3 columns (mean + std per action)."""
    action_names = ["Push Left", "No Push", "Push Right"]
    
    fig, axs = plt.subplots(nrows=2, ncols=3, constrained_layout=True, figsize=(15, 10))
    
    # Predict rewards and uncertainties for each action
    est_rewards_all = np.empty((num_actions, num_data))
    est_std_all = np.empty((num_actions, num_data))
    
    for a in range(num_actions):
        # Create one-hot action encoding
        action_one_hot = torch.zeros((num_data, num_actions), device=device, dtype=torch.float32)
        action_one_hot[:, a] = 1.0
        
        for i in tqdm(range(0, num_data, batch_size), desc=f"Predicting R(s, a={a})"):
            batch_obs = all_feats_torch[i:i+batch_size]
            batch_acts = action_one_hot[i:i+batch_size]
            r_batch, log_var_batch = fb_model.encoder(obs=batch_obs, acts=batch_acts, next_obs=None)
            r_batch = to_numpy(r_batch).squeeze()
            std_batch = to_numpy(log_var_to_std(log_var_batch)).squeeze()
            est_rewards_all[a, i:i+batch_size] = r_batch
            est_std_all[a, i:i+batch_size] = std_batch
    
    # Reshape rewards and uncertainties
    est_rewards_resh = np.reshape(est_rewards_all, (num_actions, resolution, resolution))
    est_std_resh = np.reshape(est_std_all, (num_actions, resolution, resolution))
    
    # Find global min/max for consistent colorbars
    reward_vmin = est_rewards_resh.min()
    reward_vmax = est_rewards_resh.max()
    std_vmin = est_std_resh.min()
    std_vmax = est_std_resh.max()
    
    # Plot each action's mean reward (top row)
    for a in range(num_actions):
        im = axs[0, a].imshow(
            est_rewards_resh[a],
            extent=[pos_range[0], pos_range[-1], vel_range[0], vel_range[-1]],
            origin='lower',
            aspect='auto',
            vmin=reward_vmin,
            vmax=reward_vmax
        )
        _add_terrain_overlay(axs[0, a], pos_range, vel_range)
        _add_goal_marker(axs[0, a], pos_range, vel_range)
        axs[0, a].set_xlabel("Position")
        axs[0, a].set_ylabel("Velocity")
        axs[0, a].set_title(f"R(s, {action_names[a]})")
        plt.colorbar(im, ax=axs[0, a])
    
    # Plot each action's uncertainty (bottom row)
    for a in range(num_actions):
        im = axs[1, a].imshow(
            est_std_resh[a],
            extent=[pos_range[0], pos_range[-1], vel_range[0], vel_range[-1]],
            origin='lower',
            aspect='auto',
            vmin=std_vmin,
            vmax=std_vmax
        )
        _add_terrain_overlay(axs[1, a], pos_range, vel_range)
        _add_goal_marker(axs[1, a], pos_range, vel_range)
        axs[1, a].set_xlabel("Position")
        axs[1, a].set_ylabel("Velocity")
        axs[1, a].set_title(f"std(s, {action_names[a]})")
        plt.colorbar(im, ax=axs[1, a])
    
    return fig


def _vis_state_action_nextstate_reward(
    all_feats_torch: torch.Tensor,
    resolution: int,
    num_data: int,
    batch_size: int,
    fb_model: MultiFeedbackTypeModel,
    num_actions: int,
    device: torch.device,
    pos_range: np.ndarray,
    vel_range: np.ndarray
):
    """Visualize state-action-nextstate reward R(s, a, s') - 2 rows x 3 columns.
    
    Uses current state as next_state (self-transition approximation).
    """
    action_names = ["Push Left", "No Push", "Push Right"]
    
    fig, axs = plt.subplots(nrows=2, ncols=3, constrained_layout=True, figsize=(15, 10))
    
    # Predict rewards and uncertainties for each action
    est_rewards_all = np.empty((num_actions, num_data))
    est_std_all = np.empty((num_actions, num_data))
    
    for a in range(num_actions):
        # Create one-hot action encoding
        action_one_hot = torch.zeros((num_data, num_actions), device=device, dtype=torch.float32)
        action_one_hot[:, a] = 1.0
        
        for i in tqdm(range(0, num_data, batch_size), desc=f"Predicting R(s, a={a}, s')"):
            batch_obs = all_feats_torch[i:i+batch_size]
            batch_acts = action_one_hot[i:i+batch_size]
            # Use current state as next_state (self-transition approximation)
            batch_next_obs = batch_obs
            r_batch, log_var_batch = fb_model.encoder(obs=batch_obs, acts=batch_acts, next_obs=batch_next_obs)
            r_batch = to_numpy(r_batch).squeeze()
            std_batch = to_numpy(log_var_to_std(log_var_batch)).squeeze()
            est_rewards_all[a, i:i+batch_size] = r_batch
            est_std_all[a, i:i+batch_size] = std_batch
    
    # Reshape rewards and uncertainties
    est_rewards_resh = np.reshape(est_rewards_all, (num_actions, resolution, resolution))
    est_std_resh = np.reshape(est_std_all, (num_actions, resolution, resolution))
    
    # Find global min/max for consistent colorbars
    reward_vmin = est_rewards_resh.min()
    reward_vmax = est_rewards_resh.max()
    std_vmin = est_std_resh.min()
    std_vmax = est_std_resh.max()
    
    # Plot each action's mean reward (top row)
    for a in range(num_actions):
        im = axs[0, a].imshow(
            est_rewards_resh[a],
            extent=[pos_range[0], pos_range[-1], vel_range[0], vel_range[-1]],
            origin='lower',
            aspect='auto',
            vmin=reward_vmin,
            vmax=reward_vmax
        )
        _add_terrain_overlay(axs[0, a], pos_range, vel_range)
        _add_goal_marker(axs[0, a], pos_range, vel_range)
        axs[0, a].set_xlabel("Position")
        axs[0, a].set_ylabel("Velocity")
        axs[0, a].set_title(f"R(s, {action_names[a]}, s')\n(s' = s)")
        plt.colorbar(im, ax=axs[0, a])
    
    # Plot each action's uncertainty (bottom row)
    for a in range(num_actions):
        im = axs[1, a].imshow(
            est_std_resh[a],
            extent=[pos_range[0], pos_range[-1], vel_range[0], vel_range[-1]],
            origin='lower',
            aspect='auto',
            vmin=std_vmin,
            vmax=std_vmax
        )
        _add_terrain_overlay(axs[1, a], pos_range, vel_range)
        _add_goal_marker(axs[1, a], pos_range, vel_range)
        axs[1, a].set_xlabel("Position")
        axs[1, a].set_ylabel("Velocity")
        axs[1, a].set_title(f"std(s, {action_names[a]}, s')\n(s' = s)")
        plt.colorbar(im, ax=axs[1, a])
    
    return fig
