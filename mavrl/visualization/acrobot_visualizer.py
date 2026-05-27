import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from mavrl.multi_fb_model import MultiFeedbackTypeModel
from mavrl.utils.math import log_var_to_std
from mavrl.utils.torch_utils import get_model_device, to_numpy
from tqdm import tqdm


def _add_success_region_contour(ax, cos_theta1_range: np.ndarray, cos_theta2_range: np.ndarray):
    """
    Add a contour showing the Acrobot success region boundary.
    
    The success condition is: -cos(theta1) - cos(theta1 + theta2) > 1.0
    
    With sin(theta1) = sin(theta2) = 0 (as used in visualization):
    cos(theta1 + theta2) = cos(theta1)*cos(theta2)
    
    So the condition simplifies to:
    -cos(theta1) - cos(theta1)*cos(theta2) > 1.0
    -cos(theta1)*(1 + cos(theta2)) > 1.0
    """
    cos1_grid, cos2_grid = np.meshgrid(cos_theta1_range, cos_theta2_range)
    
    # Success condition: -cos(theta1)*(1 + cos(theta2)) > 1.0
    success_value = -cos1_grid * (1 + cos2_grid)
    
    # Draw the boundary contour at success_value = 1.0
    contour = ax.contour(
        cos1_grid, cos2_grid, success_value,
        levels=[1.0],
        colors=['lime'],
        linewidths=[2.5],
        linestyles=['--']
    )
    ax.clabel(contour, inline=True, fontsize=8, fmt='success boundary')
    
    # Optionally shade the success region with transparency
    ax.contourf(
        cos1_grid, cos2_grid, success_value,
        levels=[1.0, np.max(success_value) + 1],
        colors=['lime'],
        alpha=0.15
    )


def vis_acrobot(
    env: gym.Env,
    fb_model: MultiFeedbackTypeModel,
    resolution: int = 64,
    batch_size: int = 1024
):
    """
    Visualize learned rewards for Acrobot-v1 environment.
    
    Creates a 2D visualization plotting cosine of theta1 (obs[0]) against 
    cosine of theta2 (obs[2]) for all three actions.
    
    Args:
        env: Acrobot environment
        fb_model: Multi-feedback model
        resolution: Resolution of the visualization grid
        batch_size: Batch size for predictions
    """
    obs_space = env.observation_space
    action_space = env.action_space
    num_actions = action_space.n
    
    # Create grid for cos(theta1) vs cos(theta2)
    # Both cosines are in [-1, 1] range
    cos_theta1_range = np.linspace(-1.0, 1.0, resolution)
    cos_theta2_range = np.linspace(-1.0, 1.0, resolution)
    cos1_grid, cos2_grid = np.meshgrid(cos_theta1_range, cos_theta2_range)
    
    # Flatten the grid
    cos1_flat = cos1_grid.flatten()
    cos2_flat = cos2_grid.flatten()
    num_data = cos1_flat.shape[0]
    
    # Create full observation vectors
    # obs[0] = cos(theta1), obs[1] = sin(theta1), obs[2] = cos(theta2), 
    # obs[3] = sin(theta2), obs[4] = angular_vel_theta1, obs[5] = angular_vel_theta2
    # We set sin values and angular velocities to 0 for visualization
    sin_theta1 = np.zeros(num_data)
    sin_theta2 = np.zeros(num_data)
    ang_vel_theta1 = np.zeros(num_data)
    ang_vel_theta2 = np.zeros(num_data)
    
    all_feats = np.stack([
        cos1_flat,      # obs[0]: cos(theta1)
        sin_theta1,     # obs[1]: sin(theta1) = 0
        cos2_flat,      # obs[2]: cos(theta2)
        sin_theta2,     # obs[3]: sin(theta2) = 0
        ang_vel_theta1, # obs[4]: angular velocity = 0
        ang_vel_theta2  # obs[5]: angular velocity = 0
    ], axis=-1)
    
    model_device = get_model_device(fb_model)
    all_feats_torch = torch.tensor(all_feats, device=model_device, dtype=torch.float32)

    # Predict Q-values
    est_q_vals = np.empty((num_data, num_actions))
    for i in tqdm(range(0, num_data, batch_size), desc="Visualizing Acrobot"):
        batch = all_feats_torch[i:i+batch_size]
        q_vals_batch = to_numpy(fb_model.q_model(batch))
        est_q_vals[i:i+batch_size] = q_vals_batch
    
    # Reshape Q-values
    est_q_vals_resh = np.reshape(est_q_vals, (resolution, resolution, num_actions))

    # Get reward domain from the encoder's feature module
    reward_domain = fb_model.encoder.features.reward_domain

    if reward_domain == 's':
        fig = _vis_state_only_reward(
            all_feats_torch, est_q_vals_resh, resolution, num_data, 
            batch_size, fb_model, num_actions, cos_theta1_range, cos_theta2_range
        )
    elif reward_domain == 'sa':
        fig = _vis_state_action_reward(
            all_feats_torch, est_q_vals_resh, resolution, num_data, 
            batch_size, fb_model, num_actions, model_device, cos_theta1_range, cos_theta2_range
        )
    elif reward_domain == 'sas':
        fig = _vis_state_action_nextstate_reward(
            all_feats_torch, est_q_vals_resh, resolution, num_data, 
            batch_size, fb_model, num_actions, model_device, cos_theta1_range, cos_theta2_range
        )
    else:
        raise ValueError(f"Unsupported reward domain: {reward_domain}")

    return fig


def _vis_state_only_reward(
    all_feats_torch: torch.Tensor,
    est_q_vals_resh: np.ndarray,
    resolution: int,
    num_data: int,
    batch_size: int,
    fb_model: MultiFeedbackTypeModel,
    num_actions: int,
    cos_theta1_range: np.ndarray,
    cos_theta2_range: np.ndarray
):
    """Visualize state-only reward R(s)."""
    # Create figure: 2 rows for Q-values, 1 row for rewards
    fig, axs = plt.subplots(nrows=3, ncols=2, constrained_layout=True, figsize=(12, 14))
    
    # Plot Q-values for all three actions
    action_names = ["torque -1", "torque 0", "torque +1"]
    for a in range(num_actions):
        row, col = divmod(a, 2)
        if a < 3:
            im = axs[row, col].imshow(
                est_q_vals_resh[..., a],
                extent=[cos_theta1_range[0], cos_theta1_range[-1], 
                        cos_theta2_range[0], cos_theta2_range[-1]],
                origin='lower',
                aspect='auto'
            )
            _add_success_region_contour(axs[row, col], cos_theta1_range, cos_theta2_range)
            axs[row, col].set_xlabel("cos(θ₁)")
            axs[row, col].set_ylabel("cos(θ₂)")
            axs[row, col].set_title(f"Q(s, {action_names[a]})")
            plt.colorbar(im, ax=axs[row, col])
    
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
    
    im_r = axs[2, 0].imshow(
        est_rewards_resh,
        extent=[cos_theta1_range[0], cos_theta1_range[-1], 
                cos_theta2_range[0], cos_theta2_range[-1]],
        origin='lower',
        aspect='auto'
    )
    _add_success_region_contour(axs[2, 0], cos_theta1_range, cos_theta2_range)
    axs[2, 0].set_xlabel("cos(θ₁)")
    axs[2, 0].set_ylabel("cos(θ₂)")
    axs[2, 0].set_title("R(s)")
    plt.colorbar(im_r, ax=axs[2, 0])
    
    im_std = axs[2, 1].imshow(
        est_std_resh,
        extent=[cos_theta1_range[0], cos_theta1_range[-1], 
                cos_theta2_range[0], cos_theta2_range[-1]],
        origin='lower',
        aspect='auto'
    )
    _add_success_region_contour(axs[2, 1], cos_theta1_range, cos_theta2_range)
    axs[2, 1].set_xlabel("cos(θ₁)")
    axs[2, 1].set_ylabel("cos(θ₂)")
    axs[2, 1].set_title("std(s)")
    plt.colorbar(im_std, ax=axs[2, 1])

    return fig


def _vis_state_action_reward(
    all_feats_torch: torch.Tensor,
    est_q_vals_resh: np.ndarray,
    resolution: int,
    num_data: int,
    batch_size: int,
    fb_model: MultiFeedbackTypeModel,
    num_actions: int,
    device: torch.device,
    cos_theta1_range: np.ndarray,
    cos_theta2_range: np.ndarray
):
    """Visualize state-action reward R(s, a)."""
    action_names = ["torque -1", "torque 0", "torque +1"]
    
    # Create figure: 2 rows for Q-values, 2 rows for rewards (mean + std per action)
    fig, axs = plt.subplots(nrows=4, ncols=2, constrained_layout=True, figsize=(12, 16))
    
    # Plot Q-values (first 2 rows)
    for a in range(num_actions):
        row, col = divmod(a, 2)
        im = axs[row, col].imshow(
            est_q_vals_resh[..., a],
            extent=[cos_theta1_range[0], cos_theta1_range[-1], 
                    cos_theta2_range[0], cos_theta2_range[-1]],
            origin='lower',
            aspect='auto'
        )
        _add_success_region_contour(axs[row, col], cos_theta1_range, cos_theta2_range)
        axs[row, col].set_xlabel("cos(θ₁)")
        axs[row, col].set_ylabel("cos(θ₂)")
        axs[row, col].set_title(f"Q(s, {action_names[a]})")
        plt.colorbar(im, ax=axs[row, col])

    # Predict rewards for each action
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

    # Reshape and compute mean/std across actions for visualization
    est_rewards_resh = np.reshape(est_rewards_all, (num_actions, resolution, resolution))
    est_std_resh = np.reshape(est_std_all, (num_actions, resolution, resolution))
    
    # Show mean reward across all actions and average std
    mean_reward = np.mean(est_rewards_resh, axis=0)
    mean_std = np.mean(est_std_resh, axis=0)
    
    im_r = axs[2, 0].imshow(
        mean_reward,
        extent=[cos_theta1_range[0], cos_theta1_range[-1], 
                cos_theta2_range[0], cos_theta2_range[-1]],
        origin='lower',
        aspect='auto'
    )
    _add_success_region_contour(axs[2, 0], cos_theta1_range, cos_theta2_range)
    axs[2, 0].set_xlabel("cos(θ₁)")
    axs[2, 0].set_ylabel("cos(θ₂)")
    axs[2, 0].set_title("Mean R(s, a) across actions")
    plt.colorbar(im_r, ax=axs[2, 0])
    
    im_std = axs[2, 1].imshow(
        mean_std,
        extent=[cos_theta1_range[0], cos_theta1_range[-1], 
                cos_theta2_range[0], cos_theta2_range[-1]],
        origin='lower',
        aspect='auto'
    )
    _add_success_region_contour(axs[2, 1], cos_theta1_range, cos_theta2_range)
    axs[2, 1].set_xlabel("cos(θ₁)")
    axs[2, 1].set_ylabel("cos(θ₂)")
    axs[2, 1].set_title("Mean std(s, a) across actions")
    plt.colorbar(im_std, ax=axs[2, 1])
    
    # Show max reward action and reward range
    max_reward_action = np.argmax(est_rewards_resh, axis=0)
    reward_range = np.max(est_rewards_resh, axis=0) - np.min(est_rewards_resh, axis=0)
    
    im_max = axs[3, 0].imshow(
        max_reward_action,
        extent=[cos_theta1_range[0], cos_theta1_range[-1], 
                cos_theta2_range[0], cos_theta2_range[-1]],
        origin='lower',
        aspect='auto',
        cmap='tab10',
        vmin=0,
        vmax=num_actions-1
    )
    _add_success_region_contour(axs[3, 0], cos_theta1_range, cos_theta2_range)
    axs[3, 0].set_xlabel("cos(θ₁)")
    axs[3, 0].set_ylabel("cos(θ₂)")
    axs[3, 0].set_title("Argmax_a R(s, a)")
    plt.colorbar(im_max, ax=axs[3, 0])
    
    im_range = axs[3, 1].imshow(
        reward_range,
        extent=[cos_theta1_range[0], cos_theta1_range[-1], 
                cos_theta2_range[0], cos_theta2_range[-1]],
        origin='lower',
        aspect='auto'
    )
    _add_success_region_contour(axs[3, 1], cos_theta1_range, cos_theta2_range)
    axs[3, 1].set_xlabel("cos(θ₁)")
    axs[3, 1].set_ylabel("cos(θ₂)")
    axs[3, 1].set_title("Max - Min R(s, a)")
    plt.colorbar(im_range, ax=axs[3, 1])

    return fig


def _vis_state_action_nextstate_reward(
    all_feats_torch: torch.Tensor,
    est_q_vals_resh: np.ndarray,
    resolution: int,
    num_data: int,
    batch_size: int,
    fb_model: MultiFeedbackTypeModel,
    num_actions: int,
    device: torch.device,
    cos_theta1_range: np.ndarray,
    cos_theta2_range: np.ndarray
):
    """Visualize state-action-nextstate reward R(s, a, s').
    
    Since next_state is continuous, we use the current state as next_state 
    (self-transition approximation) to visualize the reward landscape.
    """
    action_names = ["torque -1", "torque 0", "torque +1"]
    
    # Create figure: 2 rows for Q-values, 2 rows for rewards
    fig, axs = plt.subplots(nrows=4, ncols=2, constrained_layout=True, figsize=(12, 16))
    
    # Plot Q-values (first 2 rows)
    for a in range(num_actions):
        row, col = divmod(a, 2)
        im = axs[row, col].imshow(
            est_q_vals_resh[..., a],
            extent=[cos_theta1_range[0], cos_theta1_range[-1], 
                    cos_theta2_range[0], cos_theta2_range[-1]],
            origin='lower',
            aspect='auto'
        )
        _add_success_region_contour(axs[row, col], cos_theta1_range, cos_theta2_range)
        axs[row, col].set_xlabel("cos(θ₁)")
        axs[row, col].set_ylabel("cos(θ₂)")
        axs[row, col].set_title(f"Q(s, {action_names[a]})")
        plt.colorbar(im, ax=axs[row, col])

    # Predict rewards for each action (using s' = s as approximation)
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

    # Reshape and compute mean/std across actions for visualization
    est_rewards_resh = np.reshape(est_rewards_all, (num_actions, resolution, resolution))
    est_std_resh = np.reshape(est_std_all, (num_actions, resolution, resolution))
    
    # Show mean reward across all actions and average std
    mean_reward = np.mean(est_rewards_resh, axis=0)
    mean_std = np.mean(est_std_resh, axis=0)
    
    im_r = axs[2, 0].imshow(
        mean_reward,
        extent=[cos_theta1_range[0], cos_theta1_range[-1], 
                cos_theta2_range[0], cos_theta2_range[-1]],
        origin='lower',
        aspect='auto'
    )
    _add_success_region_contour(axs[2, 0], cos_theta1_range, cos_theta2_range)
    axs[2, 0].set_xlabel("cos(θ₁)")
    axs[2, 0].set_ylabel("cos(θ₂)")
    axs[2, 0].set_title("Mean R(s, a, s') across actions\n(s' = s)")
    plt.colorbar(im_r, ax=axs[2, 0])
    
    im_std = axs[2, 1].imshow(
        mean_std,
        extent=[cos_theta1_range[0], cos_theta1_range[-1], 
                cos_theta2_range[0], cos_theta2_range[-1]],
        origin='lower',
        aspect='auto'
    )
    _add_success_region_contour(axs[2, 1], cos_theta1_range, cos_theta2_range)
    axs[2, 1].set_xlabel("cos(θ₁)")
    axs[2, 1].set_ylabel("cos(θ₂)")
    axs[2, 1].set_title("Mean std(s, a, s') across actions\n(s' = s)")
    plt.colorbar(im_std, ax=axs[2, 1])
    
    # Show max reward action and reward range
    max_reward_action = np.argmax(est_rewards_resh, axis=0)
    reward_range = np.max(est_rewards_resh, axis=0) - np.min(est_rewards_resh, axis=0)
    
    im_max = axs[3, 0].imshow(
        max_reward_action,
        extent=[cos_theta1_range[0], cos_theta1_range[-1], 
                cos_theta2_range[0], cos_theta2_range[-1]],
        origin='lower',
        aspect='auto',
        cmap='tab10',
        vmin=0,
        vmax=num_actions-1
    )
    _add_success_region_contour(axs[3, 0], cos_theta1_range, cos_theta2_range)
    axs[3, 0].set_xlabel("cos(θ₁)")
    axs[3, 0].set_ylabel("cos(θ₂)")
    axs[3, 0].set_title("Argmax_a R(s, a, s')\n(s' = s)")
    plt.colorbar(im_max, ax=axs[3, 0])
    
    im_range = axs[3, 1].imshow(
        reward_range,
        extent=[cos_theta1_range[0], cos_theta1_range[-1], 
                cos_theta2_range[0], cos_theta2_range[-1]],
        origin='lower',
        aspect='auto'
    )
    _add_success_region_contour(axs[3, 1], cos_theta1_range, cos_theta2_range)
    axs[3, 1].set_xlabel("cos(θ₁)")
    axs[3, 1].set_ylabel("cos(θ₂)")
    axs[3, 1].set_title("Max - Min R(s, a, s')\n(s' = s)")
    plt.colorbar(im_range, ax=axs[3, 1])

    return fig

