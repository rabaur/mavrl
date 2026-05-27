import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from mavrl.multi_fb_model import MultiFeedbackTypeModel
from mavrl.utils.math import log_var_to_std
from mavrl.utils.torch_utils import get_model_device, to_numpy
from tqdm import tqdm
from mavrl.data.demonstration_dataset import DemonstrationDataset
from torch.utils.data import DataLoader
from mavrl.types import DataKey


def vis_lunarlander(
    env: gym.Env,
    fb_model: MultiFeedbackTypeModel,
    resolution: int = 64,
    batch_size: int = 1024
):
    obs_space = env.observation_space
    action_space = env.action_space
    num_actions = action_space.n
    
    x_range = np.linspace(obs_space.low[0], obs_space.high[0], resolution)
    y_range = np.linspace(obs_space.low[1], obs_space.high[1], resolution)
    xs, ys = np.meshgrid(x_range, y_range)
    xys = np.stack([xs, ys], axis=-1)
    xys_flat = np.reshape(xys, (-1, 2))
    num_data = xys_flat.shape[0]
    other_feats = np.full((num_data, 6), 0.0)
    all_feats = np.concatenate([xys_flat, other_feats], axis=1)
    model_device = get_model_device(fb_model)
    all_feats_torch = torch.tensor(all_feats, device=model_device, dtype=torch.float32)

    # Predict Q-values
    est_q_vals = np.empty((num_data, num_actions))
    for i in tqdm(range(0, num_data, batch_size), desc="Visualizing LunarLander"):
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
            batch_size, fb_model, num_actions
        )
    elif reward_domain == 'sa':
        fig = _vis_state_action_reward(
            all_feats_torch, est_q_vals_resh, resolution, num_data, 
            batch_size, fb_model, num_actions, model_device
        )
    elif reward_domain == 'sas':
        fig = _vis_state_action_nextstate_reward(
            all_feats_torch, est_q_vals_resh, resolution, num_data, 
            batch_size, fb_model, num_actions, model_device
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
    num_actions: int
):
    """Visualize state-only reward R(s)."""
    # Create figure: 2 rows for Q-values, 1 row for rewards
    fig, axs = plt.subplots(nrows=3, ncols=2, constrained_layout=True)
    
    # Plot Q-values
    for a in range(min(num_actions, 4)):
        row, col = divmod(a, 2)
        im = axs[row, col].imshow(est_q_vals_resh[..., a])
        plt.colorbar(im, ax=axs[row, col])
    
    # Set Q-value titles
    action_names = ["do nothing", "fire left", "fire main", "fire right"]
    for a in range(min(num_actions, 4)):
        row, col = divmod(a, 2)
        axs[row, col].set_title(f"Q(s, {action_names[a]})")

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
    
    im_r = axs[2, 0].imshow(est_rewards_resh)
    im_std = axs[2, 1].imshow(est_std_resh)
    axs[2, 0].set_title("R(s)")
    axs[2, 1].set_title("std(s)")
    plt.colorbar(im_r, ax=axs[2, 0])
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
    device: torch.device
):
    """Visualize state-action reward R(s, a)."""
    action_names = ["do nothing", "fire left", "fire main", "fire right"]
    
    # Create figure: 2 rows for Q-values, 2 rows for rewards (mean + std per action)
    fig, axs = plt.subplots(nrows=4, ncols=2, constrained_layout=True, figsize=(10, 14))
    
    # Plot Q-values (first 2 rows)
    for a in range(min(num_actions, 4)):
        row, col = divmod(a, 2)
        im = axs[row, col].imshow(est_q_vals_resh[..., a])
        axs[row, col].set_title(f"Q(s, {action_names[a]})")
        plt.colorbar(im, ax=axs[row, col])

    # Predict rewards for each action
    est_rewards_all = np.empty((num_data, num_actions))
    est_std_all = np.empty((num_data, num_actions))
    
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
            est_rewards_all[i:i+batch_size, a] = r_batch
            est_std_all[i:i+batch_size, a] = std_batch

    # Reshape and compute mean/std across actions for visualization
    est_rewards_resh = np.reshape(est_rewards_all, (resolution, resolution, num_actions))
    est_std_resh = np.reshape(est_std_all, (resolution, resolution, num_actions))
    
    # Show mean reward across all actions and average std
    mean_reward = np.mean(est_rewards_resh, axis=-1)
    mean_std = np.mean(est_std_resh, axis=-1)
    
    im_r = axs[2, 0].imshow(mean_reward)
    axs[2, 0].set_title("Mean R(s, a) across actions")
    plt.colorbar(im_r, ax=axs[2, 0])
    
    im_std = axs[2, 1].imshow(mean_std)
    axs[2, 1].set_title("Mean std(s, a) across actions")
    plt.colorbar(im_std, ax=axs[2, 1])
    
    # Show max reward action and reward range
    r_rgb = est_rewards_resh[..., 1:]
    # normalize
    r_rgb = (r_rgb - np.min(r_rgb)) / (np.max(r_rgb) - np.min(r_rgb))
    reward_range = np.max(est_rewards_resh, axis=-1) - np.min(est_rewards_resh, axis=-1)
    
    im_rgb = axs[3, 0].imshow(r_rgb)
    axs[3, 0].set_title("R(s, a)")
    plt.colorbar(im_rgb, ax=axs[3, 0])
    
    im_range = axs[3, 1].imshow(reward_range)
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
    device: torch.device
):
    """Visualize state-action-nextstate reward R(s, a, s').
    
    Since next_state is continuous, we use the current state as next_state 
    (self-transition approximation) to visualize the reward landscape.
    """
    action_names = ["do nothing", "fire left", "fire main", "fire right"]
    
    # Create figure: 2 rows for Q-values, 2 rows for rewards
    fig, axs = plt.subplots(nrows=4, ncols=2, constrained_layout=True, figsize=(10, 14))
    
    # Plot Q-values (first 2 rows)
    for a in range(min(num_actions, 4)):
        row, col = divmod(a, 2)
        im = axs[row, col].imshow(est_q_vals_resh[..., a])
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
    est_rewards_resh = np.reshape(est_rewards_all, (resolution, resolution, num_actions))
    est_std_resh = np.reshape(est_std_all, (resolution, resolution, num_actions))
    
    # Show mean reward across all actions and average std
    mean_reward = np.mean(est_rewards_resh, axis=-1)
    mean_std = np.mean(est_std_resh, axis=-1)
    
    im_r = axs[2, 0].imshow(mean_reward)
    axs[2, 0].set_title("Mean R(s, a, s') across actions\n(s' = s)")
    plt.colorbar(im_r, ax=axs[2, 0])
    
    im_std = axs[2, 1].imshow(mean_std)
    axs[2, 1].set_title("Mean std(s, a, s') across actions\n(s' = s)")
    plt.colorbar(im_std, ax=axs[2, 1])
    
    # Show max reward action and reward range
    r_rgb = est_rewards_resh[..., 1:]
    # normalize
    r_rgb = (r_rgb - np.min(r_rgb)) / (np.max(r_rgb) - np.min(r_rgb))
    reward_range = np.max(est_rewards_resh, axis=0) - np.min(est_rewards_resh, axis=0)
    
    im_rgb = axs[3, 0].imshow(r_rgb)
    axs[3, 0].set_title("R(s, a, s')\n(s' = s)")
    plt.colorbar(im_rgb, ax=axs[3, 0])
    
    im_range = axs[3, 1].imshow(reward_range)
    axs[3, 1].set_title("Max - Min R(s, a, s')\n(s' = s)")
    plt.colorbar(im_range, ax=axs[3, 1])

    return fig


def visualize_dataset(dataset: DemonstrationDataset):
    """Visualize a dataset of LunarLander trajectories."""
    loader = DataLoader(dataset, batch_size=1024)
    resolution = 128
    y_bins = np.linspace(-0.5, 1.5, resolution)
    x_bins = np.linspace(-2.0, 2.0, resolution)
    img_array = np.zeros((resolution, resolution, 3))
    for batch in tqdm(loader, desc="Visualizing dataset"):
        obs = batch[DataKey.OBS]
        xs = to_numpy(obs[:, 0]).flatten()
        ys = to_numpy(obs[:, 1]).flatten()
        x_idxs, y_idxs = np.digitize(xs, x_bins) - 1, np.digitize(ys, y_bins) - 1
        acts = to_numpy(batch[DataKey.ACTS].long()).flatten() - 1
        acts[acts < 0] = 0
        img_array[y_idxs, x_idxs, acts] += 1
    
    # Log counts
    img_array = np.log(img_array + 1)
    # Normalize
    img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
    plt.imshow(img_array, origin='lower')
    plt.show()