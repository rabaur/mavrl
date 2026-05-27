import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from mavrl.multi_fb_model import MultiFeedbackTypeModel
from mavrl.utils.math import log_var_to_std
from mavrl.utils.torch_utils import to_numpy, get_model_device
from tqdm import tqdm
from mavrl.data.demonstration_dataset import DemonstrationDataset
from mavrl.types import DataKey

def vis_lunarlander_data(
    dataset: DemonstrationDataset,
):
    obs = dataset.data[DataKey.OBS]
    xs = to_numpy(obs[:, 0]).flatten()
    ys = to_numpy(obs[:, 1]).flatten()
    x_bins = np.linspace(-2.0, 2.0, 256)
    y_bins = np.linspace(-0.5, 1.5, 256)
    x_idxs, y_idxs = np.digitize(xs, x_bins) - 1, np.digitize(ys, y_bins) - 1
    # Clip indices to valid range
    x_idxs = np.clip(x_idxs, 0, 255)
    y_idxs = np.clip(y_idxs, 0, 255)
    img_array = np.zeros((256, 256, 3))
    # Use np.add.at to properly handle duplicate indices and accumulate counts
    np.add.at(img_array, (y_idxs, x_idxs, 0), 1)
    np.add.at(img_array, (y_idxs, x_idxs, 1), 1)
    np.add.at(img_array, (y_idxs, x_idxs, 2), 1)
    img_array = np.log(img_array + 1)
    img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
    plt.imshow(img_array, origin='lower')
    plt.show()

def vis_lunarlander(
    env: gym.Env,
    fb_model: MultiFeedbackTypeModel,
    resolution: int = 64,
    batch_size: int = 1024
):
    """
    Create visualization plots showing how the learned reward function shapes behavior.
    
    Plot 1: Reward for main engine when falling (vy < 0) - shows reward as function of position
    Plot 2: Reward difference (left - right) when moving right-down (vx > 0, vy < 0)
    Plot 3: Reward difference (left - right) when moving left-down (vx < 0, vy < 0)
    """
    device = get_model_device(fb_model)
    reward_domain = fb_model.encoder.features.reward_domain
    
    # Get observation space bounds from environment
    obs_space = env.observation_space
    obs_low = obs_space.low
    obs_high = obs_space.high
    
    # Create position grid
    x_range = [-2, 2]
    y_range = [-0.5, 1.5]
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    xs_grid, ys_grid = np.meshgrid(xs, ys)
    xys_flat = np.stack([xs_grid.flatten(), ys_grid.flatten()], axis=-1)
    num_data = xys_flat.shape[0]
    
    #------------------------------------
    # Scenario 1: Moving down (y_vel < 0)
    # main engine should compensate
    # -----------------------------------
    
    # Set fixed values for falling scenario: vy < 0, other velocities/angles at 0
    vy_falling = -0.5  # Negative vertical velocity (falling)
    states_falling = np.zeros((num_data, 8))
    states_falling[:, 0] = xys_flat[:, 0]  # x position
    states_falling[:, 1] = xys_flat[:, 1]  # y position
    states_falling[:, 3] = vy_falling      # vy < 0 (falling)
    # vx, angle, angular_velocity, leg contacts all remain 0
    
    # Evaluate reward for main engine (action 2)
    main_engine_rewards = _evaluate_reward_for_action(
        fb_model, states_falling, action_idx=2, reward_domain=reward_domain,
        device=device, batch_size=batch_size
    )
    main_engine_rewards_reshaped = main_engine_rewards.reshape(resolution, resolution)
    
    #-----------------------------------------------------
    # Scenario 2: Moving right-down (x_vel > 0, y_vel < 0)
    # Show difference between left and right engine rewards
    # ----------------------------------------------------
    
    # Set fixed values for right-down movement
    vx_right = 0.5   # Positive x velocity (moving right)
    vy_down = -0.3   # Negative y velocity (falling)
    states_right_down = np.zeros((num_data, 8))
    states_right_down[:, 0] = xys_flat[:, 0]  # x position
    states_right_down[:, 1] = xys_flat[:, 1]  # y position
    states_right_down[:, 2] = vx_right        # vx > 0 (moving right)
    states_right_down[:, 3] = vy_down         # vy < 0 (falling)
    
    # Evaluate rewards for left engine (action 1) and right engine (action 3)
    left_engine_rewards = _evaluate_reward_for_action(
        fb_model, states_right_down, action_idx=1, reward_domain=reward_domain,
        device=device, batch_size=batch_size
    )
    right_engine_rewards = _evaluate_reward_for_action(
        fb_model, states_right_down, action_idx=3, reward_domain=reward_domain,
        device=device, batch_size=batch_size
    )
    
    # Compute difference: left - right (positive means left engine is preferred)
    reward_diff_right_down = left_engine_rewards - right_engine_rewards
    reward_diff_right_down_reshaped = reward_diff_right_down.reshape(resolution, resolution)
    
    #-----------------------------------------------------
    # Scenario 3: Moving left-down (x_vel < 0, y_vel < 0)
    # Show difference between left and right engine rewards
    # ----------------------------------------------------
    
    # Set fixed values for left-down movement
    vx_left = -0.5  # Negative x velocity (moving left)
    vy_down = -0.3  # Negative y velocity (falling)
    states_left_down = np.zeros((num_data, 8))
    states_left_down[:, 0] = xys_flat[:, 0]  # x position
    states_left_down[:, 1] = xys_flat[:, 1]  # y position
    states_left_down[:, 2] = vx_left         # vx < 0 (moving left)
    states_left_down[:, 3] = vy_down         # vy < 0 (falling)
    
    # Evaluate rewards for left engine (action 1) and right engine (action 3)
    left_engine_rewards_left_down = _evaluate_reward_for_action(
        fb_model, states_left_down, action_idx=1, reward_domain=reward_domain,
        device=device, batch_size=batch_size
    )
    right_engine_rewards_left_down = _evaluate_reward_for_action(
        fb_model, states_left_down, action_idx=3, reward_domain=reward_domain,
        device=device, batch_size=batch_size
    )
    
    # Compute difference: left - right (positive means left engine is preferred)
    # When moving left, we expect right engine to be preferred (negative difference)
    reward_diff_left_down = left_engine_rewards_left_down - right_engine_rewards_left_down
    reward_diff_left_down_reshaped = reward_diff_left_down.reshape(resolution, resolution)
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6), constrained_layout=True)
    
    # Plot 1: Main engine reward when falling
    im1 = ax1.imshow(
        main_engine_rewards_reshaped,
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        origin='lower',
        aspect='auto',
        cmap='viridis'
    )
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Learned Reward for Main Engine\nWhen Falling (vy < 0)')
    plt.colorbar(im1, ax=ax1, label='Reward')
    
    # Plot 2: Reward difference (left - right) when moving right-down
    vmax_diff = max(np.abs(reward_diff_right_down_reshaped).max(), 
                    np.abs(reward_diff_left_down_reshaped).max())
    im2 = ax2.imshow(
        reward_diff_right_down_reshaped,
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        origin='lower',
        aspect='auto',
        cmap='RdBu_r',
        vmin=-vmax_diff,
        vmax=vmax_diff
    )
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title('Reward Difference (Left - Right Engine)\nWhen Moving Right-Down (vx > 0, vy < 0)')
    plt.colorbar(im2, ax=ax2, label='Reward Difference')
    
    # Plot 3: Reward difference (left - right) when moving left-down
    im3 = ax3.imshow(
        reward_diff_left_down_reshaped,
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        origin='lower',
        aspect='auto',
        cmap='RdBu_r',
        vmin=-vmax_diff,
        vmax=vmax_diff
    )
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    ax3.set_title('Reward Difference (Left - Right Engine)\nWhen Moving Left-Down (vx < 0, vy < 0)')
    plt.colorbar(im3, ax=ax3, label='Reward Difference')
    
    return fig


def _evaluate_reward_for_action(
    fb_model: MultiFeedbackTypeModel,
    states: np.ndarray,
    action_idx: int,
    reward_domain: str,
    device: torch.device,
    batch_size: int = 1024
) -> np.ndarray:
    """
    Evaluate reward for a specific action across a batch of states.
    
    Args:
        fb_model: The multi-feedback model
        states: Array of shape (N, 8) with LunarLander states
        action_idx: Action index (0=do nothing, 1=left, 2=main, 3=right)
        reward_domain: Reward domain type ('s', 'sa', or 'sas')
        device: PyTorch device
        batch_size: Batch size for evaluation
        
    Returns:
        Array of shape (N,) with reward predictions
    """
    num_data = states.shape[0]
    num_actions = 4
    rewards = np.empty(num_data)
    
    states_torch = torch.tensor(states, device=device, dtype=torch.float32)
    
    with torch.no_grad():
        if reward_domain == 's':
            # State-only reward: same for all actions
            for i in tqdm(range(0, num_data, batch_size), desc=f"Evaluating action {action_idx}", leave=False):
                batch = states_torch[i:i+batch_size]
                mean, log_var = fb_model.encoder(obs=batch, acts=None, next_obs=None)
                rewards[i:i+batch_size] = to_numpy(mean.squeeze())
                
        elif reward_domain == 'sa':
            # State-action reward: need to provide action
            action_one_hot = torch.zeros((num_data, num_actions), device=device, dtype=torch.float32)
            action_one_hot[:, action_idx] = 1.0
            
            for i in tqdm(range(0, num_data, batch_size), desc=f"Evaluating action {action_idx}", leave=False):
                batch_states = states_torch[i:i+batch_size]
                batch_actions = action_one_hot[i:i+batch_size]
                mean, log_var = fb_model.encoder(obs=batch_states, acts=batch_actions, next_obs=None)
                rewards[i:i+batch_size] = to_numpy(mean.squeeze())
                
        elif reward_domain == 'sas':
            # State-action-nextstate reward: use same state as next state
            action_one_hot = torch.zeros((num_data, num_actions), device=device, dtype=torch.float32)
            action_one_hot[:, action_idx] = 1.0
            
            for i in tqdm(range(0, num_data, batch_size), desc=f"Evaluating action {action_idx}", leave=False):
                batch_states = states_torch[i:i+batch_size]
                batch_actions = action_one_hot[i:i+batch_size]
                # Use same state as next state (self-transition)
                mean, log_var = fb_model.encoder(obs=batch_states, acts=batch_actions, next_obs=batch_states)
                rewards[i:i+batch_size] = to_numpy(mean.squeeze())
        else:
            raise ValueError(f"Unknown reward_domain: {reward_domain}")
    
    return rewards