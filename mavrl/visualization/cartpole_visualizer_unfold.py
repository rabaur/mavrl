import gymnasium as gym
from mavrl.multi_fb_model import MultiFeedbackTypeModel
import numpy as np
import torch
from mavrl.utils.torch_utils import to_numpy
from mavrl.visualization.unfold_tensor import unfold_tensor
import matplotlib.pyplot as plt
from vsup import VSUP


def visualize_reward_cartpole_unfold(
    env: gym.Env,
    curve_type: str,
    fb_model: MultiFeedbackTypeModel,
    resolution=64,
    vsup_mode='usl',
    vsup_palette='viridis',
    vsup_quantization='linear',
    num_actions=2
):
    """
    Visualizes learned rewards for CartPole-v1 environment with uncertainty using VSUP.
    
    Args:
        env: CartPole environment
        curve_type: Type of space-filling curve ('hilbert' or 'morton')
        fb_model: Trained feedback model
        resolution: Resolution of the evaluation grid
        vsup_mode: VSUP mode - 'usl' (uncertainty to saturation+lightness), 
                   'us' (uncertainty to saturation), 'ul' (uncertainty to lightness)
        vsup_palette: Color palette for VSUP (e.g., 'viridis', 'flare', 'crest')
        vsup_quantization: Quantization method ('linear', 'square', or 'tree')
        num_actions: Number of actions in the environment
    """

    # create evaluation grid
    obs_space = env.observation_space
    ranges = [(low, high) for low, high in zip(obs_space.low, obs_space.high)]
    ranges = [(l, h) if l != float('-inf') else (-3.0, 3.0) for l, h in ranges]
    eval_grid = np.meshgrid(*[np.linspace(low, high, resolution) for low, high in ranges], indexing='ij')
    eval_grid_stacked = np.stack(eval_grid, axis=-1)
    eval_grid_flat = np.reshape(eval_grid_stacked, (-1, eval_grid_stacked.shape[-1]))
    eval_grid_flat_torch = torch.tensor(eval_grid_flat, dtype=torch.float32).to(next(iter(fb_model.parameters())).device)
    
    # Detect reward domain from the feature module
    reward_domain = fb_model.encoder.features.reward_domain
    
    # Get device
    device = next(iter(fb_model.parameters())).device
    n_states = eval_grid_flat_torch.shape[0]
    
    # Initialize VSUP
    vsup = VSUP(palette=vsup_palette, mode=vsup_mode, quantization=vsup_quantization)
    
    if reward_domain == 's':
        # State-only reward: create 1x2 plot
        with torch.no_grad():
            mean, log_var = fb_model.encoder(eval_grid_flat_torch, None, None)
        mean = to_numpy(mean)
        log_var = to_numpy(log_var)
        
        # Convert log variance to standard deviation for uncertainty
        uncertainty = np.sqrt(np.exp(log_var))
        
        # Reshape back to the original shape
        mean = mean.reshape(eval_grid_stacked.shape[:-1])
        uncertainty = uncertainty.reshape(eval_grid_stacked.shape[:-1])
        
        # Unfold mean and uncertainty
        mean_vel = unfold_tensor(mean, curve_type, [0, 2], 1)
        mean_angvel = unfold_tensor(mean, curve_type, [0, 2], 3)
        uncertainty_vel = unfold_tensor(uncertainty, curve_type, [0, 2], 1)
        uncertainty_angvel = unfold_tensor(uncertainty, curve_type, [0, 2], 3)
        
        # Apply VSUP color mapping
        colors_vel = vsup(mean_vel, uncertainty_vel)
        colors_angvel = vsup(mean_angvel, uncertainty_angvel)
        
        # Create visualization
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        
        axs[0].imshow(colors_vel, aspect='auto', origin='lower')
        axs[0].set_title(f'Pos-Angle vs Vel (VSUP - {vsup_mode})')
        axs[0].set_xlabel('Velocity dimension')
        axs[0].set_ylabel('Pos-Angle (space-filling curve)')
        
        axs[1].imshow(colors_angvel, aspect='auto', origin='lower')
        axs[1].set_title(f'Pos-Angle vs AngVel (VSUP - {vsup_mode})')
        axs[1].set_xlabel('Angular Velocity dimension')
        axs[1].set_ylabel('Pos-Angle (space-filling curve)')
        
    elif reward_domain in ['sa', 'sas']:
        # State-action reward: create 2x2 plot (2 views × 2 actions)
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        for action_idx in range(num_actions):
            # Create action tensor (one-hot encoded)
            actions = torch.zeros(n_states, num_actions, device=device)
            actions[:, action_idx] = 1.0
            
            with torch.no_grad():
                if reward_domain == 'sa':
                    mean, log_var = fb_model.encoder(eval_grid_flat_torch, actions, None)
                else:  # 'sas'
                    # For visualization, use the same state as next state
                    mean, log_var = fb_model.encoder(eval_grid_flat_torch, actions, eval_grid_flat_torch)
            
            mean = to_numpy(mean)
            log_var = to_numpy(log_var)
            
            # Convert log variance to standard deviation for uncertainty
            uncertainty = np.sqrt(np.exp(log_var))
            
            # Reshape back to the original shape
            mean = mean.reshape(eval_grid_stacked.shape[:-1])
            uncertainty = uncertainty.reshape(eval_grid_stacked.shape[:-1])
            
            # Unfold mean and uncertainty
            mean_vel = unfold_tensor(mean, curve_type, [0, 2], 1)
            mean_angvel = unfold_tensor(mean, curve_type, [0, 2], 3)
            uncertainty_vel = unfold_tensor(uncertainty, curve_type, [0, 2], 1)
            uncertainty_angvel = unfold_tensor(uncertainty, curve_type, [0, 2], 3)
            
            # Apply VSUP color mapping
            colors_vel = vsup(mean_vel, uncertainty_vel)
            colors_angvel = vsup(mean_angvel, uncertainty_angvel)
            
            # Plot: rows are actions, columns are views
            axs[action_idx, 0].imshow(colors_vel, aspect='auto', origin='lower')
            axs[action_idx, 0].set_title(f'Action {action_idx}: Pos-Angle vs Vel')
            axs[action_idx, 0].set_xlabel('Velocity dimension')
            axs[action_idx, 0].set_ylabel('Pos-Angle (space-filling curve)')
            
            axs[action_idx, 1].imshow(colors_angvel, aspect='auto', origin='lower')
            axs[action_idx, 1].set_title(f'Action {action_idx}: Pos-Angle vs AngVel')
            axs[action_idx, 1].set_xlabel('Angular Velocity dimension')
            axs[action_idx, 1].set_ylabel('Pos-Angle (space-filling curve)')
    
    else:
        raise ValueError(f"Unknown reward_domain: {reward_domain}")
    
    plt.tight_layout()
    return fig
