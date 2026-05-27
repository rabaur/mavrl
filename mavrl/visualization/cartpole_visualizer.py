import torch
import numpy as np
import matplotlib.pyplot as plt
from mavrl.multi_fb_model import MultiFeedbackTypeModel
from mavrl.utils.math import log_var_to_std
from mavrl.utils.torch_utils import get_model_device, to_numpy

def vis_cartpole(
    fb_model: MultiFeedbackTypeModel,
    dataloader=None,
    resolution=50,
    num_actions=2
):
    """
    Visualizes learned rewards for CartPole-v1 environment.
    
    Creates a figure with multiple 2D heatmap views of the 4D reward function.
    
    Args:
        fb_model: The multi-feedback model
        device: PyTorch device
        dataloader: Optional dataloader for trajectory visualization
        resolution: Resolution of heatmaps
        num_actions: Number of actions in the environment
    """
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Define state space bounds (from CartPole spec)
    cart_pos_range = (-2.4, 2.4)  # episode bounds
    cart_vel_range = (-3.0, 3.0)  # reasonable range
    pole_angle_range = (-0.2095, 0.2095)  # episode bounds (±12°)
    pole_vel_range = (-3.0, 3.0)  # reasonable range
    
    # 1. Cart Position vs Pole Angle (velocities = 0)
    cart_pos = np.linspace(*cart_pos_range, resolution)
    pole_angle = np.linspace(*pole_angle_range, resolution)
    CP, PA = np.meshgrid(cart_pos, pole_angle)
    states_1 = np.stack([
        CP.flatten(),
        np.zeros(resolution**2),  # cart_vel = 0
        PA.flatten(),
        np.zeros(resolution**2)   # pole_vel = 0
    ], axis=-1)
    device = get_model_device(fb_model)
    mean_1, logvar_1 = predict_rewards(fb_model, states_1, device, num_actions)
    plot_heatmap(axs[0, 0], mean_1.reshape(resolution, resolution),
                 cart_pos_range, pole_angle_range,
                 "Cart Position vs Pole Angle\n(vel=0)",
                 "Cart Position", "Pole Angle (rad)")
    
    # 2. Pole Angle vs Pole Angular Velocity (cart at center, stationary)
    pole_angle = np.linspace(*pole_angle_range, resolution)
    pole_vel = np.linspace(*pole_vel_range, resolution)
    PA, PV = np.meshgrid(pole_angle, pole_vel)
    states_2 = np.stack([
        np.zeros(resolution**2),  # cart_pos = 0
        np.zeros(resolution**2),  # cart_vel = 0
        PA.flatten(),
        PV.flatten()
    ], axis=-1)
    
    mean_2, logvar_2 = predict_rewards(fb_model, states_2, device, num_actions)
    plot_heatmap(axs[0, 1], mean_2.reshape(resolution, resolution),
                 pole_angle_range, pole_vel_range,
                 "Pole Angle vs Angular Velocity\n(cart at center)",
                 "Pole Angle (rad)", "Angular Velocity")
    
    # 3. Cart Position vs Cart Velocity (pole upright)
    cart_pos = np.linspace(*cart_pos_range, resolution)
    cart_vel = np.linspace(*cart_vel_range, resolution)
    CP, CV = np.meshgrid(cart_pos, cart_vel)
    states_3 = np.stack([
        CP.flatten(),
        CV.flatten(),
        np.zeros(resolution**2),  # pole_angle = 0
        np.zeros(resolution**2)   # pole_vel = 0
    ], axis=-1)
    
    mean_3, logvar_3 = predict_rewards(fb_model, states_3, device, num_actions)
    plot_heatmap(axs[0, 2], mean_3.reshape(resolution, resolution),
                 cart_pos_range, cart_vel_range,
                 "Cart Position vs Velocity\n(pole upright)",
                 "Cart Position", "Cart Velocity")
    
    # 4. Uncertainty (std) for view 1
    std_1 = to_numpy(log_var_to_std(torch.tensor(logvar_1)))
    plot_heatmap(axs[1, 0], std_1.reshape(resolution, resolution),
                 cart_pos_range, pole_angle_range,
                 "Uncertainty (σ)\nCart Pos vs Pole Angle",
                 "Cart Position", "Pole Angle (rad)")
    
    # 5. Trajectory visualization (if dataloader provided)
    if dataloader is not None:
        plot_trajectory_rewards(axs[1, 1], fb_model, dataloader, device, num_actions)
    else:
        axs[1, 1].text(0.5, 0.5, 'No trajectory data', 
                      ha='center', va='center', transform=axs[1, 1].transAxes)
        axs[1, 1].axis('off')
    
    # 6. Reward distribution across all slices
    plot_reward_distribution(axs[1, 2], [mean_1, mean_2, mean_3],
                           ['Pos-Angle', 'Angle-Vel', 'Pos-Vel'])
    
    plt.tight_layout()
    return fig

def predict_rewards(fb_model, states, device, num_actions=2):
    """
    Helper to predict rewards for a batch of states.
    
    For state-action reward models, averages over all actions.
    For state-only models, directly predicts from states.
    
    Args:
        fb_model: The multi-feedback model
        states: Numpy array of states
        device: PyTorch device
        num_actions: Number of actions in the environment
        
    Returns:
        Tuple of (mean, logvar) as numpy arrays
    """
    states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
    
    # Detect reward domain from the feature module
    reward_domain = fb_model.encoder.features.reward_domain
    
    with torch.no_grad():
        if reward_domain == 's':
            # State-only reward
            mean, logvar = fb_model.encoder(states_tensor, None, None)
        elif reward_domain == 'sa':
            # State-action reward: average over all actions
            n_states = states_tensor.shape[0]
            means = []
            logvars = []
            
            for action_idx in range(num_actions):
                # Create action tensor (one-hot encoded)
                actions = torch.zeros(n_states, num_actions, device=device)
                actions[:, action_idx] = 1.0
                
                m, lv = fb_model.encoder(states_tensor, actions, None)
                means.append(m)
                logvars.append(lv)
            
            # Average over actions
            mean = torch.stack(means).mean(dim=0)
            logvar = torch.stack(logvars).mean(dim=0)
        elif reward_domain == 'sas':
            # State-action-next_state reward: average over actions
            # For visualization, use the same state as next state
            n_states = states_tensor.shape[0]
            means = []
            logvars = []
            
            for action_idx in range(num_actions):
                actions = torch.zeros(n_states, num_actions, device=device)
                actions[:, action_idx] = 1.0
                
                m, lv = fb_model.encoder(states_tensor, actions, states_tensor)
                means.append(m)
                logvars.append(lv)
            
            # Average over actions
            mean = torch.stack(means).mean(dim=0)
            logvar = torch.stack(logvars).mean(dim=0)
        else:
            raise ValueError(f"Unknown reward_domain: {reward_domain}")
    
    return to_numpy(mean.squeeze()), to_numpy(logvar.squeeze())

def plot_heatmap(ax, data, xrange, yrange, title, xlabel, ylabel):
    """Helper to plot a 2D heatmap."""
    im = ax.imshow(data, origin='lower', aspect='auto',
                   extent=[xrange[0], xrange[1], yrange[0], yrange[1]])
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Add a marker at the origin/equilibrium
    if 0 >= xrange[0] and 0 <= xrange[1] and 0 >= yrange[0] and 0 <= yrange[1]:
        ax.plot(0, 0, 'r*', markersize=10, label='Equilibrium')

def plot_trajectory_rewards(ax, fb_model, dataloader, device, num_actions=2):
    """Plot sample state-action pairs colored by learned reward."""
    from mavrl.types import DataKey, FeedbackType
    
    # Extract samples from dataloader - get one batch
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        ax.text(0.5, 0.5, 'No data in dataloader', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return
    
    # Check if this is preference data (has trajectory structure) or demonstration data (transitions)
    is_preference = batch.get(DataKey.FEEDBACK_TYPE, [None])[0] == FeedbackType.PREF
    
    # Get observations (states or obs) from batch
    if DataKey.STATES in batch:
        states = batch[DataKey.STATES]
    elif DataKey.OBS in batch:
        states = batch[DataKey.OBS]
    else:
        ax.text(0.5, 0.5, 'No state/obs data', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return
    
    # Convert to numpy if needed
    if isinstance(states, torch.Tensor):
        states_np = to_numpy(states)
    else:
        states_np = states
    
    # Handle different batch structures:
    if is_preference or len(states_np.shape) == 4:
        # Preference dataset: (batch_size, 2, T, state_dim) - take first trajectory
        states_np = states_np[0, 0]  # (T, state_dim)
    elif len(states_np.shape) == 3:
        # Old trajectory-based demonstrations: (batch_size, T, state_dim) - take first batch
        states_np = states_np[0]  # (T, state_dim)
    # else: New transition-based demonstrations: (batch_size, state_dim) - use all samples
    
    # Now states_np is either (T, state_dim) for trajectories or (batch_size, state_dim) for transitions
    if len(states_np.shape) != 2:
        ax.text(0.5, 0.5, f'Unexpected state shape: {states_np.shape}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return
    
    # Predict rewards for all states
    mean, _ = predict_rewards(fb_model, states_np, device, num_actions)
    
    # Plot states in cart_pos vs pole_angle space, colored by reward
    scatter = ax.scatter(states_np[:, 0], states_np[:, 2], c=mean, s=20, cmap='viridis', alpha=0.6)
    ax.set_xlabel('Cart Position')
    ax.set_ylabel('Pole Angle (rad)')
    
    # Update title based on data type
    if is_preference or states_np.shape[0] <= 32:  # Heuristic: if few points, likely a trajectory
        ax.set_title('Sample trajectory colored by learned reward')
    else:
        ax.set_title('Sample state-action pairs colored by learned reward')
    
    plt.colorbar(scatter, ax=ax, label='Reward')
    ax.plot(0, 0, 'r*', markersize=10, label='Goal')

def plot_reward_distribution(ax, means_list, labels):
    """Plot histogram of reward values across different views."""
    for mean, label in zip(means_list, labels):
        ax.hist(mean.flatten(), bins=30, alpha=0.5, label=label)
    ax.set_xlabel('Learned Reward')
    ax.set_ylabel('Count')
    ax.set_title('Reward Distribution\n(different state slices)')
    ax.legend()