import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mavrl.envs.grid_env.env import GridEnv
from mavrl.utils.math import log_var_to_std
from mavrl.envs.grid_env.actions import Action
from mavrl.utils.torch_utils import to_numpy
from mavrl.utils.feature_transforms import get_feature_combinations
from mavrl.multi_fb_model import MultiFeedbackTypeModel
from mavrl.types import FeedbackType, DataKey
from mavrl.visualization.hinton import draw_hinton_diagram


HINTON_LEGEND_TEXT = (
    "\u25a1 size = certainty (1 - normalized variance), "
    "\u25a1 color = mean value"
)
HINTON_CMAP_MEAN = "coolwarm"
HINTON_CMAP_VAR = "Greys"
GT_CMAP = "coolwarm"


# Action symbols for visualization
ACTION_SYMBOLS = {
    Action.RIGHT: "→",
    Action.UP:    "↑",
    Action.LEFT:  "←",
    Action.DOWN:  "↓",
    Action.STAY:  "⊙",
}


def vis_grid_env(
    env: GridEnv,
    fb_model: MultiFeedbackTypeModel
):
    """
    Visualizes the estimated rewards for a grid environment.
    
    Returns:
        fig: matplotlib figure object that can be logged to wandb
    """
    encoder = fb_model.encoder
    grid_size = env.grid_size
    gt_rewards = np.reshape(env._R, (grid_size, grid_size, -1))
    gt_rewards = np.max(gt_rewards, axis=-1)
    reward_domain = encoder.features.reward_domain
    if reward_domain == 'sas':
        raise NotImplementedError("Visualization of s,a,s' rewards is not implemented")
    
    # Construct one-hot features for all states and actions
    n_states = grid_size * grid_size
    n_actions = env.action_space.n
    device = next(encoder.parameters()).device
    
    # One-hot state features: identity matrix of shape (n_states, n_states)
    all_obs_features = torch.eye(n_states, device=device)
    # One-hot action features: identity matrix of shape (n_actions, n_actions)
    all_act_features = torch.eye(n_actions, device=device)
    
    # Construct all state-action-next_state features to compute the estimated reward matrix
    batch_state_features, batch_action_features, batch_next_state_features = get_feature_combinations(reward_domain, all_obs_features, all_act_features)

    # Predict mean and logvar
    mean, log_var = encoder(batch_state_features, batch_action_features, batch_next_state_features)
    
    # Compute Q-values for all states
    with torch.no_grad():
        q_values = fb_model.q_model(all_obs_features)  # Shape: (n_states, n_actions)
    q_values = to_numpy(q_values)  # Shape: (n_states, n_actions)

    if reward_domain == 's':
        # Create figure with 2 rows:
        # Row 0: ground truth, Hinton diagram (mean+variance combined)
        # Row 1: Q-values for each action
        num_actions = env.action_space.n
        ncols = max(2, num_actions)
        fig, axs = plt.subplots(
            nrows=2,
            ncols=ncols,
            figsize=(5 * ncols, 10)
        )

        mean = to_numpy(mean).squeeze()
        std = to_numpy(log_var_to_std(log_var)).squeeze()
        mean_grid = mean.reshape(grid_size, grid_size)
        std_grid = std.reshape(grid_size, grid_size)

        vmin_gt, vmax_gt = np.min(gt_rewards), np.max(gt_rewards)
        vmin_mean, vmax_mean = np.min(mean), np.max(mean)
        vmin_std, vmax_std = np.min(std), np.max(std)
        vmin_q, vmax_q = np.min(q_values), np.max(q_values)

        im_gt = axs[0, 0].imshow(gt_rewards, vmin=vmin_gt, vmax=vmax_gt, cmap=GT_CMAP)
        axs[0, 0].set_title("Ground Truth", fontsize=14)
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        plt.colorbar(im_gt, ax=axs[0, 0], fraction=0.046, pad=0.04)

        sm_mean, sm_var = draw_hinton_diagram(
            axs[0, 1],
            mean_grid,
            std_grid,
            vmin_mean=vmin_mean,
            vmax_mean=vmax_mean,
            vmin_var=vmin_std,
            vmax_var=vmax_std,
            cmap_mean=HINTON_CMAP_MEAN,
            cmap_var=HINTON_CMAP_VAR,
            bg_colors=None,
        )
        axs[0, 1].set_title(r"Hinton ($\mu$, $\sigma$)", fontsize=14)
        plt.colorbar(sm_mean, ax=axs[0, 1], fraction=0.046, pad=0.04)

        for col in range(2, ncols):
            axs[0, col].axis('off')

        for action_idx in range(num_actions):
            q_grid = q_values[:, action_idx].reshape(grid_size, grid_size)
            action_enum = Action(action_idx)
            action_symbol = ACTION_SYMBOLS.get(action_enum, str(action_idx))

            im_q = axs[1, action_idx].imshow(q_grid, vmin=vmin_q, vmax=vmax_q)
            axs[1, action_idx].set_title(f"Q ({action_symbol})", fontsize=14)
            axs[1, action_idx].set_xticks([])
            axs[1, action_idx].set_yticks([])
            plt.colorbar(im_q, ax=axs[1, action_idx], fraction=0.046, pad=0.04)

        for col in range(num_actions, ncols):
            axs[1, col].axis('off')

        fig.text(0.5, 0.01, HINTON_LEGEND_TEXT, fontsize=9, ha='center', va='bottom', style='italic')
        plt.tight_layout(rect=(0, 0.03, 1, 1))
        return fig

    elif reward_domain == 'sa':
        # Create figure with 2 rows:
        # Row 0: ground truth, Hinton diagram per action (mean+variance combined)
        # Row 1: Q-values per action (column 0 is empty to align with ground truth)
        num_actions = env.action_space.n
        ncols = 1 + num_actions
        fig, axs = plt.subplots(
            nrows=2,
            ncols=ncols,
            figsize=(5 * ncols, 10)
        )

        mean = to_numpy(mean).squeeze()
        std = to_numpy(log_var_to_std(log_var)).squeeze()

        num_states = grid_size * grid_size
        mean_sa = mean.reshape(num_states, num_actions)
        std_sa = std.reshape(num_states, num_actions)

        vmin_gt, vmax_gt = np.min(gt_rewards), np.max(gt_rewards)
        vmin_mean, vmax_mean = np.min(mean), np.max(mean)
        vmin_std, vmax_std = np.min(std), np.max(std)
        vmin_q, vmax_q = np.min(q_values), np.max(q_values)

        im_gt = axs[0, 0].imshow(gt_rewards, vmin=vmin_gt, vmax=vmax_gt, cmap=GT_CMAP)
        axs[0, 0].set_title("Ground Truth", fontsize=14)
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        plt.colorbar(im_gt, ax=axs[0, 0], fraction=0.046, pad=0.04)

        axs[1, 0].axis('off')

        for action_idx in range(num_actions):
            mean_grid = mean_sa[:, action_idx].reshape(grid_size, grid_size)
            std_grid = std_sa[:, action_idx].reshape(grid_size, grid_size)

            action_enum = Action(action_idx)
            action_symbol = ACTION_SYMBOLS.get(action_enum, str(action_idx))

            sm_mean, _ = draw_hinton_diagram(
                axs[0, action_idx + 1],
                mean_grid,
                std_grid,
                vmin_mean=vmin_mean,
                vmax_mean=vmax_mean,
                vmin_var=vmin_std,
                vmax_var=vmax_std,
                cmap_mean=HINTON_CMAP_MEAN,
                cmap_var=HINTON_CMAP_VAR,
                bg_colors=None,
            )
            axs[0, action_idx + 1].set_title(
                f"Hinton ({action_symbol})", fontsize=14
            )
            plt.colorbar(sm_mean, ax=axs[0, action_idx + 1], fraction=0.046, pad=0.04)

            q_grid = q_values[:, action_idx].reshape(grid_size, grid_size)
            im_q = axs[1, action_idx + 1].imshow(q_grid, vmin=vmin_q, vmax=vmax_q)
            axs[1, action_idx + 1].set_title(f"Q ({action_symbol})", fontsize=14)
            axs[1, action_idx + 1].set_xticks([])
            axs[1, action_idx + 1].set_yticks([])
            plt.colorbar(im_q, ax=axs[1, action_idx + 1], fraction=0.046, pad=0.04)

        fig.text(0.5, 0.01, HINTON_LEGEND_TEXT, fontsize=9, ha='center', va='bottom', style='italic')
        plt.tight_layout(rect=(0, 0.03, 1, 1))
        return fig

    else:
        raise ValueError(f"Unknown reward domain: {reward_domain}")


def vis_grid_stops(
    env: GridEnv,
    datasets: dict[FeedbackType, any],
) -> plt.Figure:
    """
    Visualizes where stops occur on the grid for stop feedback datasets.
    
    Creates a heatmap showing the count of stops at each grid cell,
    plus a separate heatmap for censored trajectory endpoints.
    
    Args:
        env: The GridEnv environment (for grid_size info)
        datasets: Dictionary mapping FeedbackType to dataset objects
        
    Returns:
        fig: matplotlib figure object that can be logged to wandb
    """
    if FeedbackType.STOP not in datasets:
        raise ValueError("No stop dataset found in datasets")
    
    grid_size = env.grid_size
    n_states = grid_size * grid_size
    dataset = datasets[FeedbackType.STOP]
    
    # Get states and stop times
    states = dataset.data[DataKey.STATES]  # Shape: (num_episodes, max_len)
    stop_times = dataset.data[DataKey.STOP_TIME]  # Shape: (num_episodes,)
    valid = dataset.data[DataKey.VALID]  # Shape: (num_episodes, max_len)
    
    # Count stops at each state
    stop_counts = np.zeros(n_states)
    censored_counts = np.zeros(n_states)  # Where censored trajectories end
    
    num_stopped = 0
    num_censored = 0
    
    for ep_idx in range(states.shape[0]):
        stop_time = int(stop_times[ep_idx].item())
        
        if stop_time >= 0:
            # Stopped at this time - get the state at stop time
            state_idx = int(states[ep_idx, stop_time].item())
            if 0 <= state_idx < n_states:
                stop_counts[state_idx] += 1
            num_stopped += 1
        else:
            # Censored - find the last valid state
            valid_mask = valid[ep_idx].cpu().numpy() if hasattr(valid[ep_idx], 'cpu') else valid[ep_idx]
            last_valid_idx = np.where(valid_mask)[0][-1] if np.any(valid_mask) else 0
            state_idx = int(states[ep_idx, last_valid_idx].item())
            if 0 <= state_idx < n_states:
                censored_counts[state_idx] += 1
            num_censored += 1
    
    # Create figure with 3 subplots: ground truth, stops, and censored endpoints
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    
    # Ground truth rewards for reference
    gt_rewards = np.reshape(env._R, (grid_size, grid_size, -1))
    gt_rewards = np.max(gt_rewards, axis=-1)
    
    # Plot ground truth
    im0 = axs[0].imshow(gt_rewards)
    axs[0].set_title("Ground Truth Rewards", fontsize=14)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    
    # Plot stop locations
    stop_grid = stop_counts.reshape(grid_size, grid_size)
    im1 = axs[1].imshow(stop_grid, cmap='Reds')
    axs[1].set_title(f"Stop Locations (n={num_stopped})", fontsize=14)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    
    # Add count annotations
    for i in range(grid_size):
        for j in range(grid_size):
            count = int(stop_grid[i, j])
            if count > 0:
                text_color = 'white' if count > stop_grid.max() * 0.5 else 'black'
                axs[1].text(j, i, str(count), ha='center', va='center',
                           fontsize=max(6, 10 - grid_size // 3), color=text_color)
    
    # Plot censored endpoints
    censored_grid = censored_counts.reshape(grid_size, grid_size)
    im2 = axs[2].imshow(censored_grid, cmap='Blues')
    axs[2].set_title(f"Censored Endpoints (n={num_censored})", fontsize=14)
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    
    # Add count annotations
    for i in range(grid_size):
        for j in range(grid_size):
            count = int(censored_grid[i, j])
            if count > 0:
                text_color = 'white' if count > censored_grid.max() * 0.5 else 'black'
                axs[2].text(j, i, str(count), ha='center', va='center',
                           fontsize=max(6, 10 - grid_size // 3), color=text_color)
    
    plt.tight_layout()
    return fig


def vis_grid_occupancy(
    env: GridEnv,
    datasets: dict[FeedbackType, any],
) -> plt.Figure:
    """
    Visualizes the state occupancy of trajectories in the datasets as heatmaps.
    
    Creates one heatmap per feedback type showing how often each grid cell was visited.
    
    Args:
        env: The GridEnv environment (for grid_size info)
        datasets: Dictionary mapping FeedbackType to dataset objects
        
    Returns:
        fig: matplotlib figure object that can be logged to wandb
    """
    grid_size = env.grid_size
    n_states = grid_size * grid_size
    
    # Compute occupancy for each feedback type
    occupancies = {}
    
    for fb_type, dataset in datasets.items():
        occupancy = np.zeros(n_states)
        
        if fb_type == FeedbackType.DEMO:
            # DemonstrationDataset stores flattened transitions
            states = dataset.data[DataKey.STATES]  # Shape: (total_transitions,)
            valid = dataset.data[DataKey.VALID]    # Shape: (total_transitions,)
            for idx in range(len(states)):
                if valid[idx]:
                    state = states[idx]
                    state_idx = int(state.item() if hasattr(state, 'item') else state)
                    if 0 <= state_idx < n_states:
                        occupancy[state_idx] += 1
                        
        elif fb_type == FeedbackType.PREF:
            # PreferenceDataset stores pairs, each with shape (2, T, ...)
            states = dataset.data[DataKey.STATES]  # Shape: (num_pairs, 2, T)
            valid = dataset.data[DataKey.VALID]    # Shape: (num_pairs, 2, T)
            for pair_idx in range(states.shape[0]):
                for traj_idx in range(states.shape[1]):
                    for t in range(states.shape[2]):
                        if valid[pair_idx, traj_idx, t]:
                            state = states[pair_idx, traj_idx, t]
                            state_idx = int(state.item() if hasattr(state, 'item') else state)
                            if 0 <= state_idx < n_states:
                                occupancy[state_idx] += 1
        
        elif fb_type == FeedbackType.STOP:
            # StopDataset stores episodes with shape (num_episodes, max_len)
            states = dataset.data[DataKey.STATES]  # Shape: (num_episodes, max_len)
            valid = dataset.data[DataKey.VALID]    # Shape: (num_episodes, max_len)
            for ep_idx in range(states.shape[0]):
                for t in range(states.shape[1]):
                    if valid[ep_idx, t]:
                        state = states[ep_idx, t]
                        state_idx = int(state.item() if hasattr(state, 'item') else state)
                        if 0 <= state_idx < n_states:
                            occupancy[state_idx] += 1
        
        occupancies[fb_type] = occupancy
    
    # Create figure with one subplot per feedback type
    n_types = len(occupancies)
    fig, axs = plt.subplots(
        nrows=1,
        ncols=n_types,
        figsize=(5 * n_types, 5),
        squeeze=False
    )
    axs = axs[0]  # Flatten since we only have one row
    
    for idx, (fb_type, occupancy) in enumerate(occupancies.items()):
        occupancy_grid = occupancy.reshape(grid_size, grid_size)
        
        # Use log scale for better visualization if there's high variance
        # Add 1 to avoid log(0)
        vmin = max(1, occupancy_grid.min())
        vmax = max(vmin + 1, occupancy_grid.max())
        
        im = axs[idx].imshow(
            occupancy_grid + 1,  # +1 to avoid log(0)
            norm=LogNorm(vmin=vmin, vmax=vmax + 1),
            cmap='YlOrRd'
        )
        
        # Add text annotations showing actual counts
        for i in range(grid_size):
            for j in range(grid_size):
                count = int(occupancy_grid[i, j])
                # Choose text color based on background intensity
                text_color = 'white' if count > occupancy_grid.max() * 0.5 else 'black'
                axs[idx].text(j, i, str(count), ha='center', va='center', 
                             fontsize=max(6, 10 - grid_size // 3), color=text_color)
        
        axs[idx].set_title(f"{fb_type.value.capitalize()} Occupancy", fontsize=14)
        axs[idx].set_xticks([])
        axs[idx].set_yticks([])
        plt.colorbar(im, ax=axs[idx], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig
