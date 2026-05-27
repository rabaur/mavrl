import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter

from mavrl.types import Trajectory, DataKey

# Action arrows for visualization
ACTION_ARROWS = {
    0: (0.3, 0),   # RIGHT
    1: (0, -0.3),  # UP
    2: (-0.3, 0),  # LEFT
    3: (0, 0.3),   # DOWN
    4: (0, 0),     # STAY
}


def generate_stop_debug_gif(
    segments: list[Trajectory],
    cumsum_regrets: list[np.ndarray],
    stop_times: np.ndarray,
    regret_ref: float,
    lambd: float,
    regret_discount: float,
    grid_size: int,
    reward_matrix: np.ndarray,
    output_path: str = "stop_debug.gif",
    num_segments: int = 5,
    fps: int = 2,
) -> None:
    """
    Generate a debug GIF showing segment trajectories with regret development.
    
    Args:
        segments: List of segment trajectories
        cumsum_regrets: Discounted cumulative regrets for each segment
        stop_times: Stop time for each segment (-1 if censored)
        regret_ref: Reference regret used for lambda calibration
        lambd: Lambda parameter for hazard computation
        regret_discount: Discount factor for old regret
        grid_size: Size of the grid environment
        reward_matrix: Reward matrix of shape (n_states, n_actions, n_states) or similar
        output_path: Path to save the GIF
        num_segments: Number of segments to visualize
        fps: Frames per second
    """
    # Select segments to visualize (mix of stopped and censored if possible)
    stopped_indices = np.where(stop_times >= 0)[0]
    censored_indices = np.where(stop_times < 0)[0]
    
    # Try to get a mix
    selected = []
    n_stopped = min(len(stopped_indices), (num_segments + 1) // 2)
    n_censored = min(len(censored_indices), num_segments - n_stopped)
    
    if n_stopped > 0:
        selected.extend(np.random.choice(stopped_indices, n_stopped, replace=False).tolist())
    if n_censored > 0:
        selected.extend(np.random.choice(censored_indices, n_censored, replace=False).tolist())
    
    if len(selected) < num_segments and len(stopped_indices) + len(censored_indices) > len(selected):
        # Fill remaining with any available
        remaining = [i for i in range(len(segments)) if i not in selected]
        selected.extend(remaining[:num_segments - len(selected)])
    
    selected = selected[:num_segments]
    
    # Compute max regret across all timesteps of selected segments
    max_regret = max(cumsum_regrets[i].max() for i in selected)
    
    # Build frame data: list of (segment_idx, timestep, is_last_frame_of_segment)
    frames = []
    for seg_idx in selected:
        seg = segments[seg_idx]
        valid = seg[DataKey.VALID]
        T = len(seg[DataKey.REWS])
        for t in range(T):
            if not valid[t]:
                break  # Stop at first invalid timestep
            is_last = (t == T - 1) or (stop_times[seg_idx] == t) or (t + 1 < T and not valid[t + 1])
            frames.append((seg_idx, t, is_last))
            if stop_times[seg_idx] == t:
                break  # Stop at the stop time
    
    # Prepare reward grid for visualization (max over actions and next states)
    # reward_matrix shape is typically (n_states, n_actions, n_states)
    if reward_matrix.ndim == 3:
        reward_grid = reward_matrix.max(axis=(1, 2)).reshape(grid_size, grid_size)
    elif reward_matrix.ndim == 2:
        reward_grid = reward_matrix.max(axis=1).reshape(grid_size, grid_size)
    else:
        reward_grid = reward_matrix.reshape(grid_size, grid_size)
    
    reward_vmin, reward_vmax = reward_grid.min(), reward_grid.max()
    
    # Create figure with twin axis for hazard
    fig, (ax_grid, ax_regret) = plt.subplots(1, 2, figsize=(12, 5))
    ax_hazard = ax_regret.twinx()
    
    def update(frame_idx):
        seg_idx, t, is_last = frames[frame_idx]
        seg = segments[seg_idx]
        cum_regret = cumsum_regrets[seg_idx]
        stop_t = stop_times[seg_idx]
        valid = seg[DataKey.VALID]
        
        # Get state and action
        state = int(seg[DataKey.STATES][t])
        action = int(seg[DataKey.ACTS][t])
        row, col = state // grid_size, state % grid_size
        
        # Clear all axes
        ax_grid.clear()
        ax_regret.clear()
        ax_hazard.clear()
        
        # --- Grid plot ---
        # Show rewards as background colors
        im = ax_grid.imshow(reward_grid, cmap='RdYlGn', vmin=reward_vmin, vmax=reward_vmax)
        
        # Draw grid lines
        for i in range(grid_size + 1):
            ax_grid.axhline(i - 0.5, color='black', linewidth=0.5)
            ax_grid.axvline(i - 0.5, color='black', linewidth=0.5)
        
        # Draw trajectory up to current time (only valid positions)
        for prev_t in range(t):
            if not valid[prev_t]:
                continue
            prev_state = int(seg[DataKey.STATES][prev_t])
            prev_row, prev_col = prev_state // grid_size, prev_state % grid_size
            ax_grid.plot(prev_col, prev_row, 'o', color='lightblue', markersize=10, alpha=0.5)
        
        # Draw current position
        if stop_t == t:
            # Stop happened here - red X
            ax_grid.plot(col, row, 'X', color='red', markersize=20, markeredgewidth=3)
            status = "STOPPED"
        else:
            # Normal position - blue circle with action arrow
            ax_grid.plot(col, row, 'o', color='blue', markersize=15)
            # Draw action arrow
            dx, dy = ACTION_ARROWS.get(action, (0, 0))
            if dx != 0 or dy != 0:
                ax_grid.arrow(col, row, dx, dy, head_width=0.15, head_length=0.1, fc='green', ec='green')
            status = "RUNNING"
        
        # Title with segment info
        is_censored = stop_t < 0
        seg_status = "Censored" if is_censored else f"Stopped at t={stop_t}"
        ax_grid.set_title(f"Segment {seg_idx + 1} | t={t} | {status}\n({seg_status})", fontsize=12)
        ax_grid.set_xlim(-0.5, grid_size - 0.5)
        ax_grid.set_ylim(grid_size - 0.5, -0.5)  # Invert y for matrix-style
        ax_grid.set_xticks([])
        ax_grid.set_yticks([])
        
        # --- Regret and Hazard plot ---
        times = np.arange(t + 1)
        
        # Left y-axis: Cumulative regret
        color_regret = 'tab:blue'
        ax_regret.set_xlabel('Timestep', fontsize=11)
        ax_regret.set_ylabel('Cumulative Regret', fontsize=11, color=color_regret)
        ax_regret.plot(times, cum_regret[:t + 1], '-', color=color_regret, linewidth=2, label='Cumulative Regret')
        ax_regret.plot(t, cum_regret[t], 'o', color=color_regret, markersize=10)
        ax_regret.tick_params(axis='y', labelcolor=color_regret)
        
        # Reference lines for regret
        ax_regret.axhline(regret_ref, color='orange', linestyle='--', linewidth=2, label=f'Regret Ref ({regret_ref:.2f})')
        ax_regret.axhline(max_regret, color='darkred', linestyle=':', linewidth=2, label=f'Max Regret ({max_regret:.2f})')
        
        ax_regret.set_xlim(-0.5, max(len(cum_regret), t + 5))
        # Align y-axes so that 0 is at the same proportional position
        # Hazard: -0.05 to 1.05 -> 0 is at 0.05/1.1 from bottom
        # Regret: use same proportion so zeros align
        regret_top = max(max_regret * 1.05, 0.1)
        regret_bottom = -regret_top / 21  # Same 1/22 proportion as hazard axis
        ax_regret.set_ylim(regret_bottom, regret_top)
        ax_regret.grid(True, alpha=0.3)
        
        # Right y-axis: Hazard (aligned with regret axis at 0)
        color_hazard = 'tab:green'
        hazard_values = 1.0 - np.exp(-lambd * cum_regret[:t + 1])
        ax_hazard.yaxis.set_label_position('right')
        ax_hazard.yaxis.tick_right()
        ax_hazard.set_ylabel('Hazard h(t)', fontsize=11, color=color_hazard)
        ax_hazard.plot(times, hazard_values, '-', color=color_hazard, linewidth=2, label='Hazard')
        ax_hazard.plot(t, hazard_values[-1], 'o', color=color_hazard, markersize=10)
        ax_hazard.tick_params(axis='y', labelcolor=color_hazard)
        ax_hazard.set_ylim(-0.05, 1.05)  # 0 at 1/22 from bottom
        
        # Hazard reference line at 0.5
        ax_hazard.axhline(0.5, color=color_hazard, linestyle='--', linewidth=1, alpha=0.5)
        
        # If stopped, mark the stop point
        if stop_t == t:
            ax_regret.axvline(t, color='red', linestyle='-', linewidth=2, alpha=0.5)
            ax_regret.plot(t, cum_regret[t], 'rX', markersize=15, markeredgewidth=2)
        
        ax_regret.set_title(f'Regret & Hazard (λ={lambd:.4f}, γ_r={regret_discount:.2f})', fontsize=12)
        
        # Combined legend
        lines1, labels1 = ax_regret.get_legend_handles_labels()
        lines2, labels2 = ax_hazard.get_legend_handles_labels()
        ax_regret.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
        
        plt.tight_layout()
        return []
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps, blit=False)
    
    # Save as GIF
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved stop debug GIF to: {output_path}")