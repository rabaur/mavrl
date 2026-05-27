import numpy as np
from numpy.typing import NDArray
from mavrl.envs.grid_env.actions import Action, ACTION_DELTA
from mavrl.utils.reward import Rs_to_Rsas, shape

def convert_to_shift_and_axis(action_delta: NDArray):
    axis = np.argmax(np.abs(action_delta))
    shift = -action_delta[axis]
    return axis, shift


def succ_state(i: int, j: int, a: Action, grid_size: int):
    """Implements boundary checks and returns the successor state for a deterministic action."""
    s_diff = ACTION_DELTA[a]
    i_new = i + s_diff[0]
    j_new = j + s_diff[1]
    if i_new < 0 or i_new >= grid_size or j_new < 0 or j_new >= grid_size:
        return i, j
    return i_new, j_new

def to_flat_idx(i: int, j: int, grid_size):
    return i * grid_size + j

def reward_sparse(grid_size: int, goal_position: tuple[int, int], goal_val: float):
    R1d = np.full((grid_size, grid_size), -0.1)
    R1d[goal_position[0], goal_position[1]] = goal_val
    R1d = np.reshape(R1d, (grid_size**2))
    R3d = Rs_to_Rsas(R1d, 5)
    return R3d

def reward_dense(grid_size: int, gamma: float):

    # Base reward
    R1d = np.zeros((grid_size, grid_size))
    R1d[-1, -1] = 1.0
    R1d = R1d.reshape(grid_size**2)
    R3d_base = Rs_to_Rsas(R1d, 5)

    # State-wise scalar
    ii, jj = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='ij')
    f = np.abs(ii) + np.abs(jj)
    f = -np.rot90(np.rot90(f))
    f = f.reshape(grid_size**2)

    # Shaping
    return shape(R3d_base, f, gamma)

def reward_path(
    grid_size: int,
    base_val: float,
    top_path_val: float,
    bottom_path_val: float,
    goal_val: float):
    R1d = np.full((grid_size, grid_size), base_val)
    R1d[0, 1:] = top_path_val
    R1d[-1,:-1] = bottom_path_val
    R1d[-1, -1] = goal_val
    R1d = R1d.reshape(grid_size**2)
    return Rs_to_Rsas(R1d, 5)


def reward_trap(grid_size: int, goal_val: float = 1.0, trap_val: float = -4.0):
    """
    Creates a grid with contiguous block traps blocking direct paths to the goal.
    
    Layout example for 16x16 grid (traps are ~3x3 blocks):
        S . . . . . . . . . . . . . . .      S=start (0,0)
        . . . . . . . . . . . . . . . .      G=goal (15,15)
        . . . . . . . . . . . . . . . .      T=trap block (-4)
        . . . T T T . . . . . T T T . .      .=empty (0)
        . . . T T T . . . . . T T T . .
        . . . T T T . . . . . T T T . .
        . . . . . . . . . . . . . . . .
        ...
    
    Trap block size scales with grid: ~grid_size/5, minimum 1.
    """
    R1d = np.zeros((grid_size, grid_size))
    
    # Goal at bottom-right
    R1d[-1, -1] = goal_val
    
    # Trap block size scales with grid (e.g., 1 for 5x5, 3 for 16x16)
    trap_size = max(1, grid_size // 5)
    
    # Center positions for trap blocks at ~25% and ~75% of grid
    trap_centers = [grid_size // 4 + 1, 3 * grid_size // 4]
    
    for center_row in trap_centers:
        for center_col in trap_centers:
            # Place a block of traps centered at (center_row, center_col)
            half = trap_size // 2
            for dr in range(-half, trap_size - half):
                for dc in range(-half, trap_size - half):
                    row = center_row + dr
                    col = center_col + dc
                    # Bounds check and avoid start/goal
                    if 0 <= row < grid_size and 0 <= col < grid_size:
                        if (row, col) != (0, 0) and (row, col) != (grid_size - 1, grid_size - 1):
                            R1d[row, col] = trap_val
    
    R1d = R1d.reshape(grid_size ** 2)
    return Rs_to_Rsas(R1d, 5)

def reward_factory(grid_size: int, reward_type: str, gamma: float):
    """
    Creates reward tables for the grid environment of different types.
    
    Returns:
        R (np.ndarray): The ground-truth reward function, s.t. R[s, a] is the reward for state s under action a.
    """
    n_S = grid_size ** 2
    n_A = 5
    R = np.zeros((n_S, n_A, n_S))
    if reward_type == "sparse":
        R = reward_sparse(grid_size, goal_position=(grid_size - 1, grid_size - 1), goal_val=1.0)
    elif reward_type == "dense":
        R = reward_dense(grid_size, gamma)
    elif reward_type == "center":
        R = reward_sparse(grid_size, goal_position=(grid_size // 2, grid_size // 2), goal_val=1.0)
    elif reward_type == "penalty":
        R = reward_sparse(grid_size, goal_position=(grid_size - 2, grid_size - 2), goal_val=-1)
    elif reward_type == "path":
        R = reward_path(grid_size, 0, -1, -1, 4)
    elif reward_type == "cliff":
        R = reward_path(grid_size, 0, -1, -4, 4)
    elif reward_type == "trap":
        R = reward_trap(grid_size, goal_val=1.0, trap_val=-4.0)
    else:
        raise NotImplementedError(f"Reward type {reward_type} is not implemented")
    return R