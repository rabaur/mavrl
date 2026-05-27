"""Shared Hinton-diagram rendering helpers.

These utilities are used both for offline checkpoint comparison
(`scripts/visualize_checkpoint_comparison.py`) and for the in-training grid
visualization (`mavrl.visualization.grid_visualizer`).
"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


# MiniGrid-style colors (RGB normalized to 0-1)
MINIGRID_COLORS = {
    'floor': (0.16, 0.16, 0.16),      # Dark grey/black floor tiles
    'lava': (1.0, 0.55, 0.0),          # Orange lava (#FF8C00)
    'goal': (0.0, 0.8, 0.0),           # Green goal
    'border': (0.5, 0.5, 0.5),         # Grey border
}


def get_minigrid_background_colors(gt_rewards: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Generate MiniGrid-style background colors based on ground truth rewards.

    Args:
        gt_rewards: 2D array of ground truth reward values (grid_size x grid_size)
        alpha: Opacity for the background (1.0 = full opacity, lower = more transparent)

    Returns:
        RGBA array of shape (grid_size, grid_size, 4) with background colors
    """
    rows, cols = gt_rewards.shape
    colors = np.zeros((rows, cols, 4))

    max_reward = np.max(gt_rewards)

    for i in range(rows):
        for j in range(cols):
            reward = gt_rewards[i, j]

            if reward == max_reward and max_reward > 0:
                rgb = MINIGRID_COLORS['goal']
            elif reward < 0:
                rgb = MINIGRID_COLORS['lava']
            else:
                rgb = MINIGRID_COLORS['floor']

            colors[i, j, :3] = rgb
            colors[i, j, 3] = alpha

    return colors


def draw_minigrid_background(
    ax: plt.Axes,
    bg_colors: np.ndarray,
    draw_border: bool = False,
) -> None:
    """
    Draw MiniGrid-style background tiles.

    Args:
        ax: Matplotlib axes to draw on
        bg_colors: RGBA array of shape (rows, cols, 4) with background colors
        draw_border: Whether to draw a grey border around the grid
    """
    rows, cols = bg_colors.shape[:2]

    for y in range(rows):
        for x in range(cols):
            color = bg_colors[y, x]
            rect = plt.Rectangle(
                [x, y], 1, 1,
                facecolor=color,
                edgecolor="#6E6E6E",
                linewidth=0.3
            )
            ax.add_patch(rect)


def draw_hinton_diagram(
    ax: plt.Axes,
    mean_matrix: np.ndarray,
    var_matrix: np.ndarray,
    vmin_mean: float,
    vmax_mean: float,
    vmin_var: float,
    vmax_var: float,
    cmap_mean: str = 'Reds',
    cmap_var: str = 'Blues',
    max_square_size: float = 0.9,
    bg_colors: Optional[np.ndarray] = None,
    draw_border: bool = True,
) -> tuple[plt.cm.ScalarMappable, plt.cm.ScalarMappable]:
    """
    Draw a Hinton diagram encoding mean and variance.

    - Inner square size: proportional to certainty, a.k.a. |inverser variance| (larger = lower variance)
    - Inner square color: mean value (diverging colormap)
    - Background color: MiniGrid-style tiles (if provided) or variance value

    Args:
        ax: Matplotlib axes to draw on
        mean_matrix: 2D array of mean values
        var_matrix: 2D array of variance values
        vmin_mean, vmax_mean: Color scale limits for mean
        vmin_var, vmax_var: Color scale limits for variance
        cmap_mean: Colormap for mean (inner squares)
        cmap_var: Colormap for variance (background)
        max_square_size: Maximum size of inner square relative to cell (0-1)
        bg_colors: Optional RGBA array for MiniGrid-style background (rows, cols, 4)
        draw_border: Whether to draw a grey border around the grid

    Returns:
        Tuple of ScalarMappable objects for mean and variance colorbars
    """
    rows, cols = mean_matrix.shape

    # Use symmetric normalization for mean to keep midpoint at center
    max_abs_mean = max(abs(vmin_mean), abs(vmax_mean))
    norm_mean = plt.Normalize(-max_abs_mean, max_abs_mean)
    norm_var = plt.Normalize(vmin_var, vmax_var)

    sm_mean = plt.cm.ScalarMappable(cmap=cmap_mean, norm=norm_mean)
    sm_var = plt.cm.ScalarMappable(cmap=cmap_var, norm=norm_var)

    ax.set_aspect('equal')
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_facecolor('#2a2a2a')

    if bg_colors is not None:
        draw_minigrid_background(ax, bg_colors, draw_border=draw_border)

    var_range = vmax_var - vmin_var

    for (y, x), mean_val in np.ndenumerate(mean_matrix):
        var_val = var_matrix[y, x]

        # 1. Draw Background (Variance)
        bg_color = sm_var.to_rgba(var_val)
        rect_bg = plt.Rectangle(
            [x, y], 1, 1,
            facecolor=bg_color,
            edgecolor='#444444',
            linewidth=0.3
        )
        #ax.add_patch(rect_bg)

        # 2. Draw Inner Square (Mean)
        # Size decreases with variance (sqrt slows decay, min_size prevents disappearing)
        if max_abs_mean > 0 and var_range > 0:
            normalized_var = (abs(var_val) - vmin_var) / var_range
            min_square_size = 0.3
            size = min_square_size + (max_square_size - min_square_size) * (1 - normalized_var)
        elif max_abs_mean > 0:
            size = max_square_size
        else:
            size = 0

        if size > 0.05:
            fg_color = sm_mean.to_rgba(mean_val)
            lower_left = [x + (1 - size) / 2, y + (1 - size) / 2]
            rect_fg = plt.Rectangle(
                lower_left, size, size,
                facecolor=fg_color,
                edgecolor='none'
            )
            ax.add_patch(rect_fg)

    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    return sm_mean, sm_var
