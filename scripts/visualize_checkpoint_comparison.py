#!/usr/bin/env python
"""
Visualization script to compare reward model checkpoints across training epochs.

Uses Hinton diagrams to encode mean and variance in a single visualization:
- Inner square size: proportional to |mean| (larger = higher magnitude)
- Inner square color: mean value (diverging colormap, e.g., Reds)
- Background color: variance (sequential colormap, e.g., Blues)

Layout:
- Columns = feedback combos (runs), optionally duplicated
- Single row per feedback combo (Hinton diagram encodes both mean and variance)
- Within each cell: epochs shown as horizontal strip (training progress)
- Ground truth in first column (optional, as standard heatmap)

Supports both config file input (for per-run epoch flexibility) and CLI args.

Example usage:
    # Via config file (recommended for per-run epochs)
    python scripts/visualize_checkpoint_comparison.py --config comparison_config.json

    # Via CLI args (same epochs for all runs)
    python scripts/visualize_checkpoint_comparison.py \
        --run_dirs checkpoints/run_pref checkpoints/run_demo \
        --run_names "Preference" "Demonstration" \
        --epochs 10 50 100 \
        --env_id grid_cliff \
        --grid_size 10 \
        --duplicate_runs 2 \
        --cmap_mean Reds \
        --cmap_var Blues \
        --output comparison.pdf
"""

import argparse
import json
from pathlib import Path
from typing import Optional
from argparse import Namespace

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# Set Times New Roman font
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # Use STIX fonts for math (similar to Times)

from mavrl.envs.make_env import make_env
from mavrl.encoder.reward_encoder import RewardEncoder
from mavrl.encoder.feature_modules import MLPFeatureModule
from mavrl.utils.math import log_var_to_std
from mavrl.utils.torch_utils import to_numpy, get_device
from mavrl.utils.feature_transforms import get_feature_combinations, get_act_transform, get_obs_transform
from mavrl.utils.gym import get_obs_dim, get_act_dim
from mavrl.visualization.hinton import (
    MINIGRID_COLORS,
    draw_hinton_diagram,
    draw_minigrid_background,
    get_minigrid_background_colors,
)


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load a checkpoint file and return its contents."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location='cpu', weights_only=False)


def find_checkpoint_for_epoch(run_dir: Path, epoch: int) -> Optional[Path]:
    """Find the checkpoint file for a given epoch in a run directory."""
    for stem in ("model_epoch", "checkpoint_epoch"):
        for tmpl in (f"{stem}_{epoch:04d}.pt", f"{stem}_{epoch}.pt"):
            p = run_dir / tmpl
            if p.exists():
                return p
    return None


def reconstruct_reward_encoder(checkpoint: dict, device: torch.device) -> RewardEncoder:
    """
    Reconstruct a RewardEncoder from checkpoint data.
    
    Uses stored args to create matching architecture, then loads state dict.
    """
    args = checkpoint.get("args", {})
    
    # Create a mock args namespace
    mock_args = Namespace(**args)
    
    # Create environment to get dimensions
    env = make_env(**args)
    
    # Get transforms
    act_transform = get_act_transform(mock_args, env)
    obs_transform = get_obs_transform(mock_args, env)
    
    # Get dimensions
    obs_dim = get_obs_dim(env, obs_transform)
    act_dim = get_act_dim(env, act_transform)
    
    # Create feature module and encoder
    feature_module = MLPFeatureModule(
        obs_dim,
        act_dim,
        args.get("encoder_hidden_sizes", [256, 256, 256]),
        reward_domain=args.get("reward_domain", "s")
    )
    reward_encoder = RewardEncoder(feature_module)
    
    # Load state dict (filter to only encoder keys)
    model_state = checkpoint["model_state_dict"]
    encoder_state = {k.replace("encoder.", ""): v for k, v in model_state.items() if k.startswith("encoder.")}
    reward_encoder.load_state_dict(encoder_state)
    
    return reward_encoder.to(device)


def compute_reward_predictions(encoder: RewardEncoder, grid_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std reward predictions for all states in a grid environment.
    
    Args:
        encoder: The reward encoder model
        grid_size: Size of the grid
        device: Torch device
        
    Returns:
        mean_grid: Mean rewards reshaped as (grid_size, grid_size)
        std_grid: Std rewards reshaped as (grid_size, grid_size)
    """
    n_states = grid_size * grid_size
    reward_domain = encoder.features.reward_domain
    
    # One-hot state features
    all_obs_features = torch.eye(n_states, device=device)
    # One-hot action features (assuming 5 actions for grid env)
    n_actions = 5
    all_act_features = torch.eye(n_actions, device=device)
    
    # Get feature combinations based on reward domain
    batch_state_features, batch_action_features, batch_next_state_features = get_feature_combinations(
        reward_domain, all_obs_features, all_act_features
    )
    
    # Predict mean and logvar
    with torch.no_grad():
        mean, log_var = encoder(batch_state_features, batch_action_features, batch_next_state_features)
    
    # Convert to numpy
    mean = to_numpy(mean).squeeze()
    std = to_numpy(log_var_to_std(log_var)).squeeze()
    
    # For state-action rewards, aggregate across actions (take mean or max)
    if reward_domain == 'sa':
        mean = mean.reshape(n_states, n_actions).mean(axis=1)
        std = std.reshape(n_states, n_actions).mean(axis=1)
    elif reward_domain == 'sas':
        mean = mean.reshape(n_states, n_actions, n_states).mean(axis=(1, 2))
        std = std.reshape(n_states, n_actions, n_states).mean(axis=(1, 2))
    
    # Reshape to grid
    mean_grid = mean.reshape(grid_size, grid_size)
    std_grid = std.reshape(grid_size, grid_size)
    
    return mean_grid, std_grid


def get_ground_truth_rewards(env_id: str, grid_size: int, gamma: float = 0.99) -> np.ndarray:
    """Get ground truth rewards for the grid environment."""
    args = {
        "env_id": env_id,
        "grid_size": grid_size,
        "gamma": gamma,
        "p_rand": 0.0,
        "seed": 0,
    }
    env = make_env(**args)
    gt_rewards = np.reshape(env._R, (grid_size, grid_size, -1))
    gt_rewards = np.max(gt_rewards, axis=-1)
    return gt_rewards


def create_comparison_figure(
    runs: list[dict],
    env_id: str,
    grid_size: int,
    include_ground_truth: bool = True,
    gamma: float = 0.99,
    duplicate_runs: int = 1,
    cmap_mean: str = 'Reds',
    cmap_var: str = 'Blues',
    minigrid_background: bool = True,
    bg_alpha: float = 0.3,
) -> plt.Figure:
    """
    Create a comparison figure showing reward predictions using Hinton diagrams.

    Layout:
    - Columns = feedback combos (runs), optionally duplicated
    - Single row per feedback combo (Hinton diagram encodes both mean and variance)
    - Within each cell: epochs shown as horizontal strip
    - Ground truth shown in first column (optional, as standard heatmap)

    Hinton diagram encoding:
    - Inner square size: proportional to |mean|
    - Inner square color: mean value (diverging colormap)
    - Background: MiniGrid-style tiles (floor=dark grey, lava=orange, goal=green)

    Args:
        runs: List of run configs
        env_id: Environment ID
        grid_size: Size of the grid
        include_ground_truth: Whether to include ground truth column
        gamma: Discount factor
        duplicate_runs: Number of times to duplicate runs
        cmap_mean: Colormap for mean values
        cmap_var: Colormap for variance values
        minigrid_background: Whether to use MiniGrid-style background tiles
        bg_alpha: Opacity for background in prediction plots (0-1, ground truth uses 1.0)

    Returns:
        matplotlib Figure
    """
    device = get_device()

    # Duplicate runs to simulate additional feedback combos
    original_runs = runs
    if duplicate_runs > 1:
        runs = []
        for i in range(duplicate_runs):
            for run in original_runs:
                suffix = f" ({i+1})" if duplicate_runs > 1 else ""
                runs.append({
                    "name": run["name"] + suffix,
                    "dir": run["dir"],
                    "epochs": run["epochs"],
                    "_original_idx": original_runs.index(run),
                })
    else:
        runs = [dict(run, _original_idx=i) for i, run in enumerate(original_runs)]

    # Determine the maximum number of epochs across all runs
    max_epochs = max(len(run["epochs"]) for run in original_runs)
    n_runs = len(runs)

    # Get ground truth rewards and MiniGrid background colors
    gt_rewards = get_ground_truth_rewards(env_id, grid_size, gamma)
    vmin_gt, vmax_gt = np.min(gt_rewards), np.max(gt_rewards)

    # Generate MiniGrid-style background colors
    if minigrid_background:
        bg_colors_full = get_minigrid_background_colors(gt_rewards, alpha=1.0)
        bg_colors_faded = get_minigrid_background_colors(gt_rewards, alpha=bg_alpha)
    else:
        bg_colors_full = None
        bg_colors_faded = None

    # Load all predictions and compute per-run normalization
    predictions = {}
    run_scales = {}

    for run_idx, run in enumerate(original_runs):
        run_dir = Path(run["dir"])
        predictions[run_idx] = {}
        run_means = []
        run_stds = []

        for epoch_idx, epoch in enumerate(run["epochs"]):
            checkpoint_path = find_checkpoint_for_epoch(run_dir, epoch)
            if checkpoint_path is None:
                print(f"Warning: Checkpoint not found for {run['name']} epoch {epoch}")
                continue

            checkpoint = load_checkpoint(checkpoint_path)
            encoder = reconstruct_reward_encoder(checkpoint, device)
            mean_grid, std_grid = compute_reward_predictions(encoder, grid_size, device)

            predictions[run_idx][epoch_idx] = (mean_grid, std_grid, epoch)
            run_means.append(mean_grid)
            run_stds.append(std_grid)

        if run_means:
            run_scales[run_idx] = (
                min(m.min() for m in run_means),
                max(m.max() for m in run_means),
                min(s.min() for s in run_stds),
                max(s.max() for s in run_stds),
            )
        else:
            run_scales[run_idx] = (0, 1, 0, 1)

    # Figure sizing
    cell_width = 1.5 * max_epochs
    cell_height = 1.5
    header_height = 0.1
    colorbar_height = 0.4
    legend_height = 0.6

    n_cols = (1 if include_ground_truth else 0) + n_runs
    n_rows = 2  # header + content + colorbar

    fig_width = n_cols * cell_width + 0.5
    fig_height = header_height + cell_height + legend_height

    fig = plt.figure(figsize=(fig_width, fig_height))

    gt_width = 1.5 if include_ground_truth else 0
    width_ratios = ([gt_width] if include_ground_truth else []) + [cell_width] * n_runs
    height_ratios = [header_height, cell_height]

    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        wspace=0.15,
        hspace=0.15,
        left=0.06, right=0.94, top=0.88, bottom=0.15
    )

    col_offset = 1 if include_ground_truth else 0

    # Store scalar mappables for legend
    last_sm_mean = None
    last_sm_var = None

    # --- Ground Truth Column ---
    if include_ground_truth:
        ax_gt_header = fig.add_subplot(gs[0, 0])
        ax_gt_header.text(0.5, 0.5, "Ground Truth", ha='center', va='center',
                         fontsize=10, fontweight='bold')
        ax_gt_header.axis('off')

        ax_gt = fig.add_subplot(gs[1, 0])

        if minigrid_background:
            # Use MiniGrid-style background at full opacity for ground truth
            rows, cols = gt_rewards.shape
            ax_gt.set_aspect('equal')
            ax_gt.set_xlim(0, cols)
            ax_gt.set_ylim(0, rows)
            draw_minigrid_background(ax_gt, bg_colors_full, draw_border=True)
            ax_gt.invert_yaxis()
            ax_gt.set_xticks([])
            ax_gt.set_yticks([])
        else:
            # Fallback to heatmap
            im_gt = ax_gt.imshow(gt_rewards, vmin=vmin_gt, vmax=vmax_gt, cmap='coolwarm')
            ax_gt.set_xticks([])
            ax_gt.set_yticks([])
            ax_gt.set_aspect('equal')

    # --- Run Columns ---
    for run_idx, run in enumerate(runs):
        col = col_offset + run_idx
        original_idx = run["_original_idx"]
        vmin_mean, vmax_mean, vmin_std, vmax_std = run_scales[original_idx]

        # Header
        ax_header = fig.add_subplot(gs[0, col])
        ax_header.text(0.5, 0.5, run['name'], ha='center', va='center',
                      fontsize=10, fontweight='bold')
        ax_header.axis('off')

        # Content row - create inner grid for epochs
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, max_epochs,
            subplot_spec=gs[1, col],
            wspace=0.08
        )

        for epoch_idx in range(max_epochs):
            ax_epoch = fig.add_subplot(inner_gs[0, epoch_idx])

            if epoch_idx in predictions.get(original_idx, {}):
                mean_grid, std_grid, epoch = predictions[original_idx][epoch_idx]

                last_sm_mean, last_sm_var = draw_hinton_diagram(
                    ax_epoch,
                    mean_grid, std_grid,
                    vmin_mean, vmax_mean,
                    vmin_std, vmax_std,
                    cmap_mean=cmap_mean,
                    cmap_var=cmap_var,
                    bg_colors=bg_colors_faded,
                    draw_border=True,
                )
            else:
                ax_epoch.axis('off')

        # Colorbar row - show both mean and variance colorbars
        """if last_sm_mean is not None and last_sm_var is not None:
            inner_gs_cbar = gridspec.GridSpecFromSubplotSpec(
                1, 2,
                subplot_spec=gs[2, col],
                wspace=0.4
            )

            ax_cbar_mean = fig.add_subplot(inner_gs_cbar[0, 0])
            cbar_mean = fig.colorbar(last_sm_mean, cax=ax_cbar_mean, orientation='horizontal')
            cbar_mean.ax.tick_params(labelsize=6)
            cbar_mean.set_label('Mean (μ)', fontsize=7)

            ax_cbar_var = fig.add_subplot(inner_gs_cbar[0, 1])
            cbar_var = fig.colorbar(last_sm_var, cax=ax_cbar_var, orientation='horizontal')
            cbar_var.ax.tick_params(labelsize=6)
            cbar_var.set_label('Variance (σ²)', fontsize=7)
        """

    # Add legend explaining the Hinton diagram encoding
    legend_text = "□ size = |1 - variance|, □ color = mean value"
    fig.text(0.5, 0.02, legend_text, fontsize=8, ha='center', va='bottom', style='italic')

    return fig


def load_config(config_path: str) -> dict:
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Compare reward model checkpoints across training epochs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Config file option
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    
    # CLI options (used if --config not provided)
    parser.add_argument("--run_dirs", type=str, nargs="+", help="Paths to checkpoint directories")
    parser.add_argument("--run_names", type=str, nargs="+", help="Display names for each run")
    parser.add_argument("--epochs", type=int, nargs="+", help="Epochs to visualize (same for all runs)")
    parser.add_argument("--env_id", type=str, default="grid_cliff", help="Environment ID")
    parser.add_argument("--grid_size", type=int, default=10, help="Grid size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--output", type=str, default="checkpoint_comparison.pdf", help="Output file path")
    parser.add_argument("--no_ground_truth", action="store_true", help="Don't include ground truth column")
    parser.add_argument("--duplicate_runs", type=int, default=1,
                        help="Number of times to duplicate runs (to simulate more feedback combos)")
    parser.add_argument("--cmap_mean", type=str, default="coolwarm",
                        help="Colormap for mean values (inner squares)")
    parser.add_argument("--cmap_var", type=str, default="Greens",
                        help="Colormap for variance values (background)")
    parser.add_argument("--no_minigrid_background", action="store_true",
                        help="Disable MiniGrid-style background tiles")
    parser.add_argument("--bg_alpha", type=float, default=0.3,
                        help="Background opacity for prediction plots (0-1, ground truth uses 1.0)")

    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        runs = config["runs"]
        env_id = config.get("env_id", "grid_cliff")
        grid_size = config.get("grid_size", 10)
        gamma = config.get("gamma", 0.99)
        output_path = config.get("output", "checkpoint_comparison.pdf")
        include_ground_truth = config.get("include_ground_truth", True)
        duplicate_runs = config.get("duplicate_runs", 1)
        cmap_mean = config.get("cmap_mean", "coolwarm")
        cmap_var = config.get("cmap_var", "Greys_r")
        if "no_minigrid_background" in config:
            minigrid_background = not config["no_minigrid_background"]
        else:
            minigrid_background = config.get("minigrid_background", True)
        bg_alpha = config.get("bg_alpha", 0.0)
    else:
        if not args.run_dirs:
            parser.error("Either --config or --run_dirs must be provided")
        if not args.epochs:
            parser.error("Either --config or --epochs must be provided")

        run_names = args.run_names or [f"Run {i+1}" for i in range(len(args.run_dirs))]
        if len(run_names) != len(args.run_dirs):
            parser.error("Number of run_names must match number of run_dirs")

        runs = [
            {"name": name, "dir": dir_path, "epochs": args.epochs}
            for name, dir_path in zip(run_names, args.run_dirs)
        ]
        env_id = args.env_id
        grid_size = args.grid_size
        gamma = args.gamma
        output_path = args.output
        include_ground_truth = not args.no_ground_truth
        duplicate_runs = args.duplicate_runs
        cmap_mean = args.cmap_mean
        cmap_var = args.cmap_var
        minigrid_background = not args.no_minigrid_background
        bg_alpha = args.bg_alpha

    # Create the comparison figure
    print(f"Creating comparison figure for {len(runs)} runs...")
    for run in runs:
        print(f"  {run['name']}: {run['dir']} (epochs: {run['epochs']})")
    
    fig = create_comparison_figure(
        runs=runs,
        env_id=env_id,
        grid_size=grid_size,
        include_ground_truth=include_ground_truth,
        gamma=gamma,
        duplicate_runs=duplicate_runs,
        cmap_mean=cmap_mean,
        cmap_var=cmap_var,
        minigrid_background=minigrid_background,
        bg_alpha=bg_alpha,
    )
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400, bbox_inches='tight')
    print(f"Saved comparison figure to: {output_path}")
    
    plt.close(fig)


if __name__ == "__main__":
    main()

