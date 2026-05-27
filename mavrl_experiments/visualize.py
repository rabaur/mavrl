"""Visualization tools for experiment results from file-based queue.

Generates interactive Plotly visualizations to analyze hyperparameter impact,
with special emphasis on feedback type combinations (pref-only, demo-only, pref+demo).
"""

import argparse
from pathlib import Path
from typing import Optional
import numpy as np

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

from mavrl_experiments.utils import (
    AGGREGATE_METHODS,
    load_experiment_data,
    get_available_metrics,
    parse_config_filters,
    apply_config_filters,
    add_aggregated_metric_column,
)

# =============================================================================
# Parallel Coordinates Plot
# =============================================================================

def create_parallel_coordinates(
    df: pd.DataFrame,
    color_by: str = "best_regret",
    env_filter: Optional[str] = None,
    feedback_filter: Optional[str] = None,
    jitter: float = 0.0,
) -> go.Figure:
    """
    Create a parallel coordinates plot for hyperparameter analysis.
    
    Args:
        df: DataFrame with experiment data
        color_by: Column to use for line coloring ("best_regret" or "best_epic_distance")
        env_filter: Filter to specific environment (e.g., "grid_sparse")
        feedback_filter: Filter to specific feedback type ("pref_only", "demo_only", "pref+demo")
        jitter: Amount of random jitter to add to categorical/discrete variables (0-1).
                Helps reduce overplotting. E.g., 0.15 spreads lines ±15% around category centers.
        
    Returns:
        Plotly Figure object
    """
    # Apply filters
    plot_df = df.copy()
    
    if env_filter and "config.env_id" in plot_df.columns:
        plot_df = plot_df[plot_df["config.env_id"] == env_filter]
    
    if feedback_filter and "feedback_type" in plot_df.columns:
        plot_df = plot_df[plot_df["feedback_type"] == feedback_filter]
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data matching filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Select dimensions for parallel coordinates
    # Key hyperparameters that are likely to vary
    dimension_cols = [
        "config.lr",
        "config.batch_size",
        "config.n_pref_samples",
        "config.n_demo_samples",
        "config.gamma",
    ]
    
    # Add preference-specific params if they exist and have variance
    optional_cols = [
        "config.pref_seg_len",
        "config.pref_data_rationality",
        "config.pref_model_rationality",
        "config.pref_trajectory_rationality",
        "config.demo_rationality",
        "config.encoder_hidden_sizes",
    ]
    
    for col in optional_cols:
        if col in plot_df.columns and plot_df[col].nunique() > 1:
            dimension_cols.append(col)
    
    # Filter to columns that exist and have variance
    valid_dimensions = []
    for col in dimension_cols:
        if col in plot_df.columns:
            # Check if column has at least 2 unique non-null values
            non_null = plot_df[col].dropna()
            if len(non_null) > 0 and non_null.nunique() >= 1:
                valid_dimensions.append(col)
    
    if not valid_dimensions:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid dimensions for parallel coordinates",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Check if color column exists
    if color_by not in plot_df.columns:
        # Fallback to first available metric
        for alt in ["best_regret", "best_epic_distance", "final_regret"]:
            if alt in plot_df.columns:
                color_by = alt
                break
        else:
            color_by = None
    
    # Build dimensions list for Parcoords
    dimensions = []
    
    # Set random seed for reproducible jitter
    rng = np.random.default_rng(42)
    
    for col in valid_dimensions:
        dim_data = plot_df[col].copy()
        
        # Handle string columns (like encoder_hidden_sizes)
        if dim_data.dtype == object:
            # Convert to categorical codes
            categories = dim_data.dropna().unique()
            cat_map = {cat: i for i, cat in enumerate(categories)}
            dim_data = dim_data.map(lambda x: cat_map.get(x, -1) if pd.notna(x) else -1)
            
            # Apply jitter to reduce overplotting
            if jitter > 0 and len(categories) > 1:
                noise = rng.uniform(-jitter, jitter, size=len(dim_data))
                dim_data = dim_data + noise
            
            dimensions.append(dict(
                label=col.replace("config.", ""),
                values=dim_data,
                tickvals=list(range(len(categories))),
                ticktext=[str(c)[:20] for c in categories],  # Truncate long labels
            ))
        else:
            # Numeric column - check if discrete (few unique values)
            unique_vals = dim_data.dropna().nunique()
            is_discrete = unique_vals <= 10  # Treat as discrete if ≤10 unique values
            
            # Apply jitter to discrete numeric columns too
            if jitter > 0 and is_discrete and unique_vals > 1:
                # Scale jitter relative to spacing between values
                sorted_unique = np.sort(dim_data.dropna().unique())
                if len(sorted_unique) > 1:
                    min_spacing = np.min(np.diff(sorted_unique))
                    noise = rng.uniform(-jitter * min_spacing, jitter * min_spacing, size=len(dim_data))
                    dim_data = dim_data + noise
            
            dimensions.append(dict(
                label=col.replace("config.", ""),
                values=dim_data,
                range=[dim_data.min(), dim_data.max()] if dim_data.notna().any() else [0, 1],
            ))
    
    # Add the metric as the final dimension
    if color_by and color_by in plot_df.columns:
        metric_data = plot_df[color_by]
        dimensions.append(dict(
            label=color_by.replace("_", " ").title(),
            values=metric_data,
            range=[metric_data.min(), metric_data.max()] if metric_data.notna().any() else [0, 1],
        ))
    
    # Create figure
    if color_by and color_by in plot_df.columns:
        line_config = dict(
            color=plot_df[color_by],
            colorscale="Viridis_r",  # Reversed so lower (better) is brighter
            showscale=True,
            cmin=plot_df[color_by].min(),
            cmax=plot_df[color_by].max(),
        )
    else:
        line_config = dict(color="steelblue")
    
    fig = go.Figure(data=
        go.Parcoords(
            line=line_config,
            dimensions=dimensions,
            unselected=dict(line=dict(color="lightgray", opacity=0.3)),
        )
    )
    
    # Update layout
    title = "Hyperparameter Impact Analysis"
    if env_filter:
        title += f" - {env_filter}"
    if feedback_filter:
        title += f" ({feedback_filter})"
    
    fig.update_layout(
        title=title,
        font=dict(size=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=100, r=100, t=80, b=50),
    )
    
    return fig


# =============================================================================
# Complementary Visualizations
# =============================================================================

def create_violin_by_feedback_type(
    df: pd.DataFrame,
    metric: str = "best_regret",
    env_filter: Optional[str] = None,
) -> go.Figure:
    """
    Create violin plots comparing metric distributions across feedback type combinations.
    
    Args:
        df: DataFrame with experiment data
        metric: Metric to plot ("best_regret" or "best_epic_distance")
        env_filter: Filter to specific environment
        
    Returns:
        Plotly Figure object
    """
    plot_df = df.copy()
    
    if env_filter and "config.env_id" in plot_df.columns:
        plot_df = plot_df[plot_df["config.env_id"] == env_filter]
    
    if metric not in plot_df.columns or "feedback_type" not in plot_df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Missing required columns: {metric} or feedback_type",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
        return fig
    
    # Filter out rows with missing metric
    plot_df = plot_df.dropna(subset=[metric])
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
        return fig
    
    # Create violin plot
    fig = px.violin(
        plot_df,
        x="feedback_type",
        y=metric,
        color="feedback_type",
        box=True,
        points="all",
        hover_data=["experiment_id", "seed", "config.env_id"] if "config.env_id" in plot_df.columns else ["experiment_id", "seed"],
        category_orders={"feedback_type": ["demo_only", "pref_only", "pref+demo"]},
        color_discrete_map={
            "demo_only": "#636EFA",
            "pref_only": "#EF553B",
            "pref+demo": "#00CC96",
        },
    )
    
    title = f"{metric.replace('_', ' ').title()} by Feedback Type"
    if env_filter:
        title += f" - {env_filter}"
    
    fig.update_layout(
        title=title,
        xaxis_title="Feedback Type",
        yaxis_title=metric.replace("_", " ").title(),
        showlegend=False,
    )
    
    return fig


def create_heatmap_pref_vs_demo(
    df: pd.DataFrame,
    metric: str = "best_regret",
    env_filter: Optional[str] = None,
    agg_func: str = "mean",
) -> go.Figure:
    """
    Create a heatmap showing metric as function of n_pref_samples vs n_demo_samples.
    
    Args:
        df: DataFrame with experiment data
        metric: Metric to aggregate
        env_filter: Filter to specific environment
        agg_func: Aggregation function ("mean", "min", "median")
        
    Returns:
        Plotly Figure object
    """
    plot_df = df.copy()
    
    if env_filter and "config.env_id" in plot_df.columns:
        plot_df = plot_df[plot_df["config.env_id"] == env_filter]
    
    required_cols = [metric, "config.n_pref_samples", "config.n_demo_samples"]
    for col in required_cols:
        if col not in plot_df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Missing required column: {col}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
            )
            return fig
    
    # Filter out rows with missing values
    plot_df = plot_df.dropna(subset=required_cols)
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
        return fig
    
    # Aggregate by (n_pref_samples, n_demo_samples)
    agg_funcs = {"mean": "mean", "min": "min", "median": "median"}
    pivot = plot_df.pivot_table(
        values=metric,
        index="config.n_demo_samples",
        columns="config.n_pref_samples",
        aggfunc=agg_funcs.get(agg_func, "mean"),
    )
    
    # Sort index and columns
    pivot = pivot.sort_index(ascending=True)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    
    fig = px.imshow(
        pivot,
        labels=dict(
            x="n_pref_samples",
            y="n_demo_samples",
            color=f"{agg_func} {metric}",
        ),
        aspect="auto",
        color_continuous_scale="Viridis_r",  # Lower is better
    )
    
    title = f"{agg_func.title()} {metric.replace('_', ' ').title()}: Pref vs Demo Samples"
    if env_filter:
        title += f" - {env_filter}"
    
    fig.update_layout(
        title=title,
        xaxis_title="n_pref_samples",
        yaxis_title="n_demo_samples",
    )
    
    # Add text annotations
    fig.update_traces(
        text=pivot.round(4).values,
        texttemplate="%{text}",
        textfont=dict(size=10),
    )
    
    return fig


def create_learning_curves(
    df: pd.DataFrame,
    metric: str = "regret",
    env_filter: Optional[str] = None,
    group_by: str = "feedback_type",
    show_individual: bool = True,
) -> go.Figure:
    """
    Create learning curves showing metric over epochs.
    
    Args:
        df: DataFrame with experiment data
        metric: Base metric name ("regret" or "epic_distance")
        env_filter: Filter to specific environment
        group_by: Column to group/color by
        show_individual: If True, show individual runs; otherwise show mean with std band
        
    Returns:
        Plotly Figure object
    """
    plot_df = df.copy()
    
    if env_filter and "config.env_id" in plot_df.columns:
        plot_df = plot_df[plot_df["config.env_id"] == env_filter]
    
    series_col = f"series.{metric}"
    if series_col not in plot_df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No {metric} series data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
        return fig
    
    # Filter to rows with series data
    plot_df = plot_df.dropna(subset=[series_col])
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No series data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
        return fig
    
    fig = go.Figure()
    
    color_map = {
        "demo_only": "#636EFA",
        "pref_only": "#EF553B",
        "pref+demo": "#00CC96",
        "none": "#AB63FA",
    }
    
    if show_individual:
        # Plot individual curves
        for _, row in plot_df.iterrows():
            series = row[series_col]
            epochs = row.get("epochs", list(range(len(series))))
            group_val = row.get(group_by, "unknown")
            
            fig.add_trace(go.Scatter(
                x=epochs,
                y=series,
                mode="lines",
                name=f"{group_val} (exp {row['experiment_id']})",
                line=dict(
                    color=color_map.get(group_val, "#888888"),
                    width=1,
                ),
                opacity=0.5,
                legendgroup=group_val,
                showlegend=False,
            ))
    else:
        # Aggregate by group
        groups = plot_df[group_by].unique() if group_by in plot_df.columns else ["all"]
        
        for group_val in sorted(groups):
            if group_by in plot_df.columns:
                group_df = plot_df[plot_df[group_by] == group_val]
            else:
                group_df = plot_df
            
            # Find common epoch range
            all_series = group_df[series_col].tolist()
            all_epochs = group_df["epochs"].tolist() if "epochs" in group_df.columns else [list(range(len(s))) for s in all_series]
            
            # Get max length for padding
            max_len = max(len(s) for s in all_series)
            
            # Pad series to same length and compute stats
            import numpy as np
            padded = np.full((len(all_series), max_len), np.nan)
            for i, (series, epochs) in enumerate(zip(all_series, all_epochs)):
                padded[i, :len(series)] = series
            
            mean_series = np.nanmean(padded, axis=0)
            std_series = np.nanstd(padded, axis=0)
            epoch_range = list(range(max_len))
            
            color = color_map.get(group_val, "#888888")
            
            # Add mean line
            fig.add_trace(go.Scatter(
                x=epoch_range,
                y=mean_series,
                mode="lines",
                name=str(group_val),
                line=dict(color=color, width=2),
            ))
            
            # Add std band
            fig.add_trace(go.Scatter(
                x=epoch_range + epoch_range[::-1],
                y=list(mean_series + std_series) + list(mean_series - std_series)[::-1],
                fill="toself",
                fillcolor=color,
                opacity=0.2,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))
    
    title = f"{metric.replace('_', ' ').title()} Over Epochs"
    if env_filter:
        title += f" - {env_filter}"
    
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title=metric.replace("_", " ").title(),
        legend_title=group_by.replace("_", " ").title(),
    )
    
    return fig


def create_bar_with_ci(
    df: pd.DataFrame,
    metric: str = "best_regret",
    group_by: str = "feedback_type",
    env_filter: Optional[str] = None,
) -> go.Figure:
    """
    Create bar chart showing mean metric per group with confidence intervals.
    
    Args:
        df: DataFrame with experiment data
        metric: Metric to aggregate
        group_by: Column to group by
        env_filter: Filter to specific environment
        
    Returns:
        Plotly Figure object
    """
    import numpy as np
    
    plot_df = df.copy()
    
    if env_filter and "config.env_id" in plot_df.columns:
        plot_df = plot_df[plot_df["config.env_id"] == env_filter]
    
    if metric not in plot_df.columns or group_by not in plot_df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Missing required columns",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
        return fig
    
    plot_df = plot_df.dropna(subset=[metric])
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
        return fig
    
    # Compute stats per group
    stats = plot_df.groupby(group_by)[metric].agg(["mean", "std", "count"]).reset_index()
    stats["se"] = stats["std"] / np.sqrt(stats["count"])  # Standard error
    stats["ci95"] = 1.96 * stats["se"]  # 95% CI
    
    # Sort by desired order
    order = ["demo_only", "pref_only", "pref+demo"]
    stats["sort_order"] = stats[group_by].apply(lambda x: order.index(x) if x in order else len(order))
    stats = stats.sort_values("sort_order")
    
    color_map = {
        "demo_only": "#636EFA",
        "pref_only": "#EF553B",
        "pref+demo": "#00CC96",
    }
    colors = [color_map.get(g, "#888888") for g in stats[group_by]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=stats[group_by],
        y=stats["mean"],
        error_y=dict(
            type="data",
            array=stats["ci95"],
            visible=True,
        ),
        marker_color=colors,
        text=stats["count"].apply(lambda x: f"n={x}"),
        textposition="outside",
    ))
    
    title = f"Mean {metric.replace('_', ' ').title()} by {group_by.replace('_', ' ').title()}"
    if env_filter:
        title += f" - {env_filter}"
    
    fig.update_layout(
        title=title,
        xaxis_title=group_by.replace("_", " ").title(),
        yaxis_title=f"Mean {metric.replace('_', ' ').title()} (95% CI)",
        showlegend=False,
    )
    
    return fig


# =============================================================================
# Transfer Experiment Visualization
# =============================================================================

def create_transfer_plot(
    df: pd.DataFrame,
    x_param: str = "wind_power",
    group_by: str = "feedback_combo",
    metric: str = "mean_reward",
    env_filter: Optional[str] = None,
    error_type: str = "std",
) -> plt.Figure:
    """
    Create a line plot showing mean reward across different conditions for transfer experiments.
    
    Shows one line per feedback_combo (e.g., 'demo+pref', 'imitation') with error bars
    representing uncertainty across seeds.
    
    Args:
        df: DataFrame with experiment data (from load_experiment_data)
        x_param: Config parameter to plot on x-axis (e.g., "wind_power")
        group_by: Config parameter to group by (e.g., "feedback_combo")
        metric: Metric to plot (default: "mean_reward")
        env_filter: Filter to specific environment
        error_type: Type of error bars - "std" (standard deviation) or "se" (standard error)
        
    Returns:
        Matplotlib Figure object
    """
    import numpy as np
    
    plot_df = df.copy()
    
    # Filter to transfer experiments (have feedback_combo or reward_model_path/imitation_model_path)
    is_transfer = (
        (plot_df.get("config.feedback_combo").notna()) |
        (plot_df.get("config.reward_model_path").notna()) |
        (plot_df.get("config.imitation_model_path").notna())
    )
    plot_df = plot_df[is_transfer]
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No transfer experiment data found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
        return fig
    
    # Apply environment filter
    if env_filter and "config.env_id" in plot_df.columns:
        plot_df = plot_df[plot_df["config.env_id"] == env_filter]
    
    # Check required columns
    x_col = f"config.{x_param}"
    group_col = f"config.{group_by}"
    metric_series_col = f"series.{metric}"
    
    required_cols = [x_col, group_col]
    if metric_series_col not in plot_df.columns:
        # Try final metric instead
        final_metric_col = f"final.{metric}"
        if final_metric_col in plot_df.columns:
            # Create a series column from final metric
            plot_df[metric_series_col] = plot_df[final_metric_col].apply(lambda x: [x] if pd.notna(x) else [])
        else:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Metric '{metric}' not found. Available metrics: {get_available_metrics(plot_df)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
            )
            return fig
    
    for col in required_cols:
        if col not in plot_df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Missing required column: {col}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
            )
            return fig
    
    # Extract metric values from series (transfer experiments log at epoch 0, so single value)
    def extract_metric_value(series):
        if pd.isna(series) or not isinstance(series, list) or len(series) == 0:
            return np.nan
        # Take the first (and typically only) value
        return float(series[0]) if len(series) > 0 else np.nan
    
    plot_df["metric_value"] = plot_df[metric_series_col].apply(extract_metric_value)

    # Remove all rows where enable_wind == True but wind_power is 0.0
    plot_df = plot_df[~((plot_df["config.enable_wind"] == True) & (plot_df["config.wind_power"] == 0.0))]

    # If fill NaN wind_power values with 0
    plot_df["config.wind_power"] = plot_df["config.wind_power"].fillna(0.0)
    print(plot_df[["config.enable_wind", "config.wind_power"]])
    
    plot_df = plot_df.dropna(subset=[x_col, group_col, "metric_value"])
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available after filtering",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
        return fig
    
    # Color map for different feedback combos
    color_map = {
        "demo+pref": "#00CC96",
        "pref+demo": "#00CC96",  # Alternative naming
        "imitation": "#636EFA",
        "demo_only": "#EF553B",
        "pref_only": "#AB63FA",
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    

    sns.lineplot(
        data=plot_df,
        x=x_col,
        y="metric_value",
        hue=group_col,
        palette=color_map,
        ax=ax,
        marker="o",
        markersize=8,
        linewidth=2,
        errorbar="sd",
        err_style="bars",
    )
    
    # Update labels and title
    title = f"Transfer Experiment: {metric.replace('_', ' ').title()} vs {x_param.replace('_', ' ').title()}"
    if env_filter:
        title += f" ({env_filter})"
    
    ax.set_title(title)
    ax.set_xlabel(x_param.replace("_", " ").title())
    ax.set_ylabel(f"Mean {metric.replace('_', ' ').title()}")
    ax.legend(title=group_by.replace("_", " ").title())

    plt.tight_layout()

    # save to png
    plt.savefig(f"transfer_plot_{x_param}_{group_by}_{metric}.png")

    return fig

# =============================================================================
# Dashboard Generation
# =============================================================================

def create_dashboard(
    df: pd.DataFrame,
    env_filter: Optional[str] = None,
    metric_col: str = "min_regret",
    metric_base: str = "regret",
) -> go.Figure:
    """
    Create a comprehensive dashboard with multiple visualizations.
    
    Args:
        df: DataFrame with experiment data
        env_filter: Filter to specific environment
        metric_col: Aggregated metric column name (e.g., "min_regret")
        metric_base: Base metric name for learning curves (e.g., "regret")
        
    Returns:
        Plotly Figure object with subplots
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Metric Distribution by Feedback Type",
            "Mean Metric with 95% CI",
            "Pref vs Demo Samples Heatmap",
            "Learning Curves by Feedback Type",
        ),
        specs=[
            [{"type": "violin"}, {"type": "bar"}],
            [{"type": "heatmap"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )
    
    # 1. Violin plot
    violin_fig = create_violin_by_feedback_type(df, metric_col, env_filter)
    for trace in violin_fig.data:
        fig.add_trace(trace, row=1, col=1)
    
    # 2. Bar chart with CI
    bar_fig = create_bar_with_ci(df, metric_col, "feedback_type", env_filter)
    for trace in bar_fig.data:
        fig.add_trace(trace, row=1, col=2)
    
    # 3. Heatmap
    heatmap_fig = create_heatmap_pref_vs_demo(df, metric_col, env_filter)
    for trace in heatmap_fig.data:
        fig.add_trace(trace, row=2, col=1)
    
    # 4. Learning curves
    curves_fig = create_learning_curves(df, metric_base, env_filter)
    for trace in curves_fig.data:
        fig.add_trace(trace, row=2, col=2)
    
    title = "Experiment Analysis Dashboard"
    if env_filter:
        title += f" - {env_filter}"
    
    fig.update_layout(
        title=title,
        height=900,
        showlegend=True,
    )
    
    return fig


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize experiment results from file-based queue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - generates parallel coordinates plot
  python -m mavrl_experiments.visualize /path/to/queue_dir

  # Filter by environment and save to file
  python -m mavrl_experiments.visualize /path/to/queue_dir \\
    --env-filter grid_sparse --output-html results.html

  # Filter by config attributes (e.g., learning rate, batch size)
  python -m mavrl_experiments.visualize /path/to/queue_dir \\
    --config-filter lr=0.0001 batch_size=64

  # Use custom metric and aggregation
  python -m mavrl_experiments.visualize /path/to/queue_dir \\
    --metric regret --aggregate min

  # Different aggregations
  python -m mavrl_experiments.visualize /path/to/queue_dir \\
    --metric regret --aggregate mean  # average regret across epochs
  python -m mavrl_experiments.visualize /path/to/queue_dir \\
    --metric epic_distance --aggregate last  # final epic distance

  # List available metrics
  python -m mavrl_experiments.visualize /path/to/queue_dir --list-metrics
        """
    )
    
    parser.add_argument(
        "queue_dir",
        type=str,
        help="Path to the file queue directory",
    )
    
    parser.add_argument(
        "--env-filter",
        type=str,
        default=None,
        help="Filter to specific environment (e.g., 'grid_sparse', 'CartPole-v1')",
    )
    
    parser.add_argument(
        "--feedback-filter",
        type=str,
        choices=["pref_only", "demo_only", "pref+demo"],
        default=None,
        help="Filter to specific feedback type combination",
    )
    
    parser.add_argument(
        "--config-filter",
        type=str,
        nargs="+",
        default=None,
        metavar="KEY=VALUE",
        help="Filter by config attributes (e.g., --config-filter lr=0.0001 batch_size=64)",
    )
    
    parser.add_argument(
        "--include-running",
        action="store_true",
        help="Include experiments that are still running (partial results)",
    )
    
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include failed experiments",
    )
    
    parser.add_argument(
        "--metric",
        type=str,
        default="regret",
        help="Metric to use for coloring/analysis (default: regret). Use --list-metrics to see available options.",
    )
    
    parser.add_argument(
        "--aggregate",
        type=str,
        default="min",
        choices=AGGREGATE_METHODS,
        help="Aggregation method across epochs (default: min). Options: min, max, mean, median, last, first",
    )
    
    parser.add_argument(
        "--plot-type",
        type=str,
        default="parallel",
        choices=["parallel", "violin", "heatmap", "learning", "bar", "dashboard", "transfer", "all"],
        help="Type of plot to generate (default: parallel)",
    )
    
    parser.add_argument(
        "--output-html",
        type=str,
        default=None,
        help="Save plot to HTML file instead of showing interactively",
    )
    
    parser.add_argument(
        "--list-envs",
        action="store_true",
        help="List available environments and exit",
    )
    
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List available metrics and exit",
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics and exit",
    )
    
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.15,
        help="Jitter amount for categorical/discrete variables in parallel coords (0-1). "
             "Reduces overplotting by spreading lines around category centers. Default: 0.15. Use 0 to disable.",
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading experiments from: {args.queue_dir}")
    df = load_experiment_data(
        args.queue_dir,
        include_running=args.include_running,
        include_failed=args.include_failed,
    )
    
    if df.empty:
        print("No experiments found!")
        return
    
    print(f"Loaded {len(df)} experiments")
    
    # Apply config filters early (before --list-envs/--list-metrics so they show filtered data)
    config_filters = parse_config_filters(args.config_filter)
    if config_filters:
        df = apply_config_filters(df, config_filters)
        if df.empty:
            print("No experiments match the specified config filters!")
            return
        print(f"After config filters: {len(df)} experiments")
    
    # Handle --list-envs
    if args.list_envs:
        if "config.env_id" in df.columns:
            envs = df["config.env_id"].value_counts()
            print("\nAvailable environments:")
            for env, count in envs.items():
                print(f"  {env}: {count} experiments")
        else:
            print("No env_id found in config")
        return
    
    # Handle --list-metrics
    if args.list_metrics:
        metrics = get_available_metrics(df)
        print("\nAvailable metrics:")
        for m in metrics:
            # Count how many experiments have this metric
            series_col = f"series.{m}"
            count = df[series_col].notna().sum() if series_col in df.columns else 0
            print(f"  {m}: {count} experiments")
        print(f"\nAggregation methods: {', '.join(AGGREGATE_METHODS)}")
        return
    
    # Compute the aggregated metric column
    df, metric_col = add_aggregated_metric_column(df, args.metric, args.aggregate)
    
    # Check if the metric has valid data
    valid_count = df[metric_col].notna().sum()
    if valid_count == 0:
        print(f"\nWarning: No valid data for metric '{args.metric}'. Use --list-metrics to see available metrics.")
        return
    
    print(f"Using metric: {metric_col} ({valid_count} experiments with valid data)")
    
    # Handle --summary
    if args.summary:
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        print(f"\nTotal experiments: {len(df)}")
        
        if "status" in df.columns:
            print("\nBy status:")
            for status, count in df["status"].value_counts().items():
                print(f"  {status}: {count}")
        
        if "config.env_id" in df.columns:
            print("\nBy environment:")
            for env, count in df["config.env_id"].value_counts().items():
                print(f"  {env}: {count}")
        
        if "feedback_type" in df.columns:
            print("\nBy feedback type:")
            for fb, count in df["feedback_type"].value_counts().items():
                print(f"  {fb}: {count}")
        
        metric_data = df[metric_col].dropna()
        if len(metric_data) > 0:
            print(f"\n{metric_col}:")
            print(f"  Mean:   {metric_data.mean():.6f}")
            print(f"  Std:    {metric_data.std():.6f}")
            print(f"  Min:    {metric_data.min():.6f}")
            print(f"  Max:    {metric_data.max():.6f}")
            print(f"  Median: {metric_data.median():.6f}")
        
        return
    
    # Generate plot(s)
    figures = []
    
    if args.plot_type in ["parallel", "all"]:
        fig = create_parallel_coordinates(
            df,
            color_by=metric_col,
            env_filter=args.env_filter,
            feedback_filter=args.feedback_filter,
            jitter=args.jitter,
        )
        fig.update_layout(title=f"Parallel Coordinates: Hyperparameter Impact ({metric_col})")
        figures.append(("parallel_coords", fig))
    
    if args.plot_type in ["violin", "all"]:
        fig = create_violin_by_feedback_type(df, metric_col, args.env_filter)
        figures.append(("violin", fig))
    
    if args.plot_type in ["heatmap", "all"]:
        fig = create_heatmap_pref_vs_demo(df, metric_col, args.env_filter)
        figures.append(("heatmap", fig))
    
    if args.plot_type in ["learning", "all"]:
        fig = create_learning_curves(df, args.metric, args.env_filter)
        figures.append(("learning_curves", fig))
    
    if args.plot_type in ["bar", "all"]:
        fig = create_bar_with_ci(df, metric_col, "feedback_type", args.env_filter)
        figures.append(("bar_ci", fig))
    
    if args.plot_type == "dashboard":
        fig = create_dashboard(df, args.env_filter, metric_col, args.metric)
        figures.append(("dashboard", fig))
    
    if args.plot_type in ["transfer", "all"]:
        # For transfer plots, we use mean_reward by default
        fig = create_transfer_plot(
            df,
            x_param="wind_power",
            group_by="feedback_combo",
            metric="mean_reward",
            env_filter=args.env_filter,
        )
        figures.append(("transfer", fig))
    
    # Output
    if args.output_html:
        output_path = Path(args.output_html)
        
        if len(figures) == 1:
            # Single figure - save directly
            figures[0][1].write_html(output_path)
            print(f"Saved plot to: {output_path}")
        else:
            # Multiple figures - save each with suffix
            for name, fig in figures:
                path = output_path.with_stem(f"{output_path.stem}_{name}")
                fig.write_html(path)
                print(f"Saved {name} to: {path}")
    else:
        # Show interactively
        for name, fig in figures:
            print(f"Showing: {name}")
            fig.show()


if __name__ == "__main__":
    main()

