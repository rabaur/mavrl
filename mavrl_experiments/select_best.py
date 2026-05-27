"""Helper functions to select best experiments per feedback combination.

Usage:
    from mavrl_experiments.select_best import get_best_config_per_feedback_combo
    
    best = get_best_config_per_feedback_combo("tasks", metric="regret", aggregate="min")
    # Returns: {"pref_only": {"config_hash": "...", "model_paths": ["/path/1", "/path/2", ...], ...}, ...}
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from mavrl_experiments.utils import (
    load_experiment_data,
    apply_config_filters,
)
from mavrl_experiments.utils import (
    derive_feedback_combination,
    compute_aggregated_metric,
)

def get_best_config_per_feedback_combo(
    queue_dir: str | Path,
    metric: str = "regret",
    aggregate: str = "min",
    config_filters: Optional[dict[str, str]] = None,
) -> dict[str, dict]:
    """
    Find the best configuration for each feedback combination, returning ALL seeds.
    
    For each feedback combination (e.g., "pref_only", "demo_only", "pref+demo"),
    finds the configuration with the best average metric across seeds, then returns
    ALL model paths from that configuration (all seeds).
    
    Args:
        queue_dir: Path to the file queue directory
        metric: Metric name (e.g., "regret", "epic_distance")
        aggregate: Aggregation method over epochs ("min", "max", "mean", etc.)
        env_id: Optional environment ID to filter by. If None, returns best
                across all environments.
        config_filters: Optional dict of config key -> value filters
                       (e.g., {"use_importance_weights": "True"})
                
    Returns:
        Dictionary mapping feedback_combo to configuration info with ALL model paths:
        {
            "pref_only": {
                "config_hash": "abc123...",
                "metric_value": 0.123,  # avg across seeds
                "n_seeds": 4,
                "model_paths": ["/path/seed0.pt", "/path/seed1.pt", ...],
                "policy_paths": ["/path/seed0.zip", "/path/seed1.zip", ...],
                "experiment_ids": [42, 43, 44, 45],
                "config": {...},
            },
            ...
        }
    """
    df = load_experiment_data(queue_dir)
    
    if df.empty:
        return {}
    
    # Apply config filters if specified
    if config_filters:
        df = apply_config_filters(df, config_filters)
        if df.empty:
            return {}
    
    # Compute aggregated metric per experiment
    df = df.copy()
    metric_col = f"{aggregate}_{metric}"
    df[metric_col] = compute_aggregated_metric(df, metric, aggregate)
    
    # Derive feedback combination for each experiment
    df["feedback_combo"] = df.apply(derive_feedback_combination, axis=1)
    
    # Filter to rows with valid metric data
    df_valid = df.dropna(subset=[metric_col])
    
    if df_valid.empty:
        return {}
    
    results = {}
    
    # For each feedback combination, find the best configuration
    for combo in df_valid["feedback_combo"].unique():
        combo_df = df_valid[df_valid["feedback_combo"] == combo]
        
        if combo_df.empty:
            continue
        
        # Group by config_hash to average over seeds
        # Then find the config with best average metric
        config_means = combo_df.groupby("config_hash")[metric_col].mean()
        
        # For regret-like metrics, lower is better
        if aggregate == "min" or metric in ["regret", "epic_distance"]:
            best_config_hash = config_means.idxmin()
        else:
            best_config_hash = config_means.idxmax()
        
        best_metric_value = config_means[best_config_hash]
        
        # Get ALL experiments with this config hash (all seeds)
        best_config_df = combo_df[combo_df["config_hash"] == best_config_hash]
        
        # Sort by seed for consistent ordering
        best_config_df = best_config_df.sort_values("seed")
        
        # Collect all model paths
        model_paths = []
        policy_paths = []
        experiment_ids = []
        
        for _, row in best_config_df.iterrows():
            experiment_ids.append(int(row["experiment_id"]))
            
            model_path = row.get("best_model_path")
            if pd.notna(model_path) and model_path:
                model_paths.append(model_path)
            
            policy_path = row.get("best_policy_path")
            if pd.notna(policy_path) and policy_path:
                policy_paths.append(policy_path)
        
        # Extract config columns from first row
        first_row = best_config_df.iloc[0]
        config = {}
        for col in first_row.index:
            if col.startswith("config."):
                key = col.replace("config.", "")
                config[key] = first_row[col]
        
        results[combo] = {
            "config_hash": best_config_hash,
            "metric_value": float(best_metric_value),
            "n_seeds": len(best_config_df),
            "model_paths": model_paths,
            "policy_paths": policy_paths,
            "experiment_ids": experiment_ids,
            "config": config,
        }
    
    return results


def print_best_experiments(
    queue_dir: str | Path,
    metric: str = "regret",
    aggregate: str = "min",
    config_filters: Optional[dict[str, str]] = None,
) -> None:
    """
    Print best configurations per feedback combination in a human-readable format.
    
    Shows ALL model paths from the best configuration (all seeds).
    
    Args:
        queue_dir: Path to the file queue directory
        metric: Metric name (e.g., "regret", "epic_distance")
        aggregate: Aggregation method over epochs ("min", "max", "mean", etc.)
        config_filters: Optional dict of config key -> value filters
    """
    results = get_best_config_per_feedback_combo(
        queue_dir, metric, aggregate, config_filters
    )
    
    if not results:
        print("No completed experiments found.")
        return
    
    print(f"\nBest configurations per feedback combination")
    print(f"Metric: {aggregate}({metric})")
    print("=" * 80)
    
    for combo, info in sorted(results.items()):
        print(f"\n{combo}:")
        print(f"  config_hash: {info['config_hash']}")
        print(f"  {metric} (avg over {info['n_seeds']} seeds): {info['metric_value']:.4f}")
        print(f"  experiment_ids: {info['experiment_ids']}")
        print(f"  model_paths ({len(info['model_paths'])}):")
        for path in info['model_paths']:
            print(f"    - {path}")
