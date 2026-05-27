"""Shared utilities for feedback type handling in experiments."""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from mavrl.types import FeedbackType
from mavrl_experiments.file_queue import FileTaskQueue, ExperimentStatus


# Supported aggregation methods
AGGREGATE_METHODS = ["min", "max", "mean", "median", "last", "first"]


def load_experiment_data(
    queue_dir: str | Path,
    include_running: bool = False,
    include_failed: bool = False,
) -> pd.DataFrame:
    """
    Load experiment data from file queue and prepare for visualization.
    
    Args:
        queue_dir: Path to the file queue directory
        include_running: Include experiments that are still running (partial results)
        include_failed: Include failed experiments
        
    Returns:
        DataFrame with flattened config, metric series, and derived fields.
        Metric series are stored as lists in columns like "series.<metric_name>".
    """
    queue = FileTaskQueue(queue_dir)
    
    # Determine which statuses to include
    statuses_to_include = [ExperimentStatus.COMPLETED]
    if include_running:
        statuses_to_include.append(ExperimentStatus.RUNNING)
    if include_failed:
        statuses_to_include.append(ExperimentStatus.FAILED)
    
    records = []
    
    for status in statuses_to_include:
        dir_path = queue.queue_dir / status.value
        if not dir_path.exists():
            continue
            
        for task_file in dir_path.glob("exp_*.json"):
            try:
                data = json.loads(task_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            
            config = data.get("config", {})
            evaluations = data.get("evaluations", [])
            
            # Build base record with experiment metadata
            record = {
                "experiment_id": data.get("id"),
                "config_hash": data.get("config_hash"),
                "seed": data.get("seed"),
                "status": data.get("status"),
                "worker_id": data.get("worker_id"),
                "started_at": data.get("started_at"),
                "completed_at": data.get("completed_at"),
                "error_message": data.get("error_message"),
                "best_model_path": data.get("best_model_path"),
                "best_epoch": data.get("best_epoch"),
                "wandb_run_id": data.get("wandb_run_id"),
            }
            
            # Extract final metrics (regret, mean_rew, etc.)
            final_metrics = data.get("final_metrics", {})
            for metric_name, value in final_metrics.items():
                record[f"final.{metric_name}"] = value
            
            # Flatten config into record
            for k, v in config.items():
                # Convert lists to strings for display
                if isinstance(v, list):
                    record[f"config.{k}"] = str(v)
                else:
                    record[f"config.{k}"] = v
            
            # Extract all metric series across epochs
            if evaluations:
                # Sort by epoch
                sorted_evals = sorted(evaluations, key=lambda x: x.get("epoch", 0))
                
                # Collect all unique metric keys
                all_metric_keys = set()
                for eval_entry in sorted_evals:
                    all_metric_keys.update(eval_entry.get("metrics", {}).keys())
                
                # Extract series for each metric
                epochs = []
                metric_series = {key: [] for key in all_metric_keys}
                
                for eval_entry in sorted_evals:
                    metrics = eval_entry.get("metrics", {})
                    epoch = eval_entry.get("epoch", 0)
                    epochs.append(epoch)
                    
                    for key in all_metric_keys:
                        if key in metrics:
                            val = metrics[key]
                            if val is not None:
                                metric_series[key].append(val)
                
                # Store epochs
                record["epochs"] = epochs
                record["num_epochs_completed"] = len(sorted_evals)
                
                # Store each metric series
                for metric_name, series in metric_series.items():
                    if series:  # Only store non-empty series
                        record[f"series.{metric_name}"] = series
            
            # Derive feedback type combination
            record["feedback_combo"] = derive_feedback_combination(pd.Series(record))
            
            records.append(record)
    
    df = pd.DataFrame(records)
    
    # Sort by experiment_id
    if not df.empty and "experiment_id" in df.columns:
        df = df.sort_values("experiment_id").reset_index(drop=True)
    
    return df


def get_available_metrics(df: pd.DataFrame) -> list[str]:
    """Get list of available metrics from the DataFrame (series and final metrics)."""
    metrics = set()
    for col in df.columns:
        if col.startswith("series."):
            metrics.add(col.replace("series.", ""))
        elif col.startswith("final."):
            metrics.add(col.replace("final.", ""))
    return sorted(metrics)


def get_numeric_columns(df: pd.DataFrame, prefix: str = "config.") -> list[str]:
    """Get numeric columns with a given prefix."""
    cols = []
    for col in df.columns:
        if col.startswith(prefix) and pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def parse_config_filters(filter_args: list[str] | None) -> dict[str, str]:
    """
    Parse config filter arguments into a dictionary.
    
    Args:
        filter_args: List of "key=value" strings
        
    Returns:
        Dictionary mapping config keys to filter values (as strings)
    """
    if not filter_args:
        return {}
    
    filters = {}
    for arg in filter_args:
        if "=" not in arg:
            print(f"Warning: Ignoring invalid filter '{arg}' (expected KEY=VALUE format)")
            continue
        key, value = arg.split("=", 1)
        filters[key.strip()] = value.strip()
    
    return filters


def apply_config_filters(df: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    """
    Apply config filters to the DataFrame.
    
    Args:
        df: DataFrame with experiment data
        filters: Dictionary mapping config keys to filter values
        
    Returns:
        Filtered DataFrame
    """
    if not filters:
        return df
    
    filtered_df = df.copy()
    
    for key, value in filters.items():
        col_name = f"config.{key}" if not key.startswith("config.") else key
        
        if col_name not in filtered_df.columns:
            print(f"Warning: Config key '{key}' not found in data. Available keys: "
                  f"{[c.replace('config.', '') for c in filtered_df.columns if c.startswith('config.')]}")
            continue
        
        # Try to convert value to appropriate type for comparison
        col_dtype = filtered_df[col_name].dtype
        
        try:
            if pd.api.types.is_numeric_dtype(col_dtype):
                # Try float first, then int
                try:
                    typed_value = float(value)
                    # If the column is integer type and value is whole number, use int comparison
                    if pd.api.types.is_integer_dtype(col_dtype) and typed_value == int(typed_value):
                        typed_value = int(typed_value)
                except ValueError:
                    # Handle boolean-like strings for numeric columns (e.g., 0/1 stored as int)
                    if value.lower() in ("true", "yes"):
                        typed_value = 1
                    elif value.lower() in ("false", "no"):
                        typed_value = 0
                    else:
                        print(f"Warning: Cannot convert '{value}' to number for column '{key}'")
                        continue
            elif pd.api.types.is_bool_dtype(col_dtype):
                typed_value = value.lower() in ("true", "1", "yes")
            else:
                # String comparison
                typed_value = value
            
            before_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df[col_name] == typed_value]
            after_count = len(filtered_df)
            
            print(f"Filter {key}={value}: {before_count} -> {after_count} experiments")
            
        except Exception as e:
            print(f"Warning: Error applying filter {key}={value}: {e}")
            continue
    
    return filtered_df


def add_aggregated_metric_column(
    df: pd.DataFrame,
    metric: str,
    aggregate: str,
    at_epoch: int | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Add an aggregated metric column to the DataFrame.
    
    Args:
        df: DataFrame with experiment data
        metric: Metric name (e.g., "regret", "epic_distance")
        aggregate: Aggregation method (ignored if at_epoch is specified)
        at_epoch: If specified, get value at this epoch instead of aggregating
        
    Returns:
        Tuple of (modified DataFrame, column name)
    """
    if at_epoch is not None:
        col_name = f"epoch{at_epoch}_{metric}"
    else:
        col_name = f"{aggregate}_{metric}"
    df = df.copy()
    df[col_name] = compute_aggregated_metric(df, metric, aggregate, at_epoch)
    return df, col_name


def compute_aggregated_metric(
    df: pd.DataFrame,
    metric: str,
    aggregate: str = "min",
    at_epoch: int | None = None,
) -> pd.Series:
    """
    Compute an aggregated metric from the series data or final metrics.
    
    Args:
        df: DataFrame with experiment data (must have series.<metric> or final.<metric> column)
        metric: Metric name (e.g., "regret", "epic_distance")
        aggregate: Aggregation method ("min", "max", "mean", "median", "last", "first")
            Ignored if at_epoch is specified.
        at_epoch: If specified, return the metric value at this specific epoch.
            The epoch number is looked up in the 'epochs' column, not by index.
        
    Returns:
        Series with aggregated metric values
    """
    series_col = f"series.{metric}"
    final_col = f"final.{metric}"
    
    # If we have a series column, aggregate it
    if series_col in df.columns:
        def aggregate_series(row):
            series = row[series_col]
            if not isinstance(series, list) or len(series) == 0:
                return np.nan
            
            arr = np.array(series)
            
            # If at_epoch is specified, look up the value at that epoch
            if at_epoch is not None:
                epochs = row.get("epochs", [])
                if not isinstance(epochs, list) or len(epochs) == 0:
                    return np.nan
                
                try:
                    idx = epochs.index(at_epoch)
                    if idx < len(arr):
                        return arr[idx]
                    return np.nan
                except ValueError:
                    # Epoch not found in epochs list
                    return np.nan
            
            if aggregate == "min":
                return np.nanmin(arr)
            elif aggregate == "max":
                return np.nanmax(arr)
            elif aggregate == "mean":
                return np.nanmean(arr)
            elif aggregate == "median":
                return np.nanmedian(arr)
            elif aggregate == "last":
                return arr[-1]
            elif aggregate == "first":
                return arr[0]
            else:
                raise ValueError(f"Unknown aggregate method: {aggregate}")
        
        return df.apply(aggregate_series, axis=1)
    
    # If we have a final metric column (single value, not a series), use it directly
    if final_col in df.columns:
        return df[final_col]
    
    return pd.Series([np.nan] * len(df), index=df.index)


def get_feedback_config_key(fb_type: FeedbackType) -> str:
    """Map FeedbackType enum to its corresponding config key for sample count."""
    mapping = {
        FeedbackType.PREF: "config.n_pref_samples",
        FeedbackType.DEMO: "config.n_demo_samples",
        FeedbackType.RATE: "config.n_rating_samples",
        FeedbackType.STOP: "config.n_stop_samples",
    }
    return mapping.get(fb_type, f"config.n_{fb_type.value}_samples")


def derive_feedback_combination(row: pd.Series) -> str:
    """
    Derive feedback combination string from config columns.
    
    Returns a string like "pref+demo" or "pref+rating" based on which
    sample counts are > 0.
    """
    active_types = []
    for fb_type in FeedbackType:
        config_key = get_feedback_config_key(fb_type)
        if config_key in row.index:
            val = row[config_key]
            if pd.notna(val) and val > 0:
                active_types.append(fb_type.value)
    
    if not active_types:
        return "none"
    elif len(active_types) == 1:
        return f"{active_types[0]}_only"
    else:
        return "+".join(sorted(active_types))

