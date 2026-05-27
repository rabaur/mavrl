"""
Command-line interface for managing experiments.

Usage:
    python -m mavrl_experiments.cli status --queue-dir tasks
    python -m mavrl_experiments.cli add-grid feedback_mix --queue-dir tasks
    python -m mavrl_experiments.cli retry-failed --queue-dir tasks
    python -m mavrl_experiments.cli reset-running --queue-dir tasks
    python -m mavrl_experiments.cli export --queue-dir tasks --output results.csv
    python -m mavrl_experiments.cli table-concise --queue-dir tasks --output results.tex
    python -m mavrl_experiments.cli table-exhaustive --queue-dir tasks --output appendix.csv
    python -m mavrl_experiments.cli select-best --queue-dir tasks --metric regret
    
    # Filter tables by config attributes:
    python -m mavrl_experiments.cli table-concise --queue-dir tasks --config-filter use_importance_weights=true
    python -m mavrl_experiments.cli table-exhaustive --queue-dir tasks --config-filter lr=0.0001 batch_size=64
    
    # Purge experiments by config filter (with dry-run preview):
    python -m mavrl_experiments.cli purge --queue-dir tasks --config-filter feedback_type=broken --dry-run
    python -m mavrl_experiments.cli purge --queue-dir tasks --config-filter feedback_type=broken
    python -m mavrl_experiments.cli purge --queue-dir tasks --config-filter feedback_type=broken --status pending
    
    # Purge experiments NOT matching a grid definition:
    python -m mavrl_experiments.cli purge --queue-dir tasks --grid sweep_grid_cliff --dry-run
    python -m mavrl_experiments.cli purge --queue-dir tasks --grid sweep_grid_cliff --status pending
"""

import argparse
import json
import csv
import sys
from pathlib import Path

from mavrl_experiments.config_loader import (
    load_experiment_config,
    list_configs,
    config_root,
)
from mavrl_experiments.file_queue import FileTaskQueue, ExperimentStatus
from mavrl_experiments.select_best import (
    get_best_config_per_feedback_combo,
    print_best_experiments,
)
from mavrl_experiments.utils import (
    AGGREGATE_METHODS,
    load_experiment_data,
    get_available_metrics,
    parse_config_filters,
    apply_config_filters,
    derive_feedback_combination,
)
from mavrl_experiments.table import (
    build_concise_table_df,
    build_exhaustive_table_df,
    export_table,
    cmd_status,
)


def cmd_add_grid(args: argparse.Namespace) -> None:
    """Add experiments from a grid definition to the queue."""
    try:
        module = load_experiment_config(args.grid_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not hasattr(module, "grid"):
        print(f"Error: Config '{args.grid_name}' does not define a 'grid' variable")
        print("The grid module should define: grid = ExperimentGrid(...)")
        sys.exit(1)

    grid = module.grid

    # Show summary
    print(f"\n{grid.summary()}")
    print()
    
    if args.dry_run:
        print("[DRY RUN] No changes made to queue")
        return
    
    # Populate queue
    queue = FileTaskQueue(args.queue_dir)
    result = grid.populate_db(queue)
    
    print(f"Queue updated:")
    print(f"  Inserted: {result['inserted']}")
    print(f"  Skipped (already exist): {result['skipped']}")
    print()


def cmd_retry_failed(args: argparse.Namespace) -> None:
    """Reset failed experiments to pending status."""
    queue = FileTaskQueue(args.queue_dir)
    
    if args.dry_run:
        summary = queue.get_status_summary()
        print(f"[DRY RUN] Would reset {summary['failed']} failed experiments to pending")
        return
    
    count = queue.reset_failed_to_pending()
    print(f"Reset {count} failed experiments to pending")


def cmd_reset_running(args: argparse.Namespace) -> None:
    """Reset running experiments to pending (for recovery after crashes)."""
    queue = FileTaskQueue(args.queue_dir)
    
    if args.dry_run:
        summary = queue.get_status_summary()
        print(f"[DRY RUN] Would reset {summary['running']} running experiments to pending")
        return
    
    count = queue.reset_running_to_pending()
    print(f"Reset {count} running experiments to pending")


def cmd_export(args: argparse.Namespace) -> None:
    """Export completed experiment results to CSV or JSON."""
    queue = FileTaskQueue(args.queue_dir)
    results = queue.export_results()
    
    if not results:
        print("No completed experiments to export")
        return
    
    output_path = Path(args.output)
    
    if output_path.suffix == ".json":
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Exported {len(results)} experiments to {output_path}")
        
    elif output_path.suffix == ".csv":
        # Get all unique keys across all results
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())
        all_keys = sorted(all_keys)
        
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"Exported {len(results)} experiments to {output_path}")
        
    else:
        print(f"Error: Unsupported output format '{output_path.suffix}'")
        print("Supported formats: .json, .csv")
        sys.exit(1)


def cmd_show(args: argparse.Namespace) -> None:
    """Show details of a specific experiment."""
    queue = FileTaskQueue(args.queue_dir)
    exp = queue.get_experiment(args.experiment_id)
    
    if exp is None:
        print(f"Experiment {args.experiment_id} not found")
        sys.exit(1)
    
    print(f"\nExperiment {exp.id}")
    print("=" * 40)
    print(f"  Status: {exp.status.value}")
    print(f"  Config Hash: {exp.config_hash}")
    print(f"  Seed: {exp.seed}")
    print(f"  Worker: {exp.worker_id or 'N/A'}")
    print(f"  Started: {exp.started_at or 'N/A'}")
    print(f"  Completed: {exp.completed_at or 'N/A'}")
    print(f"  Best Model: {exp.best_model_path or 'N/A'}")
    print(f"  WandB Run: {exp.wandb_run_id or 'N/A'}")
    
    print(f"\nConfiguration:")
    for k, v in sorted(exp.config.items()):
        print(f"  {k}: {v}")
    
    if exp.error_message:
        print(f"\nError Message:")
        print(exp.error_message)
    
    # Show evaluations
    evals = queue.get_evaluations(exp.id)
    if evals:
        print(f"\nEvaluations ({len(evals)} epochs):")
        for e in evals[-5:]:  # Show last 5
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in e.metrics.items())
            print(f"  Epoch {e.epoch}: {metrics_str}")
        if len(evals) > 5:
            print(f"  ... and {len(evals) - 5} earlier epochs")
    print()


def cmd_list(args: argparse.Namespace) -> None:
    """List experiments with optional status filter."""
    queue = FileTaskQueue(args.queue_dir)
    
    status = None
    if args.status:
        try:
            status = ExperimentStatus(args.status)
        except ValueError:
            print(f"Invalid status: {args.status}")
            print(f"Valid options: {', '.join(s.value for s in ExperimentStatus)}")
            sys.exit(1)
    
    experiments = queue.get_all_experiments(status)
    
    if not experiments:
        print("No experiments found")
        return
    
    print(f"\n{'ID':>6} {'Status':>10} {'Seed':>5} {'Config Hash':>18} {'Worker':<20}")
    print("-" * 65)
    
    for exp in experiments[:args.limit]:
        worker = (exp.worker_id or "")[:20]
        print(f"{exp.id:>6} {exp.status.value:>10} {exp.seed:>5} {exp.config_hash:>18} {worker:<20}")
    
    if len(experiments) > args.limit:
        print(f"... and {len(experiments) - args.limit} more (use --limit to show more)")
    print()


def cmd_table_concise(args: argparse.Namespace) -> None:
    """Generate concise summary table with best results per feedback combination."""
    print(f"Loading experiments from: {args.queue_dir}")
    df = load_experiment_data(args.queue_dir)
    
    if df.empty:
        print("No experiments found!")
        return
    
    print(f"Loaded {len(df)} experiments")
    
    # Apply config filters if specified
    if args.config_filter:
        filters = parse_config_filters(args.config_filter)
        df = apply_config_filters(df, filters)
        if df.empty:
            print("No experiments remaining after filtering!")
            return
    
    # Check if metric is available
    available = get_available_metrics(df)
    if args.metric not in available:
        print(f"Warning: Metric '{args.metric}' not found. Available metrics: {', '.join(available)}")
        if available:
            args.metric = available[0]
            print(f"Using '{args.metric}' instead")
        else:
            print("No metrics available!")
            return
    
    # Show diagnostic info about data per environment
    if args.verbose:
        env_col = "config.env_id"
        series_col = f"series.{args.metric}"
        if env_col in df.columns:
            print(f"\nDiagnostic: experiments per environment with '{args.metric}' data:")
            for env in sorted(df[env_col].unique()):
                env_df = df[df[env_col] == env]
                has_metric = env_df[series_col].notna().sum() if series_col in env_df.columns else 0
                print(f"  {env}: {len(env_df)} total, {has_metric} with {args.metric} metric")
            
            # Show feedback combinations per environment
            print(f"\nDiagnostic: feedback combinations per environment:")
            # Compute feedback_combo for diagnosis
            df_diag = df.copy()
            df_diag["feedback_combo"] = df_diag.apply(derive_feedback_combination, axis=1)
            for env in sorted(df[env_col].unique()):
                env_df = df_diag[df_diag[env_col] == env]
                combos = env_df["feedback_combo"].value_counts()
                print(f"  {env}:")
                for combo, count in combos.items():
                    print(f"    {combo}: {count}")
    
    # Build the table
    table_df = build_concise_table_df(
        df,
        metric=args.metric,
        aggregate=args.aggregate,
        precision=args.precision,
    )
    
    if table_df.empty:
        print("No data to generate table")
        return
    
    # Print summary
    print(f"\nConcise table: best {args.aggregate} {args.metric} per feedback type")
    print("-" * 60)
    print(table_df.to_string())
    print()
    
    # Export if output specified
    if args.output:
        export_table(
            table_df,
            Path(args.output),
            table_type="concise",
            metric=args.metric,
            aggregate=args.aggregate,
        )


def cmd_table_exhaustive(args: argparse.Namespace) -> None:
    """Generate exhaustive table with all configurations averaged over seeds."""
    print(f"Loading experiments from: {args.queue_dir}")
    df = load_experiment_data(args.queue_dir)
    
    if df.empty:
        print("No experiments found!")
        return
    
    print(f"Loaded {len(df)} experiments")
    
    # Apply config filters if specified
    if args.config_filter:
        filters = parse_config_filters(args.config_filter)
        df = apply_config_filters(df, filters)
        if df.empty:
            print("No experiments remaining after filtering!")
            return
    
    # Check if metric is available
    available = get_available_metrics(df)
    if args.metric not in available:
        print(f"Warning: Metric '{args.metric}' not found. Available metrics: {', '.join(available)}")
        if available:
            args.metric = available[0]
            print(f"Using '{args.metric}' instead")
        else:
            print("No metrics available!")
            return
    
    # Build the table
    table_df = build_exhaustive_table_df(
        df,
        metric=args.metric,
        aggregate=args.aggregate,
        precision=args.precision,
    )
    
    if table_df.empty:
        print("No data to generate table")
        return
    
    # Print summary
    n_configs = len(table_df)
    print(f"\nExhaustive table: {args.aggregate} {args.metric} for {n_configs} configurations")
    print("-" * 80)
    
    # Show first few rows as preview
    preview_rows = min(10, n_configs)
    print(table_df.head(preview_rows).to_string(index=False))
    if n_configs > preview_rows:
        print(f"... and {n_configs - preview_rows} more rows")
    print()
    
    # Export if output specified
    if args.output:
        export_table(
            table_df,
            Path(args.output),
            table_type="exhaustive",
            metric=args.metric,
            aggregate=args.aggregate,
        )


def cmd_select_best(args: argparse.Namespace) -> None:
    """Show best experiment IDs and model paths per feedback combination."""
    print(f"Loading experiments from: {args.queue_dir}")
    
    # Parse config filters if provided
    config_filters = None
    if args.config_filter:
        config_filters = parse_config_filters(args.config_filter)
        print(f"Config filters: {config_filters}")
    
    # Use the helper function to print results
    print_best_experiments(
        queue_dir=args.queue_dir,
        metric=args.metric,
        aggregate=args.aggregate,
        config_filters=config_filters,
    )
    
    # Also output in a format suitable for copying to transfer grid
    results = get_best_config_per_feedback_combo(
        queue_dir=args.queue_dir,
        metric=args.metric,
        aggregate=args.aggregate,
        config_filters=config_filters,
    )
    
    if results:
        print("\n" + "=" * 80)
        print("Copy the following to your transfer grid config:")
        print("=" * 80)
        print()
        
        # Output feedback_combo values to sweep over
        combos = sorted(results.keys())
        print(f'grid.add("feedback_combo", {combos})')
        print()
        
        # Output conditional reward_model_path for each feedback combo
        for combo in combos:
            info = results[combo]
            paths = info.get("model_paths", [])
            if paths:
                # Create a Python-safe variable name for the condition
                combo_safe = combo.replace("+", "_").replace("-", "_")
                print(f'has_{combo_safe} = lambda c: c.get("feedback_combo") == "{combo}"')
                print(f'grid.add_conditional("reward_model_path", [')
                for path in paths:
                    print(f'    "{path}",')
                print(f'], condition=has_{combo_safe})')
                print()


def cmd_purge(args: argparse.Namespace) -> None:
    """Purge experiments matching config filters or not matching a grid."""
    queue = FileTaskQueue(args.queue_dir)
    
    # Validate that either --config-filter or --grid is provided (but not both)
    has_config_filter = args.config_filter is not None
    has_grid = args.grid is not None
    
    if not has_config_filter and not has_grid:
        print("Error: Must provide either --config-filter or --grid")
        print("Examples:")
        print("  --config-filter feedback_type=broken_value")
        print("  --grid sweep_grid_cliff")
        sys.exit(1)
    
    if has_config_filter and has_grid:
        print("Error: Cannot use both --config-filter and --grid at the same time")
        sys.exit(1)
    
    # Parse status filter if provided
    status = None
    if args.status:
        try:
            status = ExperimentStatus(args.status)
        except ValueError:
            print(f"Invalid status: {args.status}")
            print(f"Valid options: {', '.join(s.value for s in ExperimentStatus)}")
            sys.exit(1)
    
    if has_grid:
        # Grid-based purge: delete experiments NOT matching the grid
        count, matched = _purge_by_grid(
            queue=queue,
            grid_name=args.grid,
            status=status,
            dry_run=args.dry_run,
        )
    else:
        # Config-filter based purge: delete experiments matching the filters
        filters = parse_config_filters(args.config_filter)
        if not filters:
            print("Error: No valid filters provided")
            sys.exit(1)
        
        # Show what we're filtering by
        print(f"\nPurging experiments matching:")
        for key, value in filters.items():
            print(f"  {key} = {value}")
        if status:
            print(f"  status = {status.value}")
        print()
        
        # Run the purge (dry_run if requested)
        count, matched = queue.delete_experiments_by_filter(
            config_filter=filters,
            status=status,
            dry_run=args.dry_run,
        )
    
    if count == 0:
        print("No experiments matched the purge criteria")
        return
    
    # Show matched experiments
    if args.dry_run:
        print(f"[DRY RUN] Would delete {count} experiments:")
    else:
        print(f"Deleted {count} experiments:")
    
    # Group by status for display
    by_status = {}
    for exp in matched:
        exp_status = exp.get("status", "unknown")
        if exp_status not in by_status:
            by_status[exp_status] = []
        by_status[exp_status].append(exp)
    
    for exp_status, exps in sorted(by_status.items()):
        print(f"\n  {exp_status}: {len(exps)} experiments")
        # Show a few examples
        for exp in exps[:5]:
            config_parts = [f"{k}={v}" for k, v in exp.items() if k.startswith("config.")]
            config_str = ", ".join(config_parts) if config_parts else ""
            hash_str = exp.get('config_hash', '')[:8]
            print(f"    - ID {exp['id']} (seed {exp['seed']}, hash {hash_str}): {config_str}")
        if len(exps) > 5:
            print(f"    ... and {len(exps) - 5} more")
    
    print()
    if args.dry_run:
        print("Run without --dry-run to actually delete these experiments")


def _purge_by_grid(
    queue: FileTaskQueue,
    grid_name: str,
    status: ExperimentStatus | None,
    dry_run: bool,
) -> tuple[int, list[dict]]:
    """
    Purge experiments that don't match any configuration in the given grid.
    
    Returns (count, matched_experiments) tuple.
    """
    from mavrl_experiments.file_queue import compute_config_hash

    try:
        module = load_experiment_config(grid_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not hasattr(module, "grid"):
        print(f"Error: Config '{grid_name}' does not define a 'grid' variable")
        print("The grid module should define: grid = ExperimentGrid(...)")
        sys.exit(1)

    grid = module.grid
    
    # Generate all valid configurations and extract unique hashes
    print(f"\nLoading grid '{grid_name}'...")
    try:
        all_configs = grid.generate_configs()
        valid_hashes = {compute_config_hash(config) for config, seed in all_configs}
    except Exception as e:
        print(f"Error generating configs from grid: {e}")
        sys.exit(1)
    
    print(f"Grid has {len(valid_hashes)} unique configurations")
    
    # Show what we're doing
    print(f"\nPurging experiments NOT matching grid '{grid_name}'")
    if status:
        print(f"  (only {status.value} experiments)")
    print()
    
    # Run the purge
    return queue.delete_experiments_not_in_hashes(
        valid_hashes=valid_hashes,
        status=status,
        dry_run=dry_run,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="python -m mavrl_experiments.cli",
        description="Manage distributed experiments"
    )
    parser.add_argument(
        "--queue-dir",
        type=str,
        default="tasks",
        help="Path to the task queue directory"
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # status command
    status_parser = subparsers.add_parser("status", help="Show queue status")
    status_parser.add_argument("-v", "--verbose", action="store_true", help="Show more details")
    status_parser.set_defaults(func=cmd_status)
    
    # add-grid command
    add_parser = subparsers.add_parser("add-grid", help="Add experiments from a grid definition")
    add_parser.add_argument("grid_name", type=str, help="Name of the grid module (e.g., 'feedback_mix')")
    add_parser.add_argument("--dry-run", action="store_true", help="Show what would be added without modifying queue")
    add_parser.set_defaults(func=cmd_add_grid)
    
    # retry-failed command
    retry_parser = subparsers.add_parser("retry-failed", help="Reset failed experiments to pending")
    retry_parser.add_argument("--dry-run", action="store_true", help="Show what would be reset")
    retry_parser.set_defaults(func=cmd_retry_failed)
    
    # reset-running command
    reset_parser = subparsers.add_parser("reset-running", help="Reset running experiments to pending")
    reset_parser.add_argument("--dry-run", action="store_true", help="Show what would be reset")
    reset_parser.set_defaults(func=cmd_reset_running)
    
    # export command
    export_parser = subparsers.add_parser("export", help="Export results to CSV or JSON")
    export_parser.add_argument("-o", "--output", type=str, default="results.csv", help="Output file path")
    export_parser.set_defaults(func=cmd_export)
    
    # show command
    show_parser = subparsers.add_parser("show", help="Show details of a specific experiment")
    show_parser.add_argument("experiment_id", type=int, help="Experiment ID to show")
    show_parser.set_defaults(func=cmd_show)
    
    # list command
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--status", type=str, help="Filter by status")
    list_parser.add_argument("--limit", type=int, default=50, help="Maximum number to show")
    list_parser.set_defaults(func=cmd_list)
    
    # table-concise command
    table_concise_parser = subparsers.add_parser(
        "table-concise",
        help="Generate concise summary table with best results per feedback combination"
    )
    table_concise_parser.add_argument(
        "--metric", type=str, default="regret",
        help="Metric to summarize (default: regret)"
    )
    table_concise_parser.add_argument(
        "--aggregate", type=str, default="min", choices=AGGREGATE_METHODS,
        help="Aggregation method over epochs (default: min)"
    )
    table_concise_parser.add_argument(
        "--precision", type=int, default=1,
        help="Decimal places for metric values (default: 1)"
    )
    table_concise_parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file path (.csv or .tex)"
    )
    table_concise_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show diagnostic information about data per environment"
    )
    table_concise_parser.add_argument(
        "--config-filter",
        type=str,
        nargs="+",
        default=None,
        metavar="KEY=VALUE",
        help="Filter by config attributes (e.g., --config-filter use_importance_weights=true lr=0.0001)",
    )
    table_concise_parser.set_defaults(func=cmd_table_concise)
    
    # table-exhaustive command
    table_exhaustive_parser = subparsers.add_parser(
        "table-exhaustive",
        help="Generate exhaustive table with all configurations averaged over seeds"
    )
    table_exhaustive_parser.add_argument(
        "--metric", type=str, default="regret",
        help="Metric to summarize (default: regret)"
    )
    table_exhaustive_parser.add_argument(
        "--aggregate", type=str, default="min", choices=AGGREGATE_METHODS,
        help="Aggregation method over epochs (default: min)"
    )
    table_exhaustive_parser.add_argument(
        "--precision", type=int, default=4,
        help="Decimal places for metric values (default: 4)"
    )
    table_exhaustive_parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file path (.csv or .tex)"
    )
    table_exhaustive_parser.add_argument(
        "--config-filter",
        type=str,
        nargs="+",
        default=None,
        metavar="KEY=VALUE",
        help="Filter by config attributes (e.g., --config-filter use_importance_weights=true lr=0.0001)",
    )
    table_exhaustive_parser.set_defaults(func=cmd_table_exhaustive)
    
    # select-best command
    select_best_parser = subparsers.add_parser(
        "select-best",
        help="Show best experiment IDs and model paths per feedback combination"
    )
    select_best_parser.add_argument(
        "--metric", type=str, default="regret",
        help="Metric to optimize (default: regret)"
    )
    select_best_parser.add_argument(
        "--aggregate", type=str, default="min", choices=AGGREGATE_METHODS,
        help="Aggregation method over epochs (default: min)"
    )
    select_best_parser.add_argument(
        "--env-id", type=str, default=None,
        help="Filter by environment ID (optional)"
    )
    select_best_parser.add_argument(
        "--config-filter",
        type=str,
        nargs="+",
        default=None,
        metavar="KEY=VALUE",
        help="Filter by config attributes (e.g., --config-filter use_importance_weights=true)",
    )
    select_best_parser.set_defaults(func=cmd_select_best)
    
    # purge command
    purge_parser = subparsers.add_parser(
        "purge",
        help="Delete experiments matching config filters OR not matching a grid"
    )
    purge_parser.add_argument(
        "--config-filter",
        type=str,
        nargs="+",
        default=None,
        metavar="KEY=VALUE",
        help="Purge experiments MATCHING these config attributes (e.g., --config-filter feedback_type=broken)",
    )
    purge_parser.add_argument(
        "--grid",
        type=str,
        default=None,
        metavar="GRID_NAME",
        help="Purge experiments NOT matching this grid (e.g., --grid sweep_grid_cliff)",
    )
    purge_parser.add_argument(
        "--status",
        type=str,
        default=None,
        help="Only purge experiments with this status (pending, running, completed, failed)",
    )
    purge_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    purge_parser.set_defaults(func=cmd_purge)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

