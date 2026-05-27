"""
Worker process for distributed experiment execution.

Each worker:
1. Claims a pending task from the queue
2. Runs the experiment with wandb logging
3. Logs evaluation metrics to the queue
4. Saves the best model to scratch space
5. Repeats until no pending tasks remain
"""

import os
import sys
import socket
import traceback
import argparse
from pathlib import Path
from typing import Optional

import wandb

from mavrl_experiments.file_queue import FileTaskQueue


def get_worker_id() -> str:
    """Generate a unique worker identifier."""
    hostname = socket.gethostname()
    pid = os.getpid()
    return f"{hostname}-{pid}"


def get_scratch_path() -> Path:
    """Get the scratch directory path from environment or default."""
    scratch = os.environ.get("SCRATCH", os.environ.get("HOME", "/tmp"))
    return Path(scratch)


def get_run_directory(project_name: str) -> Path:
    """
    Get the run directory for this experiment batch.
    
    The run directory is based on the project name and the date when the
    first experiment in this worker started. This groups all experiments
    from the same run together.
    
    Returns:
        Path like $SCRATCH/mavrl/<wandb_project>_<date>/
    """
    from datetime import datetime
    
    scratch = get_scratch_path()
    date_str = datetime.now().strftime("%Y-%m-%d")
    run_dir = scratch / "mavrl" / f"{project_name}"
    return run_dir


def get_model_save_dir(run_dir: Path, config_hash: str, seed: int) -> Path:
    """Get the directory to save models and policies for an experiment.
    
    Uses config_hash + seed to uniquely identify models, making them
    recoverable even if experiment IDs change (e.g., after purging).
    """
    model_dir = run_dir / "models" / f"{config_hash}_seed_{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


class QueueEvalCallback:
    """
    Callback to log evaluation metrics to the queue during training.
    
    This is passed to run_experiment to enable periodic metric logging.
    """
    
    def __init__(self, queue: FileTaskQueue, experiment_id: int):
        self.queue = queue
        self.experiment_id = experiment_id
        self.best_regret = float("inf")
        self.best_epoch = 0
    
    def __call__(self, epoch: int, metrics: dict) -> None:
        """Log evaluation metrics to the queue."""
        # Extract key metrics for storage (subset of wandb metrics)
        queue_metrics = {}
        for key, value in metrics.items():
            # Only store numeric metrics, skip things like GIFs
            if isinstance(value, (int, float)) and key.startswith("eval/"):
                # Remove eval/ prefix for cleaner storage
                clean_key = key.replace("eval/", "")
                queue_metrics[clean_key] = value
        
        if queue_metrics:
            self.queue.log_evaluation(self.experiment_id, epoch, queue_metrics)
        
        # Track best model based on regret
        regret = metrics.get("eval/regret")
        if regret is not None and regret < self.best_regret:
            self.best_regret = regret
            self.best_epoch = epoch


def run_worker(
    queue_dir: str,
    max_experiments: Optional[int] = None,
    dry_run: bool = False,
    verbose: bool = True,
    wandb_project: Optional[str] = None,
    wandb_log_dir: Optional[str] = None,
) -> dict:
    """
    Run the worker loop to process experiments.
    
    Args:
        queue_dir: Path to the task queue directory.
        max_experiments: Maximum number of experiments to run (None for unlimited).
        dry_run: If True, claim tasks but don't actually run them.
        verbose: Print progress information.
        wandb_project: Override wandb project name for all experiments (overrides config).
        wandb_log_dir: Override wandb log directory for all experiments (overrides config).
        
    Returns:
        Dict with counts: {"completed": N, "failed": M}
    """
    # Import here to avoid circular imports and ensure train.py is available
    from train import run_experiment, get_default_args
    
    queue = FileTaskQueue(queue_dir)
    worker_id = get_worker_id()
    
    completed = 0
    failed = 0
    
    if verbose:
        print(f"Worker {worker_id} starting...")
        status = queue.get_status_summary()
        print(f"  Queue status: {status}")
    
    while True:
        # Check if we've hit the limit
        if max_experiments is not None and (completed + failed) >= max_experiments:
            if verbose:
                print(f"Reached max experiments limit ({max_experiments})")
            break
        
        # Try to claim a task
        experiment = queue.claim_pending_task(worker_id)
        
        if experiment is None:
            if verbose:
                print(f"No pending tasks remaining. Worker shutting down.")
            break
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Claimed experiment {experiment.id} (seed={experiment.seed})")
            print(f"Config hash: {experiment.config_hash}")
            print(f"{'='*60}")
        
        if dry_run:
            if verbose:
                print(f"  [DRY RUN] Would run: {experiment.config}")
            queue.mark_completed(experiment.id)
            completed += 1
            continue
        
        try:
            # Determine effective wandb project (CLI takes precedence over config)
            effective_wandb_project = wandb_project or experiment.config.get("wandb_project")
            if not effective_wandb_project:
                raise ValueError(
                    f"Experiment {experiment.id} has no wandb_project set. "
                    "Set via --wandb-project CLI arg or in experiment config."
                )
            
            # Set up save directory for models and policies
            run_dir = get_run_directory(effective_wandb_project)
            model_dir = get_model_save_dir(run_dir, experiment.config_hash, experiment.seed)
            
            # Create the queue callback for logging evaluations
            queue_callback = QueueEvalCallback(queue, experiment.id)
            
            # Merge experiment config with defaults
            args = get_default_args()
            
            # Apply experiment configuration
            for key, value in experiment.config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
            
            # Override seed from experiment
            args.seed = experiment.seed
            
            # Enable wandb logging if not explicitly disabled in config
            if "log_wandb" not in experiment.config:
                args.log_wandb = True
            
            # Override wandb settings from worker CLI args (takes precedence over config)
            if wandb_project is not None:
                args.wandb_project = wandb_project
            if wandb_log_dir is not None:
                args.wandb_log_dir = wandb_log_dir
            
            # Set wandb run name to include experiment ID for tracking
            args.wandb_run_name = f"exp-{experiment.id}-seed-{experiment.seed}"
            
            # Set model save directory and behavior via args
            args.model_save_dir = str(model_dir)
            args.save_behavior = "best"  # Workers save only the best model
            
            # Run the experiment
            result = run_experiment(
                args,
                db_callback=queue_callback,
            )
            
            # Mark as completed
            best_model_path = result.get("best_model_path")
            best_policy_path = result.get("best_policy_path")
            wandb_run_id = result.get("wandb_run_id")
            final_metrics = result.get("final_metrics")
            best_epoch = result.get("best_epoch")
            
            queue.mark_completed(
                experiment.id,
                best_model_path=str(best_model_path) if best_model_path else None,
                best_policy_path=str(best_policy_path) if best_policy_path else None,
                wandb_run_id=wandb_run_id,
                final_metrics=final_metrics,
                best_epoch=best_epoch
            )
            
            completed += 1
            if verbose:
                print(f"Experiment {experiment.id} completed successfully.")
                
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            queue.mark_failed(experiment.id, error_msg)
            failed += 1
            
            # Clean up wandb run if it was left active due to the exception
            if wandb.run is not None:
                if verbose:
                    print(f"Cleaning up wandb run '{wandb.run.id}' after failure")
                wandb.finish(exit_code=1)
            
            if verbose:
                print(f"Experiment {experiment.id} failed: {e}")
                traceback.print_exc()
    
    if verbose:
        print(f"\nWorker {worker_id} finished.")
        print(f"  Completed: {completed}")
        print(f"  Failed: {failed}")
    
    return {"completed": completed, "failed": failed}


def main():
    """CLI entry point for the worker."""
    parser = argparse.ArgumentParser(
        description="Run a worker to process experiments from the queue"
    )
    parser.add_argument(
        "--queue-dir",
        type=str,
        default="tasks",
        help="Path to the task queue directory"
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Maximum number of experiments to run (default: unlimited)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Claim tasks but don't actually run them"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Override wandb project name for all experiments (overrides config)"
    )
    parser.add_argument(
        "--wandb-log-dir",
        type=str,
        default=None,
        help="Override wandb log directory for all experiments (overrides config)"
    )
    
    args = parser.parse_args()
    
    result = run_worker(
        queue_dir=args.queue_dir,
        max_experiments=args.max_experiments,
        dry_run=args.dry_run,
        verbose=not args.quiet,
        wandb_project=args.wandb_project,
        wandb_log_dir=args.wandb_log_dir,
    )
    
    # Exit with error code if any experiments failed
    if result["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

