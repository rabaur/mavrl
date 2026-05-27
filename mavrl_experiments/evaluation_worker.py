"""
Worker process for distributed reward model evaluation.

Each worker:
1. Claims a pending task from the evaluation queue
2. Runs the evaluation experiment (mavrl, average, ground_truth, or imitation)
3. Logs evaluation metrics (regret, mean_reward) to the queue
4. Repeats until no pending tasks remain

Supports {seed} templates in model paths, resolved at runtime from the
experiment seed. This allows grids to specify per-seed model checkpoints
without duplicating configurations.

Usage:
    python -m mavrl_experiments.evaluation_worker --queue-dir tasks_eval
"""

import json
import os
import socket
import sys
import traceback
import argparse
from typing import Optional

import wandb

from mavrl_experiments.file_queue import FileTaskQueue
from mavrl.utils.retrain_ppo_cli import merge_retrain_ppo_defaults_into_args


def get_worker_id() -> str:
    """Generate a unique worker identifier."""
    hostname = socket.gethostname()
    pid = os.getpid()
    return f"{hostname}-{pid}"


def _resolve_seed_templates(value, seed: int):
    """Replace ``{seed}`` placeholders in strings or lists of strings."""
    if isinstance(value, str):
        return value.replace("{seed}", str(seed))
    if isinstance(value, list):
        return [_resolve_seed_templates(v, seed) for v in value]
    return value


class EvaluationQueueCallback:
    """Callback to log evaluation metrics to the queue.

    Evaluation experiments produce a single set of metrics (no epochs), so
    results are logged once at epoch 0.
    """

    def __init__(self, queue: FileTaskQueue, experiment_id: int):
        self.queue = queue
        self.experiment_id = experiment_id

    def log_results(self, metrics: dict) -> None:
        self.queue.log_evaluation(self.experiment_id, epoch=0, metrics=metrics)


def _build_args(config: dict, seed: int, experiment_id: int = 0, queue_dir: str = ""):
    """Map a queue config dict to an args namespace for ``run_evaluation``."""

    class EvalArgs:
        pass

    args = EvalArgs()

    # --- Environment ---
    env_params = {}
    for key, value in config.items():
        if key.startswith("env_params."):
            env_params[key[len("env_params."):]] = value
    args.env_id = config.get("env_id")
    args.env_params = json.dumps(env_params) if env_params else "{}"
    args.optimal_policy_path = _resolve_seed_templates(
        config.get("optimal_policy_path"), seed,
    )

    # --- Mode and checkpoints ---
    mode = config.get("mode", "reward_model")
    if mode == "reward_model":
        mode = "mavrl"
    args.mode = mode

    fb_model_path = _resolve_seed_templates(config.get("fb_model_path"), seed)
    checkpoint_paths = _resolve_seed_templates(config.get("checkpoint_paths"), seed)

    if mode == "mavrl":
        args.checkpoint_path = fb_model_path
        args.checkpoint_paths = None
    elif mode == "average":
        args.checkpoint_path = None
        args.checkpoint_paths = checkpoint_paths
    elif mode == "ground_truth":
        args.checkpoint_path = None
        args.checkpoint_paths = None
    elif mode == "imitation":
        args.checkpoint_path = fb_model_path
        args.checkpoint_paths = None
    else:
        args.checkpoint_path = fb_model_path
        args.checkpoint_paths = checkpoint_paths

    # --- Evaluation parameters ---
    args.seed = seed
    args.num_samples = config.get("num_samples", 100)
    args.max_num_steps = config.get("max_num_steps", 1000)
    args.gamma = config.get("gamma", 0.99)
    # --- Retrain PPO hparams (forward all retrain_* keys from config) ---
    for key, value in config.items():
        if key.startswith("retrain_"):
            setattr(args, key, value)
    if not hasattr(args, "retrain_verbose"):
        args.retrain_verbose = 0
    if not hasattr(args, "retrain_pbar"):
        args.retrain_pbar = not config.get("no_progress_bar", True)

    # --- Model architecture ---
    args.hidden_sizes = config.get("hidden_sizes", [256, 256])
    args.reward_domain = config.get("reward_domain", "sa")

    # --- Feature transforms ---
    args.act_transform = config.get("act_transform", None)
    args.obs_transform = config.get("obs_transform", None)

    # --- Regret ---
    args.use_avg_state_regret = config.get("use_avg_state_regret", False)
    args.regret_softmax_beta = config.get("regret_softmax_beta", None)

    # --- Weights & Biases ---
    args.log_wandb = config.get("log_wandb", False)
    args.wandb_project = config.get("wandb_project", "mavrl-eval")
    args.wandb_entity = config.get("wandb_entity", None)
    args.wandb_name = config.get("wandb_name", None)

    # --- Output ---
    args.output = None
    if queue_dir:
        from pathlib import Path
        curves_dir = Path(queue_dir) / "training_curves"
        curves_dir.mkdir(parents=True, exist_ok=True)
        args.training_curve_output = str(
            curves_dir / f"exp_{experiment_id:06d}_seed_{seed}.csv"
        )
    else:
        args.training_curve_output = None

    merge_retrain_ppo_defaults_into_args(args)

    return args


def run_evaluation_worker(
    queue_dir: str,
    max_experiments: Optional[int] = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run the worker loop to process evaluation experiments.

    Args:
        queue_dir: Path to the task queue directory.
        max_experiments: Maximum number of experiments to run (None for unlimited).
        dry_run: If True, claim tasks but don't actually run them.
        verbose: Print progress information.

    Returns:
        Dict with counts: {"completed": N, "failed": M}
    """
    from evaluate_reward_model import run_evaluation

    queue = FileTaskQueue(queue_dir)
    worker_id = get_worker_id()

    completed = 0
    failed = 0

    if verbose:
        print(f"Evaluation Worker {worker_id} starting...")
        status = queue.get_status_summary()
        print(f"  Queue status: {status}")

    while True:
        if max_experiments is not None and (completed + failed) >= max_experiments:
            if verbose:
                print(f"Reached max experiments limit ({max_experiments})")
            break

        experiment = queue.claim_pending_task(worker_id)

        if experiment is None:
            if verbose:
                print("No pending tasks remaining. Worker shutting down.")
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
            args = _build_args(
                experiment.config, experiment.seed,
                experiment_id=experiment.id, queue_dir=queue_dir,
            )

            if verbose:
                print(f"  env_id: {args.env_id}")
                print(f"  env_params: {args.env_params}")
                print(f"  mode: {args.mode}")
                print(f"  checkpoint_path: {args.checkpoint_path}")
                print(f"  checkpoint_paths: {args.checkpoint_paths}")
                print(f"  optimal_policy_path: {args.optimal_policy_path}")

            results = run_evaluation(args)

            callback = EvaluationQueueCallback(queue, experiment.id)
            callback.log_results(results)
            queue.mark_completed(experiment.id)

            completed += 1
            if verbose:
                print(f"Experiment {experiment.id} completed:")
                print(f"  Regret: {results['regret']:.4f}")
                mean_rew = results.get("mean_reward")
                if mean_rew is not None:
                    print(f"  Mean reward: {mean_rew:.4f}")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            queue.mark_failed(experiment.id, error_msg)
            failed += 1

            if wandb.run is not None:
                wandb.finish(exit_code=1)

            if verbose:
                print(f"Experiment {experiment.id} failed: {e}")
                traceback.print_exc()

    if verbose:
        print(f"\nEvaluation Worker {worker_id} finished.")
        print(f"  Completed: {completed}")
        print(f"  Failed: {failed}")

    return {"completed": completed, "failed": failed}


def main():
    """CLI entry point for the evaluation worker."""
    parser = argparse.ArgumentParser(
        description="Run a worker to process evaluation experiments from the queue"
    )
    parser.add_argument(
        "--queue-dir",
        type=str,
        default="tasks_eval",
        help="Path to the task queue directory (default: tasks_eval)",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Maximum number of experiments to run (default: unlimited)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Claim tasks but don't actually run them",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    result = run_evaluation_worker(
        queue_dir=args.queue_dir,
        max_experiments=args.max_experiments,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )

    if result["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
