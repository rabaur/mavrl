import argparse
import wandb
import torch
import datetime
import math
import time

from typing import Optional, Callable, Any
from pathlib import Path
from numbers import Number

import stable_baselines3 as sb3

from mavrl.envs.env_types import TabularEnv
from mavrl.envs.make_env import make_env
from mavrl.utils.reproducibility import seed_everything
from mavrl.utils.logging import console_log_eval_metrics
from mavrl.types import FeedbackType
from mavrl.utils.gym import get_obs_dim, get_act_dim
from mavrl.utils.policies import TabularQValueModel, DQNQValueModel
from mavrl.utils.feature_transforms import get_act_transform, get_obs_transform
from mavrl.data.make_dataset import make_dataset
from mavrl.utils.retrain_ppo_cli import add_retrain_ppo_arguments
from mavrl.utils.train_utils import (
    get_git_commit_hash,
    validate_args,
    train_epoch,
    EarlyStopMonitor,
    run_validation,
    run_test,
    compute_total_val_loss,
    visualize_epoch,
    save_model_checkpoint,
    load_model_checkpoint,
    wandb_run,
    _create_policies,
    _visualize_dataset_occupancy,
    _create_model,
    _compute_importance_weights,
    _create_optimal_policy,
    _print_training_info
)


def get_default_args() -> argparse.Namespace:
    """
    Get an argparse.Namespace with all default argument values.
    
    This is useful for programmatic usage where you want to start with
    defaults and only override specific values.
    
    Returns:
        argparse.Namespace with default values for all arguments.
    """
    parser = create_parser()
    return parser.parse_args([])


def run_experiment(
    args: argparse.Namespace,
    db_callback: Optional[Callable[[int, dict], None]] = None,
) -> dict[str, Any]:
    """
    Run a single experiment with the given configuration.
    
    This is the main entry point for programmatic experiment execution.
    It supports optional database callbacks for logging evaluations.
    
    Model saving is controlled via args:
        - args.model_save_dir: Directory to save models and policies (None to disable)
        - args.save_behavior: "best" (save only best model) or "all" (save at every eval)
    
    Args:
        args: Namespace containing all experiment configuration.
        db_callback: Optional callback function called at each evaluation epoch.
                    Signature: db_callback(epoch: int, metrics: dict) -> None
                       
    Returns:
        Dict containing:
            - "best_model_path": Path to saved best reward model (or None)
            - "best_policy_path": Path to saved best estimated policy trained on learned reward (or None)
            - "wandb_run_id": The wandb run ID (or None if not logging)
            - "final_metrics": Dict of final evaluation metrics
    """

    # Hard-code device to cpu
    device = "cpu"

    print(f"\n[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] Starting run (seed={args.seed})")

    # Reproducibility
    seed_everything(args.seed)
    
    # Get git commit hash for reproducibility
    git_commit_hash = get_git_commit_hash()
    if git_commit_hash:
        print(f"Git commit hash: {git_commit_hash}")
    else:
        print("Warning: Could not determine git commit hash (not in a git repo or git not available)")
    
    # Initialize result tracking
    result = {
        "best_model_path": None,
        "best_policy_path": None,
        "best_epoch": None,
        "wandb_run_id": None,
        "final_metrics": {},
        "git_commit_hash": git_commit_hash,
        "early_stop_epoch": None,
        "early_stop_reason": None,
    }
    
    # Prepare feedback configuration
    feedback_config = {
        FeedbackType.PREF: args.n_pref_samples,
        FeedbackType.DEMO: args.n_demo_samples,
        FeedbackType.RATE: args.n_rating_samples,
        FeedbackType.STOP: args.n_stop_samples,
    }
    
    # Validate arguments
    active_feedback_types = validate_args(args, feedback_config)
    
    # Prepare wandb config
    wandb_config = vars(args).copy()
    if git_commit_hash:
        wandb_config["git_commit_hash"] = git_commit_hash
    
    with wandb_run(args, wandb_config) as run:
        if run is not None:
            result["wandb_run_id"] = run.id
        
        # Setup experiment
        def make_env_fn():
            return make_env(**vars(args))
    
        env = make_env_fn()
        is_tabular = isinstance(env.unwrapped, TabularEnv)
        
        # Create policies for each feedback type (distinct RNG streams per split)
        print("Creating policies...")
        policies_train = _create_policies(args, env, active_feedback_types, "train")

        policies_val_dict = (
            _create_policies(args, env, active_feedback_types, "val")
            if args.val_every_n_epochs else None
        )
        
        # Define transforms
        act_transform = get_act_transform(args, env)
        obs_transform = get_obs_transform(args, env)
        
        # Get dimensions
        obs_dim = get_obs_dim(env, obs_transform)
        act_dim = get_act_dim(env, act_transform)

        # Create reference q-value model for stop feedback
        q_true = None
        if FeedbackType.STOP in active_feedback_types:
            if is_tabular:
                q_true = TabularQValueModel(env.unwrapped, gamma=args.gamma)
            else:
                stop_q_path = str(Path(args.stop_q_value_model).expanduser())
                q_true = DQNQValueModel(sb3.DQN.load(stop_q_path, env=env, device=device))
        
        # Create datasets and dataloaders
        print("Creating train datasets and dataloaders...")
        train_datasets, train_dataloaders = make_dataset(
            active_feedback_types,
            args,
            make_env_fn,
            policies_train,
            device,
            obs_transform,
            act_transform,
            name="train",
            q_true=q_true
        )
        
        val_dataloaders = None
        if args.val_every_n_epochs:
            print("Creating val datasets and dataloaders...")
            _, val_dataloaders = make_dataset(
                active_feedback_types,
                args,
                make_env_fn,
                policies_val_dict,
                device,
                obs_transform,
                act_transform,
                name="val",
                q_true=q_true,
            )
        
        # Visualize dataset occupancy for grid environments
        _visualize_dataset_occupancy(args, env, train_datasets)
        
        # Create model
        fb_model, reward_encoder = _create_model(args, env, obs_dim, act_dim, active_feedback_types, device)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(lr=args.lr, params=fb_model.parameters())
        
        # Calculate training info
        steps_per_epoch = max(len(dl) for dl in train_dataloaders.values())
        importance_weights = _compute_importance_weights(args, train_datasets, active_feedback_types)
        
        _print_training_info(train_dataloaders, importance_weights)
        
        # Create optimal policy for evaluation (skip if not needed)
        needs_optimal_policy = args.val_every_n_epochs or not getattr(args, "skip_final_eval", False)
        optimal_policy = _create_optimal_policy(args, env, is_tabular) if needs_optimal_policy else None
        
        # Initialize training state
        dloader_iters = {k: iter(train_dataloaders[k]) for k in active_feedback_types}
        best_val_loss = float("inf")
        early_stop_monitor = EarlyStopMonitor(
            patience=args.early_stop_patience,
            metric_name=args.early_stop_metric,
            min_delta=args.early_stop_min_delta,
        )
        
        print(f"Starting training for {args.num_epochs} epochs")
        
        t_train_start = time.perf_counter()
        
        # Training loop
        for epoch in range(args.num_epochs):
            global_step = train_epoch(
                fb_model,
                steps_per_epoch,
                optimizer,
                train_dataloaders,
                active_feedback_types,
                importance_weights,
                args,
                epoch,
                dloader_iters
            )
            relative_step = (global_step + 1) / steps_per_epoch
            
            # Validation: loss metrics (and tabular-only regret / EPIC). For non-tabular
            # envs, regret and mean reward are not computed here—only in run_test below.
            should_eval = (
                args.val_every_n_epochs
                and (epoch + 1) % args.val_every_n_epochs == 0
                and (not args.skip_first_val_epoch or epoch > 0)
            )
            
            if should_eval:
                # Used for clearer stdout logs on cluster runs.
                args._current_epoch = epoch + 1
                eval_metrics = run_validation(
                    fb_model,
                    val_dataloaders,
                    active_feedback_types,
                    is_tabular,
                    env,
                    make_env_fn,
                    optimal_policy,
                    args
                )
                
                # Compute total validation loss for model selection
                val_loss = compute_total_val_loss(eval_metrics, args.kl_weight, args.td_error_weight)
                eval_metrics["eval/total_val_loss"] = val_loss
                
                # Logging
                if args.log_wandb:
                    eval_metrics |= {"epoch": epoch, "relative_step": relative_step}
                    wandb.log(eval_metrics, step=global_step)
                
                console_log_eval_metrics(eval_metrics)
                
                if db_callback is not None:
                    db_callback(epoch, eval_metrics)
                
                # Model saving (based on validation loss, not regret)
                best_val_loss = save_model_checkpoint(
                    args,
                    fb_model,
                    optimizer,
                    epoch,
                    val_loss,
                    eval_metrics,
                    best_val_loss,
                    result
                )

                # If any numeric eval/* metric is NaN, stop early and jump to final evaluation.
                numeric_eval_values = []
                for key, value in eval_metrics.items():
                    if not key.startswith("eval/"):
                        continue
                    if torch.is_tensor(value):
                        if value.numel() != 1:
                            continue
                        value = value.item()
                    if isinstance(value, bool):
                        continue
                    if isinstance(value, Number):
                        numeric_eval_values.append(float(value))

                any_eval_metric_nan = (
                    len(numeric_eval_values) > 0
                    and any(math.isnan(v) for v in numeric_eval_values)
                )
                if any_eval_metric_nan:
                    result["early_stop_epoch"] = epoch
                    result["early_stop_reason"] = "nan_validation_metric_detected"
                    print(f"\n{'='*60}")
                    print(f"Early stopping at epoch {epoch}: at least one numeric validation metric is NaN.")
                    print("Skipping remaining training epochs and proceeding to final evaluation.")
                    print(f"{'='*60}\n")
                    break

                # Patience-based early stop on the configured validation metric.
                if early_stop_monitor.should_stop(eval_metrics, epoch):
                    result["early_stop_epoch"] = epoch
                    result["early_stop_reason"] = "val_loss_no_improvement"
                    print("Skipping remaining training epochs and proceeding to final evaluation.")
                    break
                
                fb_model.train()
            
            # Visualization
            should_visualize = args.vis_every_n_epochs and (epoch + 1) % args.vis_every_n_epochs == 0
            if should_visualize:
                visualize_epoch(args, env, fb_model, epoch, global_step)
        
        t_train_end = time.perf_counter()
        train_elapsed = t_train_end - t_train_start
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Training wall-clock time: {train_elapsed:.2f}s ({train_elapsed/60:.2f}min)")
        print(f"{'='*60}\n")
        
        result["training_time_sec"] = train_elapsed
        
        # Final evaluation: load best model and compute regret
        if getattr(args, "skip_final_eval", False):
            print("Skipping final evaluation (--skip_final_eval).")
        else:
            final_eval_model = getattr(args, "final_eval_model", "best")
            if final_eval_model == "last":
                print("Using final-epoch model for final evaluation (final_eval_model='last').")
                if args.model_save_dir is not None:
                    save_dir = Path(args.model_save_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    last_path = save_dir / "best_model.pt"
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": fb_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "args": vars(args),
                    }, last_path)
                    result["best_model_path"] = str(last_path)
                    result["best_epoch"] = epoch
            elif result["best_model_path"] is not None:
                print("Loading best model for final evaluation...")
                load_model_checkpoint(result["best_model_path"], fb_model)
            else:
                print("No best model saved - using final model for evaluation...")
            
            print("Running final evaluation (computing regret)...")
            final_metrics, _ = run_test(
                fb_model=fb_model,
                is_tabular=is_tabular,
                make_env_fn=make_env_fn,
                act_transform=act_transform,
                obs_transform=obs_transform,
                optimal_policy=optimal_policy,
                args=args
            )
            
            result["final_metrics"] = final_metrics.copy()
            
            if args.log_wandb:
                final_log = {f"final/{k.replace('eval/', '')}": v for k, v in final_metrics.items()}
                wandb.log(final_log)
            
            console_log_eval_metrics(final_metrics)
            
            print(f"\n{'='*60}")
            print("Final evaluation completed!")
            print(f"  Regret: {final_metrics.get('eval/regret', 'N/A')}")
            print(f"  Mean Reward: {final_metrics.get('eval/mean_rew', 'N/A')}")
            print(f"{'='*60}\n")

    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] Run finished (seed={args.seed})")
    
    return result


def main(args: argparse.Namespace):
    """
    Main entry point for CLI usage.
    
    This is a thin wrapper around run_experiment for backwards compatibility.
    """
    run_experiment(args)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all training arguments."""
    parser = argparse.ArgumentParser()

    g_repro = parser.add_argument_group("reproducibility")
    g_repro.add_argument("--seed", type=int, default=0, help="Global seed")

    g_rm = parser.add_argument_group("reward model")
    g_rm.add_argument("--reward_domain", type=str, default="s", help="Either state-only ('s'), state-action ('sa'), state-action-next-state ('sas')")
    g_rm.add_argument("--td_error_weight", type=float, default=1.0, help="Weight for TD-error constraint in demonstrations")
    g_rm.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    g_ds = parser.add_argument_group("dataset")
    g_ds.add_argument("--step_offset", type=int, default=1, help="Offset applied to next_obs and next_act")
    g_ds.add_argument("--subsample_factor", type=int, default=1, help="Subsample factor for demonstrations")

    g_cache = parser.add_argument_group("dataset caching")
    g_cache.add_argument("--dataset_cache_dir", type=str, default=None, help="Directory for cached datasets. None disables caching.")
    g_cache.add_argument("--dataset_cache_gen_samples", type=int, default=None, help="When set, generate this many samples per modality when populating the cache (for pre-generation)")

    g_opt = parser.add_argument_group("optimal policy (evaluation)")
    g_opt.add_argument("--optimal_policy_path", type=str, default=None, help="Path to optimal policy for generating evaluations")

    g_pref = parser.add_argument_group("preferences")
    g_pref.add_argument("--n_pref_episodes", type=int, default=0, help="Number of episodes to extract preference segments from")
    g_pref.add_argument("--n_pref_samples", type=int, default=0, help="Number of preference samples (0 to disable)")
    g_pref.add_argument("--pref_policy_path", type=str, default="expert_policies/ppo/LunarLander-v3_1/best_model.zip", help="Path to preference policy")
    g_pref.add_argument("--pref_trajectory_rationality", type=float, default=0.5, help="Rationality of the expert policy generating the comparison trajectories")
    g_pref.add_argument("--pref_data_rationality", type=float, default=1.0, help="Rationality (beta) for Bradley-Terry data generation")
    g_pref.add_argument("--pref_model_rationality", type=float, default=1.0, help="Rationality (beta) assumed by the model in the preference loss")
    g_pref.add_argument("--pref_seg_len", type=lambda x: None if x.lower() == "none" else int(x), default=128, help="Length of the extracted segments from episodes. None for full episodes")
    g_pref.add_argument("--min_reward_pref", type=lambda x: None if x.lower() == "none" else float(x), default=None, help="If not 'None', preferences with a lower reward will be rejected if their cumulative reward is lower than this threshold")

    g_demo = parser.add_argument_group("demonstrations")
    g_demo.add_argument("--n_demo_samples", type=int, default=0, help="Number of demonstration samples (0 to disable)")
    g_demo.add_argument("--demo_policy_path", type=str, default="expert_policies/dqn/LunarLander-v3_1/best_model.zip", help="Path to demonstration policy")
    g_demo.add_argument("--min_reward_demo", type=lambda x: None if x.lower() == "none" else float(x), default=None, help="If not 'None', demonstrations with a lower reward will be rejected if their cumulative reward is lower than this threshold")
    g_demo.add_argument("--demo_rationality", type=float, default=float("inf"), help="Rationality for expert policy (data-generation beta)")
    g_demo.add_argument("--demo_model_rationality", type=float, default=None, help="Rationality (beta) assumed by the model in the demo loss. None = use demo_rationality (well-specified)")

    g_rate = parser.add_argument_group("ratings")
    g_rate.add_argument("--n_rating_episodes", type=int, default=0, help="Number of episodes to extract rating segments from")
    g_rate.add_argument("--n_rating_samples", type=int, default=0, help="Number of rating samples (0 to disable)")
    g_rate.add_argument("--rating_policy_path", type=str, default="expert_policies/ppo/LunarLander-v3_1/best_model.zip", help="Path to rating policy")
    g_rate.add_argument("--rating_trajectory_rationality", type=float, default=5.0, help="Rationality of the expert policy generating rated trajectories")
    g_rate.add_argument("--rating_seg_len", type=lambda x: None if x.lower() == "none" else int(x), default=128, help="Length of the extracted segments for ratings. None for full episodes")
    g_rate.add_argument("--min_reward_rating", type=lambda x: None if x.lower() == "none" else float(x), default=None, help="If not 'None', ratings with a lower reward will be rejected")
    g_rate.add_argument("--rating_noise_std", type=float, default=0.0, help="Std-dev of Gaussian noise added to segment returns before rating assignment (0.0 = no noise)")

    g_stop = parser.add_argument_group("stops")
    g_stop.add_argument("--n_stop_samples", type=int, default=0, help="Number of stop samples (0 to disable)")
    g_stop.add_argument("--stop_policy_path", type=str, default="expert_policies/ppo/LunarLander-v3_1/best_model.zip", help="Path to stop policy")
    g_stop.add_argument("--stop_trajectory_rationality", type=float, default=0.5, help="Rationality of the expert policy generating the comparison trajectories")
    g_stop.add_argument("--stop_c", type=float, default=1.0, help="Calibration constant for lambda (higher = more aggressive stopping)")
    g_stop.add_argument("--stop_model_c", type=float, default=None, help="Model-assumed c for stop loss. None = use stop_c (well-specified)")
    g_stop.add_argument("--stop_regret_percentile", type=float, default=75.0, help="Percentile of final regrets to use as reference")
    g_stop.add_argument("--stop_regret_discount", type=float, default=0.8, help="Discount factor for old regret (0-1). Lower = faster forgetting.")
    g_stop.add_argument("--min_reward_stop", type=lambda x: None if x.lower() == "none" else float(x), default=None, help="If not 'None', stops with a lower reward will be rejected if their cumulative reward is lower than this threshold")
    g_stop.add_argument("--n_stop_episodes", type=int, default=0, help="Number of episodes to extract stop segments from")
    g_stop.add_argument("--stop_seg_len", type=lambda x: None if x.lower() == "none" else int(x), default=128, help="Length of the extracted segments for stops. None for full episodes")
    g_stop.add_argument("--stop_q_value_model", type=str, default=None, help="Path to q-value model used to estimate immediate regret in dataset generation. Does not need to be provided for tabular environments.")

    g_train = parser.add_argument_group("training")
    g_train.add_argument("--num_epochs", type=int, default=2000)
    g_train.add_argument("--val_every_n_epochs", type=lambda x: None if x.lower() == "none" else int(x), default=None)
    g_train.add_argument("--vis_every_n_epochs", type=lambda x: None if x.lower() == "none" else int(x), default=None)
    g_train.add_argument("--batch_size", type=int, default=128)
    g_train.add_argument("--lr", type=float, default=1e-4, help="Learning rate for model")
    g_train.add_argument("--kl_weight", type=float, default=1.0, help="KL weight")
    g_train.add_argument("--encoder_hidden_sizes", type=int, nargs="+", default=[256, 256, 256], help="Hidden sizes for encoder MLP")
    g_train.add_argument("--skip_first_val_epoch", action="store_true", help="Skip the first validation epoch")
    g_train.add_argument("--early_stop_patience", type=lambda x: None if x.lower() == "none" else int(x), default=30, help="Stop after this many validation rounds without metric improvement (None disables)")
    g_train.add_argument("--early_stop_metric", type=str, default="eval/total_val_loss", help="Validation metric name to monitor for early stopping")
    g_train.add_argument("--early_stop_min_delta", type=float, default=0.0, help="Minimum improvement required to reset early-stop patience")
    g_train.add_argument("--use_importance_weights", action="store_true", help="Reweight loss components by importance weights derived from dataset sizes")
    g_train.add_argument("--model_save_dir", type=str, default=None, help="Directory to save models and policies")
    g_train.add_argument("--save_behavior", type=str, choices=["best", "all"], default="best", help="Save behavior: 'best' saves only when val loss improves, 'all' saves at every eval epoch")
    g_train.add_argument("--final_eval_model", type=str, choices=["best", "last"], default="best", help="Which model to use for final evaluation: 'best' loads the lowest-val-loss checkpoint, 'last' uses the model state at the end of training")
    g_train.add_argument("--n_regret_samples", type=int, default=1000, help="Number of samples for regret computation")

    g_env = parser.add_argument_group("environment")
    g_env.add_argument("--grid_size", type=int, default=10)
    g_env.add_argument("--env_id", type=str, default="CartPole-v1")
    g_env.add_argument("--p_rand", type=float, default=0.0, help="Randomness in transitions (0 for deterministic)")
    g_env.add_argument("--obs_transform", choices=["one_hot", "continuous_coordinate", "dct", None], default=None, help="Apply a transform to the observation space")
    g_env.add_argument("--act_transform", choices=["one_hot", None], default=None, help="Apply a transform to the action space")
    g_env.add_argument("--exploration_epsilon", type=float, default=0.0, help="Probability of taking a random action during data collection (epsilon-greedy)")

    g_wb = parser.add_argument_group("weights & biases")
    g_wb.add_argument("--log_wandb", action="store_true", help="Log to weights and biases")
    g_wb.add_argument("--log_every_n_steps", type=int, default=10, help="Log every n steps")
    g_wb.add_argument("--wandb_project", type=str, default="var-rew-learning", help="Wandb project name")
    g_wb.add_argument("--wandb_run_name", type=str, default=None, help="Custom wandb run name")
    g_wb.add_argument("--wandb_log_dir", type=str, default="wandb", help="Wandb log directory")

    g_bench = parser.add_argument_group("benchmarking")
    g_bench.add_argument("--skip_final_eval", action="store_true", help="Skip final evaluation (regret computation). Useful for timing the training loop only.")

    g_base = parser.add_argument_group("baselines")
    g_base.add_argument("--use_imitation_learning", action="store_true", help="Use imitation learning")

    add_retrain_ppo_arguments(parser)

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
