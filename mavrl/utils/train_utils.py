import argparse
import numpy as np
import os
import tempfile
import torch
import wandb
import subprocess

import matplotlib.pyplot as plt
import gymnasium as gym
import stable_baselines3 as sb3

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Any, Iterator
from torch.utils.data import DataLoader

from mavrl.evaluation.epic import evaluate_epic_distance
from mavrl.evaluation.regret import compute_regret
from mavrl.utils.retrain_ppo_cli import retrain_ppo_settings_from_args
from mavrl.evaluation.spearmanr import evaluate_spearmanr
from mavrl.multi_fb_model import MultiFeedbackTypeModel
from mavrl.encoder.reward_encoder import RewardEncoder
from mavrl.encoder.feature_modules import MLPFeatureModule
from mavrl.feedback.make_nll import make_nll
from mavrl.data.utils import derive_seed
from mavrl.utils.logging import console_log_batch_metrics, losses_to_flat_dict
from mavrl.losses import elbo_loss
from mavrl.visualization.get_visualization import get_visualization
from mavrl.visualization.grid_visualizer import vis_grid_occupancy
from mavrl.visualization.rollout_gif import create_rollout_gif
from mavrl.utils.policies import create_policy, EpsilonGreedyWrapper
from mavrl.types import FeedbackType
from mavrl.envs.grid_env.env import GridEnv
from mavrl.envs.env_types import TabularEnv
from mavrl.learned_reward_wrapper import LearnedRewardWrapper


# =============================================================================
# Argument Validation
# =============================================================================

def validate_args(args: argparse.Namespace, feedback_config: dict[FeedbackType, int]) -> dict[FeedbackType, int]:
    """
    Validate argument combinations and return active feedback types.
    
    Args:
        args: Experiment configuration.
        feedback_config: Mapping of feedback types to sample counts.
        
    Returns:
        Dictionary of active feedback types (those with samples > 0).
        
    Raises:
        ValueError: If argument combinations are invalid.
    """
    # Validate imitation learning settings
    if args.use_imitation_learning:
        if args.td_error_weight != 0:
            print(f"Warning: TD-error weight is {args.td_error_weight} when using imitation learning. Setting to 0.")
            args.td_error_weight = 0
        if args.n_demo_samples == 0:
            print(f"Warning: n_demo_samples is {args.n_demo_samples} when using imitation learning. Setting to 32.")
            args.n_demo_samples = 32
            feedback_config[FeedbackType.DEMO] = 32
        if any(v > 0 for k, v in feedback_config.items() if k != FeedbackType.DEMO):
            raise ValueError("Imitation learning may only use demonstration feedback.")
    
    # Filter out feedback types with 0 samples
    active_feedback_types = {k: v for k, v in feedback_config.items() if v > 0}
    
    if not active_feedback_types:
        raise ValueError("At least one feedback type must have samples > 0")
    
    return active_feedback_types


# =============================================================================
# WandB Context Manager
# =============================================================================

@contextmanager
def wandb_run(args: argparse.Namespace, config: dict) -> Iterator[Optional[Any]]:
    """
    Context manager for wandb run lifecycle.
    
    Handles initialization and cleanup of wandb runs, including cleanup
    of any orphaned runs from previous crashes. Falls back to no-wandb
    mode if initialization fails (e.g. network timeout on cluster nodes).
    
    Args:
        args: Experiment configuration (must have log_wandb, wandb_project, wandb_log_dir).
        config: Configuration dictionary to log to wandb.
        
    Yields:
        The wandb run object if logging is enabled, None otherwise.
    """
    if not args.log_wandb:
        yield None
        return
    
    # Clean up any orphaned run from a previous crash
    if wandb.run is not None:
        print(f"Warning: Found existing wandb run '{wandb.run.id}', finishing it before starting new run")
        wandb.finish()
    
    try:
        run = wandb.init(
            project=args.wandb_project,
            config=config,
            name=getattr(args, "wandb_run_name", None),
            dir=args.wandb_log_dir,
            settings=wandb.Settings(init_timeout=300)
        )
    except Exception as e:
        print(f"Warning: wandb.init() failed ({type(e).__name__}: {e}). "
              f"Continuing without wandb logging.")
        args.log_wandb = False
        yield None
        return
    
    try:
        yield run
    finally:
        wandb.finish()


def _create_policies(
    args: argparse.Namespace,
    env: gym.Env,
    active_feedback_types: dict[FeedbackType, int],
    dataset_split_name: str = "train",
) -> dict[FeedbackType, Any]:
    """Create policies for each active feedback type.

    Parameters
    ----------
    dataset_split_name:
        Logical split (``\"train\"`` / ``\"val\"``). Used to derive a NumPy RNG
        seed per modality so stochastic tabular/DQN policies use **different**
        independent streams across splits (train vs val datasets are not
        copies of identical rollouts when the MDP is deterministic).
    """
    policies = {}
    
    if FeedbackType.PREF in active_feedback_types:
        print(f"  Creating PREFERENCE policy from: {args.pref_policy_path}")
        rng_seed = derive_seed(args.seed, FeedbackType.PREF.value, dataset_split_name)
        policies[FeedbackType.PREF] = create_policy(
            args.pref_policy_path,
            args.pref_trajectory_rationality,
            env,
            args.gamma,
            rng_seed=rng_seed,
        )
    if FeedbackType.DEMO in active_feedback_types:
        print(f"  Creating DEMONSTRATION policy from: {args.demo_policy_path}")
        rng_seed = derive_seed(args.seed, FeedbackType.DEMO.value, dataset_split_name)
        policies[FeedbackType.DEMO] = create_policy(
            args.demo_policy_path,
            args.demo_rationality,
            env,
            args.gamma,
            rng_seed=rng_seed,
        )
    if FeedbackType.RATE in active_feedback_types:
        print(f"  Creating RATING policy from: {args.rating_policy_path}")
        rng_seed = derive_seed(args.seed, FeedbackType.RATE.value, dataset_split_name)
        policies[FeedbackType.RATE] = create_policy(
            args.rating_policy_path,
            args.rating_trajectory_rationality,
            env,
            args.gamma,
            rng_seed=rng_seed,
        )
    if FeedbackType.STOP in active_feedback_types:
        print(f"  Creating STOP policy from: {args.stop_policy_path}")
        rng_seed = derive_seed(args.seed, FeedbackType.STOP.value, dataset_split_name)
        policies[FeedbackType.STOP] = create_policy(
            args.stop_policy_path,
            args.stop_trajectory_rationality,
            env,
            args.gamma,
            rng_seed=rng_seed,
        )

    epsilon = getattr(args, "exploration_epsilon", 0.0)
    if epsilon > 0:
        print(f"  Wrapping all policies with epsilon-greedy exploration (ε={epsilon})")
        policies = {
            k: EpsilonGreedyWrapper(
                v,
                env.action_space,
                epsilon,
                exploration_rng=np.random.default_rng(
                    derive_seed(
                        args.seed,
                        k.value,
                        dataset_split_name + ":epsilon",
                    )
                ),
            )
            for k, v in policies.items()
        }

    return policies


def _create_model(
    args: argparse.Namespace,
    env: gym.Env,
    obs_dim: int,
    act_dim: int,
    active_feedback_types: dict[FeedbackType, int],
    device: str
) -> tuple[MultiFeedbackTypeModel, RewardEncoder]:
    """Create the multi-feedback model and reward encoder."""
    feature_module = MLPFeatureModule(
        obs_dim, act_dim, args.encoder_hidden_sizes, reward_domain=args.reward_domain
    )
    reward_encoder = RewardEncoder(feature_module)
    
    q_model = MLPFeatureModule(
        state_dim=obs_dim,
        action_dim=None,
        hidden_sizes=args.encoder_hidden_sizes + [act_dim],
        reward_domain='s',
        activate_last_layer=False
    )
    
    decoders = {fb_type: make_nll(fb_type) for fb_type in active_feedback_types}
    
    fb_model = MultiFeedbackTypeModel(
        encoder=reward_encoder,
        q_model=q_model,
        decoders=decoders,
    )
    fb_model.to(device)
    
    return fb_model, reward_encoder


def _visualize_dataset_occupancy(args: argparse.Namespace, env: gym.Env, train_datasets: dict) -> None:
    """Visualize dataset occupancy for grid environments."""
    if not args.vis_every_n_epochs:
        return
    

    if isinstance(env.unwrapped, GridEnv): 
        print("Generating dataset occupancy visualization...")
        fig = vis_grid_occupancy(env.unwrapped, train_datasets)
        if args.log_wandb:
            wandb.log({"visualizations/dataset_occupancy": wandb.Image(fig)}, step=0)
        plt.close(fig)
    else:
        return


def _compute_importance_weights(
    args: argparse.Namespace,
    train_datasets: dict,
    active_feedback_types: dict[FeedbackType, int]
) -> Optional[dict[FeedbackType, float]]:
    """Compute importance weights for each feedback type if requested."""
    if not args.use_importance_weights:
        return None
    
    max_dataset_size = max(len(train_datasets[fb_type]) for fb_type in active_feedback_types)
    weights = {fb_type: len(train_datasets[fb_type]) / max_dataset_size for fb_type in active_feedback_types}
    print(f"Importance weights: {weights}")
    return weights


def _print_training_info(train_dataloaders: dict, importance_weights: Optional[dict]) -> None:
    """Print training information."""
    print("Training info:")
    for fb_type, dl in train_dataloaders.items():
        print(f"  {fb_type.value}: {len(dl)} batches per epoch")
    print(f"  Batches processed per update: {len(train_dataloaders)}")


def _create_optimal_policy(args: argparse.Namespace, env: gym.Env, is_tabular: bool) -> Any:
    """Create the optimal policy for evaluation."""
    print("Creating optimal policy for evaluation...")
    if is_tabular:
        return None
    if args.optimal_policy_path is None:
        raise ValueError("An optimal policy path must be provided for evaluation.")
    return create_policy(args.optimal_policy_path, float("inf"), env, args.gamma)


# =============================================================================
# Training Step
# =============================================================================

def train_step(
    fb_model: MultiFeedbackTypeModel,
    optimizer: torch.optim.Optimizer,
    dloader_iters: dict,
    train_dataloaders: dict,
    active_feedback_types: dict[FeedbackType, int],
    args: argparse.Namespace,
    importance_weights: Optional[dict[FeedbackType, float]] = None,
) -> tuple[float, dict]:
    """
    Execute one training step across all feedback types.
    
    Args:
        fb_model: The multi-feedback model.
        optimizer: The optimizer.
        dloader_iters: Dictionary of dataloader iterators.
        train_dataloaders: Dictionary of training dataloaders.
        active_feedback_types: Active feedback types.
        args: Experiment configuration.
        importance_weights: Optional importance weights per feedback type.
        
    Returns:
        Tuple of (total_loss, losses_dict) where losses_dict contains per-feedback-type losses.
    """
    total_loss = 0.0
    losses = {}
    
    for fb_type in active_feedback_types:
        batch = get_batch(dloader_iters, train_dataloaders, fb_type)
        loss_dict = fb_model(**batch)
        
        # Extract losses for logging
        losses[fb_type.value] = {
            "nll": loss_dict["negative_log_likelihood"].item() if torch.is_tensor(loss_dict["negative_log_likelihood"]) else loss_dict["negative_log_likelihood"],
            "kl": loss_dict["kl_divergence"].item() if torch.is_tensor(loss_dict["kl_divergence"]) else loss_dict["kl_divergence"],
            "td": loss_dict["td_error"].item() if torch.is_tensor(loss_dict["td_error"]) else loss_dict["td_error"],
        }
        
        # Compute ELBO loss
        elbo = elbo_loss(
            loss_dict["negative_log_likelihood"],
            loss_dict["kl_divergence"],
            kl_weight=args.kl_weight
        )
        
        # Add TD regularization
        loss = elbo + args.td_error_weight * loss_dict["td_error"]
        
        # Apply importance weighting if requested
        if importance_weights is not None:
            loss *= importance_weights[fb_type]
        
        total_loss += loss
    
    # Optimization step
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(fb_model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return total_loss.item(), losses


# =============================================================================
# Evaluation
# =============================================================================

def run_validation(
    fb_model: MultiFeedbackTypeModel,
    val_dataloaders,
    active_feedback_types,
    is_tabular,
    env,
    make_env_fn,
    optimal_policy,
    args: argparse.Namespace,
) -> dict:
    """
    Run periodic validation during training for model selection (cheap metrics only).

    Always computes loss-based metrics (NLL, KL, TD) and Spearman correlations.
    EPIC distance is computed only for tabular environments; for non-tabular envs it
    is set to None.

    Regret and mean reward: for **non-tabular** (high-dimensional) environments,
    these are **not** computed here—`compute_regret` would require training a
    policy on the learned reward each time, which is too expensive for every
    validation step. Those quantities appear only in :func:`run_test`, which
    ``train.py`` calls once after training (on the best or last checkpoint).

    For **tabular** environments, regret, mean reward, and discounted value are
    included here because they can be computed from the model without PPO.

    Model selection uses :func:`compute_total_val_loss` (losses), not regret.

    Returns:
        Dictionary of validation metrics.
    """
    epoch = getattr(args, "_current_epoch", None)
    if epoch is None:
        print("  Running validation...")
    else:
        print(f"  Running validation... (epoch={epoch})")
    fb_model.eval()
    eval_metrics = {}
    
    with torch.no_grad():
        eval_metrics |= compute_eval_loss(
            val_dataloaders,
            active_feedback_types,
            fb_model
        )
        
        # Spearman correlation
        for fb_type, val_dl in val_dataloaders.items():
            spearman_corr = evaluate_spearmanr(fb_model.encoder, val_dl)
            eval_metrics[f"eval/spearman_{fb_type.value}"] = spearman_corr
        
        # EPIC distance for tabular environments
        if is_tabular:
            eval_metrics["eval/epic_distance"] = evaluate_epic_distance(
                env.unwrapped, fb_model.encoder, args.gamma
            )
            
            # For tabular environments, also compute regret and discounted_value
            # since these are fast to compute without training a policy
            def make_train_env(seed: int):
                return make_env_fn()
            
            regret, mean_rew, discounted_value, _, _ = compute_regret(
                ppo_seed=args.seed,
                true_optimal_policy=optimal_policy,
                train_env_fn=make_train_env,
                eval_env_fn=make_env_fn,
                is_tabular=True,
                is_imitation=args.use_imitation_learning,
                fb_model=fb_model,
                gamma=args.gamma,
                true_reward_threshold=args.retrain_reward_thresh,
                verbose=args.retrain_verbose,
                progress_bar=args.retrain_pbar,
                retrain_ppo_settings=retrain_ppo_settings_from_args(args),
                n_regret_samples=args.n_regret_samples,
                seed_fn=lambda i: args.seed * args.n_regret_samples + i,
            )
            eval_metrics["eval/regret"] = regret
            eval_metrics["eval/mean_rew"] = mean_rew
            eval_metrics["eval/discounted_value"] = discounted_value
        else:
            eval_metrics["eval/epic_distance"] = None
    
    return eval_metrics


def run_test(
    fb_model: MultiFeedbackTypeModel,
    is_tabular: bool,
    make_env_fn,
    act_transform,
    obs_transform,
    optimal_policy,
    args: argparse.Namespace,
) -> tuple[dict, Any]:
    """
    Final evaluation after training: compute regret and mean reward via ``compute_regret``.

    For **non-tabular** environments this is the **only** stage in the training
    pipeline where regret and mean reward are estimated (PPO on the learned
    reward, then rollouts vs the true optimal policy). Periodic validation
    (:func:`run_validation`) intentionally skips them.

    For **tabular** environments, regret can also be computed during
    :func:`run_validation`; this call still runs once at the end on the loaded
    checkpoint so ``final_metrics`` matches what downstream code expects.

    This is expensive and should only be called once at the end of training
    after loading the best model checkpoint.

    Args:
        fb_model: Reward model to evaluate (typically best checkpoint).
        args: Experiment configuration (env id, PPO settings, etc.).

    Returns:
        Tuple of (metrics dict including eval/regret and eval/mean_rew, estimated_optimal_policy).
    """
    print("  Running final evaluation (computing regret)...")
    fb_model.eval()
    eval_metrics = {}
    
    # Create train environment factory with learned reward
    def make_train_env(seed: int):
        if is_tabular:
            return make_env_fn()
        env = gym.make(args.env_id)
        env.reset(seed=seed)
        return LearnedRewardWrapper(
            env,
            fb_model.encoder,
            seed=seed,
            act_transform=act_transform,
            obs_transform=obs_transform
        )
    
    # Compute regret
    regret, mean_rew, discounted_value, est_optimal_policy, ppo_best_step = compute_regret(
        ppo_seed=args.seed,
        true_optimal_policy=optimal_policy,
        train_env_fn=make_train_env,
        eval_env_fn=make_env_fn,
        is_tabular=is_tabular,
        is_imitation=args.use_imitation_learning,
        fb_model=fb_model,
        gamma=args.gamma,
        true_reward_threshold=args.retrain_reward_thresh,
        verbose=args.retrain_verbose,
        progress_bar=args.retrain_pbar,
        retrain_ppo_settings=retrain_ppo_settings_from_args(args),
        n_regret_samples=args.n_regret_samples,
        seed_fn=lambda i: args.seed * args.n_regret_samples + i,  # ensure distinct results across seeds
    )
    eval_metrics["eval/regret"] = regret
    eval_metrics["eval/mean_rew"] = mean_rew

    if ppo_best_step is not None:
        eval_metrics["eval/ppo_best_step"] = ppo_best_step

    # Add discounted value for tabular environments
    if discounted_value is not None:
        eval_metrics["eval/discounted_value"] = discounted_value
    
    return eval_metrics, est_optimal_policy


def compute_total_val_loss(eval_metrics: dict, kl_weight: float, td_weight: float) -> float:
    """
    Compute total validation loss for model selection.
    
    Total loss = NLL + kl_weight * KL + td_weight * TD, summed across all feedback types.
    This matches the training loss computation and should be used for model selection
    instead of regret to avoid selection bias in reported metrics.
    
    Args:
        eval_metrics: Dictionary of evaluation metrics from run_validation().
        kl_weight: Weight for KL divergence term (same as training).
        td_weight: Weight for TD error term (same as training).
        
    Returns:
        Total validation loss (lower is better).
    """
    nll = eval_metrics.get("eval/negative_log_likelihood", 0.0)
    kl = eval_metrics.get("eval/kl_divergence", 0.0)
    td = eval_metrics.get("eval/td_error", 0.0)
    return nll + kl_weight * kl + td_weight * td


# =============================================================================
# Model Saving
# =============================================================================

def save_model_checkpoint(
    args: argparse.Namespace,
    fb_model: MultiFeedbackTypeModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    eval_metrics: dict,
    best_val_loss: float,
    result: dict,
) -> float:
    """
    Save model checkpoint based on save_behavior setting.
    
    Args:
        args: Experiment configuration.
        fb_model: The model to save.
        optimizer: The optimizer to save.
        epoch: Current epoch.
        val_loss: Current validation loss (from compute_total_val_loss).
        eval_metrics: Current evaluation metrics.
        best_val_loss: Best validation loss seen so far.
        result: Result dictionary to update with model path.
        
    Returns:
        Updated best_val_loss value.
    """
    if args.model_save_dir is None:
        return best_val_loss
    
    save_dir = Path(args.model_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_state = {
        "epoch": epoch,
        "model_state_dict": fb_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "eval_metrics": eval_metrics.copy(),
        "args": vars(args)
    }
    
    if args.save_behavior == "all":
        # Save checkpoint for this epoch
        model_path = save_dir / f"model_epoch_{epoch:04d}.pt"
        torch.save(model_state, model_path)
        print(f"  Saved model to: {model_path}")
        
        # Also track the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            result["best_model_path"] = str(model_path)
            result["best_epoch"] = epoch
            print(f"  ^ This is the new best model (val_loss: {val_loss:.4f})")
    
    elif args.save_behavior == "best" and val_loss < best_val_loss:
        best_val_loss = val_loss
        model_path = save_dir / "best_model.pt"
        torch.save(model_state, model_path)
        result["best_model_path"] = str(model_path)
        result["best_epoch"] = epoch
        print(f"  Saved best model to: {model_path} (epoch={epoch}, val_loss: {val_loss:.4f})")
    
    return best_val_loss


def load_model_checkpoint(
    model_path: str,
    fb_model: MultiFeedbackTypeModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> dict:
    """
    Load a model checkpoint.
    
    Args:
        model_path: Path to the checkpoint file.
        fb_model: The model to load weights into.
        optimizer: Optional optimizer to load state into.
        device: Device to load the model to.
        
    Returns:
        Dictionary containing checkpoint metadata (epoch, val_loss, eval_metrics, args).
    """
    print(f"  Loading model checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    fb_model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return {
        "epoch": checkpoint.get("epoch"),
        "val_loss": checkpoint.get("val_loss"),
        "eval_metrics": checkpoint.get("eval_metrics", {}),
        "args": checkpoint.get("args", {}),
    }


# =============================================================================
# Visualization
# =============================================================================

def visualize_epoch(
    args: argparse.Namespace,
    env: gym.Env,
    fb_model: MultiFeedbackTypeModel,
    epoch: int,
    global_step: int
) -> None:
    """Generate and log visualization for the current epoch."""
    print("  Generating visualization...")
    fb_model.eval()
    
    with torch.no_grad():
        fig = get_visualization(env, fb_model)
        
        if args.log_wandb:
            wandb.log({
                "visualizations/rewards": wandb.Image(fig),
                "epoch": epoch,
            }, step=global_step)
        
        plt.close(fig)
    
    fb_model.train()


# =============================================================================
# Main Training Loop
# =============================================================================

def train_epoch(
    fb_model: MultiFeedbackTypeModel,
    steps_per_epoch: int,
    optimizer: torch.optim.Optimizer,
    train_dataloaders,
    active_feedback_types,
    importance_weights,
    args: argparse.Namespace,
    epoch: int,
    dloader_iters: dict,
) -> int:
    """
    Run one training epoch.
    
    Args:
        components: Experiment components.
        args: Experiment configuration.
        epoch: Current epoch number.
        dloader_iters: Dictionary of dataloader iterators.
        
    Returns:
        The global_step at the end of the epoch.
    """
    fb_model.train()
    
    for step in range(steps_per_epoch):
        global_step = epoch * steps_per_epoch + step
        relative_step = global_step / steps_per_epoch
        
        total_loss, losses = train_step(
            fb_model,
            optimizer,
            dloader_iters,
            train_dataloaders,
            active_feedback_types,
            args,
            importance_weights,
        )
        
        # Log every N steps
        if global_step % args.log_every_n_steps == 0:
            console_log_batch_metrics(losses, step, steps_per_epoch, total_loss)
            
            if args.log_wandb:
                wandb_log_dict = {f"batch/{k}": v for k, v in losses_to_flat_dict(losses).items()}
                wandb_log_dict["batch/total_loss"] = total_loss
                wandb_log_dict["relative_step"] = relative_step
                wandb.log(wandb_log_dict, step=global_step)
    
    return (epoch + 1) * steps_per_epoch - 1


def get_git_commit_hash() -> Optional[str]:
    """
    Get the current git commit hash.
    
    Returns:
        The commit hash as a string, or None if git is not available
        or we're not in a git repository.
    """
    try:
        # Find the git repository root by walking up from the current file
        current_path = Path(__file__).resolve()
        
        # Walk up the directory tree to find .git directory
        repo_root = None
        for parent in [current_path] + list(current_path.parents):
            if (parent / ".git").exists():
                repo_root = parent
                break
        
        # If no .git found, try using git rev-parse --show-toplevel
        if repo_root is None:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=True,
                cwd=current_path.parent
            )
            repo_root = Path(result.stdout.strip())
        
        # Get the commit hash from the repo root
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_batch(dloader_iters: dict[FeedbackType, Iterator], dataloaders: dict[FeedbackType, DataLoader], fb_type: FeedbackType) -> dict[str, Any]:
    try:
        batch = next(dloader_iters[fb_type])
    except StopIteration:
        # Restart iterator if we've exhausted this dataloader
        dloader_iters[fb_type] = iter(dataloaders[fb_type])
        batch = next(dloader_iters[fb_type])
    return batch


def compute_eval_loss(val_dataloaders: dict[FeedbackType, DataLoader], active_feedback_types: list[FeedbackType], multi_fb_model: MultiFeedbackTypeModel):
    assert not multi_fb_model.training, "Model is not in evaluation mode"
    dloader_iters = {fb_type: iter(val_dataloaders[fb_type]) for fb_type in active_feedback_types}
    eval_loss_dict = {}
    for fb_type in active_feedback_types:
        batch = get_batch(dloader_iters, val_dataloaders, fb_type)
        loss_dict = multi_fb_model(**batch)

        # Log total loss metrics
        for k, v in loss_dict.items():
            if k not in eval_loss_dict:
                eval_loss_dict[k] = (0, 0.0)
            count, agg_val = eval_loss_dict[k]
            eval_loss_dict[k] = (count + 1, agg_val + v)
        
        # Log feedback-specific metrics
        for k, v in loss_dict.items():
            key = f"{k}_{fb_type.value}"
            if key not in eval_loss_dict:
                eval_loss_dict[key] = (0, 0.0)
            count, agg_val = eval_loss_dict[key]
            eval_loss_dict[key] = (count + 1, agg_val + v)
    
    # Average loss metrics
    final_dict = {}
    for k, (count, agg_val) in eval_loss_dict.items():
        if count > 0:
            final_dict[f"eval/{k}"] = agg_val / count
    return final_dict


class EarlyStopMonitor:
    """
    Monitor for early stopping based on a tracked metric.
    
    Assumes lower metric values are better. If patience is None, 
    early stopping is disabled and should_stop() always returns False.
    
    Example:
        monitor = EarlyStopMonitor(patience=5, metric_name="eval/regret")
        
        for epoch in range(num_epochs):
            # ... training ...
            eval_metrics = evaluate()
            
            if monitor.should_stop(eval_metrics, epoch):
                break
    """
    
    def __init__(
        self,
        patience: Optional[int],
        metric_name: str = "eval/regret",
        min_delta: float = 0.0,
    ):
        """
        Args:
            patience: Number of eval epochs without improvement before stopping.
                     If None, early stopping is disabled.
            metric_name: Name of the metric to monitor in eval_metrics dict.
            min_delta: Minimum decrease required to count as an improvement.
        """
        self.patience = patience
        self.metric_name = metric_name
        self.min_delta = min_delta
        self.best_metric = float("inf")
        self.epochs_without_improvement = 0
    
    @property
    def enabled(self) -> bool:
        """Whether early stopping is enabled."""
        return self.patience is not None
    
    def should_stop(self, eval_metrics: dict[str, Any], epoch: int) -> bool:
        """
        Check if training should stop based on the current metrics.
        
        Args:
            eval_metrics: Dictionary of evaluation metrics.
            epoch: Current epoch number (for logging).
            
        Returns:
            True if training should stop, False otherwise.
            
        Raises:
            ValueError: If the monitored metric is not found in eval_metrics.
        """
        if not self.enabled:
            return False
        
        current_metric = eval_metrics.get(self.metric_name)
        if current_metric is None:
            raise ValueError(
                f"Early stop metric '{self.metric_name}' not found in eval_metrics. "
                f"Available: {list(eval_metrics.keys())}"
            )
        
        # Lower is better. Require a decrease larger than min_delta.
        if current_metric < (self.best_metric - self.min_delta):
            self.best_metric = current_metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        if self.epochs_without_improvement >= self.patience:
            print(f"\n{'='*60}")
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"  Metric '{self.metric_name}' has not improved for {self.patience} eval epochs")
            print(f"  Best value: {self.best_metric:.6f}")
            print(f"{'='*60}\n")
            return True
        
        return False
