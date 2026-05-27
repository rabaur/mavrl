"""Evaluate reward-model and imitation transfer experiments.

Modes:
- `average`: train on an averaged reward model from multiple checkpoints
- `mavrl`: train on a single reward-model checkpoint
- `ground_truth`: train directly on environment reward
- `imitation`: evaluate policy derived from checkpoint q-model
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import torch
import wandb

from mavrl.encoder.averaged_reward_encoder import (
    load_averaged_encoder,
    reconstruct_reward_encoder,
)
from mavrl.encoder.feature_modules import MLPFeatureModule
from mavrl.encoder.reward_encoder import RewardEncoder
from mavrl.envs.env_types import TabularEnv
from mavrl.envs.make_env import make_env
from mavrl.evaluation.epic import evaluate_epic_distance
from mavrl.evaluation.regret import (
    regret_non_tabular,
    regret_tabular,
    regret_tabular_imitation,
)
from mavrl.learned_reward_wrapper import LearnedRewardWrapper
from mavrl.feedback.make_nll import make_nll
from mavrl.multi_fb_model import MultiFeedbackTypeModel
from mavrl.types import FeedbackType
from mavrl.utils.feature_transforms import get_act_transform, get_obs_transform
from mavrl.utils.gym import get_act_dim, get_obs_dim
from mavrl.utils.policies import (
    NeuralQValueModel,
    PPOExpert,
    QValueExpert,
    create_policy,
)
from mavrl.utils.reproducibility import seed_everything
from mavrl.utils.retrain_ppo_cli import add_retrain_ppo_arguments, retrain_ppo_settings_from_args
from mavrl.utils.sb3 import train_ppo


def load_checkpoint(checkpoint_path: Path) -> dict:
    checkpoint_path = checkpoint_path.expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def reconstruct_multi_fb_model(
    checkpoint: dict,
    env: gym.Env,
    act_transform: Optional[Callable] = None,
    obs_transform: Optional[Callable] = None,
    device: str = "cpu",
) -> MultiFeedbackTypeModel:
    """Reconstruct full MultiFeedbackTypeModel from a training checkpoint."""
    obs_dim = get_obs_dim(env, obs_transform)
    act_dim = get_act_dim(env, act_transform)

    train_args = checkpoint.get("args", {})
    if not train_args:
        raise ValueError("Checkpoint does not contain 'args'. Cannot reconstruct model architecture.")

    encoder_hidden_sizes = train_args.get("encoder_hidden_sizes", [256, 256])
    reward_domain = train_args.get("reward_domain", "sa")

    feedback_config = {
        FeedbackType.PREF: train_args.get("n_pref_samples", 0),
        FeedbackType.DEMO: train_args.get("n_demo_samples", 0),
        FeedbackType.RATE: train_args.get("n_rating_samples", 0),
        FeedbackType.STOP: train_args.get("n_stop_samples", 0),
    }
    active_feedback_types = [fb_type for fb_type, n in feedback_config.items() if n > 0]
    if not active_feedback_types:
        raise ValueError("No active feedback types found in checkpoint args.")

    feature_module = MLPFeatureModule(
        obs_dim,
        act_dim,
        encoder_hidden_sizes,
        reward_domain=reward_domain,
    )
    reward_encoder = RewardEncoder(feature_module)

    q_model = MLPFeatureModule(
        state_dim=obs_dim,
        action_dim=None,
        hidden_sizes=encoder_hidden_sizes + [act_dim],
        reward_domain="s",
        activate_last_layer=False,
    )
    decoders = {fb_type: make_nll(fb_type) for fb_type in active_feedback_types}

    fb_model = MultiFeedbackTypeModel(
        encoder=reward_encoder,
        q_model=q_model,
        decoders=decoders,
    )
    fb_model.to(device)
    fb_model.load_state_dict(checkpoint["model_state_dict"])
    return fb_model


def make_perturbed_env(env_id: str, env_params: dict, seed: int = 0) -> gym.Env:
    """Create a transfer environment using env-specific perturbation handling."""
    if "acrobot" in env_id.lower():
        from mavrl.envs.wrappers.acrobot_transfer_env import AcrobotTransferEnv

        base_env = gym.make(env_id)
        env = AcrobotTransferEnv(base_env, **env_params)
        env.reset(seed=seed)
    elif env_id.startswith("grid"):
        from mavrl.envs.grid_env.env import GridEnv

        rew_type = env_id.split("_")[1]
        grid_params = {**env_params, "reward_type": rew_type, "seed": seed}
        env = GridEnv(**grid_params)
    else:
        env = gym.make(
            env_id,
            render_mode=None,
            **{k: v for k, v in env_params.items() if k not in ["env_id", "seed"]},
        )
        env.reset(seed=seed)

    return env


def _load_single_encoder(checkpoint_path: str, env, act_transform, obs_transform):
    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    encoder = reconstruct_reward_encoder(
        ckpt,
        env,
        act_transform=act_transform,
        obs_transform=obs_transform,
        device="cpu",
    )
    encoder.eval()
    return encoder


def _validate_checkpoint_args(args: argparse.Namespace) -> None:
    if args.mode == "average":
        if not args.checkpoint_paths or len(args.checkpoint_paths) < 2:
            raise ValueError("--mode average requires at least two --checkpoint_paths.")
        if args.checkpoint_path is not None:
            raise ValueError("--mode average does not use --checkpoint_path.")
        return

    if args.mode in {"mavrl", "imitation"}:
        if args.checkpoint_path is None:
            raise ValueError(f"--mode {args.mode} requires --checkpoint_path.")
        if args.checkpoint_paths is not None:
            raise ValueError(f"--mode {args.mode} does not use --checkpoint_paths.")
        return

    if args.mode == "ground_truth":
        if args.checkpoint_path is not None or args.checkpoint_paths is not None:
            raise ValueError("--mode ground_truth does not use checkpoints.")
        return

    raise ValueError(f"Unknown mode: {args.mode}")


def run_evaluation(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)

    env_params: dict = {}
    if args.env_params:
        env_params = json.loads(args.env_params)

    _validate_checkpoint_args(args)

    if getattr(args, "log_wandb", False):
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config={
                "env_id": args.env_id,
                "env_params": env_params,
                "mode": args.mode,
                "checkpoint_path": args.checkpoint_path,
                "checkpoint_paths": args.checkpoint_paths,
                "optimal_policy_path": args.optimal_policy_path,
                "gamma": args.gamma,
                "num_samples": args.num_samples,
                "max_num_steps": args.max_num_steps,
                "seed": args.seed,
                "act_transform": args.act_transform,
                "obs_transform": args.obs_transform,
            },
        )

    base_env = make_env(**(env_params | {"env_id": args.env_id, "seed": args.seed}))
    act_transform = get_act_transform(args, base_env)
    obs_transform = get_obs_transform(args, base_env)

    encoder = None
    if args.mode == "mavrl":
        encoder = _load_single_encoder(
            args.checkpoint_path,
            base_env,
            act_transform,
            obs_transform,
        )
    elif args.mode == "average":
        encoder = load_averaged_encoder(
            checkpoint_paths=args.checkpoint_paths,
            env=base_env,
            act_transform=act_transform,
            obs_transform=obs_transform,
            device="cpu",
        )
        encoder.eval()

    fb_model = None
    if args.mode == "imitation":
        checkpoint = load_checkpoint(Path(args.checkpoint_path))
        fb_model = reconstruct_multi_fb_model(
            checkpoint=checkpoint,
            env=base_env,
            act_transform=act_transform,
            obs_transform=obs_transform,
            device="cpu",
        )
        fb_model.eval()

    is_tabular = isinstance(base_env, TabularEnv)
    results: dict = {
        "env_id": args.env_id,
        "mode": args.mode,
        "is_tabular": is_tabular,
    }

    if is_tabular:
        if args.mode == "imitation":
            regret, discounted_value = regret_tabular_imitation(base_env, fb_model.q_model, args.gamma)
            results.update(
                epic_distance=float("nan"),
                regret=float(regret),
                discounted_value=float(discounted_value),
            )
            print("\nResults (tabular imitation):")
            print(f"  Regret:            {results['regret']:.4f}")
            print(f"  Discounted value:  {results['discounted_value']:.4f}")
        elif args.mode == "ground_truth":
            results.update(epic_distance=0.0, regret=0.0, discounted_value=float("nan"))
            print("\nResults (tabular, ground_truth): regret is 0 by definition.")
        else:
            epic = evaluate_epic_distance(base_env, encoder, args.gamma)
            regret, discounted_value = regret_tabular(base_env, encoder, args.gamma)
            results.update(
                epic_distance=float(epic),
                regret=float(regret),
                discounted_value=float(discounted_value),
            )
            softmax_beta = getattr(args, "regret_softmax_beta", None)
            if softmax_beta is not None:
                soft_regret, soft_value = regret_tabular(
                    base_env, encoder, args.gamma, softmax_beta=softmax_beta,
                )
                results.update(
                    soft_regret=float(soft_regret),
                    soft_discounted_value=float(soft_value),
                    regret_softmax_beta=softmax_beta,
                )
            print("\nResults (tabular):")
            print(f"  EPIC distance:     {results['epic_distance']:.4f}")
            print(f"  Regret:            {results['regret']:.4f}")
            print(f"  Discounted value:  {results['discounted_value']:.4f}")
            if softmax_beta is not None:
                print(f"  Soft regret (β={softmax_beta}):  {results['soft_regret']:.4f}")
                print(f"  Soft disc. value:  {results['soft_discounted_value']:.4f}")
    else:
        if args.optimal_policy_path is None:
            raise ValueError("--optimal_policy_path is required for non-tabular environments")

        true_policy = create_policy(
            args.optimal_policy_path,
            float("inf"),
            env=base_env,
            gamma=args.gamma,
        )

        make_eval_env = lambda: make_perturbed_env(args.env_id, env_params, seed=args.seed)

        eval_history = []
        eval_summary = []
        if args.mode in {"mavrl", "average", "ground_truth"}:
            if encoder is not None:

                def make_train_env(seed: int):
                    return LearnedRewardWrapper(
                        make_eval_env(),
                        encoder,
                        seed=seed,
                        act_transform=act_transform,
                        obs_transform=obs_transform,
                    )

            else:

                def make_train_env(seed: int):
                    return make_perturbed_env(args.env_id, env_params, seed=seed)

            print(f"Training PPO with mode={args.mode} on {args.env_id} ...")
            ppo_model, eval_history, eval_summary = train_ppo(
                make_train_env_fn=make_train_env,
                make_eval_env_fn=make_eval_env,
                seed=args.seed,
                retrain_ppo_settings=retrain_ppo_settings_from_args(args),
                true_reward_threshold=args.retrain_reward_thresh,
                verbose=args.retrain_verbose,
                progress_bar=args.retrain_pbar,
                return_training_stats=True,
            )
            est_policy = PPOExpert(ppo_model)
        else:
            q_model = NeuralQValueModel(fb_model.q_model)
            est_policy = QValueExpert(q_model, beta=float("inf"))

        regret, mean_rew = regret_non_tabular(
            true_optimal_policy=true_policy,
            est_optimal_policy=est_policy,
            eval_env_fn=make_eval_env,
            gamma=args.gamma,
            num_samples=args.num_samples,
            max_num_steps=args.max_num_steps,
            seed_fn=lambda i: args.seed * args.num_samples + i,
        )
        results.update(regret=float(regret), mean_reward=float(mean_rew))

        print("\nResults (non-tabular):")
        print(f"  Regret:       {results['regret']:.4f}")
        print(f"  Mean reward:  {results['mean_reward']:.4f}")

        if args.training_curve_output and eval_history:
            curve_path = Path(args.training_curve_output)
            curve_path.parent.mkdir(parents=True, exist_ok=True)

            # Save compact summary CSV (timestep, mean_reward)
            with open(curve_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestep", "eval_mean_reward"])
                writer.writerows(eval_summary)
            print(f"Training curve ({len(eval_summary)} evals) saved to {curve_path}")

            # Save full per-episode detail alongside it
            detail_path = curve_path.with_suffix(".detail.json")
            with open(detail_path, "w") as f:
                json.dump(
                    [{"timestep": ts, "episode_rewards": rews} for ts, rews in eval_history],
                    f, indent=2,
                )
            print(f"Per-episode eval detail saved to {detail_path}")

        if getattr(args, "log_wandb", False) and eval_summary:
            for ts, mean_r in eval_summary:
                wandb.log({"eval/timestep": ts, "eval/mean_reward": mean_r}, step=ts)

    base_env.close()

    if getattr(args, "log_wandb", False):
        wandb.log(results)
        wandb.finish()

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"Results saved to {out}")

    return results


def run_transfer(args: argparse.Namespace) -> dict:
    """Compatibility wrapper for older callers."""
    if getattr(args, "mode", None) == "reward_model":
        args.mode = "mavrl"
    if getattr(args, "fb_model_path", None) and not getattr(args, "checkpoint_path", None):
        args.checkpoint_path = args.fb_model_path
    if not hasattr(args, "checkpoint_paths"):
        args.checkpoint_paths = None
    return run_evaluation(args)


def main():
    p = argparse.ArgumentParser(
        description="Evaluate reward-model and imitation transfer experiments.",
    )

    p.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["average", "mavrl", "ground_truth", "imitation"],
        help="Evaluation mode.",
    )
    p.add_argument(
        "--checkpoint_paths",
        nargs="+",
        default=None,
        help="Multiple checkpoints for --mode average.",
    )
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Single checkpoint for --mode mavrl or --mode imitation.",
    )

    p.add_argument(
        "--env_id",
        type=str,
        required=True,
        help="Environment ID (e.g. grid_trap, LunarLander-v3)",
    )
    p.add_argument("--env_params", type=str, default="{}", help="JSON dict of env parameters")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--act_transform", type=str, default=None)
    p.add_argument("--obs_transform", type=str, default=None)

    p.add_argument(
        "--optimal_policy_path",
        type=str,
        default=None,
        help="Expert policy for regret (required for non-tabular envs)",
    )
    p.add_argument("--num_samples", type=int, default=1000, help="MC samples for regret estimation")
    p.add_argument("--max_num_steps", type=int, default=1000)
    p.add_argument(
        "--regret_softmax_beta",
        type=float,
        default=None,
        help="If set, also compute soft (Boltzmann) regret with this beta (tabular only).",
    )

    p.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    p.add_argument(
        "--training_curve_output",
        type=str,
        default=None,
        help="Path to save training curve CSV (non-tabular training modes only)",
    )

    p.add_argument("--log_wandb", action="store_true", help="Enable Weights and Biases logging")
    p.add_argument("--wandb_project", type=str, default="mavrl-transfer")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)

    add_retrain_ppo_arguments(p)

    args = p.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
