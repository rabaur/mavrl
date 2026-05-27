"""Generate baseline evaluation configs from results checkpoints.

Reads per-seed checkpoints from results/models/fixed_allocation/
and writes eval_*_baseline_v2.py configs that point at the current fixed_paper
regime checkpoints (replacing the old sweep_* paths).

Run:
    python scripts/generate_baseline_configs.py
"""

import json
import re
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS_ROOT = REPO / "results" / "models" / "fixed_allocation"
CONFIGS_DIR = REPO / "mavrl_experiments" / "configs" / "experiments"

SINGLE_MODALITIES = ["demo", "pref", "rating", "stop"]

ENV_CONFIGS = {
    "grid_cliff": {
        "env_id": "grid_cliff",
        "base_config": {
            "env_id": "grid_cliff",
            "log_wandb": True,
            "encoder_hidden_sizes": [64, 64],
            "reward_domain": "s",
            "env_params.grid_size": 10,
            "env_params.p_rand": 0.0,
            "env_params.gamma": 0.95,
            "gamma": 0.95,
            "act_transform": "one_hot",
            "obs_transform": "one_hot",
            "regret_softmax_beta": 10.0,
        },
        "eval_conditions": ["average"],
        "is_tabular": True,
    },
    "grid_sparse": {
        "env_id": "grid_sparse",
        "base_config": {
            "env_id": "grid_sparse",
            "log_wandb": True,
            "encoder_hidden_sizes": [64, 64],
            "reward_domain": "s",
            "env_params.grid_size": 10,
            "env_params.p_rand": 0.0,
            "env_params.gamma": 0.95,
            "gamma": 0.95,
            "act_transform": "one_hot",
            "obs_transform": "one_hot",
            "regret_softmax_beta": 10.0,
        },
        "eval_conditions": ["average"],
        "is_tabular": True,
    },
    "grid_trap": {
        "env_id": "grid_trap",
        "base_config": {
            "env_id": "grid_trap",
            "log_wandb": True,
            "encoder_hidden_sizes": [64, 64],
            "reward_domain": "s",
            "env_params.grid_size": 10,
            "env_params.p_rand": 0.0,
            "env_params.gamma": 0.95,
            "gamma": 0.95,
            "act_transform": "one_hot",
            "obs_transform": "one_hot",
            "regret_softmax_beta": 10.0,
        },
        "eval_conditions": ["average"],
        "is_tabular": True,
    },
    "lunar_lander_v3": {
        "env_id": "lunar_lander_v3",
        "base_config": {
            "env_id": "LunarLander-v3",
            "num_samples": 1000,
            "max_num_steps": 1000,
            "gamma": 0.999,
            "retrain_verbose": 0,
            "no_progress_bar": True,
            "log_wandb": True,
            "act_transform": "one_hot",
            "encoder_hidden_sizes": [256, 256],
            "reward_domain": "sa",
        },
        "eval_conditions": ["average"],
        "is_tabular": False,
    },
}

RETRAIN_HPARAM_KEYS = [
    "retrain_learning_rate",
    "retrain_clip_range",
    "retrain_clip_range_vf",
    "retrain_ent_coef",
    "retrain_vf_coef",
    "retrain_gae_lambda",
    "retrain_max_grad_norm",
    "retrain_gamma",
    "retrain_n_steps",
    "retrain_n_minibatches",
    "retrain_n_epochs",
    "retrain_batch_size",
    "retrain_n_envs",
    "retrain_n_timesteps",
    "retrain_reward_thresh",
]


def discover_seeds(env_dir: Path, subset: str) -> set[int]:
    subset_dir = env_dir / subset
    seeds = set()
    for f in subset_dir.glob("r_model_seed*.pt"):
        m = re.match(r"r_model_seed(\d+)\.pt", f.name)
        if m:
            seeds.add(int(m.group(1)))
    return seeds


def compute_seed_intersection(env_dir: Path) -> list[int]:
    seed_sets = []
    for subset in SINGLE_MODALITIES:
        seeds = discover_seeds(env_dir, subset)
        if not seeds:
            raise ValueError(f"No seeds found for {env_dir / subset}")
        seed_sets.append(seeds)
    return sorted(set.intersection(*seed_sets))


def load_retrain_hparams(env_dir: Path) -> dict:
    hparams_path = env_dir / "pdrs" / "hparams.json"
    with open(hparams_path) as f:
        hparams = json.load(f)
    return {k: hparams[k] for k in RETRAIN_HPARAM_KEYS if k in hparams}


def format_base_config(base_config: dict, indent: int = 8) -> str:
    lines = []
    for k, v in base_config.items():
        lines.append(f"{' ' * indent}{k!r}: {v!r},")
    return "\n".join(lines)


def generate_config(env_name: str, env_cfg: dict) -> str:
    env_dir = MODELS_ROOT / env_name
    seeds = compute_seed_intersection(env_dir)
    models_base_rel = f"results/models/fixed_allocation/{env_name}"

    base_config = dict(env_cfg["base_config"])

    if not env_cfg["is_tabular"]:
        retrain_hparams = load_retrain_hparams(env_dir)
        base_config.update(retrain_hparams)

    eval_conditions = env_cfg["eval_conditions"]
    has_ground_truth = "ground_truth" in eval_conditions

    lines = []
    lines.append('"""')
    lines.append(f"Post-hoc averaging baseline evaluation grid for {base_config['env_id']}.")
    lines.append("")
    lines.append("Evaluates the naive average of four single-modality reward encoders")
    lines.append("on the unperturbed environment.")
    if has_ground_truth:
        lines.append("Also includes a ground-truth PPO baseline for comparison.")
    lines.append("")
    lines.append("Checkpoint source: results/models/fixed_allocation/")
    lines.append("(fixed_paper regime, optuna_fixalloc_gridseed20 best trials)")
    lines.append("")
    lines.append("Generated by scripts/generate_baseline_configs.py")
    lines.append('"""')
    lines.append("")
    lines.append("from pathlib import Path")
    lines.append("from mavrl_experiments.config import ExperimentGrid")
    lines.append("")
    lines.append("")
    lines.append("# ============================================================================")
    lines.append("# Checkpoint paths from results")
    lines.append("# ============================================================================")
    lines.append("")
    lines.append(f'REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)')
    lines.append(f'MODELS_BASE = f"{{REPO_ROOT}}/{models_base_rel}"')
    lines.append("")
    lines.append(f"SEEDS = {seeds}")
    lines.append("")
    lines.append("")
    lines.append("def _model_path(subset: str, seed: int) -> str:")
    lines.append('    return f"{MODELS_BASE}/{subset}/r_model_seed{seed}.pt"')
    lines.append("")
    lines.append("")
    lines.append("# ============================================================================")
    lines.append("# Grid definition")
    lines.append("# ============================================================================")
    lines.append("")
    lines.append("grid = ExperimentGrid(")
    lines.append("    base_config={")
    lines.append(format_base_config(base_config))
    lines.append("    }")
    lines.append(")")
    lines.append("")
    lines.append("grid.add(\"seed\", SEEDS)")
    lines.append("")

    if not env_cfg["is_tabular"]:
        lines.append("")
        lines.append("# ============================================================================")
        lines.append("# Environment (unperturbed)")
        lines.append("# ============================================================================")
        lines.append("")
        lines.append("grid.add(\"optimal_policy_path\", [")
        lines.append('    "~/mavrl/expert_policies/ppo/LunarLander-v3_2/best_model.zip",')
        lines.append("])")
        lines.append("")

    lines.append("")
    lines.append("# ============================================================================")
    lines.append("# Evaluation conditions")
    lines.append("# ============================================================================")
    lines.append("")
    if has_ground_truth:
        lines.append(f"grid.add(\"eval_condition\", {eval_conditions})")
        lines.append("")
        lines.append("# --- Average ensemble: average of four single-modality encoders ---")
        lines.append("grid.add_conditional(")
        lines.append('    "mode", ["average"],')
        lines.append('    condition=lambda c: c.get("eval_condition") == "average",')
        lines.append(")")
        lines.append("for s in SEEDS:")
        lines.append("    grid.add_conditional(")
        lines.append('        "checkpoint_paths",')
        lines.append("        [[")
        lines.append('            _model_path("demo", s),')
        lines.append('            _model_path("pref", s),')
        lines.append('            _model_path("rating", s),')
        lines.append('            _model_path("stop", s),')
        lines.append("        ]],")
        lines.append("        condition=lambda c, _s=s: (")
        lines.append('            c.get("eval_condition") == "average" and c.get("seed") == _s')
        lines.append("        ),")
        lines.append("    )")
    else:
        lines.append("# --- Average ensemble: average of four single-modality encoders ---")
        lines.append('grid.add("mode", ["average"])')
        lines.append("for s in SEEDS:")
        lines.append("    grid.add_conditional(")
        lines.append('        "checkpoint_paths",')
        lines.append("        [[")
        lines.append('            _model_path("demo", s),')
        lines.append('            _model_path("pref", s),')
        lines.append('            _model_path("rating", s),')
        lines.append('            _model_path("stop", s),')
        lines.append("        ]],")
        lines.append("        condition=lambda c, _s=s: c.get(\"seed\") == _s,")
        lines.append("    )")

    if has_ground_truth:
        lines.append("")
        lines.append("# --- Ground truth: train PPO on the true environment reward ---")
        lines.append("grid.add_conditional(")
        lines.append('    "mode", ["ground_truth"],')
        lines.append('    condition=lambda c: c.get("eval_condition") == "ground_truth",')
        lines.append(")")

    lines.append("")
    lines.append("")
    lines.append("# ============================================================================")
    lines.append("# Validation")
    lines.append("# ============================================================================")
    lines.append("")
    lines.append("def _validate_checkpoint_consistency(c: dict) -> bool:")
    lines.append('    mode = c.get("mode")')
    lines.append('    has_cp = c.get("checkpoint_paths") is not None')
    lines.append('    if mode == "average":')
    lines.append("        return has_cp")
    if has_ground_truth:
        lines.append('    if mode == "ground_truth":')
        lines.append("        return not has_cp")
    lines.append("    return True")
    lines.append("")
    lines.append("")
    lines.append("grid.add_validator(_validate_checkpoint_consistency)")
    lines.append("")
    lines.append("")
    lines.append('if __name__ == "__main__":')
    lines.append("    print(grid.summary())")
    lines.append("")

    return "\n".join(lines)


def main():
    config_names = {
        "grid_cliff": "eval_grid_cliff_baseline_v2",
        "grid_sparse": "eval_grid_sparse_baseline_v2",
        "grid_trap": "eval_grid_trap_baseline_v2",
        "lunar_lander_v3": "eval_lander_baseline_v2",
    }

    for env_name, env_cfg in ENV_CONFIGS.items():
        config_name = config_names[env_name]
        out_path = CONFIGS_DIR / f"{config_name}.py"

        print(f"Generating {config_name} ...")
        env_dir = MODELS_ROOT / env_name

        seeds = compute_seed_intersection(env_dir)
        print(f"  Seeds (intersection of 4 single-modality subsets): {seeds} ({len(seeds)} seeds)")

        content = generate_config(env_name, env_cfg)
        out_path.write_text(content)
        print(f"  Written to {out_path}")

    print("\nDone. Verify with:")
    for config_name in config_names.values():
        print(f"  python -c \"from mavrl_experiments.config_loader import load_experiment_config; "
              f"m = load_experiment_config('{config_name}'); print(m.grid.summary())\"")


if __name__ == "__main__":
    main()
