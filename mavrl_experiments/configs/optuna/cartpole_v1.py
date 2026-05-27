"""Optuna search configuration for CartPole-v1.

Training hyperparameters follow configs/experiments/sweep_cartpole.py; the
HYPERPARAM_SEARCH_SPACE mirrors lunar_lander_v3.py so that PPO retraining is
co-tuned with the reward-model td/kl weights and the Dirichlet feedback-budget
allocation.
"""

BASE_CONFIG = {
    # Environment
    "env_id": "CartPole-v1",

    # Data
    "subsample_factor": 1,
    "step_offset": 1,
    "obs_transform": None,
    "act_transform": "one_hot",

    # Model architecture (encoder_hidden_sizes is searched)
    "reward_domain": "sa",

    # Training parameters (sweep_cartpole)
    "num_epochs": 100,
    "gamma": 0.999,
    "n_regret_samples": 1000,

    "optimal_policy_path": "~/mavrl/expert_policies/dqn/CartPole-v1_1/best_model.zip",

    "use_imitation_learning": False,

    # Validation
    "retrain_verbose": 0,
    "retrain_pbar": False,
    "log_every_n_steps": 1000000,
    "val_every_n_epochs": 5,
    "vis_every_n_epochs": None,

    # ppo
    "retrain_n_envs": 4,
    "retrain_n_timesteps": 100_000,
}

# Conditional hyperparameters applied when a modality has samples > 0.
# Keys must match train.py argument names (see sweep_cartpole.py).
MODALITY_PARAMS = {
    "pref": {
        "n_pref_episodes": 1024,
        "pref_seg_len": 32,
        "pref_trajectory_rationality": 5.0,
        "pref_data_rationality": 5.0,
        "pref_model_rationality": 5.0,
        "pref_policy_path": "~/mavrl/expert_policies/dqn/CartPole-v1_1/best_model.zip",
    },
    "demo": {
        "demo_policy_path": "~/mavrl/expert_policies/dqn/CartPole-v1_1/best_model.zip",
        "demo_rationality": 5.0,
        "demo_model_rationality": 1.0,
    },
    "rating": {
        "n_rating_episodes": 1024,
        "rating_seg_len": 32,
        "rating_trajectory_rationality": 5.0,
        "rating_policy_path": "~/mavrl/expert_policies/dqn/CartPole-v1_1/best_model.zip",
    },
    "stop": {
        "n_stop_episodes": 1024,
        "stop_seg_len": 32,
        "stop_regret_percentile": 50.0,
        "stop_regret_discount": 0.1,
        "stop_c": 1.0,
        "stop_trajectory_rationality": 1.0,
        "stop_policy_path": "~/mavrl/expert_policies/dqn/CartPole-v1_1/best_model.zip",
        "stop_q_value_model": "~/mavrl/expert_policies/dqn/CartPole-v1_1/best_model.zip",
    },
}

# Hyperparameter search space: (low, high, log) per parameter.
HYPERPARAM_SEARCH_SPACE = {

    # KL/TD (log-uniform; floor ≈ 0)
    "td_error_weight": (1e-3, 2.0, True),
    "kl_weight": (1e-3, 2.0, True),

    # reward model network architecture (string-encoded; decoded to list[int] in optuna_search)
    "encoder_hidden_sizes": ["64,64", "128,128", "256,256"],

    # PPO retraining (joint search)
    "retrain_learning_rate":  (1e-5, 3e-3, True),    # log-uniform; cast to str automatically
    "retrain_clip_range":     (0.1, 0.4, False),
    "retrain_ent_coef":       (1e-8, 1e-1, True),
    "retrain_vf_coef":        (0.2, 1.0, False),
    "retrain_gae_lambda":     (0.8, 1.0, False),
    "retrain_max_grad_norm":  (0.3, 1.0, False),
    "retrain_gamma":          [0.95, 0.99, 0.995],

    # Rollout / minibatch structure.
    # batch_size is DERIVED as (n_envs * n_steps) // n_minibatches.
    "retrain_n_steps":         [256, 512, 1024, 2048],
    "retrain_n_minibatches":   [1, 2, 4, 8, 16, 32],
    "retrain_n_epochs":        [3, 5, 10, 20],

    # Reward-model optimizer + importance-weighting toggle
    "lr": (1e-5, 1e-3, True),
    "batch_size": [32, 64, 128, 256],
    "use_importance_weights": [True, False],
}

# Per-modality maximum sample count (short-name keyed). Demonstrations
# saturate quickly and become numerically unstable at higher counts; cap at
# ~1/8 of the equal-budget table budget (128).
MODALITY_MAX_SAMPLES = {"demo": 16}
