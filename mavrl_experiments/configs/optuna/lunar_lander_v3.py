"""Optuna search configuration for LunarLander-v3.

Training hyperparameters follow configs/experiments/sweep_lander.py; feedback
sample counts and TD/KL search match grid_trap for equal-budget table experiments.
"""

BASE_CONFIG = {
    # Environment
    "env_id": "LunarLander-v3",

    # Data
    "subsample_factor": 1,
    "step_offset": 1,
    "obs_transform": None,
    "act_transform": "one_hot",

    # Model architecture
    "reward_domain": "sa",

    # Training parameters (sweep_lander)
    "num_epochs": 250,  # reduced from previous 1000
    "gamma": 0.999,
    "n_regret_samples": 1000,

    "optimal_policy_path": "~/mavrl/expert_policies/ppo/LunarLander-v3_2/best_model.zip",
    "min_reward_demo": 200,
    "retrain_reward_thresh": -600.0,

    "use_imitation_learning": False,

    # validation
    "retrain_verbose": 0,
    "retrain_pbar": False,
    "log_every_n_steps": 1000000,
    "val_every_n_epochs": 10,
    "vis_every_n_epochs": None,

    # ppo
    "retrain_n_envs": 4,
    "retrain_n_timesteps": 500_000,
}

# Conditional hyperparameters applied when a modality has samples > 0.
# Keys must match train.py argument names (see sweep_lander.py).
MODALITY_PARAMS = {
    "pref": {
        "n_pref_episodes": 1024,
        "pref_seg_len": 32,
        "pref_trajectory_rationality": 5.0,
        "pref_data_rationality": 5.0,
        "pref_model_rationality": 5.0,
        "pref_policy_path": "~/mavrl/expert_policies/dqn/LunarLander-v3_1/best_model.zip",
    },
    "demo": {
        "demo_policy_path": "~/mavrl/expert_policies/ppo/LunarLander-v3_2/best_model.zip",
        "demo_rationality": 1.0,
        "demo_model_rationality": 1.0,
    },
    "rating": {
        "n_rating_episodes": 1024,
        "rating_seg_len": 32,
        "rating_trajectory_rationality": 5.0,
        "rating_policy_path": "~/mavrl/expert_policies/dqn/LunarLander-v3_1/best_model.zip",
    },
    "stop": {
        "n_stop_episodes": 1024,
        "stop_seg_len": 32,
        "stop_regret_percentile": 50.0,
        "stop_regret_discount": 0.1,
        "stop_c": 1.0,
        "stop_trajectory_rationality": 1.0,
        "stop_policy_path": "~/mavrl/expert_policies/dqn/LunarLander-v3_1/best_model.zip",
        "stop_q_value_model": "~/mavrl/expert_policies/dqn/LunarLander-v3_1/best_model.zip",
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
# ~1/8 of the equal-budget table budget (256).
MODALITY_MAX_SAMPLES = {"demo": 32}
