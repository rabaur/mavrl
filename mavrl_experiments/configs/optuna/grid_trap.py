"""Optuna search configuration for grid_trap environment.

Extracted from configs/experiments/sweep_grid_trap.py.
Reward-model training hparams (lr, batch_size, use_importance_weights) and
td/kl weights are searched alongside the Dirichlet feedback-budget allocation.
"""

BASE_CONFIG = {
    # Environment
    "env_id": "grid_trap",
    "grid_size": 10,

    # Data
    "subsample_factor": 1,
    "step_offset": 1,
    "obs_transform": "one_hot",
    "act_transform": "one_hot",

    # Model architecture (encoder_hidden_sizes is searched)
    "reward_domain": "s",

    # Training parameters
    "num_epochs": 301,
    "gamma": 0.95,

    "use_imitation_learning": False,

    # Validation
    "retrain_verbose": 0,
    "retrain_pbar": False,
    "log_every_n_steps": 1000000, # effectively disabled
    "val_every_n_epochs": 50,
    "vis_every_n_epochs": None,
}

# Conditional hyperparameters applied when a modality has samples > 0.
# Keys must match train.py argument names.
MODALITY_PARAMS = {
    "pref": {
        "n_pref_episodes": 256,
        "pref_seg_len": 10,
        "pref_trajectory_rationality": 0.0,
        "pref_data_rationality": 5.0,
        "pref_model_rationality": 5.0,
    },
    "demo": {
        "demo_rationality": 10.0,
    },
    "rating": {
        "n_rating_episodes": 256,
        "rating_seg_len": 10,
        "rating_trajectory_rationality": 0.0,
    },
    "stop": {
        "n_stop_episodes": 256,
        "stop_seg_len": 10,
        "stop_regret_percentile": 50.0,
        "stop_regret_discount": 0.1,
        "stop_c": 2.0,
        "stop_trajectory_rationality": 0.0,
    },
}

# Hyperparameter search space.
# list  → categorical (trial.suggest_categorical)
# tuple → continuous  (trial.suggest_float with (low, high, log))
HYPERPARAM_SEARCH_SPACE = {
    "td_error_weight": (1e-3, 2.0, True),  # log-uniform; floor ≈ 0
    "kl_weight": (1e-3, 2.0, True),

    # reward model network architecture (string-encoded; decoded to list[int] in optuna_search)
    "encoder_hidden_sizes": ["16,16", "32,32", "64,64"],

    # Reward-model optimizer + importance-weighting toggle
    "lr": (1e-5, 1e-3, True),
    "batch_size": [16, 32, 64, 128],
    "use_importance_weights": [True, False],
}

# Per-modality maximum sample count (short-name keyed). Demonstrations
# saturate quickly and become numerically unstable at higher counts; cap at
# ~1/4 of the equal-budget table budget (128).
MODALITY_MAX_SAMPLES = {"demo": 16}
