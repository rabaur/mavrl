"""Fixed-allocation config for CartPole-v1 using the original paper budgets.

Per-modality counts mirror the main-text allocation used in the paper:
``n_pref = n_rating = 256, n_demo = 4, n_stop = 64``. The full hyperparameter
search space (reward-model td/kl/lr/batch_size/IW + PPO retraining hparams) is
inherited from cartpole_v1.py.

Companion to cartpole_v1_fixed.py (equal-share allocation).
"""

from mavrl_experiments.config_loader import load_optuna_config

_base = load_optuna_config("cartpole_v1")

BASE_CONFIG = _base.BASE_CONFIG
MODALITY_PARAMS = _base.MODALITY_PARAMS
MODALITY_MAX_SAMPLES = getattr(_base, "MODALITY_MAX_SAMPLES", {})
HYPERPARAM_SEARCH_SPACE = _base.HYPERPARAM_SEARCH_SPACE

FIXED_SAMPLE_COUNTS = {"pref": 256, "demo": 4, "rating": 256, "stop": 64}
