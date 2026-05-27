"""Fixed-allocation config for LunarLander-v3 using the published paper budgets.

Per-modality counts mirror the main-text allocation used in the paper:
``n_pref = n_rating = n_stop = 256, n_demo = 32``. The full hyperparameter
search space (reward-model td/kl/lr/batch_size/IW + PPO retraining hparams)
is inherited from lunar_lander_v3.py.
"""

from mavrl_experiments.config_loader import load_optuna_config

_base = load_optuna_config("lunar_lander_v3")

BASE_CONFIG = _base.BASE_CONFIG
MODALITY_PARAMS = _base.MODALITY_PARAMS
MODALITY_MAX_SAMPLES = getattr(_base, "MODALITY_MAX_SAMPLES", {})
HYPERPARAM_SEARCH_SPACE = _base.HYPERPARAM_SEARCH_SPACE

FIXED_SAMPLE_COUNTS = {"pref": 256, "demo": 32, "rating": 256, "stop": 256}
