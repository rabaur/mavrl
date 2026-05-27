"""Fixed-allocation config for grid_sparse using the original paper budgets.

See grid_cliff_fixed_paper.py for the rationale shared across all
``*_fixed_paper.py`` configs.
"""

from mavrl_experiments.config_loader import load_optuna_config

_base = load_optuna_config("grid_sparse")

BASE_CONFIG = _base.BASE_CONFIG
MODALITY_PARAMS = _base.MODALITY_PARAMS
MODALITY_MAX_SAMPLES = getattr(_base, "MODALITY_MAX_SAMPLES", {})
HYPERPARAM_SEARCH_SPACE = _base.HYPERPARAM_SEARCH_SPACE

FIXED_SAMPLE_COUNTS = {"pref": 64, "demo": 1, "rating": 64, "stop": 256}
