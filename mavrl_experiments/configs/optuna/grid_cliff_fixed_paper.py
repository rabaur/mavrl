"""Fixed-allocation config for grid_cliff using the original paper budgets.

Per-modality counts mirror the main-text allocation used in the paper:
``n_pref = n_rating = 64, n_demo = 1, n_stop = 256``. The full hyperparameter
search space (td/kl weights, encoder size, lr, batch_size, importance-weighting
toggle) is inherited from grid_cliff.py.

Companion to grid_cliff_fixed.py (equal-share allocation). Select between them
via the ``VARIANT`` env var / ``--variant`` flag in the fixed-allocation
launcher and table builder.
"""

from mavrl_experiments.config_loader import load_optuna_config

_base = load_optuna_config("grid_cliff")

BASE_CONFIG = _base.BASE_CONFIG
MODALITY_PARAMS = _base.MODALITY_PARAMS
MODALITY_MAX_SAMPLES = getattr(_base, "MODALITY_MAX_SAMPLES", {})
HYPERPARAM_SEARCH_SPACE = _base.HYPERPARAM_SEARCH_SPACE

FIXED_SAMPLE_COUNTS = {"pref": 64, "demo": 1, "rating": 64, "stop": 256}
