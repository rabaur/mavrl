"""Shared helpers for loading transfer-experiment data.

Used by the rb_*vis-transfer-*.ipynb notebooks and by plot_combined_transfer.py
to overlay the new fixalloc queue (for the non-imitation rows) with the old
queue (for the unchanged imitation row).

Usage in a notebook:

    from scripts.transfer_data_helpers import load_transfer_data
    df = load_transfer_data("grid_cliff")
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mavrl_experiments.utils import load_experiment_data  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# New fixalloc queues (non-imitation rows only).
NEW_QUEUES = {
    "grid_cliff":      EXPERIMENTS_DIR / "transfer_fixalloc_grid_cliff",
    "grid_trap":       EXPERIMENTS_DIR / "transfer_fixalloc_grid_trap",
    "acrobot_v1":      EXPERIMENTS_DIR / "transfer_fixalloc_acrobot_v1",
    "lunar_lander_v3": EXPERIMENTS_DIR / "transfer_fixalloc_lunar_lander_v3",
}
# Old queues — only used to overlay the unchanged imitation row.
OLD_QUEUES = {
    "grid_cliff":      EXPERIMENTS_DIR / "transfer_grid_cliff_25012026",
    "grid_trap":       EXPERIMENTS_DIR / "transfer_grid_trap_25012026",
    "acrobot_v1":      EXPERIMENTS_DIR / "transfer_acrobot_27012026_new_imit",
    "lunar_lander_v3": EXPERIMENTS_DIR / "transfer_lander_24012026",
}


def load_transfer_data(env_key: str, *, fallback_to_old: bool = True) -> pd.DataFrame:
    """Return the per-experiment DataFrame for a transfer env.

    Combines:
      * non-imitation rows from `NEW_QUEUES[env_key]` (the fixalloc re-runs)
      * the imitation row from `OLD_QUEUES[env_key]` (unchanged baseline)

    If the new queue does not yet exist or is empty, falls back to the OLD
    queue for *all* rows (so notebooks keep working before the cluster runs
    finish). Set ``fallback_to_old=False`` to disable that fallback.
    """
    if env_key not in NEW_QUEUES:
        raise KeyError(f"Unknown env_key: {env_key!r}. Valid: {list(NEW_QUEUES)}")

    new_q = NEW_QUEUES[env_key]
    old_q = OLD_QUEUES[env_key]

    df_new = pd.DataFrame()
    if new_q.exists():
        df_new = load_experiment_data(new_q)
    if not df_new.empty:
        df_new = df_new[df_new["config.feedback_combo"] != "imitation"]

    df_imit = pd.DataFrame()
    if old_q.exists():
        df_old = load_experiment_data(old_q)
        if not df_old.empty and "config.feedback_combo" in df_old.columns:
            df_imit = df_old[df_old["config.feedback_combo"] == "imitation"]

    if df_new.empty and fallback_to_old:
        # New queue not populated yet — return the OLD queue verbatim so the
        # notebook still runs against the pre-fixalloc baseline.
        print(f"[load_transfer_data] new queue empty → falling back to OLD: {old_q}",
              file=sys.stderr)
        return load_experiment_data(old_q) if old_q.exists() else pd.DataFrame()

    return pd.concat([df_new, df_imit], ignore_index=True)
