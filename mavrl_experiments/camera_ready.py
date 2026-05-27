"""Utilities for loading results from the camera-ready directory layout.

The camera-ready layout stores per-(setting, env, subset) cells as:

    {root}/models/{setting}/{env}/{subset}/hparams.json
    {root}/models/{setting}/{env}/{subset}/r_model_seed*.pt

This module provides ``load_camera_ready_best`` which returns ``(value, trial)``
in the same shape as ``optuna_budget_table.load_study_best``, so existing table
and EPIC code can consume camera-ready results without changes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace


def _pseudo_trial(hparams: dict) -> SimpleNamespace:
    """Build a minimal object mimicking an Optuna FrozenTrial."""
    meta = hparams.get("_meta", {})
    return SimpleNamespace(
        number=meta.get("best_trial_number"),
        value=meta.get("best_trial_value"),
        params={k: v for k, v in hparams.items() if k != "_meta"},
        user_attrs={
            "per_seed_values": meta.get("per_seed_values", []),
            "sample_counts": meta.get("sample_counts", {}),
        },
    )


def load_camera_ready_best(
    camera_ready_root: Path,
    setting: str,
    env: str,
    subset: str,
):
    """Load best-trial value and pseudo-trial from camera-ready layout.

    Returns ``(value, pseudo_trial)`` or ``(None, None)`` if the cell is
    missing, matching the signature of ``load_study_best``.
    """
    hparams_path = camera_ready_root / "models" / setting / env / subset / "hparams.json"
    if not hparams_path.exists():
        return None, None
    hparams = json.loads(hparams_path.read_text())
    meta = hparams.get("_meta", {})
    value = meta.get("best_trial_value")
    if value is None:
        return None, None
    return value, _pseudo_trial(hparams)


def list_model_paths(
    camera_ready_root: Path,
    setting: str,
    env: str,
    subset: str,
) -> list[Path]:
    """Return sorted model checkpoint paths for a camera-ready cell."""
    cell_dir = camera_ready_root / "models" / setting / env / subset
    if not cell_dir.is_dir():
        return []
    paths = sorted(
        cell_dir.glob("r_model_seed*.pt"),
        key=lambda p: int(re.search(r"seed(\d+)", p.stem).group(1)),
    )
    return paths
