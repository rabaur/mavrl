"""File-path-based loader for experiment / optuna config files.

Configs live alongside this module (at ``mavrl_experiments/configs/`` by
default). They are still Python modules — we load them by file path rather
than by import path so that the loader works whether the package is
installed or run in-place.

Resolution order for the config root:
  1. ``$MAVRL_CONFIG_ROOT`` if set
  2. ``mavrl_experiments/configs/`` (sibling of this module)
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def config_root() -> Path:
    env = os.environ.get("MAVRL_CONFIG_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # mavrl_experiments/config_loader.py -> mavrl_experiments/configs/
    return Path(__file__).resolve().parent / "configs"


def _load_file(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_config(subdir: str, name: str) -> ModuleType:
    """Load ``<config_root>/<subdir>/<name>.py`` as a module.

    Raises ``FileNotFoundError`` with a list of available configs if missing.
    """
    root = config_root()
    path = root / subdir / f"{name}.py"
    if not path.exists():
        avail = sorted(p.stem for p in (root / subdir).glob("*.py")
                       if not p.name.startswith("_"))
        raise FileNotFoundError(
            f"Config '{name}' not found at {path}.\n"
            f"Available in {root / subdir}/: {avail}"
        )
    return _load_file(path, f"_mavrl_config_{subdir}_{name}")


def load_experiment_config(name: str) -> ModuleType:
    """Load a grid/sweep config from ``configs/experiments/``."""
    return load_config("experiments", name)


def load_optuna_config(name: str) -> ModuleType:
    """Load an Optuna search config from ``configs/optuna/``."""
    return load_config("optuna", name)


def list_configs(subdir: str) -> list[str]:
    root = config_root() / subdir
    if not root.exists():
        return []
    return sorted(p.stem for p in root.glob("*.py") if not p.name.startswith("_"))
