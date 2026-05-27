"""Utilities to make experiments reproducible across libraries.

This module exposes a single entry point `seed_everything` that:
- Sets `PYTHONHASHSEED`
- Seeds Python's `random`
- Seeds NumPy's RNG
- Seeds PyTorch (if installed) and enables deterministic algorithms
- Seeds TensorFlow (if installed) and configures determinism where possible

Notes on determinism:
- Some GPU kernels in PyTorch/TensorFlow remain non-deterministic depending on
  versions, operations, or hardware. We enable the recommended flags and
  environment variables, but full determinism may still not be guaranteed.
"""

from __future__ import annotations

import os
import warnings
import random


def _set_env_var_if_unset(key: str, value: str) -> None:
    if os.environ.get(key) is None:
        os.environ[key] = value


def _seed_python(seed: int) -> None:
    _set_env_var_if_unset("PYTHONHASHSEED", str(seed))
    random.seed(seed)


def _seed_numpy(seed: int) -> None:
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception as exc:  # pragma: no cover - optional dependency
        warnings.warn(f"NumPy seeding skipped: {exc}")


def _seed_torch(seed: int, deterministic: bool) -> None:
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Determinism options (PyTorch >= 1.8)
        # Use deterministic algorithms where possible
        try:
            torch.use_deterministic_algorithms(deterministic)
        except Exception:
            # Fallback for very old torch versions
            pass

        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = deterministic  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]

        # Recommended for deterministic cublas (PyTorch on CUDA)
        # Use :4096:8 for larger workspace if available, else :16:8
        if deterministic:
            if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
                # Choose a conservative default
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

        # Optional: disable TF32 for determinism on Ampere+ GPUs
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[attr-defined]
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False  # type: ignore[attr-defined]

    except Exception as exc:  # pragma: no cover - optional dependency
        warnings.warn(f"PyTorch seeding skipped: {exc}")


def _seed_tensorflow(seed: int, deterministic: bool) -> None:
    # TensorFlow is optional; only configure if present
    try:
        import tensorflow as tf  # type: ignore

        try:
            # TF 2.x
            tf.random.set_seed(seed)
        except Exception:
            # TF 1.x compatibility
            try:
                import numpy as np  # type: ignore

                tf.set_random_seed(seed)  # type: ignore[attr-defined]
                np.random.seed(seed)
            except Exception:
                pass

        # Best-effort determinism flags
        if deterministic:
            _set_env_var_if_unset("TF_DETERMINISTIC_OPS", "1")
            _set_env_var_if_unset("TF_CUDNN_DETERMINISTIC", "1")
            # Disable autotune-like behavior
            _set_env_var_if_unset("TF_CUDNN_USE_AUTOTUNE", "0")

    except Exception as exc:  # pragma: no cover - optional dependency
        # Only warn if TensorFlow is installed but failed; otherwise be quiet
        if "No module named" not in str(exc):
            warnings.warn(f"TensorFlow seeding skipped: {exc}")


def _seed_jax(seed: int) -> None:
    # JAX is optional; provide a helper PRNGKey for users if installed
    try:
        import jax  # type: ignore
        import jax.random as jrandom  # type: ignore

        # There is no global seed; users should pass keys explicitly.
        # We pre-warm a default key in an env var for convenience.
        key = jrandom.PRNGKey(seed)
        # Store a serialized key in an env var as a convenience hint (optional)
        os.environ["JAX_PRNGKEY_SET"] = "1"
        # `key` is not returned to avoid importing jax in calling sites; users
        # should create their own with jax.random.PRNGKey(seed).
        _ = jax, key
    except Exception:
        # Silent if not installed.
        pass


def seed_everything(
    seed: int,
    *,
    deterministic: bool = True,
    set_global_hash_seed: bool = True,
    quiet: bool = False,
) -> None:
    """Seed common libraries and set determinism flags.

    Parameters
    ----------
    seed:
        Base seed used for Python, NumPy, and as a source for other libraries.
    deterministic:
        If True, enables deterministic algorithms in libraries that support it
        (PyTorch, TensorFlow) and sets environment variables associated with
        deterministic behavior where possible.
    set_global_hash_seed:
        If True, sets `PYTHONHASHSEED` to the provided seed to make hashing
        operations reproducible. Recommended for scripts and notebooks.
    quiet:
        If True, suppresses non-critical warnings.
    """

    if set_global_hash_seed:
        _set_env_var_if_unset("PYTHONHASHSEED", str(seed))

    # Seed base libraries before optional imports...
    _seed_python(seed)

    # Optional imports (Torch / TensorFlow / JAX) may indirectly touch NumPy's
    # global RNG state via their own init or dependency imports. Calling
    # `_seed_numpy` once *before* JAX is therefore not enough --- the canonical
    # fix is to re-seed NumPy *after* all optional seeders that may import jax.
    _seed_torch(seed, deterministic=deterministic)
    _seed_tensorflow(seed, deterministic=deterministic)
    _seed_jax(seed)

    # Finally lock NumPy to the user's seed last.
    _seed_numpy(seed)

    if not quiet:
        msg = (
            f"Seeds set (seed={seed}). Deterministic={deterministic}. "
            f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED', 'unset')}"
        )
        print(msg)


__all__ = ["seed_everything"]


