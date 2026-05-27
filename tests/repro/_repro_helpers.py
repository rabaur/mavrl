"""Helpers for reproducibility tests.

Provides equality predicates for the kinds of objects the test suite checks
across two runs of the same experiment:

- ``state_dict_equal``: bitwise (``torch.equal``) comparison of ``state_dict``s
  (model or optimizer).
- ``dataset_equal``: bitwise comparison of every tensor in
  ``BaseFeedbackDataset.data``, plus modality-specific scalar extras
  (``cutpoints``, ``lambd``, ``regret_discount``, ``rationality``,
  ``episode_lengths``).
- ``metrics_allclose``: recursive **approximate** matching for floats in the
  same dict shape (same keys, nested dicts), using PEP-485 /
  ``math.isclose``-style semantics. Intended for non-tabular final metrics
  where Box2D, SB3, and threaded MC regret introduce tiny FP jitter.
- ``set_pinned_threading``: best-effort pinning of CPU thread count and
  determinism env vars. Idempotent and safe to call multiple times.

All comparisons return a ``(ok, reason)`` tuple so callers (especially
``tests/repro/repro_diff.py``) can surface the first divergence with useful context.
"""

from __future__ import annotations

import math
import os
from numbers import Number, Integral
from typing import Any

import numpy as np
import torch


# ── Threading / determinism env ──────────────────────────────────────────────


_THREADS_PINNED = False


def set_pinned_threading() -> None:
    """Pin CPU thread count and set determinism env vars.

    Must be called as early as possible in a test session. Re-calls are no-ops
    so it is safe to invoke from multiple fixtures. ``set_num_interop_threads``
    can only be called once per process, before any parallel work starts; we
    swallow the resulting ``RuntimeError`` if torch refuses.
    """
    global _THREADS_PINNED
    if _THREADS_PINNED:
        return

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("PYTHONHASHSEED", "0")

    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Already initialized; cannot be changed mid-process.
        pass

    _THREADS_PINNED = True


# ── Equality predicates ──────────────────────────────────────────────────────


def _tensor_equal(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, str]:
    if a.shape != b.shape:
        return False, f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"
    if a.dtype != b.dtype:
        return False, f"dtype mismatch {a.dtype} vs {b.dtype}"
    if not torch.equal(a, b):
        diff = (a.float() - b.float()).abs()
        return False, f"max abs diff {diff.max().item():.6e}"
    return True, "ok"


def state_dict_equal(
    sd1: dict[str, torch.Tensor], sd2: dict[str, torch.Tensor]
) -> tuple[bool, str]:
    """Bitwise compare two state_dicts. Returns ``(ok, reason)``."""
    if set(sd1.keys()) != set(sd2.keys()):
        diff = set(sd1.keys()).symmetric_difference(sd2.keys())
        return False, f"key mismatch: {sorted(diff)[:5]}"
    for k in sd1:
        v1, v2 = sd1[k], sd2[k]
        if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
            ok, reason = _tensor_equal(v1, v2)
            if not ok:
                return False, f"{k}: {reason}"
        elif v1 != v2:
            return False, f"{k}: {v1!r} vs {v2!r}"
    return True, "ok"


def dataset_equal(ds1, ds2) -> tuple[bool, str]:
    """Compare two ``BaseFeedbackDataset`` instances of the same modality."""
    if type(ds1) is not type(ds2):
        return False, f"type mismatch {type(ds1).__name__} vs {type(ds2).__name__}"

    keys1, keys2 = set(ds1.data.keys()), set(ds2.data.keys())
    if keys1 != keys2:
        return False, f"data key mismatch: {sorted(keys1.symmetric_difference(keys2))}"

    for k in ds1.data:
        v1, v2 = ds1.data[k], ds2.data[k]
        if isinstance(v1, torch.Tensor):
            ok, reason = _tensor_equal(v1, v2)
            if not ok:
                return False, f"data[{k}]: {reason}"
        elif isinstance(v1, np.ndarray):
            if not np.array_equal(v1, v2):
                return False, f"data[{k}]: ndarray mismatch"
        elif v1 != v2:
            return False, f"data[{k}]: {v1!r} vs {v2!r}"

    # Modality-specific extras
    for attr in ("cutpoints", "lambd", "regret_discount", "rationality",
                 "episode_lengths", "beta"):
        if hasattr(ds1, attr) and hasattr(ds2, attr):
            v1, v2 = getattr(ds1, attr), getattr(ds2, attr)
            if isinstance(v1, np.ndarray):
                if not np.array_equal(v1, v2):
                    return False, f"{attr}: ndarray mismatch"
            elif v1 != v2:
                return False, f"{attr}: {v1!r} vs {v2!r}"

    return True, "ok"


def _scalar_equal(a: Any, b: Any) -> tuple[bool, str]:
    """Compare two scalar metric values (NaN-aware)."""
    if a is None and b is None:
        return True, "ok"
    if a is None or b is None:
        return False, f"None mismatch ({a!r} vs {b!r})"
    if torch.is_tensor(a):
        a = a.item() if a.numel() == 1 else a
    if torch.is_tensor(b):
        b = b.item() if b.numel() == 1 else b
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return _tensor_equal(a, b)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            return False, f"shape {a.shape} vs {b.shape}"
        if not np.array_equal(a, b, equal_nan=True):
            return False, f"max abs diff {np.abs(a - b).max():.6e}"
        return True, "ok"
    if isinstance(a, Number) and isinstance(b, Number):
        if isinstance(a, float) and math.isnan(a) and math.isnan(b):
            return True, "ok"
        if a != b:
            return False, f"{a!r} vs {b!r}"
        return True, "ok"
    if a != b:
        return False, f"{a!r} vs {b!r}"
    return True, "ok"


def metrics_equal(m1: dict, m2: dict) -> tuple[bool, str]:
    """Recursive exact equality on metrics dicts (NaN-aware)."""
    if set(m1.keys()) != set(m2.keys()):
        diff = set(m1.keys()).symmetric_difference(m2.keys())
        return False, f"key mismatch: {sorted(diff)[:5]}"
    for k in m1:
        v1, v2 = m1[k], m2[k]
        if isinstance(v1, dict) and isinstance(v2, dict):
            ok, reason = metrics_equal(v1, v2)
            if not ok:
                return False, f"{k} -> {reason}"
        else:
            ok, reason = _scalar_equal(v1, v2)
            if not ok:
                return False, f"{k}: {reason}"
    return True, "ok"


def _scalar_allclose(a: Any, b: Any, rtol: float, atol: float) -> tuple[bool, str]:
    """Approximate equality for numeric leaves (mixed numpy / torch scalars).

    Booleans compare exactly. Integer metrics compare exactly—no rtol drift.
    Floating metrics use ``math.isclose(rel_tol=rtol, abs_tol=atol)``.
    NaN compares equal only to NaN.
    """
    if a is None and b is None:
        return True, "ok"
    if a is None or b is None:
        return False, f"None mismatch ({a!r} vs {b!r})"
    if isinstance(a, bool) or isinstance(b, bool):
        return (True, "ok") if a is b else (False, f"{a!r} vs {b!r}")
    if torch.is_tensor(a):
        a = a.item() if a.numel() == 1 else a
    if torch.is_tensor(b):
        b = b.item() if b.numel() == 1 else b
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        ok, reason = _tensor_equal(a, b)
        if ok:
            return True, "ok"
        return False, f"tensor: {reason}"
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            return False, f"shape {a.shape} vs {b.shape}"
        close = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        if not close:
            return False, f"max abs diff {np.abs(a - b).max():.6e}"
        return True, "ok"
    # Exact integers (epochs, discrete counts in metrics)—no FP slack.
    if isinstance(a, Integral) and isinstance(b, Integral):
        return (True, "ok") if a == b else (False, f"{a!r} vs {b!r}")
    if isinstance(a, Number) and isinstance(b, Number):
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            return True, "ok"
        if math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol):
            return True, "ok"
        return False, f"{fa!r} vs {fb!r} (rtol={rtol}, atol={atol})"
    if a != b:
        return False, f"{a!r} vs {b!r}"
    return True, "ok"


def metrics_allclose(
    m1: dict,
    m2: dict,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> tuple[bool, str]:
    """Recursive approximation on metrics dicts (same topology as ``metrics_equal``).

    Intended for stochastic non-tabular evaluation (physics sim + SB3 PPO +
    Monte Carlo regret), where exact bitwise reproducibility is not attainable
    but run-to-run agreement should agree within Monte Carlo sampling noise and
    floating-point error. Defaults follow a common pragmatic choice:

    ``rtol=1e-3`` (~0.1% relative difference), ``atol=1e-6`` (absolute floor
    for near-zero quantities). Comparable in spirit to ``numpy.testing.assert_allclose``
    with lenient tolerance for flaky CI on Box2d stacks.
    """
    if set(m1.keys()) != set(m2.keys()):
        diff = set(m1.keys()).symmetric_difference(m2.keys())
        return False, f"key mismatch: {sorted(diff)[:5]}"
    for k in m1:
        v1, v2 = m1[k], m2[k]
        if isinstance(v1, dict) and isinstance(v2, dict):
            ok, reason = metrics_allclose(v1, v2, rtol=rtol, atol=atol)
            if not ok:
                return False, f"{k} -> {reason}"
        else:
            ok, reason = _scalar_allclose(v1, v2, rtol, atol)
            if not ok:
                return False, f"{k}: {reason}"
    return True, "ok"


def first_param_diff(
    sd1: dict[str, torch.Tensor], sd2: dict[str, torch.Tensor]
) -> str | None:
    """Return a one-line description of the first parameter that differs.

    Used by the diagnostic CLI to surface which weight diverges first.
    """
    for k in sd1:
        if k not in sd2:
            return f"{k}: missing in second state_dict"
        v1, v2 = sd1[k], sd2[k]
        if not isinstance(v1, torch.Tensor):
            continue
        if not torch.equal(v1, v2):
            diff = (v1.float() - v2.float()).abs()
            return (
                f"{k}: shape={tuple(v1.shape)} max_abs_diff={diff.max().item():.6e}"
            )
    return None
