"""Layer 1: reproducibility primitives.

Tests the bottom of the stack -- ``seed_everything`` and ``derive_seed`` --
in isolation from any environment, dataset, or model code. Failures here mean
the rest of the suite is meaningless.
"""

from __future__ import annotations

import random
import subprocess
import sys

import numpy as np
import torch

from mavrl.data.utils import derive_seed
from mavrl.types import FeedbackType
from mavrl.utils.reproducibility import seed_everything


# ── seed_everything ──────────────────────────────────────────────────────────


def test_seed_everything_torch_idempotent():
    seed_everything(42, quiet=True)
    a = torch.randn(8)
    seed_everything(42, quiet=True)
    b = torch.randn(8)
    assert torch.equal(a, b), "torch.randn output not reproducible after seed_everything"


def test_seed_everything_numpy_idempotent():
    seed_everything(42, quiet=True)
    a = np.random.rand(8)
    seed_everything(42, quiet=True)
    b = np.random.rand(8)
    assert np.array_equal(a, b), "numpy output not reproducible after seed_everything"


_FRESH_PROCESS_PROBE = """\
import numpy as np
from mavrl.utils.reproducibility import seed_everything

seed_everything(0, quiet=True)
draws = np.random.rand(3).tolist()
print(','.join(f'{d:.17g}' for d in draws))
"""


def test_seed_everything_numpy_matches_np_seed_fresh_process():
    """In a fresh Python interpreter, ``seed_everything(s)`` must produce
    exactly the same numpy draws as a bare ``np.random.seed(s)``.

    We run the probe via ``subprocess`` because the bug we are pinning --
    a side-effect of importing jax/tensorflow inside ``seed_everything``
    that advances the global numpy state *after* ``np.random.seed(seed)``
    -- only manifests on the *first* call to ``seed_everything`` in a
    process. Once those modules are imported, subsequent calls behave
    consistently. An in-process idempotency test cannot reliably catch
    this regression because earlier tests (or even pytest's own
    discovery) may have already triggered the import.

    Anchor values are the canonical first draws of NumPy's MT19937 after
    ``np.random.seed(0)``.
    """
    out = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS_PROBE],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, f"probe failed: {out.stderr}"
    draws = [float(x) for x in out.stdout.strip().splitlines()[-1].split(",")]
    expected = [0.5488135039273248, 0.7151893663724195, 0.6027633760716439]
    assert np.allclose(draws, expected, atol=1e-12), (
        f"seed_everything(0) in a fresh process does not match np.random.seed(0). "
        f"Got {draws}, expected {expected}. "
        f"This means an import side-effect inside seed_everything (typically jax) "
        f"advances numpy state after np.random.seed(seed)."
    )


def test_seed_everything_python_idempotent():
    seed_everything(42, quiet=True)
    a = [random.random() for _ in range(8)]
    seed_everything(42, quiet=True)
    b = [random.random() for _ in range(8)]
    assert a == b, "random.random() output not reproducible after seed_everything"


def test_seed_everything_different_seeds_differ():
    seed_everything(0, quiet=True)
    a = torch.randn(8)
    seed_everything(1, quiet=True)
    b = torch.randn(8)
    assert not torch.equal(a, b), "different seeds produced identical output"


def test_seed_everything_enables_deterministic_algorithms():
    """``torch.use_deterministic_algorithms(True)`` must actually be set."""
    seed_everything(0, quiet=True)
    assert torch.are_deterministic_algorithms_enabled() is True


# ── derive_seed ──────────────────────────────────────────────────────────────


def test_derive_seed_is_stable():
    a = derive_seed(123, FeedbackType.PREF.value, "train")
    b = derive_seed(123, FeedbackType.PREF.value, "train")
    assert a == b


def test_derive_seed_distinct_per_modality():
    base = 42
    seeds = {
        fb: derive_seed(base, fb.value, "train")
        for fb in (FeedbackType.PREF, FeedbackType.DEMO,
                   FeedbackType.RATE, FeedbackType.STOP)
    }
    assert len(set(seeds.values())) == 4, f"collisions in derived seeds: {seeds}"


def test_derive_seed_distinct_per_split():
    train = derive_seed(42, FeedbackType.PREF.value, "train")
    val = derive_seed(42, FeedbackType.PREF.value, "val")
    assert train != val


def test_derive_seed_distinct_per_base_seed():
    a = derive_seed(0, FeedbackType.PREF.value, "train")
    b = derive_seed(1, FeedbackType.PREF.value, "train")
    assert a != b


def test_derive_seed_returns_nonnegative_int():
    s = derive_seed(0, FeedbackType.PREF.value, "train")
    assert isinstance(s, int)
    assert s >= 0
    # Must fit in a 32-bit unsigned int so it can be used as a torch seed.
    assert s < 2**32
