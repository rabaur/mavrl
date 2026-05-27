"""Layer 2: per-modality dataset reproducibility.

For each feedback modality, asserts that
1. Building the dataset twice with the same seed produces bitwise-identical
   tensors and modality-specific extras (``cutpoints``, ``lambd``, ...).
2. A different seed produces *different* tensors (sanity).
3. Loading from cache yields a dataset that is bitwise-equal to a freshly
   generated one (covers the cache key + payload + slicing logic).

The tests use the tabular ``grid_trap`` env so they don't need any external
policy files and run in <2 s total.
"""

from __future__ import annotations

import argparse

import pytest

from mavrl.data.make_dataset import make_dataset
from mavrl.envs.make_env import make_env
from mavrl.types import FeedbackType
from mavrl.utils.feature_transforms import get_act_transform, get_obs_transform
from mavrl.utils.policies import TabularQValueModel
from mavrl.utils.reproducibility import seed_everything
from mavrl.utils.train_utils import _create_policies, validate_args

from tests.repro._repro_helpers import dataset_equal


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_all_datasets(args: argparse.Namespace, name: str = "train") -> dict:
    """Run the full ``make_dataset`` pipeline and return ``{fb_type: dataset}``.

    Reseeds globals before each call so that the only entropy source is the
    seed inside ``args``.
    """
    seed_everything(args.seed, quiet=True)

    def make_env_fn():
        return make_env(**vars(args))

    env = make_env_fn()
    feedback_config = {
        FeedbackType.PREF: args.n_pref_samples,
        FeedbackType.DEMO: args.n_demo_samples,
        FeedbackType.RATE: args.n_rating_samples,
        FeedbackType.STOP: args.n_stop_samples,
    }
    active = validate_args(args, feedback_config)
    policies = _create_policies(args, env, active, name)
    obs_tr = get_obs_transform(args, env)
    act_tr = get_act_transform(args, env)
    q_true = TabularQValueModel(env.unwrapped, gamma=args.gamma)

    datasets, _ = make_dataset(
        active, args, make_env_fn, policies, "cpu",
        obs_tr, act_tr, name=name, q_true=q_true,
    )
    return datasets


# ── Same seed -> bitwise equal datasets ──────────────────────────────────────


@pytest.mark.parametrize(
    "fb_type",
    [FeedbackType.PREF, FeedbackType.DEMO, FeedbackType.RATE, FeedbackType.STOP],
    ids=lambda fb: fb.value,
)
def test_dataset_same_seed_bitwise_equal(tabular_args, fb_type):
    """Same seed + same args -> identical dataset tensors and extras."""
    ds_a = _build_all_datasets(tabular_args)[fb_type]
    ds_b = _build_all_datasets(tabular_args)[fb_type]
    ok, reason = dataset_equal(ds_a, ds_b)
    assert ok, f"{fb_type.value}: {reason}"


# ── Different seed -> different datasets (sanity) ────────────────────────────


@pytest.mark.parametrize(
    "fb_type",
    [FeedbackType.PREF, FeedbackType.DEMO, FeedbackType.RATE, FeedbackType.STOP],
    ids=lambda fb: fb.value,
)
def test_dataset_different_seed_differs(tabular_args, fb_type):
    """Sanity: changing the seed must change the dataset."""
    ds_a = _build_all_datasets(tabular_args)[fb_type]

    other = argparse.Namespace(**vars(tabular_args))
    other.seed = tabular_args.seed + 1
    ds_b = _build_all_datasets(other)[fb_type]

    ok, _ = dataset_equal(ds_a, ds_b)
    assert not ok, f"{fb_type.value}: seed change did not change dataset"


# ── Cache round-trip ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fb_type",
    [FeedbackType.PREF, FeedbackType.DEMO, FeedbackType.RATE, FeedbackType.STOP],
    ids=lambda fb: fb.value,
)
def test_dataset_cache_roundtrip(tabular_args, tmp_path, fb_type):
    """Cache hit returns a dataset bitwise-equal to a fresh generation."""
    fresh = _build_all_datasets(tabular_args)[fb_type]

    # Same args, but with caching enabled.
    cached_args = argparse.Namespace(**vars(tabular_args))
    cached_args.dataset_cache_dir = str(tmp_path / "cache")
    cached_args.dataset_cache_gen_samples = None

    # First call populates the cache.
    _build_all_datasets(cached_args)
    # Second call hits the cache.
    cached = _build_all_datasets(cached_args)[fb_type]

    ok, reason = dataset_equal(fresh, cached)
    assert ok, f"{fb_type.value}: {reason}"


# ── Train vs val splits differ ───────────────────────────────────────────────


def test_train_val_splits_use_distinct_seeds(tabular_args):
    """``train`` and ``val`` splits must produce different tensors.

    Pins ``derive_seed(seed, fb, name)``: if anyone collapses the per-split
    suffix, train and val data become identical and EPIC-style overfit is
    invisible.
    """
    train_ds = _build_all_datasets(tabular_args, name="train")
    val_ds = _build_all_datasets(tabular_args, name="val")

    for fb_type in train_ds:
        ok, _ = dataset_equal(train_ds[fb_type], val_ds[fb_type])
        assert not ok, f"{fb_type.value}: train and val datasets are identical"
