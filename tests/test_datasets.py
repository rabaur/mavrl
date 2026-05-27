"""Smoke tests for the BaseFeedbackDataset hierarchy.

Covers, for each modality:
- Empty construction via the regular constructor (``policy=None``).
- ``__len__`` is 0 on an empty dataset, 1 after a single ``add_sample``.
- ``__getitem__`` injects the expected ``_scalars()`` keys plus the data keys.
- Cache round-trip via ``_cache_payload`` / ``_from_cache_payload`` reproduces
  the data tensors and modality-specific extras.
"""

import numpy as np
import pytest
import torch

from mavrl.data.demonstration_dataset import DemonstrationDataset
from mavrl.data.preference_dataset import PreferenceDataset
from mavrl.data.rating_dataset import RatingDataset
from mavrl.data.stop_dataset import StopDataset
from mavrl.types import DataKey, FeedbackType


DEVICE = "cpu"


# ── Sample fixtures ──────────────────────────────────────────────────────────


def _pref_sample(seg_len: int = 4, obs_dim: int = 3) -> dict:
    """A single preference pair (shape (2, seg_len, ...))."""
    return {
        DataKey.OBS: torch.randn(2, seg_len, obs_dim),
        DataKey.ACTS: torch.randint(0, 4, (2, seg_len)),
        DataKey.REWS: torch.randn(2, seg_len),
        DataKey.VALID: torch.ones(2, seg_len, dtype=torch.bool),
        DataKey.PREFERENCE: torch.tensor(0.7),
    }


def _rate_sample(seg_len: int = 4, obs_dim: int = 3) -> dict:
    return {
        DataKey.OBS: torch.randn(seg_len, obs_dim),
        DataKey.ACTS: torch.randint(0, 4, (seg_len,)),
        DataKey.REWS: torch.randn(seg_len),
        DataKey.VALID: torch.ones(seg_len, dtype=torch.bool),
        DataKey.RATING: torch.tensor(2),
    }


def _stop_sample(seg_len: int = 4, obs_dim: int = 3) -> dict:
    return {
        DataKey.OBS: torch.randn(seg_len, obs_dim),
        DataKey.ACTS: torch.randint(0, 4, (seg_len,)),
        DataKey.REWS: torch.randn(seg_len),
        DataKey.VALID: torch.ones(seg_len, dtype=torch.bool),
        DataKey.STOP_TIME: torch.tensor(2),
    }


def _demo_sample(obs_dim: int = 3) -> dict:
    """A single demo transition (no batch dim)."""
    return {
        DataKey.OBS: torch.randn(obs_dim),
        DataKey.ACTS: torch.tensor(1),
        DataKey.REWS: torch.tensor(0.5),
        DataKey.VALID: torch.tensor(True),
    }


# ── Empty construction ──────────────────────────────────────────────────────


class TestEmptyConstruction:
    def test_pref_empty(self):
        ds = PreferenceDataset(device=DEVICE, beta=2.0)
        assert len(ds) == 0
        assert ds.data == {}
        assert ds.beta == 2.0
        assert torch.equal(ds._rationality, torch.tensor(2.0))

    def test_pref_model_beta_overrides_data_beta(self):
        ds = PreferenceDataset(device=DEVICE, beta=1.0, model_beta=3.5)
        assert torch.equal(ds._rationality, torch.tensor(3.5))
        assert ds.beta == 1.0  # data beta is preserved separately

    def test_rate_empty(self):
        ds = RatingDataset(device=DEVICE, gamma=0.9)
        assert len(ds) == 0
        assert ds.data == {}
        assert ds.cutpoints is None

    def test_stop_empty(self):
        ds = StopDataset(device=DEVICE, lambd=2.5, regret_discount=0.7)
        assert len(ds) == 0
        assert ds.data == {}
        assert ds.lambd == 2.5
        assert ds.regret_discount == 0.7
        assert torch.equal(ds._lambda, torch.tensor(2.5))
        assert torch.equal(ds._regret_discount, torch.tensor(0.7))

    def test_demo_empty(self):
        ds = DemonstrationDataset(device=DEVICE, beta=4.0, model_beta=2.0)
        assert len(ds) == 0
        assert ds.data == {}
        assert ds.episode_lengths == []
        assert torch.equal(ds._rationality, torch.tensor(2.0))


# ── add_sample + __len__ + __getitem__ ──────────────────────────────────────


class TestAddSampleAndGetItem:
    def test_pref_add_and_get(self):
        ds = PreferenceDataset(device=DEVICE, beta=2.0)
        sample = _pref_sample()
        ds.add_sample(sample)
        assert len(ds) == 1
        item = ds[0]
        # Scalar metadata
        assert item[DataKey.FEEDBACK_TYPE] == FeedbackType.PREF
        assert torch.equal(item[DataKey.GAMMA], ds._gamma)
        assert torch.equal(item[DataKey.TD_ERROR_WEIGHT], ds._td_error_weight)
        assert torch.equal(item[DataKey.RATIONALITY], ds._rationality)
        # Data fields preserved (shape stripped of batch dim)
        assert torch.equal(item[DataKey.OBS], sample[DataKey.OBS])
        assert torch.equal(item[DataKey.PREFERENCE], sample[DataKey.PREFERENCE])

    def test_rate_add_and_get(self):
        ds = RatingDataset(device=DEVICE)
        sample = _rate_sample()
        ds.add_sample(sample)
        assert len(ds) == 1
        item = ds[0]
        assert item[DataKey.FEEDBACK_TYPE] == FeedbackType.RATE
        assert DataKey.RATIONALITY not in item  # rate has no rationality scalar
        assert torch.equal(item[DataKey.RATING], sample[DataKey.RATING])

    def test_stop_add_and_get(self):
        ds = StopDataset(device=DEVICE, lambd=2.5, regret_discount=0.7)
        sample = _stop_sample()
        ds.add_sample(sample)
        assert len(ds) == 1
        item = ds[0]
        assert item[DataKey.FEEDBACK_TYPE] == FeedbackType.STOP
        assert torch.equal(item[DataKey.LAMBDA], ds._lambda)
        assert torch.equal(item[DataKey.REGRET_DISCOUNT], ds._regret_discount)
        assert torch.equal(item[DataKey.STOP_TIME], sample[DataKey.STOP_TIME])

    def test_demo_add_and_get(self):
        ds = DemonstrationDataset(device=DEVICE, beta=2.0)
        sample = _demo_sample()
        ds.add_sample(sample)
        assert len(ds) == 1  # __len__ uses DataKey.OBS
        item = ds[0]
        assert item[DataKey.FEEDBACK_TYPE] == FeedbackType.DEMO
        assert torch.equal(item[DataKey.RATIONALITY], ds._rationality)
        assert torch.equal(item[DataKey.OBS], sample[DataKey.OBS])

    def test_add_sample_accepts_numpy(self):
        ds = StopDataset(device=DEVICE)
        sample = {
            DataKey.OBS: np.zeros((4, 3), dtype=np.float32),
            DataKey.STOP_TIME: np.int64(2),
        }
        ds.add_sample(sample)
        assert isinstance(ds.data[DataKey.OBS], torch.Tensor)
        assert ds.data[DataKey.OBS].shape == (1, 4, 3)

    def test_multiple_adds_concatenate(self):
        ds = PreferenceDataset(device=DEVICE)
        ds.add_sample(_pref_sample())
        ds.add_sample(_pref_sample())
        ds.add_sample(_pref_sample())
        assert len(ds) == 3
        assert ds.data[DataKey.OBS].shape[0] == 3


# ── Cache round-trip ────────────────────────────────────────────────────────


class TestCacheRoundTrip:
    def _populated(self, cls, sample_fn, n=2, **kwargs):
        ds = cls(device=DEVICE, **kwargs)
        for _ in range(n):
            ds.add_sample(sample_fn())
        return ds

    def test_pref_round_trip(self):
        ds = self._populated(PreferenceDataset, _pref_sample, beta=2.5)
        payload = ds._cache_payload()
        assert payload["beta"] == 2.5
        assert payload["gamma"] == ds.gamma
        ds2 = PreferenceDataset._from_cache_payload(payload, n_samples=2, device=DEVICE, td_error_weight=1.0)
        assert len(ds2) == 2
        assert torch.equal(ds2.data[DataKey.OBS], ds.data[DataKey.OBS])
        assert ds2.beta == 2.5

    def test_rate_round_trip(self):
        ds = self._populated(RatingDataset, _rate_sample)
        ds.cutpoints = np.array([-0.5, 0.0, 0.5])
        payload = ds._cache_payload()
        assert "cutpoints" in payload
        ds2 = RatingDataset._from_cache_payload(payload, n_samples=2, device=DEVICE, td_error_weight=1.0)
        assert len(ds2) == 2
        assert np.allclose(ds2.cutpoints, ds.cutpoints)

    def test_stop_round_trip(self):
        ds = self._populated(StopDataset, _stop_sample, lambd=3.0, regret_discount=0.8)
        payload = ds._cache_payload()
        assert payload["lambd"] == 3.0
        assert payload["regret_discount"] == 0.8
        ds2 = StopDataset._from_cache_payload(payload, n_samples=2, device=DEVICE, td_error_weight=1.0)
        assert len(ds2) == 2
        assert ds2.lambd == 3.0
        assert ds2.regret_discount == 0.8

    def test_demo_round_trip(self):
        ds = DemonstrationDataset(device=DEVICE, beta=1.5)
        # Two "episodes" of 3 transitions each (demo data is flat).
        for _ in range(6):
            ds.add_sample(_demo_sample())
        ds.episode_lengths = [3, 3]
        payload = ds._cache_payload()
        assert payload["rationality"] == 1.5
        assert payload["episode_lengths"] == [3, 3]
        # Slice to 1 episode -> first 3 transitions.
        ds2 = DemonstrationDataset._from_cache_payload(payload, n_samples=1, device=DEVICE, td_error_weight=1.0)
        assert ds2.episode_lengths == [3]
        assert ds2.data[DataKey.OBS].shape[0] == 3
        assert torch.equal(ds2.data[DataKey.OBS], ds.data[DataKey.OBS][:3])


# ── Sanity: base hooks fire in correct order ────────────────────────────────


def test_post_generate_only_runs_with_policy():
    """When policy is None, _post_generate must NOT run (would clobber lambd)."""

    class TrackingStop(StopDataset):
        post_calls = 0

        def _post_generate(self):
            type(self).post_calls += 1
            super()._post_generate()

    TrackingStop.post_calls = 0
    TrackingStop(device=DEVICE, lambd=4.0, regret_discount=0.5)
    assert TrackingStop.post_calls == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
