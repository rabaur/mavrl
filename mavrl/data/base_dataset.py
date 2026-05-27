"""Base class for feedback datasets.

Centralizes the boilerplate that was duplicated across PreferenceDataset,
RatingDataset, StopDataset, and DemonstrationDataset:

- Common scalar tensor construction (gamma, td_error_weight)
- ``add_sample`` (per-sample ``unsqueeze(0)`` + ``torch.cat``)
- ``__len__`` (size lookup via ``_size_key``)
- ``__getitem__`` (scalar metadata injection via ``_scalars``)
- Cache save/load dispatch (``_cache_payload`` / ``_from_cache_payload``)

A single constructor handles both the "generate now" and "construct empty"
paths: pass ``policy=None`` to skip data generation (used for online/active
learning and cache loading).
"""

from __future__ import annotations

import random
from typing import Any, ClassVar, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from mavrl.types import DataKey, FeedbackType
from mavrl.utils.torch_utils import to_torch


# ── Tensor-slicing helpers (used by cache load and post-cache slicing) ───────


def _slice_data(data: dict, n: int) -> dict:
    """Slice every tensor in ``data`` along dim-0 to the first ``n`` rows."""
    return {k: v[:n] if isinstance(v, torch.Tensor) else v for k, v in data.items()}


def slide_demo_data(
    data: dict, episode_lengths: list[int], n_episodes: int
) -> tuple[dict, list[int]]:
    """Slice demonstration data to the first ``n_episodes`` episodes.

    Demo tensors are flat (all transitions concatenated), so the transition
    offset is computed from ``episode_lengths``.
    """
    kept_lengths = episode_lengths[:n_episodes]
    n_transitions = sum(kept_lengths)
    sliced = {
        k: v[:n_transitions] if isinstance(v, torch.Tensor) else v
        for k, v in data.items()
    }
    return sliced, kept_lengths


class BaseFeedbackDataset(Dataset):
    """Common base for all modality datasets.

    Subclasses must set the class attributes ``feedback_type`` and (unless the
    default ``DataKey.OBS`` is correct) ``_size_key``, and implement the
    ``_generate`` hook. They typically also override ``_install_modality_scalars``
    to build modality-specific scalar tensors (e.g. ``_rationality``, ``_lambda``)
    and ``_scalars`` to inject those into each sample.
    """

    feedback_type: ClassVar[FeedbackType]
    _size_key: ClassVar[DataKey] = DataKey.OBS

    def __init__(
        self,
        *,
        device: str,
        gamma: float = 0.99,
        td_error_weight: float = 1.0,
        policy: Optional[Any] = None,
        base_seed: Optional[int] = None,
        **modality_kwargs,
    ):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.td_error_weight = td_error_weight
        self._gamma = torch.tensor(gamma, dtype=torch.float32, device=device)
        self._td_error_weight = torch.tensor(td_error_weight, dtype=torch.float32, device=device)

        self._install_modality_scalars(**modality_kwargs)

        if policy is None:
            self.data = {}
            return

        if base_seed is None:
            raise ValueError(
                f"{type(self).__name__}: base_seed is required when policy is provided"
            )
        self.base_seed = base_seed
        self.generator = random.Random(base_seed)

        data, extras = self._generate(policy)
        self.data = data
        for k, v in extras.items():
            setattr(self, k, v)
        self._post_generate()

    # ── Hooks (overridable by subclasses) ────────────────────────────────────

    def _install_modality_scalars(self, **kwargs) -> None:
        """Build modality-specific scalar tensors from constructor kwargs.

        Called once during ``__init__`` regardless of whether data is being
        generated. Subclasses should accept their modality-specific scalar
        kwargs (e.g. ``beta``, ``model_beta``, ``lambd``, ``regret_discount``)
        and assign the corresponding ``self._*`` torch tensors.
        """

    def _generate(self, policy) -> tuple[dict, dict]:
        """Generate data tensors from a policy.

        Returns
        -------
        data : dict[DataKey, torch.Tensor]
            The contiguous tensor dict to assign to ``self.data``.
        extras : dict[str, Any]
            Auxiliary attributes the base class will assign via
            ``setattr(self, k, v)`` (e.g. ``{"cutpoints": ...}``,
            ``{"lambd": ...}``, ``{"episode_lengths": [...]}``).
        """
        raise NotImplementedError

    def _post_generate(self) -> None:
        """Hook run after ``_generate`` has populated ``self.data`` and extras.

        Override when a scalar tensor depends on a value produced by
        ``_generate`` (e.g. ``StopDataset`` rebuilds ``self._lambda`` from the
        calibrated ``self.lambd``).
        """

    def _scalars(self) -> dict[DataKey, torch.Tensor]:
        """Per-sample scalar metadata to inject in ``__getitem__``.

        Default returns the common ``FEEDBACK_TYPE``, ``GAMMA`` and
        ``TD_ERROR_WEIGHT``. Subclasses should extend via
        ``{**super()._scalars(), DataKey.X: self._x}``.
        """
        return {
            DataKey.FEEDBACK_TYPE: self.feedback_type,
            DataKey.GAMMA: self._gamma,
            DataKey.TD_ERROR_WEIGHT: self._td_error_weight,
        }

    # ── Cache hooks ──────────────────────────────────────────────────────────

    def _cache_payload(self) -> dict:
        """Build the dict to ``torch.save`` for this dataset.

        Subclasses extend with their modality-specific extras (``beta``,
        ``cutpoints``, ``lambd``, ``regret_discount``, ``rationality``,
        ``episode_lengths``).
        """
        return {
            "data": self.data,
            "feedback_type": type(self).__name__,
            "gamma": self.gamma,
        }

    @classmethod
    def _from_cache_payload(
        cls,
        payload: dict,
        n_samples: int,
        device: str,
        td_error_weight: float,
    ) -> "BaseFeedbackDataset":
        """Reconstruct a dataset from a cache payload.

        Default implementation builds an empty instance via the regular
        constructor (no ``policy``) using ``payload["gamma"]`` and the supplied
        ``td_error_weight``, then slices ``payload["data"]`` to ``n_samples``.
        Subclasses override to forward additional scalar kwargs and to populate
        modality-specific extras (e.g. ``cutpoints``, ``episode_lengths``).
        """
        ds = cls(device=device, gamma=payload["gamma"], td_error_weight=td_error_weight)
        ds.data = _slice_data(payload["data"], n_samples)
        return ds

    # ── Shared API ───────────────────────────────────────────────────────────

    def add_sample(self, sample: dict) -> None:
        """Append a single sample to ``self.data``.

        Each value is unsqueezed along dim 0 (introducing a fresh batch
        dimension) and concatenated with the existing tensor under the same
        key. Numpy arrays are converted to torch tensors on ``self.device``.
        """
        for k, v in sample.items():
            if not isinstance(v, torch.Tensor):
                v = to_torch(np.asarray(v), self.device)
            v = v.unsqueeze(0)
            if k in self.data:
                self.data[k] = torch.cat([self.data[k], v], dim=0)
            else:
                self.data[k] = v

    def __len__(self) -> int:
        if not self.data:
            return 0
        return int(self.data[self._size_key].shape[0])

    def __getitem__(self, idx) -> dict:
        item = dict(self._scalars())
        for k, v in self.data.items():
            item[k] = v[idx]
        return item
