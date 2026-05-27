"""Dataset caching utilities for avoiding redundant data generation.

Provides save/load/hash functions so that large datasets can be pre-generated
once and reused across Optuna trials that only differ in sample count or
training-only hyperparameters (td_error_weight, kl_weight, etc.).

Per-modality save/load logic lives on each :class:`BaseFeedbackDataset` subclass
via the ``_cache_payload`` / ``_from_cache_payload`` hooks; this module is a thin
dispatcher.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

import torch

from mavrl.data.base_dataset import BaseFeedbackDataset, _slice_data, slide_demo_data
from mavrl.data.demonstration_dataset import DemonstrationDataset
from mavrl.data.preference_dataset import PreferenceDataset
from mavrl.data.rating_dataset import RatingDataset
from mavrl.data.stop_dataset import StopDataset
from mavrl.types import DataKey, FeedbackType


__all__ = [
    "compute_cache_key",
    "save_dataset_to_cache",
    "load_dataset_from_cache",
    "_cache_path",
    "_slice_data",
    "slide_demo_data",
]


_DATASET_CLS: dict[FeedbackType, type[BaseFeedbackDataset]] = {
    FeedbackType.PREF: PreferenceDataset,
    FeedbackType.DEMO: DemonstrationDataset,
    FeedbackType.RATE: RatingDataset,
    FeedbackType.STOP: StopDataset,
}


_SHARED_KEY_ATTRS = [
    "seed", "env_id", "grid_size", "p_rand",
    "obs_transform", "act_transform", "gamma", "exploration_epsilon",
]

_MODALITY_KEY_ATTRS: dict[FeedbackType, list[str]] = {
    FeedbackType.PREF: [
        "n_pref_episodes", "pref_policy_path", "pref_trajectory_rationality",
        "pref_data_rationality", "pref_seg_len", "min_reward_pref",
    ],
    FeedbackType.DEMO: [
        "demo_policy_path", "demo_rationality", "min_reward_demo",
        "step_offset", "subsample_factor",
    ],
    FeedbackType.RATE: [
        "n_rating_episodes", "rating_policy_path", "rating_trajectory_rationality",
        "rating_seg_len", "min_reward_rating",
        "rating_noise_std",
    ],
    FeedbackType.STOP: [
        "n_stop_episodes", "stop_policy_path", "stop_trajectory_rationality",
        "stop_seg_len", "stop_c", "stop_regret_percentile",
        "stop_regret_discount", "stop_q_value_model",
    ],
}

_FEEDBACK_PREFIX = {
    FeedbackType.PREF: "pref",
    FeedbackType.DEMO: "demo",
    FeedbackType.RATE: "rating",
    FeedbackType.STOP: "stop",
}


def compute_cache_key(
    feedback_type: FeedbackType,
    args,
    name: str,
) -> str:
    """Deterministic hash of all parameters that affect data generation.

    Excludes sample count (which determines slice size at load time) and
    training-only params like ``td_error_weight`` / ``kl_weight``.
    """
    key_dict: dict = {"name": name}
    for attr in _SHARED_KEY_ATTRS:
        key_dict[attr] = getattr(args, attr, None)
    for attr in _MODALITY_KEY_ATTRS[feedback_type]:
        key_dict[attr] = getattr(args, attr, None)

    raw = json.dumps(key_dict, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_path(
    cache_dir: Path,
    feedback_type: FeedbackType,
    cache_key: str,
) -> Path:
    return cache_dir / f"{_FEEDBACK_PREFIX[feedback_type]}_{cache_key}.pt"


# ── Save ─────────────────────────────────────────────────────────────────────


def save_dataset_to_cache(path: Path, dataset: BaseFeedbackDataset) -> None:
    """Persist a dataset's tensors and metadata to a ``.pt`` file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dataset._cache_payload()
    torch.save(payload, path)
    print(f"  [cache] Saved {type(dataset).__name__} ({len(dataset)} samples) -> {path}")


# ── Load ─────────────────────────────────────────────────────────────────────


def load_dataset_from_cache(
    path: Path,
    feedback_type: FeedbackType,
    n_samples: int,
    device: str,
    td_error_weight: float,
) -> Optional[BaseFeedbackDataset]:
    """Load a cached dataset, slice to ``n_samples``, and return it.

    Returns ``None`` if the cache file does not exist or has fewer samples
    than requested.
    """
    if not path.exists():
        return None

    payload = torch.load(path, map_location=device, weights_only=False)
    cached_len = _payload_len(payload, feedback_type)

    if cached_len < n_samples:
        print(
            f"  [cache] {path.name}: cached {cached_len} < requested {n_samples}; regenerating"
        )
        return None

    cls = _DATASET_CLS[feedback_type]
    ds = cls._from_cache_payload(payload, n_samples, device, td_error_weight)

    # Move tensors to target device (in case map_location differed).
    ds.data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in ds.data.items()}

    print(
        f"  [cache] Loaded {_FEEDBACK_PREFIX[feedback_type]} from cache "
        f"(cached={cached_len}, using={n_samples})"
    )
    return ds


# ── Helpers ──────────────────────────────────────────────────────────────────


def _payload_len(payload: dict, feedback_type: FeedbackType) -> int:
    """Infer the number of samples stored in a cache payload."""
    data = payload["data"]
    if feedback_type == FeedbackType.PREF:
        return int(data[DataKey.PREFERENCE].shape[0])
    elif feedback_type == FeedbackType.DEMO:
        return len(payload["episode_lengths"])
    elif feedback_type == FeedbackType.RATE:
        return int(data[DataKey.RATING].shape[0])
    elif feedback_type == FeedbackType.STOP:
        return int(data[DataKey.STOP_TIME].shape[0])
    raise ValueError(f"Unknown feedback type: {feedback_type}")
