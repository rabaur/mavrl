from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch

from mavrl.data.base_dataset import BaseFeedbackDataset, _slice_data
from mavrl.data.diagnostics import (
    print_preference_pair_diagnostics,
    print_preference_stats,
)
from mavrl.feedback.pref import build_queries as build_preference_queries
from mavrl.feedback.pref import sample as sample_preferences
from mavrl.types import DataKey, FeedbackType
from mavrl.utils.policies import Expert
from mavrl.utils.torch_utils import to_torch


class PreferenceDataset(BaseFeedbackDataset):
    """Dataset for preference learning with trajectory pairs and simulated preferences."""

    feedback_type = FeedbackType.PREF
    _size_key = DataKey.PREFERENCE

    def __init__(
        self,
        *,
        device: str,
        beta: float = 1.0,
        model_beta: Optional[float] = None,
        gamma: float = 0.99,
        td_error_weight: float = 1.0,
        # Generation-only kwargs (all optional; required as a bundle when generating).
        policy: Optional[Expert] = None,
        make_env_fn: Optional[Callable[[], gym.Env]] = None,
        base_seed: Optional[int] = None,
        num_episodes: Optional[int] = None,
        num_pref_pairs: Optional[int] = None,
        segment_len: Optional[int] = None,
        obs_transform: Optional[Callable] = None,
        act_transform: Optional[Callable] = None,
        name: Optional[str] = "train",
        step_offset: int = 1,
        subsample_factor: int = 1,
        min_reward_threshold: Optional[float] = None,
    ):
        # Stash generation kwargs so _generate can read them via self.*.
        self.num_episodes = num_episodes
        self.num_pref_pairs = num_pref_pairs
        self.segment_len = segment_len
        self.make_env_fn = make_env_fn
        self.obs_transform = obs_transform
        self.act_transform = act_transform
        self.name = name
        self.step_offset = step_offset
        self.subsample_factor = subsample_factor
        self.min_reward_threshold = min_reward_threshold

        super().__init__(
            device=device,
            gamma=gamma,
            td_error_weight=td_error_weight,
            policy=policy,
            base_seed=base_seed,
            beta=beta,
            model_beta=model_beta,
        )

    def _install_modality_scalars(self, *, beta: float = 1.0, model_beta: Optional[float] = None):
        self.beta = beta
        _mb = model_beta if model_beta is not None else beta
        self._rationality = torch.tensor(_mb, dtype=torch.float32, device=self.device)

    def _scalars(self) -> dict[DataKey, torch.Tensor]:
        return {**super()._scalars(), DataKey.RATIONALITY: self._rationality}

    def _generate(self, policy: Expert) -> tuple[dict, dict]:
        """Generate trajectory pairs and Bradley-Terry preferences.

        Step 1: build pair queries via :func:`mavrl.feedback.pref.build_queries`.
        Step 2: sample preference labels via :func:`mavrl.feedback.pref.sample`,
        the canonical Bradley-Terry sampler.
        """
        pairs = build_preference_queries(
            policy=policy,
            num_pairs=self.num_pref_pairs,
            num_episodes=self.num_episodes,
            segment_len=self.segment_len,
            make_env_fn=self.make_env_fn,
            base_seed=self.base_seed,
            step_offset=self.step_offset,
            subsample_factor=self.subsample_factor,
            obs_transform=self.obs_transform,
            act_transform=self.act_transform,
            print_stat_fn=lambda eps: print_preference_stats(eps, self.name),
            min_reward_threshold=self.min_reward_threshold,
            rng=self.generator,
        )

        rewards_np = np.stack([p[DataKey.REWS] for p in pairs], axis=0).astype(np.float32)
        valid_np = np.stack([p[DataKey.VALID] for p in pairs], axis=0).astype(np.float32)
        rewards_t = torch.as_tensor(rewards_np)
        valid_t = torch.as_tensor(valid_np)
        beta_t = torch.tensor(self.beta, dtype=torch.float32)
        torch_gen = torch.Generator().manual_seed(self.base_seed)
        pref_labels = sample_preferences(
            rewards_t, valid_t, beta_t, generator=torch_gen,
        ).cpu().numpy()  # (num_pairs,)

        for pair, label in zip(pairs, pref_labels):
            pair[DataKey.PREFERENCE] = label

        print_preference_pair_diagnostics(pairs, self.beta, self.name)

        tensors = {
            k: to_torch(np.stack([p[k] for p in pairs], axis=0), self.device)
            for k in pairs[0].keys()
        }
        return tensors, {}

    # ── Cache integration ────────────────────────────────────────────────────

    def _cache_payload(self) -> dict:
        return {**super()._cache_payload(), "beta": self.beta}

    @classmethod
    def _from_cache_payload(cls, payload, n_samples, device, td_error_weight):
        ds = cls(
            device=device,
            beta=payload["beta"],
            gamma=payload["gamma"],
            td_error_weight=td_error_weight,
        )
        ds.data = _slice_data(payload["data"], n_samples)
        return ds
