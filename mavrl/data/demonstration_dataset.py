from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch

from mavrl.data.base_dataset import BaseFeedbackDataset, slide_demo_data
from mavrl.data.diagnostics import print_demonstration_stats
from mavrl.feedback.demo import sample as sample_demonstrations
from mavrl.types import DataKey, FeedbackType
from mavrl.utils.torch_utils import to_torch


class DemonstrationDataset(BaseFeedbackDataset):
    """Dataset for demonstration learning with expert trajectories.

    Stores all transitions flat in ``self.data`` (concatenated across
    episodes), with ``self.episode_lengths`` tracking the per-episode length
    so trajectories can be reconstructed if needed.
    """

    feedback_type = FeedbackType.DEMO
    _size_key = DataKey.OBS  # flat per-transition layout

    def __init__(
        self,
        *,
        device: str,
        beta: float = 1.0,
        model_beta: Optional[float] = None,
        gamma: float = 0.99,
        td_error_weight: float = 1.0,
        # Generation-only kwargs.
        policy: Optional[Callable] = None,
        make_env_fn: Optional[Callable[[], gym.Env]] = None,
        base_seed: Optional[int] = None,
        num_demonstrations: Optional[int] = None,
        num_steps: Optional[int] = None,
        obs_transform: Optional[Callable] = None,
        act_transform: Optional[Callable] = None,
        name: Optional[str] = "train",
        step_offset: int = 1,
        subsample_factor: int = 1,
        min_reward_threshold: Optional[float] = None,
    ):
        self.num_demonstrations = num_demonstrations
        self.num_steps = num_steps
        self.make_env_fn = make_env_fn
        self.obs_transform = obs_transform
        self.act_transform = act_transform
        self.name = name
        self.step_offset = step_offset
        self.subsample_factor = subsample_factor
        self.min_reward_threshold = min_reward_threshold

        # ``episode_lengths`` is populated by ``_generate`` (or assigned by
        # cache load); default to ``[]`` so the empty-dataset path has the
        # attribute set (used by cache save and online sample-counting).
        self.episode_lengths: list[int] = []

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
        # Mirror PreferenceDataset: ``rationality`` is what the model trains
        # against (``model_beta`` if supplied, else ``beta`` of the data).
        self.rationality = beta
        _mb = model_beta if model_beta is not None else beta
        self._rationality = torch.tensor(_mb, dtype=torch.float32, device=self.device)

    def _scalars(self) -> dict[DataKey, torch.Tensor]:
        return {**super()._scalars(), DataKey.RATIONALITY: self._rationality}

    def _generate(self, policy: Callable) -> tuple[dict, dict]:
        """Generate expert demonstration trajectories."""
        episodes = sample_demonstrations(
            policy=policy,
            num_episodes=self.num_demonstrations,
            make_env_fn=self.make_env_fn,
            base_seed=self.base_seed,
            step_offset=self.step_offset,
            subsample_factor=self.subsample_factor,
            obs_transform=self.obs_transform,
            act_transform=self.act_transform,
            print_stat_fn=lambda eps: print_demonstration_stats(eps, self.name),
            min_reward_threshold=self.min_reward_threshold,
        )

        # Sanity-check trajectory length consistency and capture per-episode lengths.
        episode_lengths: list[int] = []
        for i, e in enumerate(episodes):
            first_len = len(e[DataKey.OBS])
            assert all(len(e[k]) == first_len for k in e.keys()), (
                f"Lengths of data for trajectory {i} don't match."
            )
            episode_lengths.append(first_len)

        tensors = {
            k: to_torch(np.concatenate([e[k] for e in episodes], axis=0), self.device)
            for k in episodes[0].keys()
        }
        return tensors, {"episode_lengths": episode_lengths}

    # ── Cache integration ────────────────────────────────────────────────────

    def _cache_payload(self) -> dict:
        return {
            **super()._cache_payload(),
            "rationality": self.rationality,
            "episode_lengths": self.episode_lengths,
        }

    @classmethod
    def _from_cache_payload(cls, payload, n_samples, device, td_error_weight):
        ds = cls(
            device=device,
            beta=payload["rationality"],
            gamma=payload["gamma"],
            td_error_weight=td_error_weight,
        )
        ds.data, ds.episode_lengths = slide_demo_data(
            payload["data"], payload["episode_lengths"], n_samples
        )
        return ds
