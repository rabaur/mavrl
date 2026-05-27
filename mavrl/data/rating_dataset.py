from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch

from mavrl.data.base_dataset import BaseFeedbackDataset, _slice_data
from mavrl.data.diagnostics import print_rating_diagnostics, print_rating_stats
from mavrl.feedback.rate import build_queries as build_rating_queries
from mavrl.feedback.rate import sample as sample_ratings
from mavrl.types import DataKey, FeedbackType
from mavrl.utils.gym import get_undiscounted_return
from mavrl.utils.policies import Expert
from mavrl.utils.torch_utils import to_torch


def compute_quantile_cutpoints(returns: np.ndarray, num_categories: int) -> np.ndarray:
    """Compute cutpoints that partition returns into approximately equal bins.

    For K categories, we need K-1 cutpoints at quantiles 1/K, 2/K, ..., (K-1)/K.

    Args:
        returns: Array of trajectory returns
        num_categories: Number of ordinal categories (K)

    Returns:
        Array of K-1 cutpoints
    """
    quantiles = [(i + 1) / num_categories * 100 for i in range(num_categories - 1)]
    cutpoints = np.percentile(returns, quantiles)
    return cutpoints


class RatingDataset(BaseFeedbackDataset):
    """Dataset for ordinal rating learning with trajectory segments.

    Cutpoints are derived from the empirical (un-noised) return distribution
    so that the K categories are approximately equally populated. Ratings are
    then sampled stochastically under the cumulative-logit model via
    :func:`mavrl.feedback.rate.sample` (i.e., the same generative model that
    :class:`RatingModule` is fitting at training time).
    """

    feedback_type = FeedbackType.RATE
    _size_key = DataKey.RATING

    def __init__(
        self,
        *,
        device: str,
        gamma: float = 0.99,
        td_error_weight: float = 1.0,
        # Generation-only kwargs (all optional; required as a bundle when generating).
        policy: Optional[Expert] = None,
        make_env_fn: Optional[Callable[[], gym.Env]] = None,
        base_seed: Optional[int] = None,
        num_episodes: Optional[int] = None,
        num_samples: Optional[int] = None,
        segment_len: Optional[int] = None,
        num_categories: int = 5,
        obs_transform: Optional[Callable] = None,
        act_transform: Optional[Callable] = None,
        name: Optional[str] = "train",
        step_offset: int = 1,
        subsample_factor: int = 1,
        min_reward_threshold: Optional[float] = None,
        noise_std: float = 0.0,
    ):
        self.num_episodes = num_episodes
        self.num_samples = num_samples
        self.segment_len = segment_len
        self.make_env_fn = make_env_fn
        self.obs_transform = obs_transform
        self.act_transform = act_transform
        self.name = name
        self.step_offset = step_offset
        self.subsample_factor = subsample_factor
        self.min_reward_threshold = min_reward_threshold
        self.num_categories = num_categories
        self.noise_std = noise_std

        # ``cutpoints`` is populated by ``_generate`` (or assigned by cache load);
        # default to ``None`` so the empty-dataset path has the attribute set.
        self.cutpoints: Optional[np.ndarray] = None

        super().__init__(
            device=device,
            gamma=gamma,
            td_error_weight=td_error_weight,
            policy=policy,
            base_seed=base_seed,
        )

    def _generate(self, policy: Expert) -> tuple[dict, dict]:
        """Generate trajectory segments and sample ordinal ratings.

        Step 1: build segment queries via :func:`mavrl.feedback.rate.build_queries`.
        Step 2: derive K-1 quantile cutpoints from the empirical (un-noised)
            mean-return distribution so the K categories are approximately
            equally populated.
        Step 3: sample ratings via :func:`mavrl.feedback.rate.sample`
            (stochastic cumulative-logit model). Optional Gaussian noise on the
            latent return is injected first.
        """
        segments = build_rating_queries(
            policy=policy,
            num_samples=self.num_samples,
            num_episodes=self.num_episodes,
            segment_len=self.segment_len,
            make_env_fn=self.make_env_fn,
            base_seed=self.base_seed,
            step_offset=self.step_offset,
            subsample_factor=self.subsample_factor,
            obs_transform=self.obs_transform,
            act_transform=self.act_transform,
            print_stat_fn=lambda eps: print_rating_stats(eps, self.name),
            min_reward_threshold=self.min_reward_threshold,
            rng=self.generator,
        )

        returns = np.array(
            [get_undiscounted_return(seg) / self.segment_len for seg in segments]
        )
        cutpoints = compute_quantile_cutpoints(returns, self.num_categories)

        # Aggregated return as a single "step" so `rate.sample`'s mean-aggregation
        # is the identity, matching the cutpoints' scale.
        latent = returns
        if self.noise_std > 0.0:
            rng_np = np.random.default_rng(self.base_seed)
            latent = latent + rng_np.normal(0.0, self.noise_std, size=latent.shape)

        rewards_t = torch.as_tensor(latent, dtype=torch.float32).unsqueeze(-1)  # (B, 1)
        valid_t = torch.ones_like(rewards_t)                                     # (B, 1)
        cutpoints_t = torch.as_tensor(cutpoints, dtype=torch.float32)
        torch_gen = torch.Generator().manual_seed(self.base_seed)
        ratings = sample_ratings(
            rewards_t, valid_t, cutpoints_t, generator=torch_gen,
        ).cpu().numpy()

        for i, seg in enumerate(segments):
            seg[DataKey.RATING] = ratings[i]

        print_rating_diagnostics(
            segments, ratings, cutpoints, self.num_categories, self.name,
        )

        tensors = {
            k: to_torch(np.stack([seg[k] for seg in segments], axis=0), self.device)
            for k in segments[0].keys()
        }

        return tensors, {"cutpoints": cutpoints}

    # ── Cache integration ────────────────────────────────────────────────────

    def _cache_payload(self) -> dict:
        return {**super()._cache_payload(), "cutpoints": self.cutpoints}

    @classmethod
    def _from_cache_payload(cls, payload, n_samples, device, td_error_weight):
        ds = cls(
            device=device,
            gamma=payload["gamma"],
            td_error_weight=td_error_weight,
        )
        ds.data = _slice_data(payload["data"], n_samples)
        ds.cutpoints = payload["cutpoints"]
        return ds
