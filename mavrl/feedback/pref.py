import random
from typing import Callable, Optional

import numpy as np
import gymnasium as gym
import torch
import torch.nn.functional as F

from mavrl.feedback.base import FeedbackModule
from mavrl.feedback._aggregate import aggregate_rewards
from mavrl.types import Trajectory
from mavrl.utils.policies import Expert


def build_queries(
    policy: Expert,
    num_pairs: int,
    num_episodes: int,
    segment_len: int,
    make_env_fn: Callable[[], gym.Env],
    base_seed: int,
    *,
    step_offset: int = 1,
    subsample_factor: int = 1,
    obs_transform: Optional[Callable] = None,
    act_transform: Optional[Callable] = None,
    print_stat_fn: Optional[Callable[[list[Trajectory]], None]] = None,
    min_reward_threshold: Optional[float] = None,
    rng: Optional[random.Random] = None,
) -> list[Trajectory]:
    """Generate `num_pairs` segment pairs ready to be labeled by `pref.sample`.

    Each returned Trajectory has shape ``(2, segment_len, ...)`` per key,
    obtained by stacking two segments sampled from a pool of episodes
    rolled out from `policy`.
    """
    # Local import to avoid a feedback <-> data import cycle.
    from mavrl.data.utils import (
        extract_segments_from_episodes,
        prepare_episodes,
    )
    rng = rng if rng is not None else random.Random(base_seed)
    episodes = prepare_episodes(
        policy=policy,
        num_episodes=num_episodes,
        make_env_fn=make_env_fn,
        base_seed=base_seed,
        step_offset=step_offset,
        subsample_factor=subsample_factor,
        obs_transform=obs_transform,
        act_transform=act_transform,
        print_stat_fn=print_stat_fn,
        min_reward_threshold=min_reward_threshold,
    )
    segments = extract_segments_from_episodes(
        episodes, segment_len, 2 * num_pairs, rng=rng,
    )
    pairs: list[Trajectory] = []
    for i in range(0, len(segments), 2):
        seg1 = segments[i]
        seg2 = segments[i + 1]
        pairs.append({k: np.stack([seg1[k], seg2[k]], axis=0) for k in seg1})
    return pairs


def sample(
    reward_samples: torch.Tensor,
    valid: torch.Tensor,
    beta: torch.Tensor,
    normalize_by_length: bool = True,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Samples a preference under the Bradley-Terry model.
    """
    lps = log_probs(reward_samples, valid, beta, normalize_by_length)
    p_first = lps[..., 0].exp()
    return torch.bernoulli(p_first, generator=generator).long()


def log_probs(reward_samples: torch.Tensor, valid: torch.Tensor, beta: torch.Tensor, normalize_by_length: bool = True) -> torch.Tensor:
    """
    Bradley-Terry model log-probabilities.
    P(y = 0 | R) = exp(beta * r1) / (exp(beta * r1) + exp(beta * r2))

    Args:
        reward_samples: Tensor with reward samples of shape (B, 2, T)
        valid: Tensor indicating whether reward samples are valid of shape (B, 2, T)
        beta: Rationality coefficient (scalar)
        normalize_by_length: If True, use mean reward instead of sum to prevent logit saturation with long segments.
    
    Returns:
        Log-probabilities of the event that first trajectory is preferred over the second and vice versa of shape (B, 2)
    """
    
    # Mask invalid timesteps before aggregating rewards
    agg_rewards = aggregate_rewards(reward_samples, valid, normalize_by_length) # (B, 2)
    
    logits = beta * (agg_rewards[..., 0] - agg_rewards[..., 1]) # (B,)

    # Compute log probabilities for all outcomes
    return torch.stack([F.logsigmoid(logits), F.logsigmoid(-logits)], dim=-1)


class PreferenceModule(FeedbackModule):
    """Head for predicting preferences.
    
    Args:
        normalize_by_length: If True, use mean reward instead of sum to prevent logit saturation with long segments.
    """

    def __init__(self, normalize_by_length: bool = True):
        super().__init__()
        self.normalize_by_length = normalize_by_length

    def forward(self, reward_samples: torch.Tensor, valid: torch.Tensor, beta: torch.Tensor, prefs: torch.Tensor) -> torch.Tensor:
        """
        Computes the NLL-loss based on the Bradley-Terry model.

        Args:
            reward_samples: Tensor with reward samples of shape (B, 2, T)
            valid: Tensor indicating whether reward samples are valid of shape (B, 2, T)
            beta: Rationality coefficient (scalar)
            prefs: Preference labels of shape (B,). 1 if first trajectory is preferred, 0 otherwise.
        
        Returns:
            Binary cross-entropy (BCE) loss.
        """
        lps = log_probs(reward_samples, valid, beta, normalize_by_length=self.normalize_by_length)
        return -(prefs * lps[..., 0] + (1 - prefs) * lps[..., 1]).mean()

