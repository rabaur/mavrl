import math
import random
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn
import gymnasium as gym

from mavrl.feedback.base import FeedbackModule
from mavrl.feedback._aggregate import aggregate_rewards
from mavrl.types import Trajectory
from mavrl.utils.policies import Expert


def build_queries(
    policy: Expert,
    num_samples: int,
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
    """Generate `num_samples` segments ready to be labeled by `rate.sample`.

    Each returned Trajectory has shape ``(segment_len, ...)`` per key,
    sampled from a pool of episodes rolled out from `policy`.
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
    return extract_segments_from_episodes(
        episodes, segment_len, num_samples, rng=rng,
    )


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x > -math.log(2),
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )


def sample(
    reward_samples: torch.Tensor,
    valid: torch.Tensor,
    cutpoints: torch.Tensor,
    normalize_by_length: bool = True,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Samples a rating under the cumulative-logit model.
    """
    lps = log_probs(reward_samples, valid, cutpoints, normalize_by_length)  # (B, K)
    return torch.multinomial(
        lps.exp(), num_samples=1, generator=generator,
    ).squeeze(-1)


def log_probs(reward_samples: torch.Tensor, valid: torch.Tensor, cutpoints: torch.Tensor, normalize_by_length: bool = True) -> torch.Tensor:
    """
    Computes:
        P(y = k | r) = sigma(theta_k - r) - sigma(theta_{k-1} - r)
    To obtain the log-probabilities, we compute the equivalent (and more numerically stable):
        log P(y = k | r) = logsigmoid(theta_{k} - r) + log1mexp(logsigmoid(theta_{k-1} - r) - logsigmoid(theta_{k} - r))
    """
    agg = aggregate_rewards(reward_samples, valid, normalize_by_length)  # (B,)
    diff = cutpoints - agg.unsqueeze(-1)  # (B, K-1)
    pad_shape = diff.shape[:-1] + (1,)
    neg_inf = diff.new_full(pad_shape, float("-inf"))
    pos_inf = diff.new_full(pad_shape, float("inf"))
    bounds = torch.cat([neg_inf, diff, pos_inf], dim=-1)  # (B, K+1)
    theta_km1 = bounds[..., :-1]  # (B, K)
    theta_k = bounds[..., 1:]  # (B, K)
    log_sig_theta_km1 = F.logsigmoid(theta_km1)  # 0 at +inf, -inf at -inf
    log_sig_theta_k = F.logsigmoid(theta_k)
    return log_sig_theta_k + _log1mexp(log_sig_theta_km1 - log_sig_theta_k)


class RatingModule(FeedbackModule):
    """Head for predicting ordinal ratings using ordered logit model.
    
    Uses the cumulative link model with logistic (sigmoid) link function.
    
    P(y = k | r) = sigma(theta_k - r) - sigma(theta_{k-1} - r)
    
    where theta_0 = -inf and theta_K = +inf by convention.
    
    To ensure theta_1 < theta_2 < ... < theta_{K-1}, we parameterize:
    - theta_1 = raw_theta_1 (free parameter)
    - theta_k = theta_{k-1} + softplus(delta_k) for k > 1
    
    Args:
        num_categories: Number of ordinal categories (K). Default is 5 for Likert scale.
        normalize_by_length: If True, use mean reward instead of sum.
        enable_diagnostics: If True, periodically log diagnostic information.
    """

    def __init__(
        self, 
        num_categories: int = 5,
        normalize_by_length: bool = True, 
        enable_diagnostics: bool = True
    ):
        super().__init__()
        self.num_categories = num_categories
        self.normalize_by_length = normalize_by_length
        self.enable_diagnostics = enable_diagnostics
        
        # K-1 cutpoints for K categories
        # Parameterize as: theta_1 (free), then K-2 increments (softplus ensures positivity)
        self.raw_theta_1 = nn.Parameter(torch.tensor(0.0))
        if num_categories > 2:
            # Initialize increments to give roughly unit spacing after softplus
            self.delta_increments = nn.Parameter(torch.zeros(num_categories - 2))
        else:
            self.register_buffer('delta_increments', torch.empty(0))
    
    def get_cutpoints(self) -> torch.Tensor:
        """Compute ordered cutpoints from learnable parameters.
        
        Returns:
            Tensor of shape (K-1,) with strictly increasing cutpoints.
        """
        if self.num_categories == 2:
            return self.raw_theta_1.unsqueeze(0)
        
        # theta_1 is the first cutpoint
        theta_1 = self.raw_theta_1
        
        # Remaining cutpoints: theta_k = theta_{k-1} + softplus(delta_k)
        increments = F.softplus(self.delta_increments)
        cumulative_increments = torch.cumsum(increments, dim=0)
        
        # Stack all cutpoints
        cutpoints = torch.cat([
            theta_1.unsqueeze(0),
            theta_1 + cumulative_increments
        ])
        
        return cutpoints

    def forward(self, reward_samples: torch.Tensor, valid: torch.Tensor, ratings: torch.Tensor) -> torch.Tensor:
        """Compute negative log-likelihood of observed ratings.
        
        Args:
            reward_samples: Reward estimates of shape (batch_size, segment_len)
            valid: Boolean mask of shape (batch_size, segment_len)
            ratings: Ordinal ratings of shape (batch_size,), values in [0, K-1]
        
        Returns:
            Scalar NLL averaged over valid entries.
        """
        
        cutpoints = self.get_cutpoints()
        lps_all = log_probs(
            reward_samples, valid, cutpoints,
            normalize_by_length=self.normalize_by_length,
        )
        lps_obs = lps_all.gather(1, ratings.long().unsqueeze(1)).squeeze(1)
        
        # Average NLL
        return -lps_obs.mean()

