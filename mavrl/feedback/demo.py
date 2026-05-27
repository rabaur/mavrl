from typing import Callable, Optional

import torch
import torch.nn.functional as F
import gymnasium as gym

from mavrl.feedback.base import FeedbackModule
from mavrl.types import Trajectory
from mavrl.utils.policies import Expert


def sample(
    policy: Expert,
    num_episodes: int,
    make_env_fn: Callable[[], gym.Env],
    base_seed: int,
    step_offset: int = 1,
    subsample_factor: int = 1,
    obs_transform: Optional[Callable] = None,
    act_transform: Optional[Callable] = None,
    print_stat_fn: Optional[Callable[[list[Trajectory]], None]] = None,
    min_reward_threshold: Optional[float] = None,
) -> list[Trajectory]:
    """Sample demonstrations by rolling out a (Boltzmann-rational) expert.

    The trajectory choice set is unbounded, so unlike `pref.sample` /
    `rate.sample` we cannot draw from `log_probs`; we draw from the
    environment-induced distribution by simulating the policy.
    """
    # Local import to avoid a feedback <-> data import cycle.
    from mavrl.data.utils import prepare_episodes
    return prepare_episodes(
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


def log_probs(q_values: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Boltzmann log-policy over the discrete action set.

        pi(a | s) = softmax(beta * Q(s, .))

    Args:
        q_values: Q-values for all actions, shape (..., n_actions)
        beta: Rationality parameter (scalar tensor)

    Returns:
        Log-probabilities of shape (..., n_actions).
    """
    return F.log_softmax(beta * q_values, dim=-1)


class DemonstrationsDecoder(FeedbackModule):

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        acts_curr: torch.Tensor,
        q_curr: torch.Tensor,
        beta: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the NLL of demonstrations under a Boltzmann-rational policy.

            pi(a | s) = exp(beta * Q(s, a)) / sum_a' exp(beta * Q(s, a'))

        Args:
            acts_curr: Integer action indices, shape (batch_size, 1)
            q_curr: Q-values for all actions, shape (batch_size, n_actions)
            beta: Rationality parameter (scalar tensor)
            valid: Boolean mask, shape (batch_size,) or (batch_size, 1)

        Returns:
            Scalar NLL averaged over valid entries only.
        """
        lps = log_probs(q_curr, beta)  # (B, n_actions)
        nll = -lps.gather(1, acts_curr.long().view(-1, 1)).squeeze(-1)  # (B,)

        valid_flat = valid.view(-1) if valid.dim() > 1 else valid
        nll_masked = nll * valid_flat
        return nll_masked.sum() / valid_flat.sum().clamp(min=1)
