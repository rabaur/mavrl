import random
from typing import Callable, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray

from mavrl.feedback.base import FeedbackModule
from mavrl.types import DataKey, Trajectory
from mavrl.utils.policies import Expert


def to_sample_inputs(
    segments: list[Trajectory],
    q_value_fn: Callable[[NDArray], NDArray],
    device: Union[str, torch.device] = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build batched ``(q_values, actions, valid)`` tensors for ``stop.sample``.

    Parameters
    ----------
    segments
        List of segment :class:`Trajectory` dicts. Each must expose ``ACTS``,
        ``VALID``, and either ``STATES`` (preferred) or ``OBS`` for state-index
        lookups.
    q_value_fn
        Callable mapping a ``(T,)`` state-index array to a ``(T, A)`` Q-value
        array. Typical sources: ``q_model.q_values`` for a fixed model, or
        ``TabularQValueModel.from_reward_matrix(R, P, gamma).q_values`` for a
        per-posterior-sample $Q^*_R$ during IG estimation.
    device
        Target device for the returned tensors.

    Returns
    -------
    q_values
        ``(B, T, A)`` float tensor.
    actions
        ``(B, T, 1)`` long tensor. The trailing singleton matches the layout
        produced by the training dataloader (rollouts use ``np.atleast_1d``
        on the per-step action) so downstream code can pass either source
        through ``log_probs`` / ``sample`` without reshaping.
    valid
        ``(B, T)`` float tensor (0/1).
    """
    if len(segments) == 0:
        raise ValueError("segments must be non-empty")

    q_per_seg: list[NDArray] = []
    acts_per_seg: list[NDArray] = []
    valid_per_seg: list[NDArray] = []
    for seg in segments:
        states = seg.get(DataKey.STATES, seg[DataKey.OBS])
        states = np.asarray(states)
        # Tabular grid envs append a trailing singleton via `np.atleast_1d`
        # during rollout. Drop it so a state-index-based `q_value_fn` sees
        # `(T,)` instead of `(T, 1)`.
        if states.ndim >= 2 and states.shape[-1] == 1:
            states = states.squeeze(-1)
        q_per_seg.append(np.asarray(q_value_fn(states)))
        # Normalize to ``(T, 1)`` so the stacked tensor matches the training
        # dataloader contract (which preserves the ``np.atleast_1d`` singleton).
        acts_per_seg.append(np.asarray(seg[DataKey.ACTS]).reshape(-1, 1))
        valid_per_seg.append(np.asarray(seg[DataKey.VALID]))

    q_np = np.stack(q_per_seg, axis=0).astype(np.float32)         # (B, T, A)
    acts_np = np.stack(acts_per_seg, axis=0).astype(np.int64)     # (B, T, 1)
    valid_np = np.stack(valid_per_seg, axis=0).astype(np.float32) # (B, T)

    q_t = torch.as_tensor(q_np, device=device)
    acts_t = torch.as_tensor(acts_np, device=device)
    valid_t = torch.as_tensor(valid_np, device=device)
    return q_t, acts_t, valid_t


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
    """Generate `num_samples` segments ready to be labeled by `stop.sample`.

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


def sample(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    valid: torch.Tensor,
    lambd: torch.Tensor,
    regret_discount: torch.Tensor,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Samples a stop time under the discrete-time hazard model."""
    lps = log_probs(q_values, actions, valid, lambd, regret_discount)  # (B, T+1)
    probs = lps.exp()
    idx = torch.multinomial(
        probs, num_samples=1, generator=generator,
    ).squeeze(-1)
    T = q_values.shape[1]
    return torch.where(idx == T, torch.full_like(idx, -1), idx)


def log_probs(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    valid: torch.Tensor,
    lambd: torch.Tensor,
    regret_discount: torch.Tensor,
) -> torch.Tensor:
    """Discrete-time hazard log-probabilities over the full outcome space.

    Outcomes are indexed ``0..T-1`` for "stop at step t" and ``T`` for
    "censored" (no stop). Each row of the returned tensor sums to 1 in
    probability space:

        P(stop=tau) = h_tau * prod_{t<tau, valid} (1 - h_t)         for valid tau
        P(stop=tau) = 0                                              for invalid tau
        P(censored) = prod_{t valid} (1 - h_t)

    where ``h_t = 1 - exp(-lambda * R_t)`` and ``R_t`` is the discounted
    cumulative regret ``R_t = regret_discount * R_{t-1} + r_t`` with
    instantaneous regret ``r_t = (max_a Q(s,a) - Q(s, a_taken)) * valid``.

    Args:
        q_values: Q-values for all actions, shape (B, T, A).
        actions: Taken actions, shape (B, T, 1). The trailing singleton
            matches the dataloader/`to_sample_inputs` contract and lets
            ``q_values.gather`` consume it directly.
        valid: Validity mask, shape (B, T). 0/1-valued or bool.
        lambd: Stop sensitivity parameter, scalar or shape (B,).
        regret_discount: Discount factor for old regret in [0, 1], scalar or
            shape (B,).

    Returns:
        Log-probabilities of shape (B, T+1) over outcomes.
    """
    batch_size, seq_len, _ = q_values.shape
    device = q_values.device

    valid_f = valid.to(q_values.dtype)

    q_max, _ = q_values.max(dim=-1)  # (B, T)
    q_taken = q_values.gather(dim=-1, index=actions.long()).squeeze(-1)
    instant_regret = (q_max - q_taken) * valid_f  # (B, T)

    if regret_discount.dim() == 0:
        regret_discount = regret_discount.unsqueeze(0)
    discount = regret_discount.view(-1)  # (B,) or (1,)

    cum_regret = torch.zeros(batch_size, seq_len, device=device, dtype=q_values.dtype)
    cum_regret[:, 0] = instant_regret[:, 0]
    for t in range(1, seq_len):
        cum_regret[:, t] = discount * cum_regret[:, t - 1] + instant_regret[:, t]

    if lambd.dim() == 0:
        lambd = lambd.unsqueeze(0)
    exponent = (lambd.unsqueeze(-1) * cum_regret).clamp(min=-20.0, max=20.0)
    hazard = (1.0 - torch.exp(-exponent)).clamp(min=1e-8, max=1.0 - 1e-8)
    # Invalid timesteps cannot be stop times: zero out their hazard so they
    # also contribute zero to the survival product (log(1-0) = 0).
    hazard = hazard * valid_f

    log_survival = torch.log1p(-hazard)  # (B, T)
    # Cumulative survival up to (and excluding) step t.
    cum_log_survival = torch.cumsum(log_survival, dim=-1)  # (B, T)
    log_survival_before = torch.cat(
        [torch.zeros(batch_size, 1, device=device, dtype=q_values.dtype),
         cum_log_survival[:, :-1]],
        dim=-1,
    )  # (B, T): sum_{s<t} log(1-h_s)

    # Hazard log-prob per step; -inf for invalid steps so they have probability 0.
    log_hazard = torch.where(
        valid_f > 0,
        torch.log(hazard.clamp(min=1e-8)),
        torch.full_like(hazard, float("-inf")),
    )

    log_p_stop = log_survival_before + log_hazard  # (B, T)
    log_p_censored = cum_log_survival[:, -1:]  # (B, 1)
    return torch.cat([log_p_stop, log_p_censored], dim=-1)  # (B, T+1)


class StopModule(FeedbackModule):
    """Thin head that wraps :func:`log_probs` for the discrete-time hazard model.

    Negative log-likelihood of observed stop times, where ``stop_times == -1``
    indicates a censored observation (no stop). Censoring is mapped to outcome
    index ``T`` and gathered alongside the regular stop indices.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        q_values: torch.Tensor,
        actions: torch.Tensor,
        stop_times: torch.Tensor,
        lambd: torch.Tensor,
        regret_discount: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        lps = log_probs(q_values, actions, valid, lambd, regret_discount)  # (B, T+1)
        T = q_values.shape[1]
        idx = torch.where(
            stop_times >= 0,
            stop_times.long(),
            torch.full_like(stop_times, T, dtype=torch.long),
        )
        observed = lps.gather(1, idx.unsqueeze(1)).squeeze(1)  # (B,)
        return -observed.mean()
