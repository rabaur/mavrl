from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch

from mavrl.data.base_dataset import BaseFeedbackDataset, _slice_data
from mavrl.data.diagnostics import print_stop_diagnostics, print_stop_stats
from mavrl.feedback.stop import build_queries, sample, to_sample_inputs
from mavrl.types import DataKey, FeedbackType, Trajectory
from mavrl.utils.policies import Expert, QValueModel, TabularQValueModel
from mavrl.utils.torch_utils import to_torch


class StopDataset(BaseFeedbackDataset):
    """Dataset for stop feedback learning using segments.

    Extracts random segments from episodes and simulates human termination
    behavior based on cumulative regret within each segment.
    """

    feedback_type = FeedbackType.STOP
    _size_key = DataKey.STOP_TIME

    def __init__(
        self,
        *,
        device: str,
        lambd: float = 1.0,
        regret_discount: float = 0.9,
        gamma: float = 0.99,
        td_error_weight: float = 1.0,
        # Generation-only kwargs.
        policy: Optional[Expert] = None,
        q_model: Optional[QValueModel] = None,
        make_env_fn: Optional[Callable[[], gym.Env]] = None,
        base_seed: Optional[int] = None,
        num_episodes: Optional[int] = None,
        num_samples: Optional[int] = None,
        segment_len: Optional[int] = None,
        c: float = 1.0,
        regret_percentile: float = 75.0,
        obs_transform: Optional[Callable] = None,
        act_transform: Optional[Callable] = None,
        name: Optional[str] = "train",
        step_offset: int = 1,
        subsample_factor: int = 1,
        min_reward_threshold: Optional[float] = None,
    ):
        self.num_episodes = num_episodes
        self.num_samples = num_samples
        self.segment_len = segment_len
        self.q_model = q_model
        self.make_env_fn = make_env_fn
        self.c = c
        self.regret_percentile = regret_percentile
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
            lambd=lambd,
            regret_discount=regret_discount,
        )

    def _install_modality_scalars(self, *, lambd: float = 1.0, regret_discount: float = 0.9):
        self.lambd = lambd
        self.regret_discount = regret_discount
        self._lambda = torch.tensor(lambd, dtype=torch.float32, device=self.device)
        self._regret_discount = torch.tensor(regret_discount, dtype=torch.float32, device=self.device)

    def _post_generate(self) -> None:
        # _generate updates self.lambd via extras; rebuild the tensor to match.
        self._lambda = torch.tensor(self.lambd, dtype=torch.float32, device=self.device)

    def _scalars(self) -> dict[DataKey, torch.Tensor]:
        return {
            **super()._scalars(),
            DataKey.LAMBDA: self._lambda,
            DataKey.REGRET_DISCOUNT: self._regret_discount,
        }

    # ── Generation helpers ───────────────────────────────────────────────────

    def compute_segment_regrets(self, segment: Trajectory) -> np.ndarray:
        """Compute discounted cumulative regret for a segment.

        Uses the recursive formula: R_t = regret_discount * R_{t-1} + r_t
        Regret starts at 0 at the beginning of each segment.
        """
        if isinstance(self.q_model, TabularQValueModel):
            obs = segment[DataKey.STATES]
        else:
            obs = segment[DataKey.OBS]
        acts = segment[DataKey.ACTS]
        valid = segment[DataKey.VALID]
        T = len(obs)

        instant_regrets = np.zeros(T)
        for t in range(T):
            if not valid[t]:
                continue
            q_values = self.q_model.q_values(obs[t]).squeeze()
            a_star = np.argmax(q_values)
            a = int(np.asarray(acts[t]).item())
            instant_regret = q_values[a_star] - q_values[a]
            instant_regrets[t] = max(0.0, instant_regret)

        discounted_cumsum = np.zeros(T)
        discounted_cumsum[0] = instant_regrets[0]
        for t in range(1, T):
            discounted_cumsum[t] = self.regret_discount * discounted_cumsum[t - 1] + instant_regrets[t]

        return discounted_cumsum

    def calibrate_lambda(self, cumsum_regrets: list[np.ndarray]) -> tuple[float, float]:
        """Calibrate lambda based on max regret distribution across segments."""
        max_regrets = np.array([cr.max() for cr in cumsum_regrets])
        regret_ref = np.percentile(max_regrets, self.regret_percentile)

        if regret_ref < 1e-8:
            regret_ref = 1.0

        lambd = self.c / regret_ref
        return lambd, regret_ref

    def _generate(self, policy: Expert) -> tuple[dict, dict]:
        """Generate segments and sample stop times.

        Step 1: build segment queries via :func:`mavrl.feedback.stop.build_queries`.
        Step 2: compute per-segment Q-values and calibrate ``lambd`` from the
            empirical regret distribution (kept here as dataset bookkeeping).
        Step 3: sample stop times via :func:`mavrl.feedback.stop.sample` -
            the canonical discrete-time hazard sampler.
        """
        if self.q_model is None:
            raise ValueError("StopDataset: q_model is required when generating data")

        segments = build_queries(
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
            print_stat_fn=lambda eps: print_stop_stats(eps, self.name),
            min_reward_threshold=self.min_reward_threshold,
            rng=self.generator,
        )

        # Prep `(q_values, actions, valid)` once for both the regret-based
        # lambda calibration and the canonical hazard sampler.
        q_values_t, actions_t, valid_t = to_sample_inputs(
            segments, self.q_model.q_values, device=self.device,
        )

        cumsum_regrets = [self.compute_segment_regrets(seg) for seg in segments]
        lambd, regret_ref = self.calibrate_lambda(cumsum_regrets)

        torch_gen = torch.Generator(device=q_values_t.device).manual_seed(self.base_seed)
        stop_times_t = sample(
            q_values_t,
            actions_t,
            valid_t,
            torch.tensor(lambd, dtype=torch.float32, device=q_values_t.device),
            torch.tensor(self.regret_discount, dtype=torch.float32, device=q_values_t.device),
            generator=torch_gen,
        )
        stop_times = stop_times_t.cpu().numpy()  # (B,), -1 means censored.

        for seg, st in zip(segments, stop_times):
            seg[DataKey.STOP_TIME] = int(st)

        max_regrets = np.array([cr.max() for cr in cumsum_regrets])
        print_stop_diagnostics(
            stop_times, self.segment_len, max_regrets,
            lambd, self.c, regret_ref, self.regret_discount, self.name,
        )

        tensors = {
            k: to_torch(np.stack([seg[k] for seg in segments], axis=0), self.device)
            for k in segments[0].keys() if k != DataKey.STOP_TIME
        }
        tensors[DataKey.STOP_TIME] = to_torch(stop_times, self.device)

        return tensors, {"lambd": lambd}

    # ── Cache integration ────────────────────────────────────────────────────

    def _cache_payload(self) -> dict:
        return {
            **super()._cache_payload(),
            "lambd": self.lambd,
            "regret_discount": self.regret_discount,
        }

    @classmethod
    def _from_cache_payload(cls, payload, n_samples, device, td_error_weight):
        ds = cls(
            device=device,
            lambd=payload["lambd"],
            regret_discount=payload["regret_discount"],
            gamma=payload["gamma"],
            td_error_weight=td_error_weight,
        )
        ds.data = _slice_data(payload["data"], n_samples)
        return ds
