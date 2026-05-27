from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Callable

import gymnasium as gym
import torch
from torch import nn

from mavrl.encoder.feature_modules import MLPFeatureModule
from mavrl.encoder.reward_encoder import BaseRewardEncoder, RewardEncoder
from mavrl.utils.math import log_var_to_std


class AveragedRewardEncoder(BaseRewardEncoder):
    """Averages predictions from multiple independently-trained RewardEncoders.

    Each sub-encoder is run in parallel and the final prediction is the
    equal-weight Gaussian mixture: mean is the average of individual means,
    variance is the full mixture variance (accounts for both individual
    variances and spread of the means).
    """

    def __init__(self, encoders: list[RewardEncoder]):
        super().__init__()
        if not encoders:
            raise ValueError("Need at least one encoder")

        domains = {e.features.reward_domain for e in encoders}
        if len(domains) != 1:
            raise ValueError(
                f"All encoders must share the same reward_domain, got {domains}"
            )

        self.encoders = nn.ModuleList(encoders)
        self.features = SimpleNamespace(reward_domain=domains.pop())

    def forward(
        self, obs: torch.Tensor, acts: torch.Tensor, next_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        means, logvars = [], []
        for enc in self.encoders:
            m, lv = enc.forward(obs, acts, next_obs)
            means.append(m)
            logvars.append(lv)

        means_stack = torch.stack(means)       # (N, ...)
        logvars_stack = torch.stack(logvars)    # (N, ...)
        variances = torch.exp(logvars_stack)

        avg_mean = means_stack.mean(dim=0)
        mixture_var = (variances + means_stack ** 2).mean(dim=0) - avg_mean ** 2
        mixture_var = mixture_var.clamp(min=1e-10)
        mixture_logvar = torch.log(mixture_var)

        return avg_mean, mixture_logvar

    def sample(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        g: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        std = log_var_to_std(logvar)
        eps = torch.randn(
            std.shape, device=std.device, dtype=std.dtype, generator=g,
        )
        return mean + std * eps

    def predict_and_sample(
        self,
        obs: torch.Tensor,
        acts: torch.Tensor,
        next_obs: torch.Tensor,
        g: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        mean, logvar = self.forward(obs, acts, next_obs)
        return mean # self.sample(mean, logvar, g)


def reconstruct_reward_encoder(
    checkpoint: dict,
    env: gym.Env,
    act_transform: Optional[Callable] = None,
    obs_transform: Optional[Callable] = None,
    device: str = "cpu",
) -> RewardEncoder:
    """Reconstruct only the RewardEncoder from a MultiFeedbackTypeModel checkpoint.

    This mirrors the encoder construction in ``transfer.reconstruct_multi_fb_model``
    but skips the Q-model and decoders, loading only the encoder weights.
    """
    from mavrl.utils.gym import get_act_dim, get_obs_dim

    obs_dim = get_obs_dim(env, obs_transform)
    act_dim = get_act_dim(env, act_transform)

    train_args = checkpoint.get("args", {})
    if not train_args:
        raise ValueError("Checkpoint missing 'args'; cannot reconstruct architecture.")

    encoder_hidden_sizes = train_args.get("encoder_hidden_sizes", [256, 256])
    reward_domain = train_args.get("reward_domain", "sa")

    feature_module = MLPFeatureModule(
        obs_dim, act_dim, encoder_hidden_sizes, reward_domain=reward_domain,
    )
    encoder = RewardEncoder(feature_module)
    encoder.to(device)

    full_state = checkpoint["model_state_dict"]
    encoder_prefix = "encoder."
    encoder_state = {
        k[len(encoder_prefix):]: v
        for k, v in full_state.items()
        if k.startswith(encoder_prefix)
    }
    encoder.load_state_dict(encoder_state)
    return encoder


def load_averaged_encoder(
    checkpoint_paths: list[str | Path],
    env: gym.Env,
    act_transform: Optional[Callable] = None,
    obs_transform: Optional[Callable] = None,
    device: str = "cpu",
) -> AveragedRewardEncoder:
    """Load multiple checkpoints and return an ``AveragedRewardEncoder``.

    Each checkpoint is expected to be a ``MultiFeedbackTypeModel`` state saved
    by ``train.py`` (containing ``model_state_dict`` and ``args``).
    """
    encoders: list[RewardEncoder] = []
    for path in checkpoint_paths:
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        encoder = reconstruct_reward_encoder(
            ckpt, env,
            act_transform=act_transform,
            obs_transform=obs_transform,
            device=device,
        )
        encoder.eval()
        encoders.append(encoder)

    return AveragedRewardEncoder(encoders).to(device)
