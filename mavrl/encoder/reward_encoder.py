import torch
from torch import nn
from typing import Optional
from mavrl.utils.math import log_var_to_std
from abc import ABC, abstractmethod

class BaseRewardEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, obs: torch.Tensor, acts: torch.Tensor, next_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...
    @abstractmethod
    def sample(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        ...
    @abstractmethod
    def predict_and_sample(self, obs: torch.Tensor, acts: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        ...

class RewardEncoder(BaseRewardEncoder):

    def __init__(self, feature_module: nn.Module):
        super().__init__()
        self.features = feature_module
        self.mean_head = nn.Linear(feature_module.out_dim, 1)
        self.logvar_head = nn.Linear(feature_module.out_dim, 1)

    def forward(self, obs: torch.Tensor, acts: torch.Tensor, next_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(obs, acts, next_obs)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        # Clamp log_var to prevent numerical instability:
        # - Upper bound: exp(2) ≈ 7.4 std is reasonable for rewards
        # - Lower bound: exp(-20) ≈ 1e-9 std prevents division by zero
        logvar = torch.clamp(logvar, min=-20.0, max=2.0)
        return mean, logvar
    
    def sample(self, mean: torch.Tensor, logvar: torch.Tensor, g: Optional[torch.Generator] = None) -> torch.Tensor:
        std = log_var_to_std(logvar)
        eps = eps = torch.randn(
            std.shape,
            device=std.device,
            dtype=std.dtype,
            generator=g,
        )
        return mean + std * eps
    
    def predict_and_sample(
        self,
        obs: torch.Tensor,
        acts: torch.Tensor,
        next_obs: torch.Tensor,
        g: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        mean, logvar = self.forward(obs, acts, next_obs)
        return self.sample(mean, logvar, g)