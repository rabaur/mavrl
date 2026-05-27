from typing import Optional
import torch
from torch import nn

class MLPFeatureModule(nn.Module):
    """Module for encoding features using a MLP."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: list[int],
        reward_domain: str = 's',
        activate_last_layer: bool = True,
    ):
        super().__init__()
        
        self.reward_domain = reward_domain

        if reward_domain == 's':
            input_dim = state_dim
        elif reward_domain == 'sa':
            input_dim = state_dim + action_dim
        elif reward_domain == 'sas':
            input_dim = state_dim + action_dim + state_dim
        else:
            raise Exception(f"Unsupported reward domain '{reward_domain}'")
        hidden_sizes = [input_dim] + hidden_sizes
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if i < len(hidden_sizes) - 2 or activate_last_layer:
                layers.append(nn.LeakyReLU())
                
        self.mlp = nn.Sequential(*layers)
        self.out_dim = hidden_sizes[-1]
    
    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        next_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.reward_domain == 's':
            input = states
        elif self.reward_domain == 'sa':
            input = torch.cat([states, actions], dim=-1)
        elif self.reward_domain == 'sas':
            input = torch.cat([states, actions, next_states], dim=-1)
        else:
            raise Exception(f"Unsupported reward domain '{self.reward_domain}'")
        
        return self.mlp(input)