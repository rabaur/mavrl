import torch
from abc import ABC, abstractmethod
from torch import nn

class FeedbackModule(nn.Module, ABC):

    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def forward(self, reward_samples: torch.Tensor, **kwargs) -> torch.Tensor:
        ...