import numpy as np
from numpy.typing import NDArray
import torch

def sigmoid(x: float) -> float:
    """
    Compute numerically stable sigmoid function.
    
    Args:
        x: Input value
        
    Returns:
        Sigmoid of x
    """
    if x < 0:
        z = np.exp(x)
        return z / (1 + z)
    else:
        return 1.0 / (1.0 + np.exp(-x))


def softmax(X: NDArray, dims=tuple[int]) -> NDArray:
    with np.errstate(all='raise'):
        X -= np.max(X)
    return np.exp(X) / np.sum(np.exp(X), axis=dims, keepdims=True)


def log_var_to_std(log_var):
    return torch.exp(0.5 * log_var)