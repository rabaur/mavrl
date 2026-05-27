import torch
import numpy as np
from scipy.stats import spearmanr
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from mavrl.encoder.reward_encoder import BaseRewardEncoder
from mavrl.types import DataKey


def spearman_correlation(R_true: NDArray, R_est: NDArray) -> float:
    """
    Computes the Spearman rank correlation between two reward arrays.

    Args:
        R_true: Ground truth reward array (any shape).
        R_est: Estimated reward array (same shape as R_true).

    Returns:
        Spearman correlation coefficient (in [-1, 1]).
    """
    corr, _ = spearmanr(R_true.flatten(), R_est.flatten())
    return float(corr)


def evaluate_spearmanr(
    encoder: BaseRewardEncoder,
    dataloader: DataLoader,
) -> float:
    """
    Computes Spearman rank correlation between predicted and true rewards
    over samples from a dataloader.

    Args:
        encoder: Reward encoder model.
        dataloader: DataLoader yielding batches with obs, acts, next_obs, and rewards.
            Supports both demonstration data (batch_size, T, dim) and 
            preference data (batch_size, 2, T, dim) - all shapes are flattened
            before computing correlation.

    Returns:
        Spearman correlation coefficient (in [-1, 1]).
    """
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in dataloader:
            obs = batch[DataKey.OBS]
            acts = batch[DataKey.ACT_FEATS]
            next_obs = batch[DataKey.NEXT_OBS]
            true_rewards = batch[DataKey.REWS]

            pred_mean, _ = encoder.forward(obs, acts, next_obs)
            
            # Squeeze out the last dimension from encoder output if present
            # pred_mean shape: (..., 1) -> (...)
            pred_mean = pred_mean.squeeze(-1)

            all_true.append(true_rewards.cpu().numpy())
            all_pred.append(pred_mean.cpu().numpy())

    R_true = np.concatenate(all_true).flatten()
    R_pred = np.concatenate(all_pred).flatten()

    return spearman_correlation(R_true, R_pred)