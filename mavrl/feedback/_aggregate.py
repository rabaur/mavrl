import torch

def aggregate_rewards(reward_samples: torch.Tensor, valid: torch.Tensor, normalize_by_length: bool = True) -> torch.Tensor:
    """
    Args:
        reward_samples: Tensor with reward samples of shape (S=(...,), T)
        valid: Validity mask, same shape as reward_samples. Bool or 0/1 valued, used multiplicatively.
        normalize_by_length: If True, use mean reward instead of sum to prevent logit saturation with long segments.
    
    Returns:
        Aggregated rewards of shape (S,)
    """
    
    # Mask invalid timesteps before aggregating rewards
    masked_rewards = reward_samples * valid  # (S, T)
    
    if normalize_by_length:
        # Use MEAN reward to prevent logit saturation with long segments
        # This keeps logits in a reasonable range regardless of segment length
        n_valid = valid.sum(dim=-1).clamp(min=1)  # (S,)
        agg = masked_rewards.sum(dim=-1) / n_valid  # (S,)
    else:
        # Use SUM of rewards (can cause saturation with long segments)
        agg = masked_rewards.sum(dim=-1)  # (S,)
    
    return agg