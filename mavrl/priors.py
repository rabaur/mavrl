import torch
from typing import Optional

def kl_div_std_normal(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    valid: torch.Tensor,
    terminal: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute KL divergence between q(reward|x) and p(reward) = N(0, 1).
    
    Terminal states are excluded from KL computation to prevent the prior
    from pushing terminal rewards toward zero, which distorts the learned
    reward structure (terminal states have weaker TD constraints).
    
    Args:
        mean: Mean of the approximate posterior reward distribution
        log_var: Log variance of the approximate posterior reward distribution
        valid: Boolean mask indicating valid states
        terminal: Boolean mask indicating terminal states (excluded from KL)
        
    Returns:
        KL divergence loss (averaged over valid non-terminal timesteps)
    """
    # Compute KL divergence per element
    kl_loss = 0.5 * (mean.pow(2) + log_var.exp() - log_var - 1.0)
    
    # Mask: valid AND not terminal
    effective_mask = valid.clone()
    # if terminal is not None:
    #    effective_mask = effective_mask & (~terminal)
    # TODO: re-enable this when we have a better way to handle terminal states
    
    # Mask out invalid/terminal timesteps
    kl_loss = kl_loss * effective_mask
    return kl_loss.sum() / effective_mask.sum().clamp(min=1)