import torch
import torch.nn.functional as F

from mavrl.utils.math import log_var_to_std


def elbo_loss(negative_log_likelihood: torch.Tensor, kl_divergence: torch.Tensor, kl_weight: float = 1.0) -> torch.Tensor:
    """
    Compute ELBO loss.

    Args:
        negative_log_likelihood: Negative log likelihood of the model (already a loss)
        kl_divergence: KL divergence between the approximate posterior and the prior
        kl_weight: Weight for the KL divergence
    Returns:
        ELBO loss (to be minimized)
    """
    return negative_log_likelihood + kl_weight * kl_divergence


def td_error_regularizer(
    acts_curr,
    acts_next,
    q_curr,
    q_next,
    r_mu,
    r_log_var,
    gamma,
    valid,
    terminal,
) -> torch.Tensor:
    """
    Computes TD error regularization for transition-level data.

    For each transition (s, a, s', a'), computes:
        TD_error = Q(s, a) - γ * Q(s', a')

    Enforces that the learned reward R(s,a) should match this TD error.
    """

    # Select Q(s_t, a_t) for current state-action pairs
    q_curr_a = torch.gather(q_curr, dim=-1, index=acts_curr).squeeze(-1)  # (batch_size,)
    q_next_a = torch.gather(q_next, dim=-1, index=acts_next).squeeze(-1)  # (batch_size,)

    # For terminal states, Q(s',a') = 0
    q_next_a = q_next_a * (~terminal)

    # Compute TD-error: R(s,a) = Q(s,a) - γ * Q(s',a')
    td_target = q_curr_a - gamma * q_next_a  # (batch_size,)

    # Compute NLL only over valid timesteps
    r_std = log_var_to_std(r_log_var)
    nll = 0.5 * (((td_target - r_mu) / r_std).pow(2) + r_log_var)

    # Compute negative log-likelihood: the learned reward should explain the TD error
    nll_masked = nll * valid
    return nll_masked.sum() / valid.sum().clamp(min=1)
