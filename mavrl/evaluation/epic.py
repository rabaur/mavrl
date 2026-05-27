import numpy as np
import torch
from typing import Optional
from numpy.typing import NDArray
from mavrl.envs.env_types import TabularEnv
from mavrl.encoder.reward_encoder import RewardEncoder
from mavrl.utils.feature_transforms import get_feature_combinations
from mavrl.utils.torch_utils import to_numpy

def canonically_shaped_reward(R_sas: NDArray, gamma: float, d_S: Optional[NDArray] = None, d_A: Optional[NDArray] = None) -> NDArray:
    """
    For a reward function R : S x A x S -> Reals the canonically shaped reward is defined as:
    C_{dist_S, dist_A}(R)(s,a,s') = R(s,a,s') + E[γR(s',A,S') - R(s,A,S') - γE[R(S,A,S')] 
                                  = R(s,a,s') + γE[R(s',A,S')] - E[R(s,A,S')] - γE[R(S,A,S')]

    Args:
        R_sas: A (|S|, |A|, |S|) array representing the tabular reward.
        gamma: The discount factor
        d_S: A (|S|,) vector representing a probability distribution over states. Uses uniform distribution if not passed.
        d_A: A (|A|,) vector representing a probability distribution over actions. Uses uniform distribution if not passed.

    Returns:
        The canonicalized reward.
    """
    assert len(R_sas.shape) == 3, f"Expect `R_sas` to be 3-dim, but was {R_sas.shape=}"
    S, A, Sp = R_sas.shape
    assert S == Sp, "Last dim must equal number of states"
    if not d_S:
        d_S = np.ones(S) / S
    if not d_A:
        d_A = np.ones(A) / A

    # M[s] = E_{A~D_a, S'~D_s}[ R(s, A, S') ]  shape (S,)
    M = np.einsum('a,p,sap->s', d_A, d_S, R_sas)

    # G = gamma * E_{S~D_s}[ M[S] ]  scalar (mean reward)
    G = np.dot(d_S, M)

    # C[s,a,s'] = R[s,a,s'] + gamma*M[s'] - M[s] - G
    C_sas = R_sas + gamma * M[None, None, :] - M[:, None, None] - gamma * G
    return C_sas


def pearson_correlation(X: NDArray, Y: NDArray) -> float:
    """
    Computes the Pearson correlation coefficient between two reward functions.
    """
    return np.corrcoef(X.flatten(), Y.flatten())[0, 1]


def pearson_distance(X: NDArray, Y: NDArray) -> float:
    """
    Computes the Pearson distance between two reward functions.
    """
    return np.sqrt(0.5 * (1 - pearson_correlation(X, Y)))


def epic_distance(R1: NDArray, R2: NDArray, gamma: float) -> float:
    """
    Computes the EPIC distance between two reward functions.
    """
    return pearson_distance(canonically_shaped_reward(R1, gamma), canonically_shaped_reward(R2, gamma))

def evaluate_epic_distance(
    env: TabularEnv,
    encoder: RewardEncoder,
    gamma: float
) -> float:
    """
    Computes the epic distance between the predicted mean of the reward distribution
    and the ground-truth reward.

    Args:
        env: A tabular environment with get_reward_matrix() and get_transition_matrix().
        encoder: The reward encoder to evaluate.
        gamma: The discount factor.
    """
    R_true = env.get_reward_matrix()
    P = env.get_transition_matrix()
    
    num_states = P.shape[0]
    num_actions = P.shape[1]
    
    # Construct one-hot features for all states and actions
    device = next(encoder.parameters()).device
    all_obs_features = torch.eye(num_states, device=device)
    all_act_features = torch.eye(num_actions, device=device)
    
    # Get feature combinations based on reward domain
    reward_domain = encoder.features.reward_domain
    expanded_s_feats, expanded_a_feats, expanded_sp_feats = \
        get_feature_combinations(reward_domain, all_obs_features, all_act_features)
    
    # Compute estimated reward
    with torch.no_grad():
        R_est_mean, _ = encoder.forward(expanded_s_feats, expanded_a_feats, expanded_sp_feats)
    
    R_est_mean = to_numpy(R_est_mean).squeeze()
    
    # Reshape to (S, A, S') format based on reward domain
    if reward_domain == 's':
        R_est_sas = np.broadcast_to(R_est_mean[:, None, None], (num_states, num_actions, num_states))
    elif reward_domain == 'sa':
        R_est = np.reshape(R_est_mean, (num_states, num_actions))
        R_est_sas = np.broadcast_to(R_est[:, :, None], (num_states, num_actions, num_states))
    else:
        R_est_sas = np.reshape(R_est_mean, (num_states, num_actions, num_states))
    
    return epic_distance(R_true, R_est_sas, gamma)


