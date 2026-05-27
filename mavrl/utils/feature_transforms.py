import numpy as np
import torch
import gymnasium as gym
from typing import Callable

def to_one_hot(discr_x: float, n: int) -> np.ndarray:
    one_hot = np.zeros(n, dtype=np.float32)
    if np.isnan(discr_x):
        return one_hot
    one_hot[int(discr_x)] = 1
    return one_hot

def get_act_transform(args, env: gym.Env) -> Callable:
    act_transform = None
    if args.act_transform:
        if args.act_transform == "one_hot":
            if not hasattr(env.action_space, "n"):
                raise ValueError(
                    f"act_transform='one_hot' requires a Discrete action space, "
                    f"but got {type(env.action_space).__name__}."
                )
            act_transform = lambda x: to_one_hot(x, env.action_space.n)
        else:
            raise NotImplementedError(f"Invalid action transform: {args.act_transform}")
    return act_transform

def get_obs_transform(args, env: gym.Env) -> Callable:
    obs_transform = None
    if args.obs_transform:
        if args.obs_transform == "one_hot":
            obs_transform = lambda x: to_one_hot(x, env.observation_space.n)
        else:
            raise NotImplementedError(f"Invalid observation transform: {args.obs_transform}")
    return obs_transform

def apply_transform(transform: Callable, x: np.ndarray) -> np.ndarray:
    """
    Apply a transform function to each element of an array.
    
    This function handles transforms that return arrays (e.g., one-hot encoding)
    where np.vectorize would fail with "setting an array element with a sequence".
    
    Args:
        transform: Function to apply to each element. Can return scalars or arrays.
        x: Input array of shape (..., feature_dim)
        
    Returns:
        Transformed array of shape (..., new_feature_dim)
        where new_feature_dim depends on the transform output.
        
    Example:
        >>> x = np.array([[[0], [1]], [[1], [0]]])  # shape: (2, 2, 1)
        >>> transform = lambda a: to_one_hot(int(a), 2)
        >>> result = apply_transform(transform, x)
        >>> result.shape
        (2, 2, 2)  # Last dimension expanded from 1 to 2
    """
    original_shape = x.shape
    
    # Flatten to 1D for easy iteration
    x_flat = x.reshape(-1)
    
    # Apply transform to each element
    transformed_list = [transform(elem) for elem in x_flat]
    
    # Check if transform returns arrays or scalars
    if len(transformed_list) > 0:
        first_result = transformed_list[0]
        
        # Determine the feature dimension of the output
        if isinstance(first_result, (np.ndarray, list, tuple)):
            new_feat_dim = len(first_result)
        else:
            new_feat_dim = 1
        
        # Stack all transformed results
        transformed_array = np.array(transformed_list)
        
        # Reshape back to original structure with new feature dimension
        # Original shape: (..., old_feat_dim) -> New shape: (..., new_feat_dim)
        new_shape = original_shape[:-1] + (new_feat_dim,)
        transformed_array = transformed_array.reshape(new_shape)
        
        return transformed_array
    else:
        # Empty array case
        return x


def get_feature_combinations(reward_domain: str, all_obs_features: torch.Tensor, all_act_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple, tuple]:
    num_states = all_obs_features.shape[0]
    num_actions = all_act_features.shape[0]
    if reward_domain == 's':
        # Reward only depends on state: R(s)
        batch_state_features = all_obs_features  # (S, state_dim)
        batch_action_features = None
        batch_next_state_features = None
        
    elif reward_domain == 'sa':
        # Reward depends on state-action: R(s,a)
        s_idx, a_idx = np.meshgrid(np.arange(num_states), np.arange(num_actions), indexing='ij')
        s_flat = s_idx.flatten()
        a_flat = a_idx.flatten()
        
        batch_state_features = all_obs_features[s_flat]    # (S*A, state_dim)
        batch_action_features = all_act_features[a_flat]  # (S*A, action_dim)
        batch_next_state_features = None
        
    elif reward_domain == 'sas':
        # Reward depends on state-action-nextstate: R(s,a,s')
        s_idx, a_idx, sp_idx = np.meshgrid(
            np.arange(num_states), 
            np.arange(num_actions), 
            np.arange(num_states), 
            indexing='ij'
        )
        s_flat = s_idx.flatten()
        a_flat = a_idx.flatten()
        sp_flat = sp_idx.flatten()
        
        batch_state_features = all_obs_features[s_flat]        # (S*A*S', state_dim)
        batch_action_features = all_act_features[a_flat]      # (S*A*S', action_dim)
        batch_next_state_features = all_obs_features[sp_flat]  # (S*A*S', state_dim)
        
    else:
        raise ValueError(f"Unknown reward domain: {reward_domain}")
    
    return batch_state_features, batch_action_features, batch_next_state_features