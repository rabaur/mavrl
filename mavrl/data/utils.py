import numpy as np
from typing import Any, Optional, Callable
from numpy.typing import NDArray
from mavrl.types import DataKey, Trajectory
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm
import random
import hashlib
from mavrl.utils.feature_transforms import apply_transform
from mavrl.utils.policies import Expert

def _valid_mean_return(seg: Trajectory) -> float:
    """Mean reward over valid timesteps (aligned with ``PreferenceDecoder``)."""
    r = np.asarray(seg[DataKey.REWS], dtype=np.float64)
    v = np.asarray(seg[DataKey.VALID], dtype=np.float64)
    return float(np.nansum(r * v) / max(np.sum(v), 1))

def derive_seed(base_seed: int, *identifiers: str) -> int:
    """Derive a deterministic seed from base seed and string identifiers."""
    key = f"{base_seed}:{':'.join(identifiers)}"
    hash_val = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return hash_val


def pad_trajectory(trajectory: Trajectory, target_length: int) -> Trajectory:
    """
    Pad a trajectory/segment to a target length with zero values and VALID=False.
    
    Args:
        trajectory: Dictionary containing trajectory data with DataKey.
        target_length: The target length to pad the trajectory to.
    
    Returns:
        Padded trajectory dictionary. If trajectory is already >= target_length,
        returns the original trajectory unchanged.
    """
    current_length = len(trajectory[DataKey.REWS])
    
    if current_length >= target_length:
        return trajectory
    
    num_pad = target_length - current_length
    padded = {}
    
    for key, value in trajectory.items():
        if len(value) == 0:
            padded[key] = value
            continue
        
        # Handle different key types
        if key == DataKey.VALID:
            pad_values = np.zeros(num_pad, dtype=bool)  # False values
        else:
            sample = value[0]
            if isinstance(sample, np.ndarray):
                pad_values = np.zeros((num_pad, *sample.shape), dtype=sample.dtype)
            else:
                pad_values = np.zeros(num_pad, dtype=type(sample))
        padded[key] = np.concatenate([value, pad_values], axis=0)
    
    return padded


def roll_with_fill(x: NDArray, shift: int, dim: int, fill: Any = 0) -> NDArray:
    pad_widths = [(0, 0) for _ in range(x.ndim)]
    if shift > 0:
        pad_widths[dim] = (shift, 0)
    elif shift < 0:
        pad_widths[dim] = (0, abs(shift))
    result = np.pad(x, pad_widths, mode='constant', constant_values=fill)
    
    # select correct slice for result
    slices = [slice(None) for _ in range(x.ndim)]
    if shift > 0:
        slices[dim] = slice(None, -shift)
    elif shift < 0:
        slices[dim] = slice(abs(shift), None)
    result = result[tuple(slices)]
    return result


def compute_next_obs_acts(trajectory: dict[DataKey, NDArray], offset: int) -> dict[DataKey, NDArray]:
    """
    Computes next observations and actions from trajectory by shifting the observations and actions by offset steps.

    Args:
        trajectory: Dictionary containing trajectory data.
        offset: Offset by which to shift the observations and actions.

    Returns:
        Dictionary containing next observations and actions.
    """
    obs = trajectory[DataKey.OBS]
    acts = trajectory[DataKey.ACTS]
    next_obs = roll_with_fill(obs, shift=offset, dim=-2, fill=0.0)
    T = obs.shape[-2]
    shared_part_prev = obs[..., abs(offset):, :]
    shared_part_next = next_obs[..., :T - abs(offset), :]
    assert np.allclose(shared_part_prev, shared_part_next), f"Obs mismatch at offset {offset}. Shared part length: {shared_part_prev.shape[-2]}"
    next_acts = roll_with_fill(acts, shift=offset, dim=-2, fill=0.0)
    return {DataKey.NEXT_OBS: next_obs, DataKey.NEXT_ACTS: next_acts}

def new_empty_trajectory() -> Trajectory:
    """Create a fresh empty trajectory with new list instances."""
    return {
        DataKey.OBS: [],
        DataKey.ACTS: [],
        DataKey.REWS: [],
        DataKey.TERMINAL: [],
        DataKey.VALID: [],
    }

def collect_episodes(
    policy: Expert,
    n_episodes: int,
    n_envs: int,
    base_seed: int,
    make_env_fn: Callable[Any, gym.Env],
    min_reward_threshold: Optional[float] = None
):

    # we don't need more envs that episodes to collect
    n_envs = max(1, min(n_episodes, n_envs))

    # Create vectorized environment
    vec_env = DummyVecEnv([make_env_fn]*n_envs)

    # Set seeds for each environment
    vec_env.seed(base_seed)

    obs = vec_env.reset()
    trajs = [new_empty_trajectory() for _ in range(n_envs)]
    collected_episodes = []
    # Silence the bar for the tiny per-iteration candidate collections
    # that clog the active-learning log; keep it for big batch collects.
    pbar = tqdm(total=n_episodes, desc="Collecting full episodes", disable=True)
    while len(collected_episodes) < n_episodes:
        actions = policy.predict(obs, deterministic=False)
        next_obs, rewards, dones, infos = vec_env.step(actions)
        for i in range(n_envs):
            trajs[i][DataKey.OBS].append(np.atleast_1d(obs[i]))
            trajs[i][DataKey.ACTS].append(np.atleast_1d(actions[i]))
            trajs[i][DataKey.REWS].append(rewards[i])
            truncated_val = infos[i].get('TimeLimit.truncated', False) if dones[i] else False
            terminated_val = dones[i] and not truncated_val
            trajs[i][DataKey.TERMINAL].append(terminated_val)
            trajs[i][DataKey.VALID].append(True)
            done = dones[i]
            if done:
                # Reject episode if cumulative reward is below threshold
                if min_reward_threshold is not None and sum(trajs[i][DataKey.REWS]) < min_reward_threshold:
                    trajs[i] = new_empty_trajectory()  # Reset trajectory before continuing
                    continue
                full_episode_data = trajs[i]
                full_episode_data = {k: np.array(v) for k, v in full_episode_data.items()}
                collected_episodes.append(full_episode_data)
                pbar.update(1)
                # Reset trajectory for this environment
                trajs[i] = new_empty_trajectory()
                if len(collected_episodes) >= n_episodes:
                    break
        if len(collected_episodes) >= n_episodes:
            break
        obs = next_obs
    pbar.close()
    vec_env.close()
    return collected_episodes

def sample_segment(episodes: list[Trajectory], segment_len: int, rng: random.Random) -> Trajectory:
    selected_episode = rng.choice(episodes)
    start_index = rng.randint(0, len(selected_episode[DataKey.REWS]))
    segment = {k: v[start_index:start_index + segment_len] for k, v in selected_episode.items()}
    # Pad segment to segment_len if it extends beyond episode boundary
    segment = pad_trajectory(segment, segment_len)
    return segment

def is_segment_valid(segment: Trajectory) -> bool:
    return np.any(segment[DataKey.VALID])

def extract_segments_from_episodes(
    episodes: list[Trajectory],
    segment_len: int,
    num_segments: int,
    rng: random.Random,
    discard_inv_segments: bool = True,
) -> list[Trajectory]:
    segments = []
    max_num_segments = 0
    for ep in episodes:
        max_num_segments += len(ep[DataKey.REWS]) - segment_len + 1
    if max_num_segments < num_segments:
        raise ValueError(f"Maximal number of distinct segments: {max_num_segments} is less than requested number of segments: {num_segments}.")
    for _ in range(num_segments):
        segment = sample_segment(episodes, segment_len, rng)
        while discard_inv_segments and not is_segment_valid(segment):
            segment = sample_segment(episodes, segment_len, rng)
        segments.append(segment)
    return segments


def prepare_episodes(
    policy: Expert,
    num_episodes: int,
    make_env_fn: Callable[Any, gym.Env],
    base_seed: int,
    step_offset: int = 1,
    subsample_factor: int = 1,
    obs_transform: Optional[Callable] = None,
    act_transform: Optional[Callable] = None,
    print_stat_fn: Optional[Callable[[list[Trajectory]], None]] = None,
    min_reward_threshold: Optional[float] = None
) -> list[Trajectory]:

    episodes = collect_episodes(policy, num_episodes, n_envs=8, make_env_fn=make_env_fn, base_seed=base_seed, min_reward_threshold=min_reward_threshold)

    if print_stat_fn:
        print_stat_fn(episodes)

    # Add next observations and actions
    for e in episodes:
        e |= compute_next_obs_acts(e, offset=-step_offset)
    
    # Initialize states and action_features as copies of observations and actions
    # (they may be transformed later)
    for e in episodes:
        e[DataKey.STATES] = e[DataKey.OBS].copy()
        e[DataKey.NEXT_STATES] = e[DataKey.NEXT_OBS].copy()
        e[DataKey.ACT_FEATS] = e[DataKey.ACTS].copy()
        e[DataKey.NEXT_ACT_FEATS] = e[DataKey.NEXT_ACTS].copy()
    
    # Subsample the dataset
    episodes = [{k: v[::subsample_factor] for k, v in e.items()} for e in episodes]
    
    # Apply transforms if provided. Transformations are applied per observation or action.
    if obs_transform:
        for e in episodes:
            e[DataKey.OBS] = apply_transform(obs_transform, e[DataKey.OBS])
            e[DataKey.NEXT_OBS] = apply_transform(obs_transform, e[DataKey.NEXT_OBS])

    if act_transform:
        for e in episodes:
            e[DataKey.ACT_FEATS] = apply_transform(act_transform, e[DataKey.ACTS])
            e[DataKey.NEXT_ACT_FEATS] = apply_transform(act_transform, e[DataKey.NEXT_ACTS])
    
    return episodes
