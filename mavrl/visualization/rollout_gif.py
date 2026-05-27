import io
from typing import Callable, Optional

import gymnasium as gym
from PIL import Image


def create_rollout_gif(
    policy: Callable,
    env_id: Optional[str] = None,
    make_env_fn: Optional[Callable[[], gym.Env]] = None,
    num_rollouts: int = 10,
    max_steps: int = 500,
    fps: int = 30,
) -> Optional[bytes]:
    """
    Create a GIF of policy rollouts by rendering each rollout sequentially.
    
    Args:
        policy: Policy to roll out (must have a .predict(obs) method)
        env_id: Gymnasium environment ID (used with gym.make).
        make_env_fn: Factory that returns a gym.Env with render_mode="rgb_array".
            Exactly one of env_id or make_env_fn must be provided.
        num_rollouts: Number of rollouts to include in the GIF
        max_steps: Maximum steps per rollout
        fps: Frames per second for the GIF
        
    Returns:
        GIF as bytes, or None if rendering is not supported
    """
    if env_id is None and make_env_fn is None:
        raise ValueError("Provide either env_id or make_env_fn")

    frames = []
    
    try:
        if make_env_fn is not None:
            env = make_env_fn()
        else:
            env = gym.make(env_id, render_mode="rgb_array")
    except Exception as e:
        print(f"Could not create environment with render_mode='rgb_array': {e}")
        return None
    
    try:
        for rollout_idx in range(num_rollouts):
            obs, _ = env.reset(seed=rollout_idx)
            done = False
            step = 0
            
            while not done and step < max_steps:
                frame = env.render()
                if frame is not None:
                    frames.append(Image.fromarray(frame))
                
                result = policy.predict(obs)
                action = result[0] if isinstance(result, tuple) else result
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                step += 1
            
            # Add a few blank frames between rollouts for visual separation
            if rollout_idx < num_rollouts - 1 and frames:
                for _ in range(int(fps * 0.3)):  # 0.3 second pause
                    frames.append(frames[-1].copy())
    finally:
        env.close()
    
    if not frames:
        print("No frames captured during rollouts")
        return None
    
    gif_buffer = io.BytesIO()
    frames[0].save(
        gif_buffer,
        format='GIF',
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),  # duration in milliseconds
        loop=0  # loop forever
    )
    gif_buffer.seek(0)
    
    return gif_buffer.getvalue()

