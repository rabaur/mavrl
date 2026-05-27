import gymnasium as gym
from mavrl.envs.grid_env.env import GridEnv
from mavrl.envs.chain_env.env import ChainEnv
from mavrl.utils.gym import is_registered_gym_env

def make_env(**kwargs) -> gym.Env:
    env_id = kwargs["env_id"]
    if is_registered_gym_env(env_id):
        env = gym.make(env_id, render_mode=None)
        # Reproducibility
        env.reset(seed=kwargs["seed"])
        return env
    elif env_id.startswith("grid"):
        rew_type = env_id.split("_")[1]
        kwargs["reward_type"] = rew_type
        env = GridEnv(**kwargs)
        return env
    elif env_id.startswith("chain"):
        # Parse chain environment parameters from env_id
        # Format: chain_N where N is the number of states
        parts = env_id.split("_")
        if len(parts) >= 2:
            n_states = int(parts[1])
        else:
            n_states = 5  # default
        kwargs["n_states"] = n_states
        env = ChainEnv(**kwargs)
        return env
    else:
        raise NotImplementedError(f"Uknown environment {env_id}")
