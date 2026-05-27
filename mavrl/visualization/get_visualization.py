import gymnasium as gym
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from mavrl.envs.grid_env.env import GridEnv
from mavrl.envs.chain_env.env import ChainEnv
from mavrl.multi_fb_model import MultiFeedbackTypeModel
from mavrl.utils.policies import Expert
from mavrl.visualization.acrobot_visualizer import vis_acrobot
from mavrl.visualization.cartpole_visualizer import vis_cartpole
from mavrl.visualization.grid_visualizer import vis_grid_env
from mavrl.visualization.chain_visualizer import vis_chain_env
from mavrl.visualization.lunarlander_visualizer2 import vis_lunarlander as vis_lunarlander2
from mavrl.visualization.mountaincar_visualizer import vis_mountaincar
from mavrl.utils.gym import get_env_name

def get_visualization(
    env: gym.Env,
    fb_model: MultiFeedbackTypeModel
):
    if isinstance(env.unwrapped, ChainEnv):
        fig = vis_chain_env(
            env.unwrapped,
            fb_model
        )
    elif isinstance(env.unwrapped, GridEnv):
        fig = vis_grid_env(
            env.unwrapped,
            fb_model
        )
    elif get_env_name(env) == "LunarLander-v3":
        fig = vis_lunarlander2(
            env=env,
            fb_model=fb_model,
        )
    elif get_env_name(env) == "CartPole-v1":
        fig = vis_cartpole(
            env=env,
            fb_model=fb_model,
        )
    elif get_env_name(env) == "Acrobot-v1":
        fig = vis_acrobot(
            env=env,
            fb_model=fb_model,
        )
    elif get_env_name(env) == "MountainCar-v0":
        fig = vis_mountaincar(
            env=env,
            fb_model=fb_model,
        )
    else: 
        raise NotImplementedError(f"Visualization for environment {get_env_name(env)} not implemented")
    return fig
