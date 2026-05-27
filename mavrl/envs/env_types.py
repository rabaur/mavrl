import gymnasium as gym
import abc
import numpy as np

class TabularEnv(gym.Env, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_transition_matrix(self):
        pass
    
    @abc.abstractmethod
    def get_reward_matrix(self):
        pass
    
    @abc.abstractmethod
    def get_init_state_dist(self) -> np.ndarray:
        """Returns the initial state distribution as a numpy array of shape (n_states,)."""
        pass