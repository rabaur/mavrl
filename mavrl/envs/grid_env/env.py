import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mavrl.envs.grid_env.actions import Action
from mavrl.envs.grid_env.rewards import reward_factory, succ_state, to_flat_idx
from mavrl.envs.env_types import TabularEnv

def construct_grid_env(
    grid_size: int,
    reward_type: str,
    p_rand: float,
    **kwargs,
):
    """
    Creates a grid environment on the [0, 1]^2 square with user-specified state-features.
    Let n_S be the number of states.

    Args:
        grid_size (int): The number of grid points in each dimension.
        feature_type (str): The type of state feature encoding. See `feature_factory` for options.
        reward_type (str): The type of ground truth reward function. See `reward_factory` for options.
        p_rand (float): The probability of transitioning to a random state.
        **kwargs: Additional arguments for `feature_factory`.
    Returns:
        P (np.ndarray): The transition probability matrix, s.t. P[s, a, s'] is the probability of transitioning from state s to state s' under action a.
        R (np.ndarray): The ground-truth reward function, s.t. R[s, a] is the reward for state s under action a.
        S (np.ndarray): The state-feature matrix, s.t. S[s, :] is the feature vector for state s.
        A (np.ndarray): The action-feature matrix, s.t. A[a, :] is the feature vector for state a.
    """
    # Create the transition probability matrix
    n_S = grid_size ** 2
    n_A = len(Action)
    P = np.zeros((n_S, n_A, n_S))

    # rows
    for i in range(grid_size):

        # cols
        for j in range(grid_size):

            # state index
            s = i * grid_size + j
            for a in range(n_A):
                for a_prime in range(n_A): # all the other actions that could be taken at random
                    i_prime, j_prime = succ_state(i, j, Action(a_prime), grid_size)
                    s_prime = i_prime * grid_size + j_prime
                    
                    # If this is the intended action, we go to the new state with probability 1 - p_rand
                    if a == a_prime:
                        P[s, a, s_prime] += 1 - p_rand
                    # If this is a random action, we go to the state corresponding
                    # to the random action with probability p_rand / (n_A - 1)
                    # (p_rand is distributed uniformly over n_A - 1 other random actions)
                    else:
                        P[s, a, s_prime] += p_rand / (n_A - 1)
    
    assert np.allclose(np.sum(P, axis=(2)), 1), "Transition probabilities do not sum to 1"
    
    # Create reward matrix
    R = reward_factory(grid_size, reward_type, gamma=kwargs["gamma"])

    return P, R

def validate_kwargs(kwargs):
    assert "grid_size" in kwargs, "grid_size must be provided"
    assert "reward_type" in kwargs, "reward_type must be provided"
    assert "p_rand" in kwargs, "p_rand must be provided"
    assert kwargs["grid_size"] > 0, "grid_size must be positive"
    assert 0 <= kwargs["p_rand"] <= 1, "p_rand must be in [0, 1]"
    if "goal_state" in kwargs:
        goal = kwargs["goal_state"]
        grid_size = kwargs["grid_size"]
        assert isinstance(goal, tuple) and len(goal) == 2, "goal_state must be a tuple of (i, j)"
        assert 0 <= goal[0] < grid_size and 0 <= goal[1] < grid_size, "goal_state must be within grid bounds"

class GridEnv(TabularEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(self, **kwargs):
        """
        Creates a discrete grid environment with coordinate features as observed features.

        Args:
            grid_size (int): The number of grid points in each dimension.
            reward_type (str): The type of ground truth reward function. Options are: "sparse", "dense", "path", "cliff", "trap".
            p_rand (float): The probability of transitioning to a random state. Must be in [0, 1].
            goal_state (tuple, optional): The goal state as (i, j) coordinates. If not provided, defaults to bottom-right corner (grid_size-1, grid_size-1).
            **kwargs: Additional arguments for `construct_grid_env`.
        """
        super().__init__()
        validate_kwargs(kwargs)
        self.grid_size = kwargs["grid_size"]
        self.reward_type = kwargs["reward_type"]
        self.p_rand = kwargs["p_rand"]
        
        # Set goal state (default to bottom-right corner)
        self.goal_state = kwargs.get("goal_state", (self.grid_size - 1, self.grid_size - 1))
        
        self._P, self._R = construct_grid_env(**kwargs)
    
        # Action space
        self.action_space = spaces.Discrete(len(Action))

        # Observation space
        self.observation_space = spaces.Discrete(self.grid_size ** 2)

        # Internal state representation
        self.state_coord = None  # will hold (i,j)
        self.state_idx = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        i = 0
        j = 0
        self.state_coord = (i, j)
        self.state_idx = i * self.grid_size + j
        return self.state_idx, {}

    def step(self, action):
        # update (i,j) deterministically or with randomness
        i, j = self.state_coord
        prev_state_idx = self.state_idx
        
        # Check if we're starting from the goal state
        terminated = (self.state_coord == self.goal_state)
        truncated = False

        i2, j2 = succ_state(i, j, Action(action), self.grid_size)
        self.state_coord = (i2, j2)
        self.state_idx = to_flat_idx(i2, j2, self.grid_size)
        reward = float(self._R[prev_state_idx, action, self.state_idx])
        
        return self.state_idx, reward, terminated, truncated, {}

    def render(self, mode="human"):
        pass  # implement optional

    def close(self):
        pass
    
    def get_transition_matrix(self):
        return self._P
    
    def get_reward_matrix(self):
        return self._R
    
    def get_init_state_dist(self) -> np.ndarray:
        n_S = self.grid_size ** 2
        dist = np.zeros(n_S)
        dist[0] = 1.0  # state 0 corresponds to (0, 0)
        return dist