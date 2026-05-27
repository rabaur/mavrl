"""
Chain MDP Environment for Debugging

A simple chain MDP: s0 -> s1 -> s2 -> ... -> s_{n-1} (terminal)

- Two actions: forward (optimal) and stay (suboptimal)
- Terminal state is the last state
- Reward is only given when in the terminal state
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mavrl.envs.env_types import TabularEnv


# Action indices
ACTION_FORWARD = 0  # Move to next state
ACTION_STAY = 1     # Stay in current state


class ChainEnv(TabularEnv):
    """
    A simple chain MDP for debugging demonstration learning.
    
    States: 0, 1, 2, ..., n_states-1
    Actions: 
        0 = FORWARD (move to next state) - optimal action
        1 = STAY (remain in current state) - suboptimal action
    Transitions: Deterministic
    Terminal: State n_states-1 is the terminal/goal state
    Reward: +1 when in terminal state, 0 otherwise (state-based)
    
    This environment is useful for debugging because:
    1. Simple linear structure makes it easy to analyze
    2. Two actions make cross-entropy loss meaningful
    3. Clear expected behavior: R(s_terminal) should be highest
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        n_states: int = 5,
        terminal_reward: float = 1.0,
        step_reward: float = 0.0,
        gamma: float = 0.99,
        **kwargs
    ):
        """
        Args:
            n_states: Number of states in the chain (including terminal state)
            terminal_reward: Reward for being in the terminal state
            step_reward: Reward for being in non-terminal states (default 0)
            gamma: Discount factor (for computing optimal values)
        """
        super().__init__()
        
        assert n_states >= 2, "Chain must have at least 2 states"
        
        self.n_states = n_states
        self.n_actions = 2  # Two actions: forward and stay
        self.terminal_reward = terminal_reward
        self.step_reward = step_reward
        self.gamma = gamma
        
        # Build transition and reward matrices
        self._P = self._build_transition_matrix()
        self._R = self._build_reward_matrix()
        
        # Gym spaces
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.n_states)
        
        # Internal state
        self.state = None
    
    def _build_transition_matrix(self) -> np.ndarray:
        """
        Build transition probability matrix P[s, a, s'].
        
        Action 0 (FORWARD): Move to next state (or stay at terminal)
        Action 1 (STAY): Stay in current state
        """
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        
        for s in range(self.n_states):
            # Action FORWARD: move to next state (or stay if at terminal)
            if s < self.n_states - 1:
                P[s, ACTION_FORWARD, s + 1] = 1.0
            else:
                P[s, ACTION_FORWARD, s] = 1.0  # Terminal is absorbing
            
            # Action STAY: stay in current state
            P[s, ACTION_STAY, s] = 1.0
        
        return P
    
    def _build_reward_matrix(self) -> np.ndarray:
        """
        Build reward matrix R[s, a, s'].
        
        Uses STATE-based rewards like GridEnv: R(s, a, s') = R(s)
        The reward is for BEING IN a state, not for transitioning to it.
        
        R[s, a, s'] = step_reward for s < n_states - 1
        R[n_states-1, a, s'] = terminal_reward (being in terminal state)
        """
        R = np.zeros((self.n_states, self.n_actions, self.n_states))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for s_prime in range(self.n_states):
                    if s == self.n_states - 1:
                        R[s, a, s_prime] = self.terminal_reward
                    else:
                        R[s, a, s_prime] = self.step_reward
        
        return R
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0  # Always start at state 0
        return self.state, {}
    
    def step(self, action):
        assert self.state is not None, "Must call reset() before step()"
        assert 0 <= action < self.n_actions, f"Invalid action {action}"
        
        prev_state = self.state
        
        # Episode terminates when we START from the terminal state (like GridEnv)
        # This ensures the terminal state is observed before the episode ends
        terminated = (self.state == self.n_states - 1)
        
        if not terminated:
            if action == ACTION_FORWARD:
                self.state = self.state + 1
            # ACTION_STAY: state stays the same
        # At terminal: state stays the same (absorbing)
        
        # Reward is R(prev_state) - state-based reward for BEING in that state
        reward = float(self._R[prev_state, action, self.state])
        
        truncated = False
        return self.state, reward, terminated, truncated, {}
    
    def render(self, mode="human"):
        """Simple text rendering of the chain."""
        chain = ["[ ]"] * self.n_states
        chain[-1] = "[G]"  # Goal
        if self.state is not None:
            if self.state == self.n_states - 1:
                chain[self.state] = "[*G]"  # Agent at goal
            else:
                chain[self.state] = "[*]"  # Agent position
        print(" -> ".join(chain))
    
    def close(self):
        pass
    
    def get_transition_matrix(self) -> np.ndarray:
        return self._P
    
    def get_reward_matrix(self) -> np.ndarray:
        return self._R
    
    def get_init_state_dist(self) -> np.ndarray:
        """Returns initial state distribution (always start at state 0)."""
        dist = np.zeros(self.n_states)
        dist[0] = 1.0
        return dist
    
    def get_optimal_q_values(self) -> np.ndarray:
        """
        Compute optimal Q-values using value iteration (same as q_opt).
        
        With STATE-based rewards R(s):
        - R(s) = step_reward for s < n_states - 1
        - R(s_terminal) = terminal_reward
        
        For the transition matrix, terminal state is absorbing:
        Q*(s_terminal, a) = R(s_terminal) / (1 - gamma)
        
        NOTE: These Q-values treat terminal as an infinite-horizon absorbing state.
        The actual episode terminates, but this is consistent with how GridEnv
        computes Q-values for policy optimization.
        
        Returns:
            Q-values of shape (n_states, n_actions)
        """
        from mavrl.utils.tabular import q_opt
        return q_opt(self._P, self._R, self.gamma)
    
    def get_optimal_rewards(self) -> np.ndarray:
        """
        Get the ground truth reward for each (state, action) pair.
        
        With STATE-based rewards: R(s, a) = R(s) (reward for being in state s)
        Same for both actions since reward only depends on state.
        
        Returns:
            Rewards of shape (n_states, n_actions)
        """
        R_sa = np.zeros((self.n_states, self.n_actions))
        
        # State-based rewards (same for both actions)
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if s == self.n_states - 1:
                    R_sa[s, a] = self.terminal_reward
                else:
                    R_sa[s, a] = self.step_reward
        
        return R_sa
