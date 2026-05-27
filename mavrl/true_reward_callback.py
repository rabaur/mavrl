import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback

class TrueRewardCallback(BaseCallback):
    def __init__(
        self, 
        window_size: int = 100, 
        verbose: int = 0, 
        true_reward_threshold: float = None,
        min_timesteps_before_save: int = 0,
    ):
        super().__init__(verbose)
        self.window_size = window_size
        self.true_reward_threshold = true_reward_threshold
        self.should_stop = False
        self.history: list[tuple[int, float]] = []

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self.ep_true_rewards = np.zeros(n_envs, dtype=np.float32)
        self.ep_true_rewards_history = deque(maxlen=self.window_size)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]      # list of info dicts, one per env
        dones = self.locals["dones"]      # np.array shape (n_envs,)
        rewards = self.locals["rewards"]  # np.array shape (n_envs,)

        any_done = False
        for i, info in enumerate(infos):
            true_r = info.get("true_reward", rewards[i])
            self.ep_true_rewards[i] += true_r

            if dones[i]:
                ep_tr = self.ep_true_rewards[i]
                self.ep_true_rewards_history.append(ep_tr)
                self.ep_true_rewards[i] = 0.0
                any_done = True

        if len(self.ep_true_rewards_history) > 0:
            mean_true_rew = np.mean(self.ep_true_rewards_history)
            self.logger.record("rollout/ep_true_rew_mean", mean_true_rew)

            if any_done:
                self.history.append((self.num_timesteps, float(mean_true_rew)))
            
            # Check if we should stop early based on true reward threshold
            if self.true_reward_threshold is not None and mean_true_rew < self.true_reward_threshold:
                print(f"\nEarly stopping: True reward {mean_true_rew:.2f} is below threshold {self.true_reward_threshold:.2f}")
                self.should_stop = True
                return False  # Stop training

        return True
