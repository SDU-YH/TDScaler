from typing import Tuple, Union
import numpy as np
import torch

from utils.random import RNG

class Context:

    def __init__(
        self,
        context_length: int,
        obs_mask: int,
        num_actions: int,
        env_obs_length: int,
        init_hidden: Tuple[torch.Tensor] = None,
    ):
        self.max_length = context_length
        self.env_obs_length = env_obs_length
        self.num_actions = num_actions
        self.obs_mask = obs_mask
        self.reward_mask = 0.0
        self.done_mask = True
        self.timestep = 0
        self.init_hidden = init_hidden

    def reset(self, obs: np.ndarray):
                            
        if isinstance(self.env_obs_length, tuple):
            self.obs = np.full(
                [self.max_length, *self.env_obs_length],
                self.obs_mask,
                dtype=np.uint8,
            )
        else:
            self.obs = np.full([self.max_length, self.env_obs_length], self.obs_mask)
                             
        self.obs[0] = obs

        self.action = RNG.rng.integers(self.num_actions, size=(self.max_length, 1))
        self.reward = np.full_like(self.action, self.reward_mask)
        self.done = np.full_like(self.reward, self.done_mask, dtype=np.int32)
        self.hidden = self.init_hidden
        self.timestep = 0

    def add_transition(
        self, o: np.ndarray, a: int, r: float, done: bool
    ) -> Tuple[Union[np.ndarray, None], Union[int, None]]:
        self.timestep += 1
        self.obs = self.roll(self.obs)
        self.action = self.roll(self.action)
        self.reward = self.roll(self.reward)
        self.done = self.roll(self.done)

        t = min(self.timestep, self.max_length - 1)

        evicted_obs = None
        evicted_action = None
        if self.is_full:
            evicted_obs = self.obs[t].copy()
            evicted_action = self.action[t]

        self.obs[t] = o
        self.action[t] = np.array([a])
        self.reward[t] = np.array([r])
        self.done[t] = np.array([done])

        return evicted_obs, evicted_action

    def export(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        current_timestep = min(self.timestep, self.max_length) - 1
        return (
            self.obs[current_timestep + 1],
            self.action[current_timestep],
            self.reward[current_timestep],
            self.done[current_timestep],
        )

    def roll(self, arr: np.ndarray) -> np.ndarray:
        return np.roll(arr, -1, axis=0) if self.timestep >= self.max_length else arr

    @property
    def is_full(self) -> bool:
        return self.timestep >= self.max_length

    @staticmethod
    def context_like(context):
        return Context(
            context.max_length,
            context.obs_mask,
            context.num_actions,
            context.env_obs_length,
            init_hidden=context.init_hidden,
        )
