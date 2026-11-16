from typing import Callable, Union, Tuple
from enum import Enum
import random

import torch
from torch.nn import Module
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from dtqn.buffers.replay_buffer import ReplayBuffer
from utils.logging_utils import RunningAverage
from utils.context import Context
from utils.epsilon_anneal import LinearAnneal
from utils.random import RNG

class TrainMode(Enum):
    TRAIN = 1
    EVAL = 2

class AgentCore:
    def __init__(
        self,
        network_factory: Callable[[], Module],
        buffer_size: int,
        device: torch.device,
        env_obs_length: int,
        max_env_steps: int,
        obs_mask: Union[int, float],
        num_actions: int,
        is_discrete_env: bool,
        learning_rate: float = 0.0003,
        batch_size: int = 32,
        context_len: int = 1,
        gamma: float = 0.99,
        grad_norm_clip: float = 1.0,
        target_update_frequency: int = 10_000,
        **kwargs,
    ):
        self.context_len = context_len
        self.env_obs_length = env_obs_length
                                           
        self.policy_network = network_factory()
        self.target_network = network_factory()
                                                  
        self.target_update()
        self.target_network.eval()

        if is_discrete_env:
            self.obs_context_type = np.int_
            self.obs_tensor_type = torch.long
        else:
            self.obs_context_type = np.float32
            self.obs_tensor_type = torch.float32

        self.device = device

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(
            buffer_size,
            env_obs_length=env_obs_length,
            obs_mask=obs_mask,
            max_episode_steps=max_env_steps,
            context_len=context_len,
        )

        self.batch_size = batch_size
        self.gamma = gamma
        self.grad_norm_clip = grad_norm_clip
        self.target_update_frequency = target_update_frequency

        self.num_train_steps = 0
        self.td_errors = RunningAverage(100)
        self.grad_norms = RunningAverage(100)
        self.qvalue_max = RunningAverage(100)
        self.target_max = RunningAverage(100)
        self.qvalue_mean = RunningAverage(100)
        self.target_mean = RunningAverage(100)
        self.qvalue_min = RunningAverage(100)
        self.target_min = RunningAverage(100)

        self.num_actions = num_actions
        self.train_mode = TrainMode.TRAIN
        self.obs_mask = obs_mask

        self.train_context = Context(
            context_len, obs_mask, self.num_actions, env_obs_length
        )
        self.eval_context = Context(
            context_len, obs_mask, self.num_actions, env_obs_length
        )

    @property
    def context(self) -> Context:
        if self.train_mode == TrainMode.TRAIN:
            return self.train_context
        elif self.train_mode == TrainMode.EVAL:
            return self.eval_context

    def eval_on(self) -> None:
        self.train_mode = TrainMode.EVAL
        self.policy_network.eval()

    def eval_off(self) -> None:
        self.train_mode = TrainMode.TRAIN
        self.policy_network.train()

    def target_update(self) -> None:
        pass
        self.target_network.load_state_dict(self.policy_network.state_dict())
