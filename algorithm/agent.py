from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from environment.config import TrainingConfig
from algorithms.networks import GRAPH_SCHEMA, GRAPH_ENCODER_VERSION, GraphLayout, build_masked_dtqn
from algorithms.reference_dtqn import ReferenceDtqnAgent, ReferenceLinearAnneal, reference_source_files

from utils.random import RNG as ReferenceRNG

class MaskedDTQNAgent(ReferenceDtqnAgent):
    pass

    # 初始化网络、缓冲区
    def __init__(
        self,
        node_count: int,
        feature_dim: int,
        action_count: int,
        config: TrainingConfig,
        seed: int,
        instance_slot_counts: Optional[tuple[int, ...]] = None,
        device: str = "cpu",
        lagrange_multiplier: float = 0.0,
        max_episode_steps: int = 1024,
    ):
        self.config = config
        self.node_count = int(node_count)
        self.layout = GraphLayout.resolve(
            self.node_count, action_count, instance_slot_counts
        )
        self.instance_type_count = self.layout.type_count
        self.instance_slot_counts = self.layout.slot_counts
        self.feature_dim = int(feature_dim)
        self.action_count = int(action_count)
        self.graph_encoder = config.graph_encoder
        self.action_masking = bool(config.action_masking)
        self.graph_observation_dim = self.node_count * self.feature_dim
        self.node_mask_offset = self.graph_observation_dim
        self.action_mask_offset = self.node_mask_offset + self.node_count
        self.combined_observation_dim = self.action_mask_offset + self.action_count
        self.environment_steps = 0
        self.epsilon_schedule = ReferenceLinearAnneal(
            config.epsilon_start,
            config.epsilon_end,
            max(1, config.epsilon_decay_steps),
        )
        self._seed = int(seed)
        self._max_episode_steps = max(1, int(max_episode_steps))
        ReferenceRNG.rng = np.random.Generator(np.random.PCG64(seed=self._seed))

        network_args = dict(
            node_count=self.node_count,
            feature_dim=self.feature_dim,
            action_count=self.action_count,
            instance_slot_counts=self.instance_slot_counts,
            hidden_dim=config.hidden_dim,
            attention_heads=config.attention_heads,
            transformer_layers=config.transformer_layers,
            transformer_dropout=config.transformer_dropout,
            graph_attention_heads=config.graph_attention_heads,
            max_context=config.context_length,
        )
        network_factory = lambda: build_masked_dtqn(
            self.graph_encoder, **network_args
        ).to(torch.device(device))
        super().__init__(
            network_factory=network_factory,
            buffer_size=config.replay_capacity,
            device=torch.device(device),
            env_obs_length=self.combined_observation_dim,
            max_env_steps=self._max_episode_steps,
            obs_mask=0.0,
            num_actions=self.action_count,
            is_discrete_env=False,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            context_len=config.context_length,
            gamma=config.gamma,
            grad_norm_clip=config.grad_norm_clip,
            target_update_frequency=config.target_update_steps,
            history=config.history_length,
            bag_size=0,
        )

        self.replay_buffer.episode_lengths = (
            self.replay_buffer.episode_lengths.astype(np.int32)
        )
        self.set_lagrange_multiplier(lagrange_multiplier)

    # 策略网络
    @property
    def online(self):
        return self.policy_network

    # 目标网络
    @property
    def target(self):
        return self.target_network

    # buffer
    @property
    def replay(self):
        return self.replay_buffer
    # 步数
    @property
    def steps(self) -> int:
        return self.environment_steps

    @property
    def updates(self) -> int:
        return self.num_train_steps

    # 探索率
    @property
    def epsilon(self) -> float:
        return float(self.epsilon_schedule.val)

    # 拉格朗日乘子
    def set_lagrange_multiplier(self, value: float) -> None:
        if not np.isfinite(value) or value < 0:
            raise ValueError("lagrange multiplier must be finite and non-negative")
        self.lagrange_multiplier = float(value)

    # 拼接观测向量
    def _combined_observation(
        self,
        node_features: np.ndarray,
        action_mask: np.ndarray,
        node_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        nodes = np.asarray(node_features, dtype=np.float32)
        if nodes.shape != (self.node_count, self.feature_dim):
            raise ValueError(
                "node feature shape mismatch: "
                f"expected {(self.node_count, self.feature_dim)}, got {nodes.shape}"
            )
        mask = self._effective_action_mask(action_mask).astype(np.float32)
        if node_mask is None:
            valid_nodes = np.ones(self.node_count, dtype=np.float32)
        else:
            valid_nodes = np.asarray(node_mask, dtype=np.float32)
            if valid_nodes.shape != (self.node_count,):
                raise ValueError(
                    f"node mask must have shape {(self.node_count,)}, got {valid_nodes.shape}"
                )
        return np.concatenate((nodes.reshape(-1), valid_nodes, mask))

    # 动作掩蔽
    def _effective_action_mask(self, action_mask: np.ndarray) -> np.ndarray:
        pass
        environment_mask = np.asarray(action_mask, dtype=bool)
        if (
            environment_mask.shape != (self.action_count,)
            or not environment_mask.any()
        ):
            raise ValueError("action mask has invalid shape or no valid action")
        if self.action_masking:
            return environment_mask
        return np.ones(self.action_count, dtype=bool)

    # 训练模式
    def begin_episode(
        self,
        node_features: np.ndarray,
        action_mask: np.ndarray,
        training: bool,
        node_mask: Optional[np.ndarray] = None,
    ) -> None:
        if training:
            self.eval_off()
        else:
            self.eval_on()
        self.context_reset(
            self._combined_observation(node_features, action_mask, node_mask)
        )

    # 更新掩码
    def _set_current_mask(self, action_mask: np.ndarray) -> np.ndarray:
        mask = self._effective_action_mask(action_mask)
        index = min(self.context.timestep, self.context.max_length - 1)
        self.context.obs[index, self.action_mask_offset :] = mask.astype(np.float32)
        return mask

    # 决策
    @torch.no_grad()
    def act(self, action_mask: np.ndarray, explore: bool = True) -> int:
        mask = self._set_current_mask(action_mask)
        valid = np.flatnonzero(mask)
        epsilon = self.epsilon if explore else 0.0
        if explore and ReferenceRNG.rng.random() < epsilon:
            action = int(ReferenceRNG.rng.choice(valid))
        else:

            action = int(super().get_action(epsilon=0.0))
        if explore:
            self.environment_steps += 1
        return action

    # 观测
    def observe(
        self,
        node_features: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        action_mask: np.ndarray,
        node_mask: Optional[np.ndarray] = None,
        anneal_epsilon: bool = True,
        **_: object,
    ) -> None:

        combined = self._combined_observation(
            node_features, action_mask, node_mask
        )
        super().observe(combined, int(action), float(reward), bool(done))
        if anneal_epsilon and self.policy_network.training:
            self.epsilon_schedule.anneal()

    # 存储经验
    def end_episode(self, training: bool) -> None:
        if training:
            self.replay_buffer.flush()

    # 训练
    def learn(self) -> Optional[float]:
        previous_updates = self.num_train_steps
        super().train()
        if self.num_train_steps == previous_updates:
            return None
        return float(self.td_errors.q[-1])

    # 保存
    def _architecture_metadata(self) -> dict[str, object]:
        return {
            "agent_family": "tgrl",
            "graph_schema": GRAPH_SCHEMA,
            "graph_encoder_version": GRAPH_ENCODER_VERSION,
            "graph_encoder": self.graph_encoder,
            "action_masking": self.action_masking,
            "node_count": self.node_count,
            "instance_type_count": self.instance_type_count,
            "instance_slot_counts": list(self.instance_slot_counts),
            "feature_dim": self.feature_dim,
            "action_count": self.action_count,
            "dtqn_source": "DTQN-main",
        }

    # 保存
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture": self._architecture_metadata(),
                "reference_source_files": reference_source_files("tgrl"),
                "online": self.policy_network.state_dict(),
                "target": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "evaluation_control": getattr(self, "evaluation_control", {}),
                "steps": self.environment_steps,
                "updates": self.num_train_steps,
                "epsilon": self.epsilon,
                "lagrange_multiplier": self.lagrange_multiplier,
            },
            path,
        )

    # 恢复
    def load(self, path: Path, load_optimizer: bool = False) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        architecture = checkpoint.get("architecture")
        expected = self._architecture_metadata()
        if architecture is not None:
            architecture = dict(architecture)
            architecture.pop("graph_encoder_parameter_count", None)
            architecture.pop("online_parameter_count", None)

            architecture.setdefault("action_masking", True)
            architecture.setdefault("agent_family", "tgrl")
        if architecture is not None and architecture != expected:
            raise ValueError(
                "checkpoint architecture does not match evaluation config: "
                f"checkpoint={architecture}, expected={expected}"
            )
        try:
            self.policy_network.load_state_dict(checkpoint["online"])
        except RuntimeError as error:
            raise ValueError(
                "checkpoint network shape does not match the DTQN-main adapter"
            ) from error
        self.target_network.load_state_dict(
            checkpoint.get("target", checkpoint["online"])
        )
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.evaluation_control = checkpoint.get("evaluation_control", {})
            if not self.evaluation_control:
                for group in self.optimizer.param_groups:
                    group["lr"] = self.config.learning_rate
        self.environment_steps = int(checkpoint.get("steps", 0))
        self.num_train_steps = int(checkpoint.get("updates", 0))
        self.epsilon_schedule.val = float(checkpoint.get("epsilon", self.epsilon))
        self.lagrange_multiplier = float(
            checkpoint.get("lagrange_multiplier", self.lagrange_multiplier)
        )
        self.target_network.eval()
