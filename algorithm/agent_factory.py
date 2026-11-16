from __future__ import annotations

from typing import Optional

from algorithms.agent import MaskedDTQNAgent
from environment.config import TrainingConfig

# 便于添加基线，指定方法

def build_agent(
    node_count: int,
    feature_dim: int,
    action_count: int,
    config: TrainingConfig,
    seed: int,
    instance_slot_counts: Optional[tuple[int, ...]] = None,
    device: str = "cpu",
    max_episode_steps: int = 1024,
):
    return MaskedDTQNAgent(
        node_count=node_count,
        feature_dim=feature_dim,
        action_count=action_count,
        config=config,
        seed=seed,
        instance_slot_counts=instance_slot_counts,
        device=device,
        max_episode_steps=max_episode_steps,
    )
