from dataclasses import replace
from pathlib import Path

import torch

from algorithm.agent import MaskedDTQNAgent
from environment.config import ExperimentConfig, set_global_seed
from environment.env import AutoscalingEnv


def _select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate(config: ExperimentConfig, checkpoint_path: Path,
             day_index: int = 0, device: str = "cpu") -> dict:
    # Full-horizon measurement: training termination does not truncate a day.
    config = replace(config,
        workload=replace(config.workload,
                         episode_start_offset_s=config.workload.episode_start_offset_s + day_index * config.duration_s,
                         episode_stride_s=0, episode_cycle_count=1),
        reward=replace(config.reward, critical_ttft_s=1e99))
    config.validate()
    torch.set_num_threads(config.training.torch_num_threads)
    set_global_seed(config.seed)
    env = AutoscalingEnv(config)
    agent = MaskedDTQNAgent(
        env.node_count, env.node_feature_dim, env.action_count,
        config.training, config.seed, instance_slot_counts=env.instance_slot_counts,
        device=_select_device(device), max_episode_steps=config.max_decision_steps)
    agent.load(checkpoint_path)
    observation, info = env.reset(0)
    scheduled = int(info["request_count"])
    if scheduled == 0:
        raise ValueError("The selected trace day contains no requests")
    agent.begin_episode(observation["node_features"], observation["action_mask"],
                        training=False, node_mask=observation["node_mask"])
    done, decisions = False, 0
    while not done:
        action = agent.act(observation["action_mask"], explore=False)
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.observe(observation["node_features"], action, reward, done,
                      observation["action_mask"], observation["node_mask"],
                      anneal_epsilon=False)
        decisions += 1
    settlement = env.finalize_episode()
    records = settlement["request_records"]
    if settlement["elapsed_time_s"] < config.duration_s or len(records) != scheduled:
        raise ValueError("Evaluation did not cover the complete planned workload")
    violations = sum(
        r["first_token_time_s"] is None or
        float(r["first_token_time_s"]) - float(r["arrival_time_s"]) >= config.reward.sla_s
        for r in records)
    return {
        "cost": float(settlement["total_cost_usd"]),
        "sla_violation_rate": violations / scheduled,
        "decision_count": decisions,
    }
