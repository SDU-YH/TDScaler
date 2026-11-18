from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from environment.backend import ClusterBackend
from algorithm.adaptive_interval import AdaptiveDecisionInterval
from environment.config import ExperimentConfig
from environment.metrics import aggregate_step
from environment.rewarding import calculate_step_reward
from environment.simulator import SimulatorClusterBackend
from environment.types import StepMetrics
from workload.workload import WorkloadGenerator

class AutoscalingEnv:

    SCALE_DOWN_TTFT_GUARD_S = 0.8
    ARRIVAL_HISTORY_LENGTH = 4

    def __init__(self, config: ExperimentConfig, backend: Optional[ClusterBackend] = None):
        self.config = config
        self.workload = WorkloadGenerator(config.workload, config.active_model, config.seed)
        self.backend = backend or SimulatorClusterBackend(
            instance_types=config.instance_types,
            models=config.models,
            active_model=config.active_model,
            load_balancer=config.load_balancer,
            seed=config.seed,
            performance_profile_path=config.performance_profile_path,
            record_iterations=False,
        )
        self.episode = -1
        self.interval_controller = AdaptiveDecisionInterval(
            config.adaptive_interval, config.scaling_interval_s, seed=config.seed
        )
        self.trace_day = 1
        self.trace_end_day = 1
        self.last_metrics: Optional[StepMetrics] = None
        self.episode_reward = 0.0
        self._episode_active = False
        self._previous_state_input_tokens: Optional[int] = None
        self._input_token_direction = 0
        self._termination_reason: Optional[str] = None
        self._scale_down_cooldown_steps = 0
        self._arrival_history: Deque[float] = deque(
            maxlen=self.ARRIVAL_HISTORY_LENGTH
        )
        self._oracle_arrival_times = np.empty(0, dtype=np.float64)
        self._oracle_prompt_prefix = np.zeros(1, dtype=np.int64)
        self._oracle_output_prefix = np.zeros(1, dtype=np.int64)

    @property
    def action_count(self) -> int:
        return self.backend.action_count

    @property
    def node_count(self) -> int:

        slots = sum(spec.max_instances for spec in self.config.instance_types.values())
        return 2 + len(self.config.instance_types) + 2 * slots

    @property
    def instance_slot_counts(self) -> Tuple[int, ...]:
        return tuple(
            spec.max_instances for spec in self.config.instance_types.values()
        )

    @property
    def node_feature_dim(self) -> int:
        return self.backend.node_feature_dim

    @property
    def next_decision_interval_s(self) -> float:
        return min(
            self.interval_controller.interval_s,
            max(0.0, self.config.duration_s - self.backend.current_time),
        )

    def action_labels(self) -> Tuple[str, ...]:
        labels = ["noop"]
        for type_name in self.config.instance_types:
            labels.extend((f"add:{type_name}", f"remove:{type_name}"))
        return tuple(labels)

    def node_labels(self) -> Tuple[str, ...]:
        names = tuple(self.config.instance_types)
        return (
            "center",
            "load",
            *(f"type:{name}" for name in names),
            *(
                f"cold:{name}:{slot + 1}"
                for name, count in zip(names, self.instance_slot_counts)
                for slot in range(count)
            ),
            *(
                f"running:{name}:{slot + 1}"
                for name, count in zip(names, self.instance_slot_counts)
                for slot in range(count)
            ),
        )

    def _observation(self) -> Dict[str, np.ndarray]:
        node_features = self.backend.node_features(
            self.last_metrics,
            self.next_decision_interval_s,
        )
        expected_shape = (self.node_count, self.node_feature_dim)
        if node_features.shape != expected_shape:
            raise ValueError(
                f"backend returned graph features {node_features.shape}, expected {expected_shape}"
            )
        node_mask = np.asarray(self.backend.node_mask(), dtype=bool)
        if node_mask.shape != (self.node_count,):
            raise ValueError(
                f"backend returned node mask {node_mask.shape}, expected {(self.node_count,)}"
            )
        return {
            "node_features": node_features,
            "node_mask": node_mask,
            "action_mask": self._action_mask(),
        }

    def _update_input_token_direction(self) -> None:
                                                                                
        current = self.backend.pending_input_tokens
        previous = self._previous_state_input_tokens
        self._input_token_direction = (
            0 if previous is None else int(current > previous) - int(current < previous)
        )
        self._previous_state_input_tokens = current

    def _oracle_horizon_s(self) -> float:
        return max(
            (spec.cold_start_s for spec in self.config.instance_types.values()),
            default=self.config.scaling_interval_s,
        )

    def _recent_arrivals(self) -> np.ndarray:
        values = list(reversed(self._arrival_history))
        values.extend([0.0] * (self.ARRIVAL_HISTORY_LENGTH - len(values)))
        return np.asarray(values, dtype=np.float32)

    def _oracle_volume(self, horizon_s: float) -> Tuple[int, int, int]:
        if not self.config.workload.use_oracle_future_load:
            return 0, 0, 0
        start = self.backend.current_time
        end = min(self.config.duration_s, start + max(0.0, horizon_s))
        left_side = "left" if start == 0.0 else "right"
        left = int(np.searchsorted(self._oracle_arrival_times, start, side=left_side))
        right = int(np.searchsorted(self._oracle_arrival_times, end, side="right"))
        return (
            right - left,
            int(self._oracle_prompt_prefix[right] - self._oracle_prompt_prefix[left]),
            int(self._oracle_output_prefix[right] - self._oracle_output_prefix[left]),
        )

    def _oracle_snapshot(self) -> Dict[str, float]:
        horizon_s = self._oracle_horizon_s()
        requests, prompt_tokens, output_tokens = self._oracle_volume(horizon_s)
        return {
            "oracle_future_load_enabled": self.config.workload.use_oracle_future_load,
            "oracle_horizon_s": horizon_s,
            "oracle_future_requests": requests,
            "oracle_future_prompt_tokens": prompt_tokens,
            "oracle_future_output_tokens": output_tokens,
        }

    def _update_load_history(self, metrics: StepMetrics) -> None:
        self._arrival_history.append(float(metrics.arrivals))

    def _scale_down_block_reasons(self) -> Dict[str, object]:
        p99_wait = self.backend.pending_ttft_age_percentile(99)
        snapshot = self.backend.resource_snapshot()
        managed_instances = sum(
            int(snapshot.get(f"ready_{name}", 0))
            + int(snapshot.get(f"cold_{name}", 0))
            for name in self.config.instance_types
        )
                                                                   
        minimum_cluster_guard_enabled = bool(
            self.config.training.agent_family == "tgrl"
            and self.config.training.action_masking
        )
        return {
            "scale_down_blocked_by_backlog": self.backend.pending_count > 0,
            "scale_down_blocked_by_ttft": (
                p99_wait > self.SCALE_DOWN_TTFT_GUARD_S
            ),
            "scale_down_blocked_by_cooldown": (
                self._scale_down_cooldown_steps > 0
            ),
            "scale_down_blocked_by_minimum_cluster": (
                minimum_cluster_guard_enabled and managed_instances <= 1
            ),
            "managed_instances_for_scale_down_guard": managed_instances,
            "minimum_cluster_guard_enabled": minimum_cluster_guard_enabled,
            "scale_down_guard_p99_ttft_age_s": p99_wait,
            "scale_down_cooldown_steps": self._scale_down_cooldown_steps,
        }

    def _action_mask(self) -> np.ndarray:
        mask = self.backend.action_mask().copy()
        reasons = self._scale_down_block_reasons()
        blocked = bool(
            reasons["scale_down_blocked_by_backlog"]
            or reasons["scale_down_blocked_by_ttft"]
            or reasons["scale_down_blocked_by_cooldown"]
            or reasons["scale_down_blocked_by_minimum_cluster"]
        )
        if blocked:
            for index in range(2, len(mask), 2):
                mask[index] = False
        return mask

    def reset(self, episode: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
        self.episode = self.episode + 1 if episode is None else int(episode)
        cycle_count = self.config.workload.episode_cycle_count
        trace_episode = (
            self.episode % cycle_count if cycle_count > 0 else self.episode
        )
        trace_start_s = (
            self.config.workload.episode_start_offset_s
            + trace_episode * self.config.workload.episode_stride_s
        )
        self.trace_day = int(np.floor(trace_start_s / 86400.0)) + 1
        trace_end_s = trace_start_s + max(0.0, self.config.duration_s - 1e-9)
        self.trace_end_day = int(np.floor(trace_end_s / 86400.0)) + 1
        self.backend.reset(self.config.initial_instances)
        requests = self.workload.build_episode(self.config.duration_s, self.episode)
        self.backend.submit(requests)
        ordered_requests = sorted(requests, key=lambda request: request.arrival_time)
        self._oracle_arrival_times = np.asarray(
            [request.arrival_time for request in ordered_requests], dtype=np.float64
        )
        prompt_tokens = np.asarray(
            [
                int(request.served_prompt_tokens or request.prompt_tokens)
                for request in ordered_requests
            ],
            dtype=np.int64,
        )
        output_tokens = np.asarray(
            [
                int(request.served_output_tokens or request.output_tokens)
                for request in ordered_requests
            ],
            dtype=np.int64,
        )
        self._oracle_prompt_prefix = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(prompt_tokens, dtype=np.int64))
        )
        self._oracle_output_prefix = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(output_tokens, dtype=np.int64))
        )
        self.last_metrics = None
        self.episode_reward = 0.0
        self._episode_active = True
        self._previous_state_input_tokens = None
        self._update_input_token_direction()

        self.interval_controller.reset(0, episode=self.episode)
        self._termination_reason = None
        self._scale_down_cooldown_steps = 0
        self._arrival_history.clear()
        info = {
            "episode": self.episode,
            "trace_day": self.trace_day,
            "trace_start_day": self.trace_day,
            "trace_end_day": self.trace_end_day,
            "request_count": len(requests),
            "action_labels": self.action_labels(),
            "node_labels": self.node_labels(),
            "next_decision_interval_s": self.next_decision_interval_s,
            "next_decision_time_s": self.backend.current_time + self.next_decision_interval_s,
        }
        return self._observation(), info

    def _reward(
        self,
        metrics: StepMetrics,
        natural_episode_end: bool,
    ) -> Tuple[float, Dict[str, object]]:
        return calculate_step_reward(
            self.config,
            p99_ttft_s=metrics.p99_ttft_s,
            has_ttft_signal=bool(metrics.p99_ttft_s > 0.0),
            natural_episode_end=natural_episode_end,
            step_cost_usd=metrics.step_cost_usd,
            step_duration_s=metrics.end_time_s - metrics.start_time_s,
            max_pending_ttft_age_s=self.backend.pending_ttft_age_percentile(100),
            remaining_duration_s=max(0.0, self.config.duration_s - metrics.end_time_s),
        )

    def step(self, action: int):
        if not self._episode_active:
            raise RuntimeError("step called without an active episode")
        start = self.backend.current_time
        if start >= self.config.duration_s:
            raise RuntimeError("step called after the episode terminated")
        mask_before = self._action_mask()
        valid = 0 <= int(action) < len(mask_before) and bool(mask_before[int(action)])
        action_applied = self.backend.apply_action(int(action)) if valid else False
        is_scale_up = int(action) > 0 and (int(action) - 1) % 2 == 0
        if self._scale_down_cooldown_steps > 0:
            self._scale_down_cooldown_steps -= 1
        if action_applied and is_scale_up:
                                                                             
            self._scale_down_cooldown_steps = 1
        scheduled_interval_s = self.interval_controller.interval_s
        end = min(self.config.duration_s, start + scheduled_interval_s)
        self.backend.advance(end)
        total_cost = self.backend.billing_cost(0.0, end)
        metrics = aggregate_step(self.backend, start, end, total_cost)
        self.last_metrics = metrics
        natural_episode_end = end >= self.config.duration_s
        reward, reward_components = self._reward(
            metrics,
            natural_episode_end=natural_episode_end,
        )
        self._update_load_history(metrics)
        self.episode_reward += reward
        critical_ttft = bool(
            reward_components["reward_critical_ttft_terminated"]
        )
        terminated = bool(natural_episode_end or critical_ttft)
        if terminated:
            self._termination_reason = (
                "critical_ttft" if critical_ttft else "natural_end"
            )
        info: Dict[str, object] = metrics.as_dict()
        info.update(
            {
                "episode": self.episode,
                "trace_day": self.trace_day,
                "trace_start_day": self.trace_day,
                "trace_end_day": self.trace_end_day,
                "action": int(action),
                "action_label": self.action_labels()[int(action)] if 0 <= int(action) < self.action_count else "invalid",
                "action_applied": action_applied,
                "terminated_early": bool(critical_ttft and not natural_episode_end),
                "termination_reason": self._termination_reason,
                "episode_reward": self.episode_reward,
                **reward_components,
                **self._oracle_snapshot(),
                **{
                    f"arrival_count_lag_{index + 1}": float(value)
                    for index, value in enumerate(self._recent_arrivals())
                },
                **self._scale_down_block_reasons(),
                **self.backend.resource_snapshot(),
            }
        )
        self._update_input_token_direction()
        self.interval_controller.update(
            metrics.prompt_tokens_arrived,
            end - start,
        )
        info.update({
            "decision_interval_s": end - start,
            "scheduled_decision_interval_s": scheduled_interval_s,
            "next_decision_interval_s": 0.0 if terminated else self.next_decision_interval_s,
            "next_decision_time_s": None if terminated else end + self.next_decision_interval_s,
        })
        return self._observation(), reward, terminated, False, info

    def request_records(self):
        return self.backend.request_records()

    def finalize_episode(self) -> Dict[str, object]:
        if not self._episode_active:
            raise RuntimeError("no active episode to finalize")
        if self._termination_reason is None:
            raise RuntimeError("cannot finalize an episode before termination")
        cutoff_s = float(self.backend.current_time)
        result: Dict[str, object] = {
            "episode": self.episode,
            "request_records": self.backend.request_records(),
            "end_backlog_requests": self.backend.pending_count,
            "episode_reward": self.episode_reward,
            "total_cost_usd": self.backend.billing_cost(0.0, cutoff_s),
            "elapsed_time_s": cutoff_s,
            "scheduled_duration_s": float(self.config.duration_s),
            "terminated_early": self._termination_reason == "critical_ttft",
            "termination_reason": self._termination_reason,
        }

        self.backend.reset({})
        self.last_metrics = None
        self.episode_reward = 0.0
        self._episode_active = False
        self._previous_state_input_tokens = None
        self._input_token_direction = 0
        self.interval_controller.reset(0)
        self._termination_reason = None
        self._scale_down_cooldown_steps = 0
        self._arrival_history.clear()
        return result
