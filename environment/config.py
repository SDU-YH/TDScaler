from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np

@dataclass(frozen=True)
class ModelConfig:
    name: str
    size_b: float
    latency_multiplier: float = 1.0
    max_total_tokens: int = 4096

@dataclass(frozen=True)
class InstanceTypeConfig:
    name: str
    hourly_cost_usd: float
    cold_start_s: float
    max_concurrency: int
    prefill_tokens_per_s: float
    decode_tokens_per_s: float
    fixed_ttft_s: float = 0.05
    concurrency_penalty: float = 0.04
    service_jitter: float = 0.08
    min_instances: int = 0
    max_instances: int = 8
    supported_models: List[str] = field(default_factory=list)
    calibration_source: str = "user-configured"
    precision: str = "fp16"
    max_batched_tokens: int = 4096
    prefill_chunk_size: int = 512
    kv_cache_tokens: int = 16384
    decode_first: bool = True
    gpu_count: int = 1
    tensor_parallel_size: int = 1
    profile_gpu_type: Optional[str] = None

    @property
    def replica_count(self) -> int:
        return self.gpu_count // self.tensor_parallel_size

@dataclass(frozen=True)
class WorkloadConfig:
    trace_path: Path
    dataset_path: Optional[Path] = None
    token_source: str = "dataset"
    timestamp_scale: float = 1.0
    request_scale: float = 1.0
    episode_start_offset_s: float = 0.0
    episode_stride_s: float = 0.0
    episode_cycle_count: int = 0
    loop_trace: bool = False
    max_trace_rows: int = 0
    dataset_sampling: str = "seeded_random"
    max_prompt_tokens: Optional[int] = None
    timestamp_origin: Optional[str] = None
    state_arrival_scale: float = 7.0
    state_prompt_token_scale: float = 18231.0
    state_output_token_scale: float = 2733.0
    use_oracle_future_load: bool = True

@dataclass(frozen=True)
class RewardConfig:
    mode: str = "sla_cost"
    cost_sla_reference_interval_s: float = 300.0
    cost_sla_reference_hourly_cost: float = 100.0
    cost_sla_cost_weight: float = 1.0
    cost_sla_violation_weight: float = 10.0
    cost_sla_excess_weight: float = 2.0
    cost_sla_terminal_weight: float = 10.0
    sla_s: float = 1.0
    critical_ttft_s: float = 5.0
    early_termination_penalty: float = 60.0
                                                                           

@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 10
    replay_capacity: int = 500000
    replay_prefill_steps: int = 50000
    replay_prefill_cache_path: Optional[Path] = None
    replay_prefill_expert_fraction: float = 0.0
    replay_prefill_expert_trajectory_path: Optional[Path] = None
    replay_prefill_expert_interval_s: Optional[float] = None
    replay_prefill_expert_episodes: int = 0
    replay_prefill_perturbed_expert_episodes: int = 0
    replay_prefill_random_episodes: int = 0
    replay_prefill_perturbation_decisions: int = 0
    batch_size: int = 32
    context_length: int = 50
    gamma: float = 0.99
    learning_rate: float = 0.0003
    target_update_steps: int = 10000
    learning_interval_steps: int = 1
    gradient_updates_per_learning_step: int = 1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 5000
    hidden_dim: int = 128
    attention_heads: int = 8
    transformer_layers: int = 2
    transformer_dropout: float = 0.0
    graph_attention_heads: int = 2
    history_length: int = 50
    grad_norm_clip: float = 1.0
    agent_family: str = "tgrl"
    graph_encoder: str = "han"
    action_masking: bool = True
    torch_num_threads: int = 1
    checkpoint_interval_episodes: int = 10
    evaluation_interval_episodes: int = 10
    evaluation_days: int = 1
    evaluation_day_duration_s: float = 0.0
    lr_plateau_enabled: bool = False
    lr_plateau_patience: int = 3
    lr_plateau_factor: float = 0.5
    lr_plateau_min_lr: float = 1e-5
    lr_plateau_min_delta: float = 0.0

@dataclass(frozen=True)
class AdaptiveIntervalConfig:
    enabled: bool = False
    mode: str = "load"
    random_step_s: float = 120.0
    adjustment_s: float = 120.0
    min_interval_s: float = 120.0
    max_interval_s: float = 1800.0
    history_length: int = 4
    high_change_threshold: float = 0.20
    low_change_threshold: float = 0.05
    stable_cycles: int = 2
    input_token_floor: float = 1024.0

    def validate(self, initial_interval_s: float) -> None:
        if self.mode not in {"fixed", "load", "rate_window", "random"}:
            raise ValueError(
                "adaptive_interval.mode must be fixed, load, rate_window or random"
            )
        if not np.isfinite(self.random_step_s) or self.random_step_s <= 0:
            raise ValueError("adaptive_interval.random_step_s must be positive and finite")
        positive = (self.adjustment_s, self.min_interval_s, self.max_interval_s,
                    self.input_token_floor, self.high_change_threshold)
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("adaptive_interval durations, floor and high threshold must be positive and finite")
        if (not np.isfinite(self.low_change_threshold)
                or not 0 <= self.low_change_threshold < self.high_change_threshold):
            raise ValueError("adaptive_interval requires 0 <= low threshold < high threshold")
        if (not isinstance(self.history_length, int) or self.history_length < 2
                or not isinstance(self.stable_cycles, int) or self.stable_cycles < 1):
            raise ValueError("adaptive_interval requires integer history_length >= 2 and stable_cycles >= 1")
        if self.max_interval_s < self.min_interval_s:
            raise ValueError("adaptive_interval maximum must be >= minimum")
        if (self.enabled and self.mode != "fixed"
                and not self.min_interval_s <= initial_interval_s <= self.max_interval_s):
            raise ValueError("scaling_interval_s must lie within adaptive_interval bounds")
        if self.enabled and self.mode == "random":
            steps = (self.max_interval_s - self.min_interval_s) / self.random_step_s
            if not np.isclose(steps, round(steps), rtol=0, atol=1e-9):
                raise ValueError("random interval range must be divisible by random_step_s")

@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    duration_s: float
    scaling_interval_s: float
    load_balancer: str
    active_model: str
    initial_instances: Dict[str, int]
    output_dir: Path
    workload: WorkloadConfig
    reward: RewardConfig
    training: TrainingConfig
    models: Dict[str, ModelConfig]
    instance_types: Dict[str, InstanceTypeConfig]
    performance_profile_path: Optional[Path] = None
    adaptive_interval: AdaptiveIntervalConfig = field(default_factory=AdaptiveIntervalConfig)

    @property
    def max_decision_steps(self) -> int:
        interval = (
            self.adaptive_interval.min_interval_s
            if self.adaptive_interval.enabled
            and self.adaptive_interval.mode != "fixed"
            else self.scaling_interval_s
        )
        return int(math.ceil(self.duration_s / interval))

    @property
    def min_decision_steps(self) -> int:
        interval = (
            self.adaptive_interval.max_interval_s
            if self.adaptive_interval.enabled
            and self.adaptive_interval.mode != "fixed"
            else self.scaling_interval_s
        )
        return int(math.ceil(self.duration_s / interval))

    def validate(self) -> None:
        source_interval = self.training.replay_prefill_expert_interval_s
        if source_interval is not None and (
            not np.isfinite(source_interval) or source_interval <= 0
        ):
            raise ValueError("replay_prefill_expert_interval_s must be positive and finite")
        if self.duration_s <= 0 or self.scaling_interval_s <= 0:
            raise ValueError("duration_s and scaling_interval_s must be positive")
        self.adaptive_interval.validate(self.scaling_interval_s)
        if self.active_model not in self.models:
            raise ValueError(f"unknown active_model: {self.active_model}")
        if self.load_balancer not in {"round_robin", "least_outstanding", "least_outstanding_tokens"}:
            raise ValueError(
                "load_balancer must be round_robin, least_outstanding or least_outstanding_tokens"
            )
        if self.workload.token_source not in {"trace", "dataset", "mixed"}:
            raise ValueError("token_source must be trace, dataset, or mixed")
        if self.workload.dataset_sampling not in {"seeded_random", "sequential"}:
            raise ValueError("dataset_sampling must be seeded_random or sequential")
        if self.workload.token_source in {"dataset", "mixed"} and not self.workload.dataset_path:
            raise ValueError("dataset_path is required by the selected token_source")
        if self.workload.timestamp_scale <= 0 or self.workload.request_scale <= 0:
            raise ValueError("workload scales must be positive")
        if min(
            self.workload.state_arrival_scale,
            self.workload.state_prompt_token_scale,
            self.workload.state_output_token_scale,
        ) <= 0:
            raise ValueError("workload state normalization scales must be positive")
        if self.workload.episode_cycle_count < 0:
            raise ValueError("episode_cycle_count must be non-negative")
        if self.reward.sla_s <= 0:
            raise ValueError("reward.sla_s must be positive")
        if self.reward.mode != "sla_cost":
            raise ValueError("reward.mode must be sla_cost")
        for name in ("cost_sla_reference_interval_s", "cost_sla_reference_hourly_cost",
                     "cost_sla_cost_weight", "cost_sla_violation_weight",
                     "cost_sla_excess_weight", "cost_sla_terminal_weight"):
            value = getattr(self.reward, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"reward.{name} must be positive and finite")
        if self.reward.critical_ttft_s <= self.reward.sla_s:
            raise ValueError("reward.critical_ttft_s must be greater than SLA")
        if (
            not np.isfinite(self.reward.early_termination_penalty)
            or self.reward.early_termination_penalty < 0
        ):
            raise ValueError(
                "reward.early_termination_penalty must be finite and non-negative"
            )
        if not self.instance_types:
            raise ValueError("at least one instance type is required")
        if self.training.checkpoint_interval_episodes <= 0:
            raise ValueError("training.checkpoint_interval_episodes must be positive")
        if self.training.evaluation_interval_episodes <= 0:
            raise ValueError("training.evaluation_interval_episodes must be positive")
        if (self.training.lr_plateau_patience < 1
                or not 0 < self.training.lr_plateau_factor < 1
                or self.training.lr_plateau_min_lr <= 0
                or self.training.lr_plateau_min_delta < 0):
            raise ValueError("invalid learning-rate plateau parameters")
        if self.training.evaluation_days <= 0:
            raise ValueError("training.evaluation_days must be positive")
        if self.training.evaluation_day_duration_s < 0:
            raise ValueError(
                "training.evaluation_day_duration_s must be non-negative"
            )
        if self.training.replay_prefill_steps < 0:
            raise ValueError("training.replay_prefill_steps must be non-negative")
        if not 0.0 <= self.training.replay_prefill_expert_fraction <= 1.0:
            raise ValueError(
                "training.replay_prefill_expert_fraction must be in [0, 1]"
            )
        if min(
            self.training.replay_prefill_expert_episodes,
            self.training.replay_prefill_perturbed_expert_episodes,
            self.training.replay_prefill_random_episodes,
            self.training.replay_prefill_perturbation_decisions,
        ) < 0:
            raise ValueError("expert prefill episode/perturbation counts must be non-negative")
        explicit_expert_episodes = (
            self.training.replay_prefill_expert_episodes
            + self.training.replay_prefill_perturbed_expert_episodes
        )
        if (
            self.training.replay_prefill_perturbed_expert_episodes > 0
            and self.training.replay_prefill_perturbation_decisions <= 0
        ):
            raise ValueError(
                "replay_prefill_perturbation_decisions must be positive when "
                "perturbed expert episodes are enabled"
            )
        if (
            self.training.replay_prefill_expert_fraction > 0.0
            or explicit_expert_episodes > 0
        ):
            expert_path = self.training.replay_prefill_expert_trajectory_path
            if expert_path is None:
                raise ValueError(
                    "training.replay_prefill_expert_trajectory_path is required "
                    "when expert prefill is enabled"
                )
            if not expert_path.is_file():
                raise ValueError(
                    f"expert prefill trajectory does not exist: {expert_path}"
                )
        if self.training.learning_rate <= 0 or self.training.target_update_steps <= 0:
            raise ValueError(
                "training learning_rate and target_update_steps must be positive"
            )
        if self.training.context_length <= 0 or self.training.transformer_layers <= 0:
            raise ValueError(
                "training context_length and transformer_layers must be positive"
            )
        if self.training.attention_heads <= 0 or self.training.graph_attention_heads <= 0:
            raise ValueError("training attention head counts must be positive")
        if self.training.hidden_dim % self.training.attention_heads:
            raise ValueError("training.hidden_dim must be divisible by attention_heads")
        if self.training.hidden_dim % self.training.graph_attention_heads:
            raise ValueError(
                "training.hidden_dim must be divisible by graph_attention_heads"
            )
        if not 0.0 <= self.training.transformer_dropout < 1.0:
            raise ValueError("training.transformer_dropout must be in [0, 1)")
        if self.training.history_length <= 0:
            raise ValueError("training.history_length must be positive")
        if self.training.grad_norm_clip <= 0:
            raise ValueError("training.grad_norm_clip must be positive")
        if (self.training.agent_family != "tgrl" or self.training.graph_encoder != "han"
                or self.training.action_masking is not True):
            raise ValueError("TDScaler requires tgrl, han, action_masking=True")
        if self.training.torch_num_threads <= 0:
            raise ValueError("training.torch_num_threads must be positive")
        if self.training.learning_interval_steps <= 0:
            raise ValueError("training.learning_interval_steps must be positive")
        if self.training.gradient_updates_per_learning_step <= 0:
            raise ValueError(
                "training.gradient_updates_per_learning_step must be positive"
            )
        if not (
            0.0
            <= self.training.epsilon_end
            <= self.training.epsilon_start
            <= 1.0
        ):
            raise ValueError(
                "training epsilon values must satisfy 0 <= end <= start <= 1"
            )
        if self.performance_profile_path is not None and not self.performance_profile_path.is_file():
            raise ValueError(
                f"performance profile does not exist: {self.performance_profile_path}"
            )
        unknown_initial = set(self.initial_instances) - set(self.instance_types)
        if unknown_initial:
            raise ValueError(f"initial_instances contains unknown types: {sorted(unknown_initial)}")
        for model in self.models.values():
            if model.size_b <= 0 or model.latency_multiplier <= 0 or model.max_total_tokens < 2:
                raise ValueError(f"invalid model values for {model.name}")
        compatible = 0
        for name, spec in self.instance_types.items():
            if spec.name != name:
                raise ValueError(f"instance type key/name mismatch: {name}/{spec.name}")
            if spec.min_instances < 0 or spec.max_instances < spec.min_instances:
                raise ValueError(f"invalid instance bounds for {name}")
            if spec.max_concurrency <= 0 or spec.prefill_tokens_per_s <= 0 or spec.decode_tokens_per_s <= 0:
                raise ValueError(f"invalid performance values for {name}")
            if spec.hourly_cost_usd < 0 or spec.cold_start_s < 0:
                raise ValueError(f"cost and cold start must be non-negative for {name}")
            if spec.fixed_ttft_s < 0 or spec.concurrency_penalty < 0 or spec.service_jitter < 0:
                raise ValueError(f"latency parameters must be non-negative for {name}")
            if spec.max_batched_tokens <= 0 or spec.prefill_chunk_size <= 0:
                raise ValueError(f"scheduler token limits must be positive for {name}")
            if spec.prefill_chunk_size > spec.max_batched_tokens:
                raise ValueError(f"prefill chunk exceeds batch token limit for {name}")
            if spec.kv_cache_tokens < 2:
                raise ValueError(f"kv_cache_tokens is too small for {name}")
            if spec.gpu_count <= 0:
                raise ValueError(f"gpu_count must be positive for {name}")
            if (not isinstance(spec.tensor_parallel_size, int)
                    or spec.tensor_parallel_size <= 0
                    or spec.gpu_count % spec.tensor_parallel_size):
                raise ValueError(f"tensor_parallel_size must be a positive divisor of gpu_count for {name}")
            initial = self.initial_instances.get(name, 0)
            if not spec.min_instances <= initial <= spec.max_instances:
                raise ValueError(f"initial count for {name} is outside min/max bounds")
            if self.active_model in spec.supported_models:
                compatible += initial
        if compatible == 0:
            raise ValueError("the initial cluster has no instance compatible with active_model")

def _resolve(base: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return path if path.is_absolute() else (base / path).resolve()

def _catalog_items(raw_catalog: object, field_name: str) -> List[Dict[str, object]]:
    if isinstance(raw_catalog, Mapping):
        items: List[Dict[str, object]] = []
        for catalog_name, raw_item in raw_catalog.items():
            if not isinstance(raw_item, Mapping):
                raise TypeError(f"{field_name}.{catalog_name} must be an object")
            item = dict(raw_item)
            item.setdefault("name", str(catalog_name))
            if item["name"] != catalog_name:
                raise ValueError(
                    f"{field_name} key {catalog_name!r} differs from item name "
                    f"{item['name']!r}"
                )
            items.append(item)
        return items
    if isinstance(raw_catalog, list):
        if not all(isinstance(item, Mapping) for item in raw_catalog):
            raise TypeError(f"every {field_name} list item must be an object")
        return [dict(item) for item in raw_catalog]
    raise TypeError(f"{field_name} must be a list or an object keyed by name")

def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    base = config_path.parent
    workload_raw = raw["workload"]
    reward_raw = dict(raw.get("reward", {}))

    reward_raw.pop("lagrangian_initial", None)
    reward_raw.pop("lagrangian_multiplier", None)
    reward_raw.pop("lagrangian_learning_rate", None)
    reward_raw.pop("lagrangian_max", None)
    for obsolete in (
        "instance_switch_penalty",
        "load_decrease_noop_reward",
        "load_decrease_remove_rewards",
        "load_increase_add_rewards",
        "noop_reward",
        "sla_violation_reward",
        "stable_episode_bonus",
        "wrong_direction_reward",
        "cost_weight",
        "latency_weight",
        "backlog_weight",
        "sla_violation_penalty",
        "sla_violation_log_cap",
    ):
        reward_raw.pop(obsolete, None)
    training_raw = dict(raw.get("training", {}))

    training_raw.pop("warmup_steps", None)
    training_raw.pop("critical_replay_fraction", None)
    replay_prefill_cache_path = _resolve(
        base, training_raw.pop("replay_prefill_cache_path", None)
    )
    replay_prefill_expert_trajectory_path = _resolve(
        base,
        training_raw.pop("replay_prefill_expert_trajectory_path", None),
    )
    workload = WorkloadConfig(
        trace_path=_resolve(base, workload_raw["trace_path"]),
        dataset_path=_resolve(base, workload_raw.get("dataset_path")),
        token_source=workload_raw.get("token_source", "dataset"),
        timestamp_scale=float(workload_raw.get("timestamp_scale", 1.0)),
        request_scale=float(workload_raw.get("request_scale", 1.0)),
        episode_start_offset_s=float(workload_raw.get("episode_start_offset_s", 0.0)),
        episode_stride_s=float(workload_raw.get("episode_stride_s", 0.0)),
        episode_cycle_count=int(workload_raw.get("episode_cycle_count", 0)),
        loop_trace=bool(workload_raw.get("loop_trace", False)),
        max_trace_rows=int(workload_raw.get("max_trace_rows", 0)),
        dataset_sampling=workload_raw.get("dataset_sampling", "seeded_random"),
        max_prompt_tokens=(int(workload_raw["max_prompt_tokens"])
                           if workload_raw.get("max_prompt_tokens") is not None else None),
        timestamp_origin=workload_raw.get("timestamp_origin"),
        state_arrival_scale=float(workload_raw.get("state_arrival_scale", 7.0)),
        state_prompt_token_scale=float(
            workload_raw.get("state_prompt_token_scale", 18231.0)
        ),
        state_output_token_scale=float(
            workload_raw.get("state_output_token_scale", 2733.0)
        ),
        use_oracle_future_load=bool(
            workload_raw.get("use_oracle_future_load", True)
        ),
    )
    model_items = _catalog_items(raw["models"], "models")
    instance_items = _catalog_items(raw["instance_types"], "instance_types")
    models = {item["name"]: ModelConfig(**item) for item in model_items}
    instance_catalog = {
        item["name"]: InstanceTypeConfig(**item) for item in instance_items
    }
    instance_types = instance_catalog
    initial_instances = {k: int(v) for k, v in raw["initial_instances"].items()}
    config = ExperimentConfig(
        seed=int(raw.get("seed", 20250805)),
        duration_s=float(raw["duration_s"]),
        scaling_interval_s=float(raw["scaling_interval_s"]),
        load_balancer=raw.get("load_balancer", "least_outstanding_tokens"),
        active_model=raw["active_model"],
        initial_instances=initial_instances,
        output_dir=_resolve(base, raw.get("output_dir", "../runs/default")),
        workload=workload,
        reward=RewardConfig(**reward_raw),
        training=TrainingConfig(
            **training_raw,
            replay_prefill_cache_path=replay_prefill_cache_path,
            replay_prefill_expert_trajectory_path=(
                replay_prefill_expert_trajectory_path
            ),
        ),
        models=models,
        instance_types=instance_types,
        performance_profile_path=_resolve(base, raw.get("performance_profile_path")),
        adaptive_interval=AdaptiveIntervalConfig(**raw.get("adaptive_interval", {})),
    )
    config.validate()
    return config

def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
