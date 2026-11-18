from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from environment.config import InstanceTypeConfig, ModelConfig

@dataclass(frozen=True)
class IterationProfile:
    gpu_type: str
    model_name: str
    precision: str
    source: str
    reference_model_latency_multiplier: float
    prefill_overhead_s: float
    mixed_batch_penalty: float
    jitter_sigma: float
    prefill_rate_points: Tuple[Tuple[int, float], ...]
    decode_iteration_points: Tuple[Tuple[int, float], ...]
    context_multiplier_points: Tuple[Tuple[int, float], ...]
    prefill_length_multiplier_points: Tuple[Tuple[int, float], ...] = ((1, 1.0),)

    def __post_init__(self) -> None:
        if self.reference_model_latency_multiplier <= 0:
            raise ValueError("reference model latency multiplier must be positive")
        if self.prefill_overhead_s < 0 or self.mixed_batch_penalty <= 0:
            raise ValueError("profile overhead/penalty is invalid")
        if self.jitter_sigma < 0:
            raise ValueError("profile jitter must be non-negative")
        for curve_name, points in (
            ("prefill rate", self.prefill_rate_points),
            ("decode iteration", self.decode_iteration_points),
            ("decode context", self.context_multiplier_points),
            ("prefill length", self.prefill_length_multiplier_points),
        ):
            if not points or any(x <= 0 or y <= 0 for x, y in points):
                raise ValueError(f"{curve_name} curve must contain positive points")

    @staticmethod
    def _interpolate(points: Sequence[Tuple[int, float]], x: float) -> float:
        if not points:
            raise ValueError("performance profile contains an empty curve")
        ordered = sorted(points)
        xs = np.asarray([point[0] for point in ordered], dtype=np.float64)
        ys = np.asarray([point[1] for point in ordered], dtype=np.float64)
        return float(np.interp(float(x), xs, ys, left=ys[0], right=ys[-1]))

    def estimate_iteration_s(
        self,
        prefill_tokens: int,
        prefill_sequences: int,
        decode_sequences: int,
        mean_context_tokens: float,
        mean_prefill_tokens_per_sequence: float,
        model_latency_multiplier: float,
        rng: np.random.Generator,
        replica_count: int = 1,
    ) -> float:
        replicas = max(1, int(replica_count))
        prefill_time = 0.0
        if prefill_tokens > 0:
            active_replicas = min(replicas, max(1, prefill_sequences))
            sequences_per_replica = max(
                1, int(np.ceil(prefill_sequences / active_replicas))
            )
            rate = active_replicas * self._interpolate(
                self.prefill_rate_points, sequences_per_replica
            )
            rate *= self._interpolate(
                self.prefill_length_multiplier_points,
                max(1.0, mean_prefill_tokens_per_sequence),
            )
            prefill_time = self.prefill_overhead_s + prefill_tokens / max(1e-9, rate)
        decode_time = 0.0
        if decode_sequences > 0:
            active_replicas = min(replicas, max(1, decode_sequences))
            sequences_per_replica = max(
                1, int(np.ceil(decode_sequences / active_replicas))
            )
            base = self._interpolate(
                self.decode_iteration_points, sequences_per_replica
            )
            context_factor = self._interpolate(
                self.context_multiplier_points, max(1.0, mean_context_tokens)
            )
            decode_time = base * context_factor
        duration = prefill_time + decode_time
        if prefill_time > 0 and decode_time > 0:
            duration *= self.mixed_batch_penalty
        duration *= model_latency_multiplier / max(
            1e-9, self.reference_model_latency_multiplier
        )
        if self.jitter_sigma > 0:
            sigma = self.jitter_sigma
            duration *= float(rng.lognormal(-0.5 * sigma * sigma, sigma))
        return max(1e-6, duration)

class PerformanceProfileCatalog:
    def __init__(self, profiles: Iterable[IterationProfile]):
        self._profiles: Dict[Tuple[str, str, str], IterationProfile] = {}
        for profile in profiles:
            key = (profile.gpu_type, profile.model_name, profile.precision)
            if key in self._profiles:
                raise ValueError(f"duplicate iteration profile: {key}")
            self._profiles[key] = profile
        if not self._profiles:
            raise ValueError("performance profile catalog is empty")

    @classmethod
    def load(cls, path: Path) -> "PerformanceProfileCatalog":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        for relative_path in raw.get("profile_files", []):
            with (path.parent / relative_path).open("r", encoding="utf-8") as handle:
                raw.setdefault("profiles", []).extend(json.load(handle)["profiles"])
        if raw.get("calibration_status") == "pending":
            raise ValueError("Measured performance profiles are not available yet: " + str(path))
        profiles = []
        for item in raw.get("profiles", []):
            profiles.append(
                IterationProfile(
                    gpu_type=item["gpu_type"],
                    model_name=item["model_name"],
                    precision=item.get("precision", "fp16"),
                    source=item.get("source", str(path)),
                    reference_model_latency_multiplier=float(
                        item.get("reference_model_latency_multiplier", 1.0)
                    ),
                    prefill_overhead_s=float(item.get("prefill_overhead_s", 0.002)),
                    mixed_batch_penalty=float(item.get("mixed_batch_penalty", 1.0)),
                    jitter_sigma=float(item.get("jitter_sigma", 0.0)),
                    prefill_rate_points=tuple(
                        (int(point["batch_size"]), float(point["tokens_per_s"]))
                        for point in item["prefill_rate_points"]
                    ),
                    decode_iteration_points=tuple(
                        (int(point["batch_size"]), float(point["time_s"]))
                        for point in item["decode_iteration_points"]
                    ),
                    context_multiplier_points=tuple(
                        (int(point["context_tokens"]), float(point["multiplier"]))
                        for point in item.get(
                            "context_multiplier_points",
                            [{"context_tokens": 1, "multiplier": 1.0}],
                        )
                    ),
                    prefill_length_multiplier_points=tuple(
                        (int(point["tokens"]), float(point["multiplier"]))
                        for point in item.get(
                            "prefill_length_multiplier_points",
                            [{"tokens": 1, "multiplier": 1.0}],
                        )
                    ),
                )
            )
        return cls(profiles)

    @classmethod
    def from_instance_configs(
        cls, instance_types: Mapping[str, InstanceTypeConfig], model_name: str
    ) -> "PerformanceProfileCatalog":
        profiles = []
        configured_profiles = set()
        for name, spec in instance_types.items():
            profile_gpu_type = spec.profile_gpu_type or name
            if profile_gpu_type in configured_profiles:
                continue
            configured_profiles.add(profile_gpu_type)
            replicas = max(1, spec.replica_count)
            maximum = max(1, int(np.ceil(spec.max_concurrency / replicas)))
            base_prefill_rate = spec.prefill_tokens_per_s / replicas
            base_decode_rate = spec.decode_tokens_per_s / replicas
            contention = 1.0 + spec.concurrency_penalty * max(0, maximum - 1)
            profiles.append(
                IterationProfile(
                    gpu_type=profile_gpu_type,
                    model_name=model_name,
                    precision=spec.precision,
                    source=(
                        f"legacy single-GPU fallback from InstanceTypeConfig({name})"
                    ),
                    reference_model_latency_multiplier=1.0,
                    prefill_overhead_s=spec.fixed_ttft_s,
                    mixed_batch_penalty=1.0,
                    jitter_sigma=spec.service_jitter,
                    prefill_rate_points=(
                        (1, base_prefill_rate),
                        (maximum, base_prefill_rate * min(4.0, maximum ** 0.5)),
                    ),
                    decode_iteration_points=(
                        (1, 1.0 / base_decode_rate),
                        (maximum, contention / base_decode_rate),
                    ),
                    context_multiplier_points=((1, 1.0),),
                    prefill_length_multiplier_points=((1, 1.0),),
                )
            )
        return cls(profiles)

    def resolve(
        self, gpu_type: str, model: ModelConfig, precision: str
    ) -> IterationProfile:
        exact = self._profiles.get((gpu_type, model.name, precision))
        if exact is not None:
            return exact
        raise KeyError(
            f"No exact measured profile for gpu={gpu_type!r}, model={model.name!r}, "
            f"precision={precision!r}; cross-model profile substitution is disabled."
        )
