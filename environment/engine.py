from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from environment.config import InstanceTypeConfig, ModelConfig
from environment.performance_profile import PerformanceProfileCatalog
from environment.types import Instance, InstanceState, Request

@dataclass(frozen=True)
class IterationPlan:
    instance_id: str
    iteration_id: int
    start_time_s: float
    duration_s: float
    prefill_allocations: Tuple[Tuple[str, int], ...]
    decode_request_ids: Tuple[str, ...]
    mean_context_tokens: float

class ContinuousBatchEngine:

    def __init__(
        self,
        profiles: PerformanceProfileCatalog,
        models: Mapping[str, ModelConfig],
        rng: np.random.Generator,
    ):
        self.profiles = profiles
        self.models = dict(models)
        self.rng = rng

    @staticmethod
    def request_reservation(request: Request) -> int:
        return int(request.served_prompt_tokens or 0) + int(request.served_output_tokens or 0)

    @staticmethod
    def request_kv_used(request: Request) -> int:
        prompt = int(request.served_prompt_tokens or 0)
        processed_prompt = prompt - max(0, request.prefill_remaining_tokens)
        return max(0, processed_prompt) + max(0, request.generated_tokens)

    def reserved_kv_tokens(self, instance: Instance) -> int:
        return sum(self.request_reservation(request) for request in instance.active.values())

    def used_kv_tokens(self, instance: Instance) -> int:
        return sum(self.request_kv_used(request) for request in instance.active.values())

    def admit_waiting(
        self, instance: Instance, spec: InstanceTypeConfig
    ) -> List[Request]:
        admitted: List[Request] = []
        if instance.state != InstanceState.READY:
            return admitted
        reserved = self.reserved_kv_tokens(instance)
        while instance.queue and len(instance.active) < spec.max_concurrency:
            request = instance.queue[0]
            reservation = self.request_reservation(request)
            if reservation > spec.kv_cache_tokens:
                raise ValueError(
                    f"request {request.request_id} needs {reservation} KV tokens, "
                    f"but {instance.type_name} only has {spec.kv_cache_tokens}"
                )
            if reserved + reservation > spec.kv_cache_tokens:
                break
            instance.queue.pop(0)
            request.status = "prefill_wait"
            instance.active[request.request_id] = request
            admitted.append(request)
            reserved += reservation
        return admitted

    @staticmethod
    def extract_unstarted(instance: Instance) -> List[Request]:
        unstarted = [
            request for request in instance.active.values() if request.start_time is None
        ]
        for request in unstarted:
            instance.active.pop(request.request_id, None)
        return unstarted

    def build_plan(
        self,
        instance: Instance,
        spec: InstanceTypeConfig,
        now_s: float,
    ) -> Optional[IterationPlan]:
        self.admit_waiting(instance, spec)
        if instance.iteration_in_flight or not instance.active:
            return None
        budget = spec.max_batched_tokens
        decode_ids: List[str] = []
        prefill_allocations: List[Tuple[str, int]] = []
        if spec.decode_first:
            for request in instance.active.values():
                if request.prefill_remaining_tokens == 0 and request.generated_tokens < int(
                    request.served_output_tokens or 0
                ):
                    if budget <= 0:
                        break
                    decode_ids.append(request.request_id)
                    budget -= 1
        for request in instance.active.values():
            if request.prefill_remaining_tokens <= 0 or budget <= 0:
                continue
            tokens = min(
                request.prefill_remaining_tokens,
                spec.prefill_chunk_size,
                budget,
            )
            if tokens > 0:
                if request.start_time is None:
                    request.start_time = now_s
                    request.prefill_start_time = now_s
                request.status = "prefilling"
                prefill_allocations.append((request.request_id, tokens))
                budget -= tokens
        if not spec.decode_first and budget > 0:
            for request in instance.active.values():
                if request.prefill_remaining_tokens == 0 and request.generated_tokens < int(
                    request.served_output_tokens or 0
                ):
                    if budget <= 0:
                        break
                    decode_ids.append(request.request_id)
                    budget -= 1
        if not prefill_allocations and not decode_ids:
            return None
        request_ids = [request_id for request_id, _ in prefill_allocations] + decode_ids
        contexts = [self.request_kv_used(instance.active[request_id]) for request_id in request_ids]
        mean_context = float(np.mean(contexts)) if contexts else 1.0
        profile = self.profiles.resolve(
            spec.profile_gpu_type or instance.type_name,
            self.models[instance.model_name],
            spec.precision,
        )
        duration = profile.estimate_iteration_s(
            prefill_tokens=sum(tokens for _, tokens in prefill_allocations),
            prefill_sequences=len(prefill_allocations),
            decode_sequences=len(decode_ids),
            mean_context_tokens=mean_context,
            mean_prefill_tokens_per_sequence=(
                sum(tokens for _, tokens in prefill_allocations)
                / max(1, len(prefill_allocations))
            ),
            model_latency_multiplier=self.models[instance.model_name].latency_multiplier,
            rng=self.rng,
            replica_count=spec.replica_count,
        )
        instance.iteration_id += 1
        instance.iteration_in_flight = True
        return IterationPlan(
            instance_id=instance.instance_id,
            iteration_id=instance.iteration_id,
            start_time_s=now_s,
            duration_s=duration,
            prefill_allocations=tuple(prefill_allocations),
            decode_request_ids=tuple(decode_ids),
            mean_context_tokens=mean_context,
        )

    def complete_plan(
        self, instance: Instance, plan: IterationPlan, completion_time_s: float
    ) -> List[Request]:
        if not instance.iteration_in_flight or plan.iteration_id != instance.iteration_id:
            return []
        instance.iteration_in_flight = False
        instance.iteration_count += 1
        completed: List[Request] = []
        for request_id, tokens in plan.prefill_allocations:
            request = instance.active.get(request_id)
            if request is None:
                continue
            request.prefill_remaining_tokens = max(
                0, request.prefill_remaining_tokens - tokens
            )
            request.iterations += 1
            if request.prefill_remaining_tokens == 0:
                request.prefill_end_time = completion_time_s

                request.generated_tokens += 1
                if request.first_token_time is None:
                    request.first_token_time = completion_time_s
                if request.generated_tokens >= int(request.served_output_tokens or 0):
                    request.completion_time = completion_time_s
                    request.status = "completed"
                    instance.active.pop(request_id, None)
                    completed.append(request)
                else:
                    request.status = "decoding"
        for request_id in plan.decode_request_ids:
            request = instance.active.get(request_id)
            if request is None:
                continue
            request.generated_tokens += 1
            request.iterations += 1
            if request.first_token_time is None:
                request.first_token_time = completion_time_s
            if request.generated_tokens >= int(request.served_output_tokens or 0):
                request.completion_time = completion_time_s
                request.status = "completed"
                instance.active.pop(request_id, None)
                completed.append(request)
        return completed
