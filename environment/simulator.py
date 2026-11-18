from __future__ import annotations

import copy
import heapq
from bisect import bisect_left, bisect_right
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from environment.backend import ClusterBackend
from environment.config import InstanceTypeConfig, ModelConfig
from environment.engine import ContinuousBatchEngine, IterationPlan
from environment.load_balancer import LoadBalancer, make_load_balancer
from environment.performance_profile import PerformanceProfileCatalog
from environment.types import Instance, InstanceState, Request, StepMetrics

class SimulatorClusterBackend(ClusterBackend):
                                                                              
    NODE_FEATURE_DIM = 2

    def __init__(
        self,
        instance_types: Mapping[str, InstanceTypeConfig],
        models: Mapping[str, ModelConfig],
        active_model: str,
        load_balancer: str,
        seed: int,
        performance_profile_path: Optional[Path] = None,
        record_iterations: bool = True,
    ):
        self.instance_types = dict(instance_types)
        pending = [name for name, spec in self.instance_types.items()
                   if spec.calibration_source.startswith("PENDING")]
        if pending:
            raise ValueError("Instance calibration is incomplete: " + ", ".join(pending))
        self.type_order = list(instance_types)
        self.models = dict(models)
        self.active_model = active_model
        self.lb: LoadBalancer = make_load_balancer(load_balancer)
        self.seed = int(seed)
        self.record_iterations = bool(record_iterations)
        self.rng = np.random.default_rng(seed)
        if performance_profile_path is None:
            raise ValueError("An explicit measured performance profile catalog is required")
        self.performance_profiles = PerformanceProfileCatalog.load(performance_profile_path)
        for name, spec in self.instance_types.items():
            self.performance_profiles.resolve(
                spec.profile_gpu_type or name, self.models[active_model], spec.precision
            )
        self.reset({name: 0 for name in self.type_order})

    @property
    def action_count(self) -> int:
        return 1 + 2 * len(self.type_order)

    @property
    def node_feature_dim(self) -> int:
        return self.NODE_FEATURE_DIM

    @property
    def current_time(self) -> float:
        return self._time

    @property
    def completed_count(self) -> int:
        return len(self._completed)

    @property
    def next_event_time(self) -> Optional[float]:
        return float(self._events[0][0]) if self._events else None

    def reset(self, initial_instances: Mapping[str, int]) -> None:
        self._time = 0.0
        self._event_seq = 0
        self._instance_seq = 0
        self._events: List[Tuple[float, int, int, str, str]] = []
        self._iteration_plans: Dict[Tuple[str, int], IterationPlan] = {}
        self._iteration_records: List[Dict[str, object]] = []
        self.instances: Dict[str, Instance] = {}
        self._live: Dict[str, Instance] = {}
        self._live_by_type: Dict[str, Dict[str, Instance]] = {
            name: {} for name in self.type_order
        }
        self.requests: Dict[str, Request] = {}
        self._global_queue: Deque[Request] = deque()
        self._completed: List[Request] = []
        self._completed_times: List[float] = []
        self._first_tokens: List[Request] = []
        self._first_token_times: List[float] = []
        self._arrived_pending: Dict[str, Request] = {}
        self._waiting_first_token: Dict[str, Request] = {}
        self._arrival_index_dirty = True
        self._arrival_times: List[float] = []
        self._arrival_prompt_prefix: List[int] = [0]
        self._arrival_output_prefix: List[int] = [0]
        self._arrivals_seen = 0
        self.rng = np.random.default_rng(self.seed)
        self.engine = ContinuousBatchEngine(
            self.performance_profiles, self.models, self.rng
        )
        self.lb.reset()
        for type_name in self.type_order:
            for _ in range(int(initial_instances.get(type_name, 0))):
                self.add_instance(type_name, ready_immediately=True)

    def _push_event(self, time_s: float, priority: int, kind: str, identifier: str) -> None:
        self._event_seq += 1
        heapq.heappush(self._events, (float(time_s), priority, self._event_seq, kind, identifier))

    def submit(self, requests: Iterable[Request]) -> None:
        for request in requests:
            if request.request_id in self.requests:
                raise ValueError(f"duplicate request id: {request.request_id}")
            model = self.models.get(request.model_name)
            if model is None:
                raise ValueError(f"request targets unknown model: {request.model_name}")
            requested_prompt = max(1, request.prompt_tokens)
            requested_output = max(1, request.output_tokens)
            requested_total = requested_prompt + requested_output
            if requested_total <= model.max_total_tokens:
                served_prompt, served_output = requested_prompt, requested_output
            else:
                served_prompt = max(
                    1, int(model.max_total_tokens * requested_prompt / requested_total)
                )
                served_prompt = min(served_prompt, model.max_total_tokens - 1)
                served_output = model.max_total_tokens - served_prompt
            request.served_prompt_tokens = served_prompt
            request.served_output_tokens = served_output
            request.prefill_remaining_tokens = served_prompt
            request.generated_tokens = 0
            request.iterations = 0
            request.start_time = None
            request.prefill_start_time = None
            request.prefill_end_time = None
            request.first_token_time = None
            request.completion_time = None
            self.requests[request.request_id] = request

            if request.arrival_time <= self._time:
                self._arrived_pending[request.request_id] = request
                self._waiting_first_token[request.request_id] = request
            self._push_event(request.arrival_time, 1, "arrival", request.request_id)
        self._arrival_index_dirty = True

    def add_instance(self, type_name: str, ready_immediately: bool = False) -> Optional[str]:
        spec = self.instance_types[type_name]
        live = len(self._live_by_type[type_name])
        if live >= spec.max_instances or self.active_model not in spec.supported_models:
            return None
        self._instance_seq += 1
        instance_id = f"{type_name}-{self._instance_seq:04d}"
        ready_at = self._time if ready_immediately else self._time + spec.cold_start_s
        state = InstanceState.READY if ready_immediately else InstanceState.COLD
        instance = Instance(
            instance_id=instance_id,
            type_name=type_name,
            model_name=self.active_model,
            created_at=self._time,
            ready_at=ready_at,
            state=state,
        )
        self.instances[instance_id] = instance
        self._live[instance_id] = instance
        self._live_by_type[type_name][instance_id] = instance
        if ready_immediately:
            self._redispatch_global()
        else:
            self._push_event(ready_at, 0, "cold_ready", instance_id)
        return instance_id

    def remove_instance(self, type_name: str) -> Optional[str]:
        spec = self.instance_types[type_name]
        live = list(self._live_by_type[type_name].values())
        candidates = [instance for instance in live if instance.state == InstanceState.READY]
        if len(live) <= spec.min_instances or not candidates:
            return None
        instance = min(candidates, key=lambda item: (len(item.active), item.outstanding, -item.created_at))
        instance.state = InstanceState.DRAINING
        queued = list(instance.queue) + self.engine.extract_unstarted(instance)
        instance.queue.clear()
        for request in queued:
            request.requeues += 1
            request.status = "requeued"
            self._route(request)
        if not instance.active:
            self._mark_removed(instance)
        else:
            self._schedule_instance(instance)
        return instance.instance_id

    def _mark_removed(self, instance: Instance) -> None:
        instance.state = InstanceState.REMOVED
        instance.removed_at = self._time
        self._live.pop(instance.instance_id, None)
        self._live_by_type[instance.type_name].pop(instance.instance_id, None)

    def apply_action(self, action: int) -> bool:
        if action == 0:
            return True
        if action < 0 or action >= self.action_count:
            return False
        type_index = (action - 1) // 2
        is_add = (action - 1) % 2 == 0
        type_name = self.type_order[type_index]
        result = self.add_instance(type_name) if is_add else self.remove_instance(type_name)
        return result is not None

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_count, dtype=bool)
        mask[0] = True
        for index, type_name in enumerate(self.type_order):
            spec = self.instance_types[type_name]
            live = list(self._live_by_type[type_name].values())
            ready = [instance for instance in live if instance.state == InstanceState.READY]
            compatible = self.active_model in spec.supported_models
            mask[1 + 2 * index] = compatible and len(live) < spec.max_instances
            mask[2 + 2 * index] = len(live) > spec.min_instances and bool(ready)
        return mask

    def _eligible_instances(self, request: Request) -> List[Instance]:
        return [
            instance
            for instance in self._live.values()
            if instance.state == InstanceState.READY and instance.model_name == request.model_name
        ]

    def _route(self, request: Request, schedule: bool = True) -> bool:
        eligible = self._eligible_instances(request)
        if not eligible:
            request.status = "global_queue"
            self._global_queue.append(request)
            return False
        instance = self.lb.choose(request, eligible)
        request.assigned_instance_id = instance.instance_id
        request.status = "instance_queue"
        instance.queue.append(request)
        if schedule:
            self._schedule_instance(instance)
        return True

    def _redispatch_global(self) -> None:
        waiting = list(self._global_queue)
        self._global_queue.clear()
        for index, request in enumerate(waiting):
            if not self._route(request, schedule=False):
                                                                                
                self._global_queue.extend(waiting[index + 1 :])
                break
        for instance in self._live.values():
            if instance.state == InstanceState.READY:
                self._schedule_instance(instance)

    def _schedule_instance(self, instance: Instance) -> None:
        if instance.state not in {InstanceState.READY, InstanceState.DRAINING}:
            return
        plan = self.engine.build_plan(
            instance, self.instance_types[instance.type_name], self._time
        )
        if plan is None:
            return
        key = (instance.instance_id, plan.iteration_id)
        self._iteration_plans[key] = plan
        identifier = f"{instance.instance_id}|{plan.iteration_id}"
        self._push_event(
            self._time + plan.duration_s, 2, "iteration_complete", identifier
        )

    def _schedule_ready_instances(self) -> None:
        for instance in self._live.values():
            if instance.state in {InstanceState.READY, InstanceState.DRAINING}:
                self._schedule_instance(instance)

    def _on_iteration_complete(self, identifier: str) -> None:
        instance_id, iteration_raw = identifier.rsplit("|", 1)
        iteration_id = int(iteration_raw)
        plan = self._iteration_plans.pop((instance_id, iteration_id), None)
        if plan is None:
            return
        instance = self.instances[instance_id]
        active_before = len(instance.active) if self.record_iterations else 0
        queued_before = len(instance.queue) if self.record_iterations else 0
        first_output_tokens = (
            sum(
                request_id in instance.active
                and instance.active[request_id].prefill_remaining_tokens <= tokens
                for request_id, tokens in plan.prefill_allocations
            )
            if self.record_iterations
            else 0
        )
        first_token_requests = []
        for request_id, tokens in plan.prefill_allocations:
            request = instance.active.get(request_id)
            if (
                request is not None
                and request.first_token_time is None
                and request.prefill_remaining_tokens <= tokens
            ):
                first_token_requests.append(request)
        for request_id in plan.decode_request_ids:
            request = instance.active.get(request_id)
            if request is not None and request.first_token_time is None:
                first_token_requests.append(request)
        completed = self.engine.complete_plan(instance, plan, self._time)
        self._completed.extend(completed)
        self._completed_times.extend(
            float(request.completion_time) for request in completed
        )
        for request in first_token_requests:
            if request.first_token_time is not None:
                instance.last_ttft_s = float(
                    request.first_token_time - request.arrival_time
                )
                self._first_tokens.append(request)
                self._first_token_times.append(float(request.first_token_time))
                self._waiting_first_token.pop(request.request_id, None)
        for request in completed:
            self._arrived_pending.pop(request.request_id, None)
            self._waiting_first_token.pop(request.request_id, None)
        if self.record_iterations:
            decode_tokens = len(plan.decode_request_ids)
            self._iteration_records.append(
                {
                    "instance_id": instance.instance_id,
                    "instance_type": instance.type_name,
                    "iteration_id": plan.iteration_id,
                    "start_time_s": plan.start_time_s,
                    "end_time_s": self._time,
                    "duration_s": plan.duration_s,
                    "prefill_tokens": sum(
                        tokens for _, tokens in plan.prefill_allocations
                    ),
                    "decode_tokens": decode_tokens,
                    "first_output_tokens": first_output_tokens,
                    "output_tokens": decode_tokens + first_output_tokens,
                    "completed_requests": len(completed),
                    "active_requests_before": active_before,
                    "active_requests_after": len(instance.active),
                    "queued_requests_before": queued_before,
                    "queued_requests_after": len(instance.queue),
                    "mean_context_tokens": plan.mean_context_tokens,
                }
            )
        if instance.state == InstanceState.DRAINING:
            if not instance.active and not instance.iteration_in_flight:
                self._mark_removed(instance)
            else:
                self._schedule_instance(instance)
            return
        self._redispatch_global()
        self._schedule_instance(instance)

    def advance(self, until_s: float) -> None:
        if until_s < self._time:
            raise ValueError("cannot move simulation time backwards")
        while self._events and self._events[0][0] <= until_s:
            time_s, _, _, kind, identifier = heapq.heappop(self._events)
            self._time = time_s
            if kind == "arrival":
                self._arrivals_seen += 1
                request = self.requests[identifier]
                request.status = "arrived"
                self._arrived_pending[request.request_id] = request
                self._waiting_first_token[request.request_id] = request

                self._route(request, schedule=False)
                more_arrivals_now = bool(
                    self._events
                    and self._events[0][0] == self._time
                    and self._events[0][3] == "arrival"
                )
                if not more_arrivals_now:
                    self._schedule_ready_instances()
            elif kind == "cold_ready":
                instance = self.instances[identifier]
                if instance.state == InstanceState.COLD:
                    instance.state = InstanceState.READY
                    self._redispatch_global()
            elif kind == "iteration_complete":
                self._on_iteration_complete(identifier)
        self._time = float(until_s)

    def billing_cost(self, start_s: float, end_s: float) -> float:
        cost = 0.0
        for instance in self.instances.values():
            billed_start = max(start_s, instance.created_at)
            lifecycle_end = instance.removed_at if instance.removed_at is not None else end_s
            billed_end = min(end_s, lifecycle_end)
            if billed_end > billed_start:
                cost += (billed_end - billed_start) * self.instance_types[instance.type_name].hourly_cost_usd / 3600.0
        return cost

    def fork_arrived_only(self) -> "SimulatorClusterBackend":

        fork = copy.deepcopy(self)
        current_time = float(fork._time)
        outstanding_ids = {
            request_id
            for request_id, request in fork.requests.items()
            if request.arrival_time <= current_time
            and request.completion_time is None
        }
        fork.requests = {
            request_id: request
            for request_id, request in fork.requests.items()
            if request_id in outstanding_ids
        }
        fork._events = [
            event
            for event in fork._events
            if event[3] != "arrival"
            or (
                event[4] in outstanding_ids
                and event[0] <= current_time
            )
        ]
        heapq.heapify(fork._events)
        fork._completed = []
        fork._completed_times = []
        fork._first_tokens = []
        fork._first_token_times = []
        fork._iteration_records = []
        fork._arrived_pending = {
            request_id: request
            for request_id, request in fork._arrived_pending.items()
            if request_id in outstanding_ids
        }
        fork._waiting_first_token = {
            request_id: request
            for request_id, request in fork._waiting_first_token.items()
            if request_id in outstanding_ids
        }
        fork._arrival_index_dirty = True
        fork._arrival_times = []
        fork._arrival_prompt_prefix = [0]
        fork._arrival_output_prefix = [0]
        fork._arrivals_seen = 0
        return fork

    def completed_between(self, start_s: float, end_s: float) -> List[Request]:
        left = bisect_right(self._completed_times, start_s)
        right = bisect_right(self._completed_times, end_s)
        return self._completed[left:right]

    def first_tokens_between(self, start_s: float, end_s: float) -> List[Request]:
        left = bisect_right(self._first_token_times, start_s)
        right = bisect_right(self._first_token_times, end_s)
        return self._first_tokens[left:right]

    def _ensure_arrival_index(self) -> None:
        if not self._arrival_index_dirty:
            return
        ordered = sorted(
            self.requests.values(),
            key=lambda request: (request.arrival_time, request.request_id),
        )
        self._arrival_times = [float(request.arrival_time) for request in ordered]
        self._arrival_prompt_prefix = [0]
        self._arrival_output_prefix = [0]
        for request in ordered:
            self._arrival_prompt_prefix.append(
                self._arrival_prompt_prefix[-1]
                + int(request.served_prompt_tokens or 0)
            )
            self._arrival_output_prefix.append(
                self._arrival_output_prefix[-1]
                + int(request.served_output_tokens or 0)
            )
        self._arrival_index_dirty = False

    def _arrival_bounds(self, start_s: float, end_s: float) -> Tuple[int, int]:
        self._ensure_arrival_index()
        left = (
            bisect_left(self._arrival_times, 0.0)
            if start_s == 0.0
            else bisect_right(self._arrival_times, start_s)
        )
        return left, bisect_right(self._arrival_times, end_s)

    def arrivals_between(self, start_s: float, end_s: float) -> int:
        left, right = self._arrival_bounds(start_s, end_s)
        return right - left

    def arrival_token_volume_between(
        self, start_s: float, end_s: float
    ) -> Tuple[int, int]:
        left, right = self._arrival_bounds(start_s, end_s)
        return (
            self._arrival_prompt_prefix[right] - self._arrival_prompt_prefix[left],
            self._arrival_output_prefix[right] - self._arrival_output_prefix[left],
        )

    @property
    def pending_count(self) -> int:
        return len(self._arrived_pending)

    @property
    def pending_input_tokens(self) -> int:
        return sum(
            int(request.served_prompt_tokens or 0)
            for request in self._arrived_pending.values()
        )

    def pending_age_percentile(self, percentile: float) -> float:
        ages = [
            max(0.0, self._time - request.arrival_time)
            for request in self._arrived_pending.values()
        ]
        return float(np.percentile(ages, percentile)) if ages else 0.0

    def pending_ttft_age_percentile(self, percentile: float) -> float:
        ages = [
            max(0.0, self._time - request.arrival_time)
            for request in self._waiting_first_token.values()
        ]
        return float(np.percentile(ages, percentile)) if ages else 0.0

    def completed_records(self) -> List[Dict[str, object]]:
        records = []
        for request in self._completed:
            record = request.as_record()
            record["instance_type"] = self.instances[request.assigned_instance_id].type_name
            records.append(record)
        return records

    def iteration_records(self) -> List[Dict[str, object]]:
        return [dict(record) for record in self._iteration_records]

    def request_records(self) -> List[Dict[str, object]]:

        records: List[Dict[str, object]] = []
        for request in sorted(self.requests.values(), key=lambda item: item.request_id):
            if request.arrival_time > self._time:
                continue
            if request.completion_time is not None:
                record = request.as_record()
                record["instance_type"] = self.instances[request.assigned_instance_id].type_name
                records.append(record)
                continue
            cutoff = self._time
            first_token = (
                request.first_token_time
                if request.first_token_time is not None and request.first_token_time <= self._time
                else None
            )
            queue_delay = (
                request.start_time - request.arrival_time
                if request.start_time is not None
                else cutoff - request.arrival_time
            )
            prefill_time = (
                0.0
                if request.prefill_start_time is None
                else max(
                    0.0,
                    (
                        request.prefill_end_time
                        if request.prefill_end_time is not None
                        else cutoff
                    )
                    - request.prefill_start_time,
                )
            )
            decode_time = (
                0.0
                if request.prefill_end_time is None
                else max(0.0, cutoff - request.prefill_end_time)
            )
            service_time = (
                0.0
                if request.start_time is None
                else max(0.0, cutoff - request.start_time)
            )
            records.append(
                {
                    "request_id": request.request_id,
                    "model_name": request.model_name,
                    "instance_id": request.assigned_instance_id,
                    "instance_type": self.instances[request.assigned_instance_id].type_name
                    if request.assigned_instance_id
                    else "",
                    "arrival_time_s": request.arrival_time,
                    "start_time_s": request.start_time,
                    "first_token_time_s": first_token,
                    "completion_time_s": None,
                    "prefill_start_time_s": request.prefill_start_time,
                    "prefill_end_time_s": request.prefill_end_time,
                    "prompt_tokens": request.prompt_tokens,
                    "output_tokens": request.output_tokens,
                    "served_prompt_tokens": request.served_prompt_tokens,
                    "served_output_tokens": request.served_output_tokens,
                    "queue_delay_s": queue_delay,
                    "prefill_time_s": prefill_time,
                    "decode_time_s": decode_time,
                    "ttft_s": (
                        first_token - request.arrival_time
                        if first_token is not None
                        else cutoff - request.arrival_time
                    ),
                    "service_time_s": service_time,
                    "latency_s": cutoff - request.arrival_time,
                    "requeues": request.requeues,
                    "iterations": request.iterations,
                    "status": request.status,
                    "is_censored": True,
                    "censor_time_s": cutoff,
                }
            )
        return records

    def resource_snapshot(self) -> Dict[str, object]:
        snapshot: Dict[str, object] = {"global_queue": len(self._global_queue)}
        for type_name in self.type_order:
            live = list(self._live_by_type[type_name].values())
            snapshot.update(
                {
                    f"live_{type_name}": len(live),
                    f"ready_{type_name}": sum(instance.state == InstanceState.READY for instance in live),
                    f"cold_{type_name}": sum(instance.state == InstanceState.COLD for instance in live),
                    f"draining_{type_name}": sum(
                        instance.state == InstanceState.DRAINING for instance in live
                    ),
                    f"active_{type_name}": sum(len(instance.active) for instance in live),
                    f"queued_{type_name}": sum(len(instance.queue) for instance in live),
                    f"kv_used_{type_name}": sum(
                        self.engine.used_kv_tokens(instance) for instance in live
                    ),
                    f"kv_reserved_{type_name}": sum(
                        self.engine.reserved_kv_tokens(instance) for instance in live
                    ),
                    f"iterations_{type_name}": sum(
                        instance.iteration_count for instance in live
                    ),
                }
            )
        return snapshot

    def _lifecycle_slot_instances(
        self,
    ) -> Tuple[Dict[str, List[Instance]], Dict[str, List[Instance]]]:
        cold: Dict[str, List[Instance]] = {}
        running: Dict[str, List[Instance]] = {}
        for type_name in self.type_order:
            live = sorted(
                self._live_by_type[type_name].values(),
                key=lambda instance: (instance.created_at, instance.instance_id),
            )
            cold[type_name] = [
                instance for instance in live if instance.state == InstanceState.COLD
            ]
            running[type_name] = [
                instance
                for instance in live
                if instance.state in {InstanceState.READY, InstanceState.DRAINING}
            ]
        return cold, running

    def node_mask(self) -> np.ndarray:
        type_count = len(self.type_order)
        total_slots = sum(
            self.instance_types[name].max_instances for name in self.type_order
        )
        mask = np.zeros(2 + type_count + 2 * total_slots, dtype=bool)
        mask[: 2 + type_count] = True
        cold_by_type, running_by_type = self._lifecycle_slot_instances()
        cold_offset = 2 + type_count
        running_offset = cold_offset + total_slots
        slot_offset = 0
        for type_name in self.type_order:
            slot_count = self.instance_types[type_name].max_instances
            mask[
                cold_offset + slot_offset : cold_offset + slot_offset + len(cold_by_type[type_name])
            ] = True
            mask[
                running_offset + slot_offset : running_offset + slot_offset + len(running_by_type[type_name])
            ] = True
            slot_offset += slot_count
        return mask

    def node_features(
        self,
        recent: StepMetrics | None,
        scaling_interval_s: float,
        sla_s: float = 1.0,
        decision_index: int = 0,
        input_token_direction: int = 0,
    ) -> np.ndarray:

        del sla_s, decision_index, input_token_direction
        type_count = len(self.type_order)
        total_slots = sum(
            self.instance_types[name].max_instances for name in self.type_order
        )
        nodes = np.zeros(
            (2 + type_count + 2 * total_slots, self.NODE_FEATURE_DIM),
            dtype=np.float32,
        )
        nodes[0, 0] = np.float32(scaling_interval_s)
        if recent is not None:
            elapsed = recent.end_time_s - recent.start_time_s
            if elapsed <= 0:
                raise ValueError("previous decision interval must have positive duration")
            nodes[0, 1] = (
                0 if np.isclose(scaling_interval_s, elapsed, rtol=0, atol=1e-9)
                else 1 if scaling_interval_s > elapsed else -1
            )
            nodes[1] = (recent.prompt_tokens_arrived,
                        recent.prompt_tokens_arrived / elapsed)

        cold_by_type, running_by_type = self._lifecycle_slot_instances()
        for type_index, type_name in enumerate(self.type_order):
            running_count = len(running_by_type[type_name])
            cold_count = len(cold_by_type[type_name])
            nodes[2 + type_index] = (cold_count, running_count)
        cold_offset = 2 + type_count
        running_offset = cold_offset + total_slots
        slot_offset = 0
        for type_name in self.type_order:
            for local_slot, instance in enumerate(cold_by_type[type_name]):
                remaining = max(0.0, instance.ready_at - self._time)
                nodes[cold_offset + slot_offset + local_slot, 0] = np.float32(
                    remaining
                )
            for local_slot, instance in enumerate(running_by_type[type_name]):
                nodes[running_offset + slot_offset + local_slot] = (
                    self.engine.used_kv_tokens(instance),
                    sum(int(request.prompt_tokens if request.served_prompt_tokens is None
                            else request.served_prompt_tokens) for request in instance.queue),
                )
            slot_offset += self.instance_types[type_name].max_instances
        return nodes
