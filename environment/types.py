from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

class InstanceState(str, Enum):
    COLD = "cold"
    READY = "ready"
    DRAINING = "draining"
    REMOVED = "removed"

@dataclass
class Request:
    request_id: str
    arrival_time: float
    model_name: str
    prompt_tokens: int
    output_tokens: int
    dataset_index: int = -1
    prompt_text: str = ""
    reference_output_text: str = ""
    served_prompt_tokens: Optional[int] = None
    served_output_tokens: Optional[int] = None
    assigned_instance_id: str = ""
    start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None
    prefill_start_time: Optional[float] = None
    prefill_end_time: Optional[float] = None
    prefill_remaining_tokens: int = 0
    generated_tokens: int = 0
    iterations: int = 0
    status: str = "created"
    requeues: int = 0

    def as_record(self) -> Dict[str, object]:
        if self.completion_time is None or self.start_time is None or self.first_token_time is None:
            raise ValueError(f"request {self.request_id} has not completed")
        return {
            "request_id": self.request_id,
            "model_name": self.model_name,
            "instance_id": self.assigned_instance_id,
            "arrival_time_s": self.arrival_time,
            "start_time_s": self.start_time,
            "first_token_time_s": self.first_token_time,
            "completion_time_s": self.completion_time,
            "prefill_start_time_s": self.prefill_start_time,
            "prefill_end_time_s": self.prefill_end_time,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "served_prompt_tokens": self.served_prompt_tokens,
            "served_output_tokens": self.served_output_tokens,
            "queue_delay_s": self.start_time - self.arrival_time,
            "prefill_time_s": (
                self.prefill_end_time - self.prefill_start_time
                if self.prefill_end_time is not None and self.prefill_start_time is not None
                else None
            ),
            "decode_time_s": self.completion_time - self.prefill_end_time
            if self.prefill_end_time is not None
            else None,
            "ttft_s": self.first_token_time - self.arrival_time,
            "service_time_s": self.completion_time - self.start_time,
            "latency_s": self.completion_time - self.arrival_time,
            "requeues": self.requeues,
            "iterations": self.iterations,
            "status": self.status,
            "is_censored": False,
            "censor_time_s": None,
        }

@dataclass
class Instance:
    instance_id: str
    type_name: str
    model_name: str
    created_at: float
    ready_at: float
    state: InstanceState
    removed_at: Optional[float] = None
    queue: List[Request] = field(default_factory=list)
    active: Dict[str, Request] = field(default_factory=dict)
    iteration_id: int = 0
    iteration_count: int = 0
    iteration_in_flight: bool = False

    last_ttft_s: float = 0.0

    @property
    def outstanding(self) -> int:
        return len(self.active) + len(self.queue)

    @property
    def outstanding_input_tokens(self) -> int:

        return sum(
            int(request.prompt_tokens if request.served_prompt_tokens is None
                else request.served_prompt_tokens)
            for requests in (self.queue, self.active.values())
            for request in requests
        )

@dataclass(frozen=True)
class StepMetrics:
    start_time_s: float
    end_time_s: float
    arrivals: int
    prompt_tokens_arrived: int
    output_tokens_arrived: int
    completions: int
    throughput_rps: float
    mean_ttft_s: float
    p99_ttft_s: float
    p99_pending_ttft_age_s: float
    p99_pending_age_s: float
    p50_latency_s: float
    p90_latency_s: float
    p95_latency_s: float
    p99_latency_s: float
    step_cost_usd: float
    total_cost_usd: float
    pending_requests: int

    def as_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)
