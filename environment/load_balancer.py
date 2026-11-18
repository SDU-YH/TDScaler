from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from environment.types import Instance, Request

class LoadBalancer(ABC):
    @abstractmethod
    def choose(self, request: Request, instances: Sequence[Instance]) -> Instance:
        raise NotImplementedError

    def reset(self) -> None:
        return None

class RoundRobinLoadBalancer(LoadBalancer):
    def __init__(self) -> None:
        self._cursor = 0

    def reset(self) -> None:
        self._cursor = 0

    def choose(self, request: Request, instances: Sequence[Instance]) -> Instance:
        if not instances:
            raise ValueError("round-robin received no eligible instances")
        ordered = sorted(instances, key=lambda instance: instance.instance_id)
        instance = ordered[self._cursor % len(ordered)]
        self._cursor += 1
        return instance

class LeastOutstandingLoadBalancer(LoadBalancer):
    def choose(self, request: Request, instances: Sequence[Instance]) -> Instance:
        if not instances:
            raise ValueError("least-outstanding received no eligible instances")
        return min(instances, key=lambda instance: (instance.outstanding, instance.instance_id))

class LeastOutstandingTokensLoadBalancer(LoadBalancer):

    def choose(self, request: Request, instances: Sequence[Instance]) -> Instance:
        if not instances:
            raise ValueError("least-outstanding-tokens received no eligible instances")
        return min(
            instances,
            key=lambda instance: (instance.outstanding_input_tokens, instance.instance_id),
        )

def make_load_balancer(name: str) -> LoadBalancer:
    if name == "round_robin":
        return RoundRobinLoadBalancer()
    if name == "least_outstanding":
        return LeastOutstandingLoadBalancer()
    if name == "least_outstanding_tokens":
        return LeastOutstandingTokensLoadBalancer()
    raise ValueError(f"unknown load-balancer policy: {name}")
