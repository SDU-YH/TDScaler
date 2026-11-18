from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np

from environment.types import Request, StepMetrics

class ClusterBackend(ABC):
    @property
    @abstractmethod
    def current_time(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def action_count(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def node_feature_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def reset(self, initial_instances: Mapping[str, int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def submit(self, requests: Iterable[Request]) -> None:
        raise NotImplementedError

    @abstractmethod
    def advance(self, until_s: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_action(self, action: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def action_mask(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def node_features(
        self,
        recent: StepMetrics | None,
        scaling_interval_s: float,
        sla_s: float = 1.0,
        decision_index: int = 0,
        input_token_direction: int = 0,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def node_mask(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def completed_records(self) -> List[Dict[str, object]]:
        raise NotImplementedError

    @abstractmethod
    def completed_between(self, start_s: float, end_s: float) -> List[Request]:
        raise NotImplementedError

    @abstractmethod
    def first_tokens_between(self, start_s: float, end_s: float) -> List[Request]:
        raise NotImplementedError

    @abstractmethod
    def arrivals_between(self, start_s: float, end_s: float) -> int:
        raise NotImplementedError

    @abstractmethod
    def arrival_token_volume_between(
        self, start_s: float, end_s: float
    ) -> Tuple[int, int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def pending_count(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def pending_input_tokens(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def pending_age_percentile(self, percentile: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def pending_ttft_age_percentile(self, percentile: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def billing_cost(self, start_s: float, end_s: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def request_records(self) -> List[Dict[str, object]]:
        raise NotImplementedError

    @abstractmethod
    def resource_snapshot(self) -> Dict[str, object]:
        raise NotImplementedError

class RealClusterBackend(ClusterBackend):

    def __init__(self, endpoints: Mapping[str, str]):
        self.endpoints = dict(endpoints)

    @property
    def current_time(self) -> float:
        self._missing()
        return 0.0

    @property
    def action_count(self) -> int:
        self._missing()
        return 0

    @property
    def node_feature_dim(self) -> int:
        self._missing()
        return 0

    def _missing(self) -> None:
        raise NotImplementedError(
            "RealClusterBackend is an integration boundary. Implement scale APIs, "
            "request dispatch, and metrics collection for the rented cluster."
        )

    def reset(self, initial_instances: Mapping[str, int]) -> None:
        self._missing()

    def submit(self, requests: Iterable[Request]) -> None:
        self._missing()

    def advance(self, until_s: float) -> None:
        self._missing()

    def apply_action(self, action: int) -> bool:
        self._missing()
        return False

    def action_mask(self) -> np.ndarray:
        self._missing()
        return np.empty(0, dtype=bool)

    def node_features(
        self,
        recent: StepMetrics | None,
        scaling_interval_s: float,
        sla_s: float = 1.0,
        decision_index: int = 0,
        input_token_direction: int = 0,
    ) -> np.ndarray:
        self._missing()
        return np.empty((0, 0), dtype=np.float32)

    def node_mask(self) -> np.ndarray:
        self._missing()
        return np.empty(0, dtype=bool)

    def completed_records(self) -> List[Dict[str, object]]:
        self._missing()
        return []

    def completed_between(self, start_s: float, end_s: float) -> List[Request]:
        self._missing()
        return []

    def first_tokens_between(self, start_s: float, end_s: float) -> List[Request]:
        self._missing()
        return []

    def arrivals_between(self, start_s: float, end_s: float) -> int:
        self._missing()
        return 0

    def arrival_token_volume_between(
        self, start_s: float, end_s: float
    ) -> Tuple[int, int]:
        self._missing()
        return 0, 0

    @property
    def pending_count(self) -> int:
        self._missing()
        return 0

    @property
    def pending_input_tokens(self) -> int:
        self._missing()
        return 0

    def pending_age_percentile(self, percentile: float) -> float:
        self._missing()
        return 0.0

    def pending_ttft_age_percentile(self, percentile: float) -> float:
        self._missing()
        return 0.0

    def billing_cost(self, start_s: float, end_s: float) -> float:
        self._missing()
        return 0.0

    def request_records(self) -> List[Dict[str, object]]:
        self._missing()
        return []

    def resource_snapshot(self) -> Dict[str, object]:
        self._missing()
        return {}
