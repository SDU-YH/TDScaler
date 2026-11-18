from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from environment.backend import ClusterBackend
from environment.types import StepMetrics

class CsvStreamWriter:

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8", newline="")
        self._writer = None

    def write(self, row: Mapping[str, object]) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._handle, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

def aggregate_step(
    backend: ClusterBackend,
    start_s: float,
    end_s: float,
    total_cost_usd: float,
) -> StepMetrics:
    completed = backend.completed_between(start_s, end_s)
    first_tokens = backend.first_tokens_between(start_s, end_s)
    prompt_tokens, output_tokens = backend.arrival_token_volume_between(
        start_s, end_s
    )
    latencies = np.asarray(
        [request.completion_time - request.arrival_time for request in completed], dtype=np.float64
    )
    ttfts = np.asarray(
        [request.first_token_time - request.arrival_time for request in first_tokens],
        dtype=np.float64,
    )
    duration = max(1e-9, end_s - start_s)

    def percentile(value: float) -> float:
        return float(np.percentile(latencies, value)) if latencies.size else 0.0

    def ttft_percentile(value: float) -> float:
        return float(np.percentile(ttfts, value)) if ttfts.size else 0.0

    return StepMetrics(
        start_time_s=start_s,
        end_time_s=end_s,
        arrivals=backend.arrivals_between(start_s, end_s),
        prompt_tokens_arrived=prompt_tokens,
        output_tokens_arrived=output_tokens,
        completions=len(completed),
        throughput_rps=len(completed) / duration,
        mean_ttft_s=float(ttfts.mean()) if ttfts.size else 0.0,
        p99_ttft_s=ttft_percentile(99),
        p99_pending_ttft_age_s=backend.pending_ttft_age_percentile(99),
        p99_pending_age_s=backend.pending_age_percentile(99),
        p50_latency_s=percentile(50),
        p90_latency_s=percentile(90),
        p95_latency_s=percentile(95),
        p99_latency_s=percentile(99),
        step_cost_usd=backend.billing_cost(start_s, end_s),
        total_cost_usd=total_cost_usd,
        pending_requests=backend.pending_count,
    )

def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
