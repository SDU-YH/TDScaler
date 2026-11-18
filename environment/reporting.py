from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value

def backlog_metrics(
    records: Iterable[Dict[str, object]], duration_s: float
) -> Dict[str, float]:
    deltas: Dict[float, int] = {}
    request_seconds = 0.0
    end_backlog = 0
    for record in records:
        arrival = float(record["arrival_time_s"])
        completion = record["completion_time_s"]
        end = duration_s if completion is None else min(duration_s, float(completion))
        end = max(arrival, end)
        request_seconds += end - arrival
        deltas[arrival] = deltas.get(arrival, 0) + 1
        deltas[end] = deltas.get(end, 0) - 1
        if bool(record["is_censored"]):
            end_backlog += 1

    outstanding = 0
    peak = 0
    for timestamp in sorted(deltas):
        outstanding += deltas[timestamp]
        peak = max(peak, outstanding)
    return {
        "end_backlog_requests": float(end_backlog),
        "mean_backlog_requests": request_seconds / max(1e-9, duration_s),
        "max_backlog_requests": float(peak),
        "backlog_request_seconds": request_seconds,
    }
