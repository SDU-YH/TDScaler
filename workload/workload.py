from __future__ import annotations

import csv
from datetime import datetime
from bisect import bisect_left
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from environment.config import WorkloadConfig
from environment.types import Request

@dataclass(frozen=True)
class TraceRow:
    timestamp_s: float
    prompt_tokens: int
    output_tokens: int

@dataclass(frozen=True)
class DatasetRow:
    prompt_tokens: int
    output_tokens: int
    prompt_text: str = ""
    output_text: str = ""

class WorkloadGenerator:

    def __init__(self, config: WorkloadConfig, model_name: str, seed: int):
        self.config = config
        self.model_name = model_name
        self.seed = int(seed)
        if config.max_prompt_tokens is not None and config.max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be positive")
        self.trace = self._load_trace(config.trace_path, config.max_trace_rows, config.timestamp_origin)
        if not self.trace:
            raise ValueError(f"workload trace is empty: {config.trace_path}")
        first = self.trace[0].timestamp_s
        self.trace = [
            TraceRow((row.timestamp_s - first) * config.timestamp_scale, row.prompt_tokens, row.output_tokens)
            for row in self.trace
        ]
        self._timestamps = [row.timestamp_s for row in self.trace]
        self.dataset = self._load_dataset(config.dataset_path) if config.dataset_path else []
        if config.token_source in {"dataset", "mixed"} and not self.dataset:
            raise ValueError(f"token dataset is empty: {config.dataset_path}")

    @staticmethod
    def _load_trace(path: Path, max_rows: int, timestamp_origin: str | None = None) -> List[TraceRow]:
        rows: List[TraceRow] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"Timestamp", "Request tokens", "Response tokens"}
            columns = set(reader.fieldnames or [])
            azure = {"TIMESTAMP", "ContextTokens", "GeneratedTokens"}.issubset(columns)
            if not required.issubset(columns) and not azure:
                raise ValueError("trace must use BurstGPT or Azure timestamp/token columns")
            azure_origin = datetime.fromisoformat(timestamp_origin) if azure and timestamp_origin else None
            for raw in reader:
                try:
                    if azure:
                        arrived = datetime.fromisoformat(raw["TIMESTAMP"].replace("Z", "+00:00"))
                        if azure_origin is None:
                            azure_origin = arrived
                        timestamp = (arrived - azure_origin).total_seconds()
                        prompt = max(1, int(float(raw["ContextTokens"])))
                        output = max(1, int(float(raw["GeneratedTokens"])))
                    else:
                        timestamp = float(raw["Timestamp"])
                        prompt = max(1, int(float(raw["Request tokens"])))
                        output = max(1, int(float(raw["Response tokens"])))
                except (TypeError, ValueError):
                    continue
                rows.append(TraceRow(timestamp, prompt, output))
                if max_rows > 0 and len(rows) >= max_rows:
                    break
        rows.sort(key=lambda row: row.timestamp_s)
        return rows

    @staticmethod
    def _load_dataset(path: Path) -> List[DatasetRow]:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        values: Sequence[dict]
        if isinstance(raw, dict):
            values = list(raw.values())
        elif isinstance(raw, list):
            values = raw
        else:
            raise ValueError("dataset must be a JSON object or array")
        result: List[DatasetRow] = []
        for item in values:
            if not isinstance(item, dict):
                continue
            prompt = item.get("len_prompt", item.get("prompt_tokens"))
            output = item.get("len_output", item.get("output_tokens"))
            if prompt is None or output is None:
                continue
            result.append(
                DatasetRow(
                    prompt_tokens=max(1, int(prompt)),
                    output_tokens=max(1, int(output)),
                    prompt_text=str(item.get("prompt", "")),
                    output_text=str(item.get("output", "")),
                )
            )
        return result

    def _window_rows(self, start: float, duration: float) -> List[TraceRow]:
        end = start + duration
        if not self.config.loop_trace:
            left = bisect_left(self._timestamps, start)
            right = bisect_left(self._timestamps, end)
            return self.trace[left:right]
        positive_gaps = np.diff(self._timestamps)
        positive_gaps = positive_gaps[positive_gaps > 0]
        final_gap = float(np.median(positive_gaps)) if positive_gaps.size else 1.0
        period = max(1e-9, self.trace[-1].timestamp_s + final_gap)
        first_cycle = int(np.floor(start / period))
        last_cycle = int(np.floor(max(start, end - 1e-12) / period))
        selected: List[TraceRow] = []
        for cycle in range(first_cycle, last_cycle + 1):
            offset = cycle * period
            local_start = max(0.0, start - offset)
            local_end = min(period, end - offset)
            left = bisect_left(self._timestamps, local_start)
            right = bisect_left(self._timestamps, local_end)
            selected.extend(
                TraceRow(row.timestamp_s + offset, row.prompt_tokens, row.output_tokens)
                for row in self.trace[left:right]
            )
        return selected

    def build_episode(self, duration_s: float, episode: int = 0) -> List[Request]:
        trace_episode = int(episode)
        if self.config.episode_cycle_count > 0:
            trace_episode %= self.config.episode_cycle_count
        origin = (
            self.config.episode_start_offset_s
            + trace_episode * self.config.episode_stride_s
        )
        rows = self._window_rows(origin, duration_s)
        rng = np.random.default_rng(self.seed + int(episode))
        result: List[Request] = []
        scale_floor = int(np.floor(self.config.request_scale))
        scale_fraction = self.config.request_scale - scale_floor

        fractional_count = min(
            len(rows), int(np.floor(len(rows) * scale_fraction + 0.5))
        )
        fractional_rows = (
            set(
                int(index)
                for index in rng.choice(
                    len(rows), size=fractional_count, replace=False
                )
            )
            if fractional_count
            else set()
        )
        dataset_cursor = episode * max(1, len(rows))
        for row_index, row in enumerate(rows):
            copies = scale_floor + int(row_index in fractional_rows)
            for copy_index in range(copies):
                dataset_index = -1
                prompt, output = row.prompt_tokens, row.output_tokens
                prompt_text = ""
                output_text = ""
                use_dataset = self.config.token_source == "dataset" or (
                    self.config.token_source == "mixed" and rng.random() < 0.5
                )
                if use_dataset:
                    if self.config.dataset_sampling == "sequential":
                        dataset_index = (dataset_cursor + row_index + copy_index) % len(self.dataset)
                    else:
                        dataset_index = int(rng.integers(0, len(self.dataset)))
                    dataset_row = self.dataset[dataset_index]
                    prompt, output = dataset_row.prompt_tokens, dataset_row.output_tokens
                    prompt_text, output_text = dataset_row.prompt_text, dataset_row.output_text
                if self.config.max_prompt_tokens is not None:
                    prompt = min(prompt, self.config.max_prompt_tokens)
                arrival = row.timestamp_s - origin
                if copy_index:
                    arrival += float(rng.uniform(0.0, 0.001))
                arrival = min(arrival, max(0.0, duration_s - 1e-9))
                request_id = f"ep{episode:04d}-r{row_index:07d}-c{copy_index:02d}"
                result.append(
                    Request(
                        request_id=request_id,
                        arrival_time=max(0.0, arrival),
                        model_name=self.model_name,
                        prompt_tokens=prompt,
                        output_tokens=output,
                        dataset_index=dataset_index,
                        prompt_text=prompt_text,
                        reference_output_text=output_text,
                    )
                )
        result.sort(key=lambda request: (request.arrival_time, request.request_id))
        return result
