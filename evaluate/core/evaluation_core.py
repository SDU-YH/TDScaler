from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from environment.config import ExperimentConfig
from environment.env import AutoscalingEnv
from environment.metrics import CsvStreamWriter, write_json
from environment.reporting import backlog_metrics, json_ready

def daily_reset_episode_config(
    config: ExperimentConfig,
    day_duration_s: Optional[float] = None,
) -> ExperimentConfig:
    if day_duration_s is None:
        day_duration_s = (
            config.training.evaluation_day_duration_s
            if config.training.evaluation_day_duration_s > 0
            else config.duration_s / config.training.evaluation_days
        )
    if day_duration_s <= 0:
        raise ValueError("day_duration_s must be positive")
    workload = replace(
        config.workload,
        episode_stride_s=day_duration_s,
        episode_cycle_count=config.training.evaluation_days,
    )
    return replace(
        config,
        duration_s=day_duration_s,
        workload=workload,
    )

def episode_metrics(
    evaluation_episode: int,
    trace_episode: int,
    reset_info: Dict[str, object],
    settlement: Dict[str, object],
    duration_s: float,
    switch_actions: int,
    step_rows: List[Dict[str, object]],
    initial_instances: Dict[str, int],
    instance_type_names: List[str],
) -> Dict[str, object]:
    records = settlement["request_records"]
    completed = [record for record in records if record["status"] == "completed"]
    unfinished = [record for record in records if bool(record["is_censored"])]
    backlog = backlog_metrics(records, duration_s)
    latencies = np.asarray(
        [record["latency_s"] for record in completed], dtype=np.float64
    )
    ttfts = np.asarray(
        [record["ttft_s"] for record in completed], dtype=np.float64
    )
    queue_delays = np.asarray(
        [record["queue_delay_s"] for record in completed], dtype=np.float64
    )
    service_times = np.asarray(
        [record["service_time_s"] for record in completed], dtype=np.float64
    )

    def percentile(values: np.ndarray, value: float) -> float:
        return float(np.percentile(values, value)) if values.size else 0.0

    request_count = len(records)
    interval_throughput = np.asarray(
        [float(row["throughput_rps"]) for row in step_rows], dtype=np.float64
    )
    row: Dict[str, object] = {
        "evaluation_day": evaluation_episode + 1,
        "evaluation_episode": evaluation_episode,
        "trace_episode": trace_episode,
        "trace_day": reset_info["trace_day"],
        "trace_start_day": reset_info["trace_start_day"],
        "trace_end_day": reset_info["trace_end_day"],
        "episode_duration_s": duration_s,
        "scheduled_duration_s": float(settlement["scheduled_duration_s"]),
        "terminated_early": bool(settlement["terminated_early"]),
        "termination_reason": settlement["termination_reason"],
        "decision_steps": len(step_rows),
        "scheduled_requests": int(reset_info["request_count"]),
        "requests": request_count,
        "completed": len(completed),
        "unfinished": len(unfinished),
        "censor_ratio": len(unfinished) / max(1, request_count),
        "end_backlog_requests": int(settlement["end_backlog_requests"]),
        "mean_backlog_requests": backlog["mean_backlog_requests"],
        "max_backlog_requests": int(backlog["max_backlog_requests"]),
        "backlog_request_seconds": backlog["backlog_request_seconds"],
        "completion_ratio": len(completed) / max(1, request_count),
        "throughput_rps": len(completed) / duration_s,
        "mean_interval_throughput_rps": (
            float(interval_throughput.mean()) if interval_throughput.size else 0.0
        ),
        "peak_interval_throughput_rps": (
            float(interval_throughput.max()) if interval_throughput.size else 0.0
        ),
        "mean_latency_s": float(latencies.mean()) if latencies.size else 0.0,
        "mean_ttft_s": float(ttfts.mean()) if ttfts.size else 0.0,
        "p50_ttft_s": percentile(ttfts, 50),
        "p90_ttft_s": percentile(ttfts, 90),
        "p95_ttft_s": percentile(ttfts, 95),
        "p99_ttft_s": percentile(ttfts, 99),
        "mean_queue_delay_s": (
            float(queue_delays.mean()) if queue_delays.size else 0.0
        ),
        "p99_queue_delay_s": percentile(queue_delays, 99),
        "mean_service_time_s": (
            float(service_times.mean()) if service_times.size else 0.0
        ),
        "p99_service_time_s": percentile(service_times, 99),
        "p50_latency_s": percentile(latencies, 50),
        "p90_latency_s": percentile(latencies, 90),
        "p95_latency_s": percentile(latencies, 95),
        "p99_latency_s": percentile(latencies, 99),
        "total_cost_usd": float(settlement["total_cost_usd"]),
        "episode_reward": float(settlement["episode_reward"]),
        "switch_actions": switch_actions,
    }
    for type_name in instance_type_names:
        interval_values = [
            int(step[f"live_{type_name}"]) for step in step_rows
        ]
        initial = int(initial_instances.get(type_name, 0))
        all_values = [initial, *interval_values]
        row[f"mean_live_{type_name}"] = (
            float(np.mean(interval_values)) if interval_values else float(initial)
        )
        row[f"max_live_{type_name}"] = max(all_values)
        row[f"final_live_{type_name}"] = all_values[-1]
    return row

def aggregate_daily_episode_metrics(
    daily_rows: Sequence[Dict[str, object]],
    request_records: Sequence[Dict[str, object]],
    instance_type_names: Sequence[str],
) -> Dict[str, object]:
    if not daily_rows:
        raise ValueError("daily_rows must not be empty")
    completed = [
        record for record in request_records if record["status"] == "completed"
    ]
    unfinished = [
        record for record in request_records if bool(record["is_censored"])
    ]
    latencies = np.asarray(
        [record["latency_s"] for record in completed], dtype=np.float64
    )
    ttfts = np.asarray(
        [record["ttft_s"] for record in completed], dtype=np.float64
    )
    queue_delays = np.asarray(
        [record["queue_delay_s"] for record in completed], dtype=np.float64
    )
    service_times = np.asarray(
        [record["service_time_s"] for record in completed], dtype=np.float64
    )

    def percentile(values: np.ndarray, value: float) -> float:
        return float(np.percentile(values, value)) if values.size else 0.0

    day_durations = np.asarray(
        [float(row["episode_duration_s"]) for row in daily_rows],
        dtype=np.float64,
    )
    total_duration_s = float(day_durations.sum())
    decision_steps = np.asarray(
        [int(row["decision_steps"]) for row in daily_rows],
        dtype=np.float64,
    )
    total_decision_steps = max(1.0, float(decision_steps.sum()))
    request_count = sum(int(row["requests"]) for row in daily_rows)
    scheduled_request_count = sum(
        int(row["scheduled_requests"]) for row in daily_rows
    )
    reset_backlog = sum(
        int(row["end_backlog_requests"]) for row in daily_rows
    )
    backlog_request_seconds = sum(
        float(row["backlog_request_seconds"]) for row in daily_rows
    )
    row: Dict[str, object] = {
        "trace_start_day": min(
            int(day["trace_start_day"]) for day in daily_rows
        ),
        "trace_end_day": max(
            int(day["trace_end_day"]) for day in daily_rows
        ),
        "episode_duration_s": total_duration_s,
        "daily_resets": len(daily_rows),
        "decision_steps": int(decision_steps.sum()),
        "scheduled_requests": scheduled_request_count,
        "requests": request_count,
        "completed": len(completed),
        "unfinished": len(unfinished),
        "censor_ratio": len(unfinished) / max(1, request_count),

        "end_backlog_requests": reset_backlog,
        "daily_reset_backlog_requests": reset_backlog,
        "final_day_end_backlog_requests": int(
            daily_rows[-1]["end_backlog_requests"]
        ),
        "mean_backlog_requests": (
            backlog_request_seconds / max(1e-9, total_duration_s)
        ),
        "max_backlog_requests": max(
            int(day["max_backlog_requests"]) for day in daily_rows
        ),
        "backlog_request_seconds": backlog_request_seconds,
        "completion_ratio": len(completed) / max(1, request_count),
        "throughput_rps": len(completed) / max(1e-9, total_duration_s),
        "mean_interval_throughput_rps": float(
            sum(
                float(day["mean_interval_throughput_rps"])
                * float(steps)
                for day, steps in zip(daily_rows, decision_steps)
            )
            / total_decision_steps
        ),
        "peak_interval_throughput_rps": max(
            float(day["peak_interval_throughput_rps"])
            for day in daily_rows
        ),
        "mean_latency_s": float(latencies.mean()) if latencies.size else 0.0,
        "mean_ttft_s": float(ttfts.mean()) if ttfts.size else 0.0,
        "p50_ttft_s": percentile(ttfts, 50),
        "p90_ttft_s": percentile(ttfts, 90),
        "p95_ttft_s": percentile(ttfts, 95),
        "p99_ttft_s": percentile(ttfts, 99),
        "mean_queue_delay_s": (
            float(queue_delays.mean()) if queue_delays.size else 0.0
        ),
        "p99_queue_delay_s": percentile(queue_delays, 99),
        "mean_service_time_s": (
            float(service_times.mean()) if service_times.size else 0.0
        ),
        "p99_service_time_s": percentile(service_times, 99),
        "p50_latency_s": percentile(latencies, 50),
        "p90_latency_s": percentile(latencies, 90),
        "p95_latency_s": percentile(latencies, 95),
        "p99_latency_s": percentile(latencies, 99),
        "total_cost_usd": sum(
            float(day["total_cost_usd"]) for day in daily_rows
        ),
        "episode_reward": sum(
            float(day["episode_reward"]) for day in daily_rows
        ),
        "switch_actions": sum(
            int(day["switch_actions"]) for day in daily_rows
        ),
    }
    for type_name in instance_type_names:
        row[f"mean_live_{type_name}"] = float(
            sum(
                float(day[f"mean_live_{type_name}"]) * float(steps)
                for day, steps in zip(daily_rows, decision_steps)
            )
            / total_decision_steps
        )
        row[f"max_live_{type_name}"] = max(
            int(day[f"max_live_{type_name}"]) for day in daily_rows
        )
        row[f"final_live_{type_name}"] = int(
            daily_rows[-1][f"final_live_{type_name}"]
        )
    return row

def run_greedy_evaluation(
    config: ExperimentConfig,
    agent: object,
    episodes: int,
    start_episode: int = 0,
    output_dir: Optional[Path] = None,
    write_detailed_logs: bool = False,
    write_request_log: bool = False,
    write_extended_step_log: bool = False,
    print_episode_rows: bool = False,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if start_episode < 0:
        raise ValueError("start_episode must be non-negative")
    if (
        write_detailed_logs or write_request_log or write_extended_step_log
    ) and output_dir is None:
        raise ValueError("output_dir is required when evaluation logs are enabled")
    if write_extended_step_log and not write_detailed_logs:
        raise ValueError(
            "write_extended_step_log requires write_detailed_logs=True"
        )

    env = AutoscalingEnv(config)
    if (agent.node_count, agent.feature_dim, agent.action_count) != (
        env.node_count,
        env.node_feature_dim,
        env.action_count,
    ):
        raise ValueError("evaluation config state/action shape differs from agent")

    resolved_output = (
        output_dir.expanduser().resolve() if output_dir is not None else None
    )
    if resolved_output is not None:
        resolved_output.mkdir(parents=True, exist_ok=True)
    step_log = (
        CsvStreamWriter(resolved_output / "evaluation_steps.csv")
        if write_detailed_logs and resolved_output is not None
        else None
    )
    group_log = (
        CsvStreamWriter(resolved_output / "evaluation_group.csv")
        if write_detailed_logs and resolved_output is not None
        else None
    )
    request_log = (
        CsvStreamWriter(resolved_output / "evaluation_requests.csv")
        if write_request_log and resolved_output is not None
        else None
    )
    episode_rows: List[Dict[str, object]] = []
    all_request_records: List[Dict[str, object]] = []
    group_row: Dict[str, object] = {}
    action_counts = {label: 0 for label in env.action_labels()}
    online_was_training = agent.online.training
    target_was_training = agent.target.training
    agent.online.eval()
    agent.target.eval()

    try:
        for evaluation_episode in range(episodes):
            trace_episode = start_episode + evaluation_episode
            observation, reset_info = env.reset(trace_episode)
            agent.begin_episode(
                observation["node_features"],
                observation["action_mask"],
                training=False,
                node_mask=observation["node_mask"],
            )
            done = False
            step_index = 0
            switch_actions = 0
            step_rows: List[Dict[str, object]] = []
            while not done:
                action = agent.act(
                    observation["action_mask"], explore=False
                )
                next_observation, reward, terminated, truncated, info = env.step(
                    action
                )
                done = bool(terminated or truncated)
                agent.observe(
                    next_observation["node_features"],
                    action,
                    reward,
                    done,
                    action_mask=next_observation["action_mask"],
                    node_mask=next_observation["node_mask"],
                )
                action_label = env.action_labels()[action]
                action_counts[action_label] += 1
                if bool(info["action_applied"]) and action != 0:
                    switch_actions += 1
                step_row = {
                    "evaluation_day": evaluation_episode + 1,
                    "evaluation_episode": evaluation_episode,
                    "trace_episode": trace_episode,
                    "step": step_index,
                    "epsilon": 0.0,
                    "exploration": False,
                    "parameter_update": False,
                    "reward": reward,
                    **info,
                }
                step_rows.append(step_row)
                if step_log is not None:
                    log_row = {
                        "evaluation_day": evaluation_episode + 1,
                        "step": step_index,
                        "start_time_s": info["start_time_s"],
                        "end_time_s": info["end_time_s"],
                        **{
                            key: info[key] for key in (
                                "decision_interval_s", "next_decision_interval_s",
                                "next_decision_time_s",
                            ) if config.adaptive_interval.enabled
                        },
                        "throughput_rps": info["throughput_rps"],
                        "pending_requests": info["pending_requests"],
                        "p99_ttft_s": info["p99_ttft_s"],
                        **{
                            f"live_{type_name}": info[
                                f"live_{type_name}"
                            ]
                            for type_name in config.instance_types
                        },
                    }
                    if write_extended_step_log:
                        log_row.update(
                            {
                                "arrivals": info["arrivals"],
                                "completed_requests": info["completions"],
                                "mean_ttft_s": info["mean_ttft_s"],
                                "p99_pending_ttft_age_s": info[
                                    "p99_pending_ttft_age_s"
                                ],
                                "interval_cost_usd": info["step_cost_usd"],
                                "cumulative_cost_usd": info[
                                    "total_cost_usd"
                                ],
                                "action": info["action"],
                                "action_label": info["action_label"],
                                "action_applied": info["action_applied"],
                                **{
                                    f"ready_{type_name}": info[
                                        f"ready_{type_name}"
                                    ]
                                    for type_name in config.instance_types
                                },
                                **{
                                    f"cold_{type_name}": info[
                                        f"cold_{type_name}"
                                    ]
                                    for type_name in config.instance_types
                                },
                                **{
                                    f"draining_{type_name}": info[
                                        f"draining_{type_name}"
                                    ]
                                    for type_name in config.instance_types
                                },
                            }
                        )
                    step_log.write(log_row)
                observation = next_observation
                step_index += 1

            settlement = env.finalize_episode()
            all_request_records.extend(settlement["request_records"])
            if request_log is not None:
                for record in settlement["request_records"]:
                    request_log.write(
                        {
                            "evaluation_day": evaluation_episode + 1,
                            "trace_episode": trace_episode,
                            **record,
                        }
                    )
                request_log.flush()
            row = episode_metrics(
                evaluation_episode,
                trace_episode,
                reset_info,
                settlement,
                float(settlement["elapsed_time_s"]),
                switch_actions,
                step_rows,
                config.initial_instances,
                list(config.instance_types),
            )
            episode_rows.append(row)
            if print_episode_rows:
                import json

                print(json.dumps(row, ensure_ascii=False))
            for writer in (step_log, request_log):
                if writer is not None:
                    writer.flush()
        group_row = {
            "evaluation_group": 0,
            "evaluation_days": episodes,
            **aggregate_daily_episode_metrics(
                episode_rows,
                all_request_records,
                list(config.instance_types),
            ),
        }
        if group_log is not None:
            group_log.write(
                {
                    "episode_reward": group_row["episode_reward"],
                    "total_cost_usd": group_row["total_cost_usd"],
                    "p99_ttft_s": group_row["p99_ttft_s"],
                }
            )
            group_log.flush()
    finally:
        if online_was_training:
            agent.eval_off()
        else:
            agent.eval_on()
        agent.target.train(target_was_training)
        for writer in (step_log, group_log, request_log):
            if writer is not None:
                writer.close()

    summary: Dict[str, object] = {
        "seed": config.seed,
        "episodes": episodes,
        "start_episode": start_episode,
        "trace_episodes": [start_episode + index for index in range(episodes)],
        "exploration": False,
        "epsilon": 0.0,
        "parameter_updates": 0,
        "state_shape": [env.node_count, env.node_feature_dim],
        "instance_slot_counts": list(env.instance_slot_counts),
        "node_mask_side_channel": True,
        "action_count": env.action_count,
        "action_masking": agent.action_masking,
        "action_counts": action_counts,
        "evaluation_days": episodes,
        "terminated_early": any(
            bool(row["terminated_early"]) for row in episode_rows
        ),
        "termination_reasons": [
            str(row["termination_reason"]) for row in episode_rows
        ],
        "episode_reward": group_row["episode_reward"],
        "scheduled_requests": group_row["scheduled_requests"],
        "observed_requests": group_row["requests"],
        "completed_requests": group_row["completed"],
        "unfinished_requests": group_row["unfinished"],
        "completion_ratio": group_row["completion_ratio"],
        "total_cost_usd": group_row["total_cost_usd"],
        "mean_ttft_s": group_row["mean_ttft_s"],
        "p50_ttft_s": group_row["p50_ttft_s"],
        "p95_ttft_s": group_row["p95_ttft_s"],
        "p99_ttft_s": group_row["p99_ttft_s"],
        "end_backlog_requests": group_row["end_backlog_requests"],
        "trace_start_day": group_row["trace_start_day"],
        "trace_end_day": group_row["trace_end_day"],
        "episode_duration_s": group_row["episode_duration_s"],
        "output_dir": str(resolved_output) if resolved_output is not None else None,
    }
    if write_detailed_logs and resolved_output is not None:
        write_json(
            resolved_output / "evaluation_config.json",
            {
                "config": json_ready(asdict(config)),
                "output_dir": str(resolved_output),
                "episodes": episodes,
                "daily_episode_duration_s": config.duration_s,
                "group_duration_s": group_row["episode_duration_s"],
                "start_episode": start_episode,
                "exploration": False,
                "epsilon": 0.0,
                "parameter_updates": False,
            },
        )
        write_json(resolved_output / "evaluation_summary.json", summary)
    return summary, episode_rows
