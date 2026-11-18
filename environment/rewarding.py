from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from environment.config import ExperimentConfig

def calculate_step_reward(
    config: ExperimentConfig, *, p99_ttft_s: float, has_ttft_signal: bool,
    step_cost_usd: float, step_duration_s: float,
    max_pending_ttft_age_s: float | None, remaining_duration_s: float,
    natural_episode_end: bool,
) -> Tuple[float, Dict[str, object]]:

    rc = config.reward
    if max_pending_ttft_age_s is None:
        raise ValueError("sla_cost requires max_pending_ttft_age_s; rebuild old replay cache")
    values = (step_cost_usd, step_duration_s, p99_ttft_s,
              max_pending_ttft_age_s, remaining_duration_s)
    if any(not np.isfinite(v) or v < 0 for v in values) or step_duration_s <= 0:
        raise ValueError("sla_cost needs finite nonnegative metrics and positive duration")
    duration_norm = step_duration_s / rc.cost_sla_reference_interval_s
    cost_norm = step_cost_usd / (
        rc.cost_sla_reference_hourly_cost * rc.cost_sla_reference_interval_s / 3600.0
    )
    signal = max(p99_ttft_s if has_ttft_signal else 0.0, max_pending_ttft_age_s)
    violated = signal >= rc.sla_s
    excess = np.log1p(max(0.0, signal / rc.sla_s - 1.0))
    cost_penalty = rc.cost_sla_cost_weight * cost_norm
    sla_penalty = duration_norm * (
        rc.cost_sla_violation_weight * float(violated)
        + rc.cost_sla_excess_weight * excess
    )
                                                                       
    critical = bool(has_ttft_signal and p99_ttft_s > rc.critical_ttft_s)
    early = critical and not natural_episode_end
    terminal_penalty = (
        rc.early_termination_penalty
        + rc.cost_sla_terminal_weight * remaining_duration_s / rc.cost_sla_reference_interval_s
    ) if early else 0.0
    return -float(cost_penalty + sla_penalty + terminal_penalty), {
        "reward_mode": "sla_cost",
        "reward_ttft_signal_s": signal,
        "reward_ttft_observed": bool(has_ttft_signal),
        "reward_max_pending_ttft_age_s": float(max_pending_ttft_age_s),
        "reward_sla_violated": bool(violated),
        "reward_critical_ttft_terminated": critical,
        "reward_terminated_early": bool(early),
        "reward_reason": "sla_violation" if violated else "actual_cost",
        "reward_base": -float(cost_penalty + sla_penalty),
        "reward_cost_penalty": float(cost_penalty),
        "reward_sla_penalty": float(sla_penalty),
        "reward_duration_norm": duration_norm,
        "reward_cost_norm": cost_norm,
        "reward_early_termination_penalty": float(terminal_penalty),
    }
