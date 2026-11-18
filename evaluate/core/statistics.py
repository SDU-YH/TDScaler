from dataclasses import replace
import json
from pathlib import Path
import statistics

from environment.config import load_config
from workload.workload import WorkloadGenerator
from evaluate.core.evaluate import evaluate

ROOT = Path(__file__).resolve().parents[2]


def evaluate_group(job):
    group, entry, requested_days, device = job
    config = load_config(ROOT / entry["config"])
    config = replace(config, training=replace(config.training, torch_num_threads=1))
    trace = WorkloadGenerator(config.workload, config.active_model, config.seed)
    available = sorted({int(r.timestamp_s // 86400) + 1 for r in trace.trace})
    days = available if requested_days is None else requested_days
    if not days or not set(days).issubset(available):
        raise ValueError(f"{group}: requested days must belong to {available}")
    references = json.loads((ROOT / "evaluate/nce_references.json").read_text())[group]
    for day in days:
        if str(day) not in references:
            raise ValueError(f"{group} day {day}: missing common NCE reference")
    daily_cost, daily_sla, daily_nce = [], [], []
    for day in days:
        result = evaluate(config, ROOT / entry["checkpoint"], day - 1, device)
        ref = references[str(day)]
        denominator = 0.5 * (result["cost"] / ref["cost_reference"] +
                             result["decision_count"] / ref["decision_reference"])
        daily_cost.append(result["cost"])
        daily_sla.append(100 * result["sla_violation_rate"])
        daily_nce.append((1 - result["sla_violation_rate"]) / denominator)
    return {
        "group": group,
        "cost": statistics.mean(daily_cost),
        "sla_violation_percent": statistics.mean(daily_sla),
        "normalized_nce": statistics.mean(daily_nce),
    }
