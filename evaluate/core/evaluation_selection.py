import math

def update_evaluation_control(agent, summary, episode, sla_s, config):
    state = getattr(agent, "evaluation_control", {})
    p99 = float(summary["p99_ttft_s"] or 0.0)
    observed = int(summary.get("observed_requests", 0))
    unfinished = int(summary.get("unfinished_requests", 0))
    complete = not summary.get("terminated_early", False) and observed > 0 and unfinished == 0
    if not math.isfinite(p99):
        p99 = float("inf")
    feasible = complete and p99 < sla_s
                                                                               
    key = ([0, float(summary["total_cost_usd"]), p99] if feasible else
           [1, int(not complete), unfinished / max(1, observed), p99,
            float(summary["total_cost_usd"])])
    improved = "best_key" not in state or key < state["best_key"]
    if improved:
        state.update(best_key=key, best_episode=episode, best_summary=dict(summary))
    reward = float(summary["episode_reward"])
    if math.isfinite(reward) and reward > state.get("peak_reward", -math.inf) + config.lr_plateau_min_delta:
        state.update(peak_reward=reward, bad_evaluations=0)
    else:
        state["bad_evaluations"] = state.get("bad_evaluations", 0) + 1
    reduced = False
    if config.lr_plateau_enabled and state["bad_evaluations"] >= config.lr_plateau_patience:
        for group in agent.optimizer.param_groups:
            old = group["lr"]
            group["lr"] = min(old, max(config.lr_plateau_min_lr, old * config.lr_plateau_factor))
            reduced = reduced or group["lr"] < old
        state["bad_evaluations"] = 0
    state["learning_rates"] = [group["lr"] for group in agent.optimizer.param_groups]
    agent.evaluation_control = state
    return improved, reduced
