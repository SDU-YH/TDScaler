from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from algorithm.agent import MaskedDTQNAgent
from environment.config import ExperimentConfig, set_global_seed
from environment.env import AutoscalingEnv
from evaluate.core.evaluation_core import aggregate_daily_episode_metrics, daily_reset_episode_config, episode_metrics, run_greedy_evaluation
from environment.metrics import CsvStreamWriter, write_json
from environment.reporting import json_ready as _json_ready
from algorithm.replay_cache import TRAJECTORY_EXPERT, TRAJECTORY_PERTURBED_EXPERT, TRAJECTORY_RANDOM, build_replay_signature, ensure_relabel_storage, load_replay_cache, record_relabel_step, save_replay_cache, set_replay_episode_kind

def _load_expert_action_labels(
    path: Path, expected_steps: int
) -> tuple[str, ...]:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        action_field = (
            "action"
            if "action" in fields
            else "planned_action"
            if "planned_action" in fields
            else None
        )
        if "step" not in fields or action_field is None:
            raise ValueError(
                "expert trajectory must contain step and action/planned_action columns"
            )
        rows = list(reader)
    if len(rows) != int(expected_steps):
        raise ValueError(
            f"expert trajectory has {len(rows)} steps; expected {expected_steps}"
        )
    actions: List[str] = []
    for expected_step, row in enumerate(rows):
        try:
            actual_step = int(row["step"])
        except (TypeError, ValueError) as error:
            raise ValueError("expert trajectory contains a non-integer step") from error
        if actual_step != expected_step:
            raise ValueError(
                "expert trajectory steps must be consecutive from zero: "
                f"expected {expected_step}, got {actual_step}"
            )
        label = str(row[action_field]).strip()
        if not label:
            raise ValueError(
                f"expert trajectory action is empty at step {expected_step}"
            )
        actions.append(label)
    return tuple(actions)

def _prefill_episode_targets(
    minimum_steps: int,
    steps_per_episode: int,
    batch_size: int,
    expert_fraction: float,
) -> tuple[int, int]:
    required = max(
        int(math.ceil(minimum_steps / max(1, steps_per_episode))),
        int(batch_size) + 1,
    )
    if expert_fraction <= 0.0:
        return required, 0
    if expert_fraction >= 1.0:
        return required, required
    ratio = Fraction(str(float(expert_fraction))).limit_denominator(1000)
    total = int(math.ceil(required / ratio.denominator) * ratio.denominator)
    expert = total * ratio.numerator // ratio.denominator
    return total, expert

def _timed_expert_action(
    actions: Sequence[str], cursor: int, now_s: float, source_interval_s: float
) -> tuple[str, int]:

    while cursor < len(actions) and cursor * source_interval_s <= now_s + 1e-9:
        label = actions[cursor]
        cursor += 1
        if label != "noop":
            return label, cursor
    return "noop", cursor

def _completed_replay_episode(replay_buffer) -> Dict[str, np.ndarray]:
    ensure_relabel_storage(replay_buffer)
    if int(replay_buffer.pos[0]) <= 0 or int(replay_buffer.pos[1]) != 0:
        raise RuntimeError("no completed replay episode is available to copy")
    source = (int(replay_buffer.pos[0]) - 1) % int(replay_buffer.max_size)
    return {
        "obss": replay_buffer.obss[source].copy(),
        "actions": replay_buffer.actions[source].copy(),
        "rewards": replay_buffer.rewards[source].copy(),
        "dones": replay_buffer.dones[source].copy(),
        "episode_length": np.asarray(
            replay_buffer.episode_lengths[source]
        ).copy(),
        "reward_step_cost_usd": replay_buffer.reward_step_cost_usd[source].copy(),
        "reward_step_duration_s": replay_buffer.reward_step_duration_s[source].copy(),
        "reward_p99_ttft_s": replay_buffer.reward_p99_ttft_s[source].copy(),
        "reward_has_ttft_signal": replay_buffer.reward_has_ttft_signal[source].copy(),
        "reward_action_applied": replay_buffer.reward_action_applied[source].copy(),
        "reward_interval_load": replay_buffer.reward_interval_load[source].copy(),
        "reward_max_pending_ttft_age_s": replay_buffer.reward_max_pending_ttft_age_s[source].copy(),
        "trajectory_kind": np.asarray(
            replay_buffer.trajectory_kinds[source]
        ).copy(),
        "perturbation_count": np.asarray(
            replay_buffer.perturbation_counts[source]
        ).copy(),
    }

def _append_replay_episode_copy(
    replay_buffer, episode: Dict[str, np.ndarray]
) -> None:
    if int(replay_buffer.pos[1]) != 0:
        raise RuntimeError("cannot append an episode while replay is partial")
    ensure_relabel_storage(replay_buffer)
    target = int(replay_buffer.pos[0]) % int(replay_buffer.max_size)
    replay_buffer.cleanse_episode(target)
    replay_buffer.reward_step_cost_usd[target].fill(0.0)
    replay_buffer.reward_step_duration_s[target].fill(0.0)
    replay_buffer.reward_p99_ttft_s[target].fill(0.0)
    replay_buffer.reward_has_ttft_signal[target].fill(False)
    replay_buffer.reward_action_applied[target].fill(False)
    replay_buffer.reward_interval_load[target].fill(0.0)
    replay_buffer.reward_max_pending_ttft_age_s[target].fill(np.nan)
    replay_buffer.obss[target] = episode["obss"]
    replay_buffer.actions[target] = episode["actions"]
    replay_buffer.rewards[target] = episode["rewards"]
    replay_buffer.dones[target] = episode["dones"]
    replay_buffer.episode_lengths[target] = episode["episode_length"]
    replay_buffer.reward_step_cost_usd[target] = episode["reward_step_cost_usd"]
    replay_buffer.reward_step_duration_s[target] = episode["reward_step_duration_s"]
    replay_buffer.reward_p99_ttft_s[target] = episode["reward_p99_ttft_s"]
    replay_buffer.reward_has_ttft_signal[target] = episode["reward_has_ttft_signal"]
    replay_buffer.reward_action_applied[target] = episode["reward_action_applied"]
    replay_buffer.reward_interval_load[target] = episode["reward_interval_load"]
    replay_buffer.reward_max_pending_ttft_age_s[target] = episode["reward_max_pending_ttft_age_s"]
    replay_buffer.trajectory_kinds[target] = episode["trajectory_kind"]
    replay_buffer.perturbation_counts[target] = episode["perturbation_count"]
    replay_buffer.flush()

def prepopulate_replay(
    env: AutoscalingEnv,
    agent: Any,
    minimum_steps: int,
    seed: int,
    expert_fraction: float = 0.0,
    expert_trajectory_path: Optional[Path] = None,
    expert_episodes: int = 0,
    perturbed_expert_episodes: int = 0,
    random_episodes: int = 0,
    perturbation_decisions: int = 0,
) -> Dict[str, object]:

    requested_steps = int(minimum_steps)
    if requested_steps < 0:
        raise ValueError("minimum_steps must be non-negative")
    requested_expert_fraction = float(expert_fraction)
    if not 0.0 <= requested_expert_fraction <= 1.0:
        raise ValueError("expert_fraction must be in [0, 1]")
    target_explicit_experts = int(expert_episodes)
    target_perturbed_experts = int(perturbed_expert_episodes)
    target_explicit_random = int(random_episodes)
    requested_perturbations = int(perturbation_decisions)
    if min(
        target_explicit_experts,
        target_perturbed_experts,
        target_explicit_random,
        requested_perturbations,
    ) < 0:
        raise ValueError("explicit expert and perturbation counts must be non-negative")
    explicit_mode = (
        target_explicit_experts
        + target_perturbed_experts
        + target_explicit_random
    ) > 0
    if explicit_mode and requested_expert_fraction > 0.0:
        raise ValueError(
            "explicit expert episode counts cannot be combined with expert_fraction"
        )
    if target_perturbed_experts > 0 and requested_perturbations <= 0:
        raise ValueError(
            "perturbation_decisions must be positive for perturbed expert episodes"
        )

    result: Dict[str, object] = {
        "requested_steps": requested_steps,
        "actual_steps": 0,
        "episodes": 0,
        "sampleable": agent.replay_buffer.can_sample(agent.batch_size),
        "expert_fraction_requested": requested_expert_fraction,
        "expert_fraction_actual": 0.0,
        "expert_episodes": 0,
        "perturbed_expert_episodes": 0,
        "random_episodes": 0,
        "expert_transitions": 0,
        "perturbed_expert_transitions": 0,
        "random_transitions": 0,
        "expert_mask_fallbacks": 0,
        "perturbation_decisions_requested": requested_perturbations,
        "perturbation_actions_actual": 0,
        "expert_trajectory_path": (
            str(expert_trajectory_path.expanduser().resolve())
            if expert_trajectory_path is not None
            else None
        ),
    }
    if requested_steps == 0:
        return result

    steps_per_episode = env.config.min_decision_steps
    source_interval_s = (
        env.config.training.replay_prefill_expert_interval_s
        or env.config.scaling_interval_s
    )
    reference_steps = int(math.ceil(env.config.duration_s / source_interval_s))
    if explicit_mode:
        target_episodes = (
            target_explicit_experts
            + target_perturbed_experts
            + target_explicit_random
        )
        target_expert_episodes = target_explicit_experts
        if target_episodes * steps_per_episode < requested_steps:
            raise ValueError(
                f"explicit prefill contains {target_episodes * steps_per_episode} "
                f"transitions, fewer than requested {requested_steps}"
            )
    else:
        target_episodes, target_expert_episodes = _prefill_episode_targets(
            requested_steps,
            steps_per_episode,
            agent.batch_size,
            requested_expert_fraction,
        )
    if target_episodes > int(agent.replay_buffer.max_size):
        raise ValueError(
            f"prefill requires {target_episodes} complete episodes but replay "
            f"capacity holds only {int(agent.replay_buffer.max_size)}"
        )
    expert_actions: Sequence[str] = ()
    if target_expert_episodes or target_perturbed_experts:
        if expert_trajectory_path is None:
            raise ValueError(
                "expert_trajectory_path is required when expert_fraction is positive"
            )
        expert_actions = _load_expert_action_labels(
            expert_trajectory_path, reference_steps
        )

    action_rng = np.random.default_rng(seed)
    if explicit_mode:
        trajectory_kinds = (
            [TRAJECTORY_EXPERT] * target_explicit_experts
            + [TRAJECTORY_PERTURBED_EXPERT] * target_perturbed_experts
            + [TRAJECTORY_RANDOM] * target_explicit_random
        )
    else:
        mix_rng = np.random.default_rng(int(seed) ^ 0x5EED50)
        expert_flags = np.asarray(
            [True] * target_expert_episodes
            + [False] * (target_episodes - target_expert_episodes),
            dtype=bool,
        )
        mix_rng.shuffle(expert_flags)
        trajectory_kinds = [
            TRAJECTORY_EXPERT if flag else TRAJECTORY_RANDOM
            for flag in expert_flags.tolist()
        ]
    steps_before = agent.steps
    updates_before = agent.updates
    epsilon_before = agent.epsilon
    actual_steps = 0
    episodes = 0
    expert_episodes = 0
    perturbed_episodes = 0
    random_episodes = 0
    expert_transitions = 0
    perturbed_transitions = 0
    random_transitions = 0
    expert_mask_fallbacks = 0
    perturbation_actions_actual = 0
    expert_template: Optional[Dict[str, np.ndarray]] = None
    expert_template_fallbacks = 0

    for trajectory_kind in trajectory_kinds:
        is_expert = trajectory_kind == TRAJECTORY_EXPERT
        is_perturbed = trajectory_kind == TRAJECTORY_PERTURBED_EXPERT
        if is_expert and expert_template is not None:
            _append_replay_episode_copy(
                agent.replay_buffer, expert_template
            )
            episodes += 1
            expert_episodes += 1
            template_steps = int(expert_template["episode_length"].item())
            expert_transitions += template_steps
            expert_mask_fallbacks += expert_template_fallbacks
            actual_steps += template_steps
            if episodes % 10 == 0:
                print(
                    json.dumps(
                        {
                            "phase": "replay_prefill",
                            "episodes": episodes,
                            "steps": actual_steps,
                            "requested_steps": requested_steps,
                            "expert_episodes": expert_episodes,
                            "perturbed_expert_episodes": perturbed_episodes,
                            "random_episodes": random_episodes,
                        },
                        ensure_ascii=False,
                    )
                )
            continue

        environment_episode = (
            0 if (is_expert or is_perturbed) else random_episodes + 1
        )
        observation, _ = env.reset(environment_episode)
        agent.begin_episode(
            observation["node_features"],
            observation["action_mask"],
            training=True,
            node_mask=observation["node_mask"],
        )
        done = False
        episode_steps = 0
        expert_cursor = 0
        episode_mask_fallbacks = 0
        episode_perturbations = 0
        perturbation_steps: set[int] = set()
        perturbation_rng: Optional[np.random.Generator] = None
        if is_perturbed:
            if requested_perturbations > steps_per_episode:
                raise ValueError(
                    "perturbation_decisions cannot exceed decisions per episode"
                )
            perturbation_rng = np.random.default_rng(
                int(seed) + 10_000 + perturbed_episodes
            )
            perturbation_steps = set(
                int(value)
                for value in perturbation_rng.choice(
                    steps_per_episode,
                    size=requested_perturbations,
                    replace=False,
                ).tolist()
            )
        label_to_action = {
            label: index for index, label in enumerate(env.action_labels())
        }
        while not done:
            valid_actions = np.flatnonzero(observation["action_mask"])
            if valid_actions.size == 0:
                raise RuntimeError("prefill observation has no valid action")
            if is_expert or is_perturbed:
                if (env.config.adaptive_interval.enabled
                        or env.config.training.replay_prefill_expert_interval_s is not None):
                    label, expert_cursor = _timed_expert_action(
                        expert_actions, expert_cursor, env.backend.current_time,
                        source_interval_s,
                    )
                else:
                    label = expert_actions[episode_steps]
                if label not in label_to_action:
                    raise ValueError(
                        f"expert action {label!r} is not in {env.action_labels()}"
                    )
                desired_action = int(label_to_action[label])
                if bool(observation["action_mask"][desired_action]):
                    reference_action = desired_action
                else:
                    reference_action = int(label_to_action["noop"])
                    expert_mask_fallbacks += 1
                    episode_mask_fallbacks += 1
                action = reference_action
                if is_perturbed and episode_steps in perturbation_steps:
                    alternatives = valid_actions[valid_actions != reference_action]
                    if alternatives.size == 0:
                        raise RuntimeError(
                            "cannot perturb an expert decision: no alternative valid action"
                        )
                    assert perturbation_rng is not None
                    action = int(
                        alternatives[
                            perturbation_rng.integers(alternatives.size)
                        ]
                    )
                    episode_perturbations += 1
            else:
                action = int(
                    valid_actions[action_rng.integers(valid_actions.size)]
                )
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            record_relabel_step(
                agent.replay_buffer,
                step_cost_usd=float(info["step_cost_usd"]),
                step_duration_s=(
                    float(info["end_time_s"]) - float(info["start_time_s"])
                ),
                p99_ttft_s=float(info["p99_ttft_s"]),
                has_ttft_signal=bool(float(info["p99_ttft_s"]) > 0.0),
                action_applied=bool(info["action_applied"]),
                interval_load=float(info["arrivals"]),
                max_pending_ttft_age_s=env.backend.pending_ttft_age_percentile(100),
            )
            agent.observe(
                next_observation["node_features"],
                action,
                reward,
                done,
                action_mask=next_observation["action_mask"],
                node_mask=next_observation["node_mask"],
                anneal_epsilon=False,
            )
            observation = next_observation
            actual_steps += 1
            episode_steps += 1

        agent.end_episode(training=True)
        set_replay_episode_kind(
            agent.replay_buffer,
            trajectory_kind,
            perturbation_count=episode_perturbations,
        )
        if is_expert:
            expert_template = _completed_replay_episode(
                agent.replay_buffer
            )
            expert_template_fallbacks = episode_mask_fallbacks
        env.finalize_episode()
        episodes += 1
        if is_expert:
            expert_episodes += 1
            expert_transitions += episode_steps
        elif is_perturbed:
            perturbed_episodes += 1
            perturbed_transitions += episode_steps
            perturbation_actions_actual += episode_perturbations
        else:
            random_episodes += 1
            random_transitions += episode_steps
        if episodes % 10 == 0 or is_perturbed:
            print(
                json.dumps(
                    {
                        "phase": "replay_prefill",
                        "episodes": episodes,
                        "steps": actual_steps,
                        "requested_steps": requested_steps,
                        "expert_episodes": expert_episodes,
                        "perturbed_expert_episodes": perturbed_episodes,
                        "random_episodes": random_episodes,
                    },
                    ensure_ascii=False,
                )
            )

    if agent.steps != steps_before or agent.updates != updates_before:
        raise RuntimeError("replay prefill changed training counters")
    if not np.isclose(agent.epsilon, epsilon_before):
        raise RuntimeError("replay prefill changed epsilon")
    if actual_steps < requested_steps:
        raise RuntimeError("replay prefill stopped before requested steps")
    if (
        not explicit_mode
        and not agent.replay_buffer.can_sample(agent.batch_size)
    ):
        raise RuntimeError("replay prefill is not sampleable after completion")

    result.update(
        {
            "actual_steps": actual_steps,
            "episodes": episodes,
            "sampleable": agent.replay_buffer.can_sample(agent.batch_size),
            "expert_fraction_actual": expert_episodes / max(1, episodes),
            "expert_episodes": expert_episodes,
            "perturbed_expert_episodes": perturbed_episodes,
            "random_episodes": random_episodes,
            "expert_transitions": expert_transitions,
            "perturbed_expert_transitions": perturbed_transitions,
            "random_transitions": random_transitions,
            "expert_mask_fallbacks": expert_mask_fallbacks,
            "perturbation_actions_actual": perturbation_actions_actual,
        }
    )
    return result

def train(
    config: ExperimentConfig,
    device: str = "cpu",
    resume_checkpoint: Optional[Path] = None,
    rebuild_replay_prefill_cache: bool = False,
    checkpoint_dir: Optional[Path] = None,
) -> Dict[str, object]:
    if checkpoint_dir is None:
        from tempfile import mkdtemp
        checkpoint_root = Path(__file__).resolve().parents[1] / "evaluate/checkpoints/training"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = Path(mkdtemp(prefix=f"{config.output_dir.name}_", dir=checkpoint_root))
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_json(checkpoint_dir / "resolved_config.json", _json_ready(asdict(config)))
    torch.set_num_threads(config.training.torch_num_threads)
    set_global_seed(config.seed)
    training_day_duration_s = (
        config.duration_s / config.training.evaluation_days
    )
    training_day_config = daily_reset_episode_config(
        config, day_duration_s=training_day_duration_s
    )
    env = AutoscalingEnv(training_day_config)
    max_episode_steps = training_day_config.max_decision_steps
    agent = MaskedDTQNAgent(
        node_count=env.node_count,
        feature_dim=env.node_feature_dim,
        action_count=env.action_count,
        config=config.training,
        seed=config.seed,
        instance_slot_counts=env.instance_slot_counts,
        device=device,
        max_episode_steps=max_episode_steps,
    )
    if resume_checkpoint is not None:
        resume_checkpoint = resume_checkpoint.expanduser().resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(
                f"resume checkpoint does not exist: {resume_checkpoint}"
            )
        agent.load(resume_checkpoint, load_optimizer=True)

    cache_path = config.training.replay_prefill_cache_path
    replay_signature = (
        build_replay_signature(
            training_day_config,
            node_count=env.node_count,
            feature_dim=env.node_feature_dim,
            action_count=env.action_count,
            instance_slot_counts=env.instance_slot_counts,
            combined_observation_dim=agent.combined_observation_dim,
            max_episode_steps=max_episode_steps,
        )
        if cache_path is not None and config.training.replay_prefill_steps > 0
        else None
    )
    print(
        json.dumps(
            {
                "phase": "replay_prefill_start",
                "requested_steps": config.training.replay_prefill_steps,
                "cache_path": str(cache_path) if cache_path is not None else None,
                "expert_fraction": (
                    config.training.replay_prefill_expert_fraction
                ),
                "expert_episodes": (
                    config.training.replay_prefill_expert_episodes
                ),
                "perturbed_expert_episodes": (
                    config.training.replay_prefill_perturbed_expert_episodes
                ),
                "random_episodes": (
                    config.training.replay_prefill_random_episodes
                ),
                "perturbation_decisions": (
                    config.training.replay_prefill_perturbation_decisions
                ),
                "expert_trajectory_path": (
                    str(config.training.replay_prefill_expert_trajectory_path)
                    if config.training.replay_prefill_expert_trajectory_path
                    is not None
                    else None
                ),
            },
            ensure_ascii=False,
        )
    )
    use_existing_cache = bool(
        cache_path is not None
        and cache_path.is_file()
        and not rebuild_replay_prefill_cache
        and config.training.replay_prefill_steps > 0
    )
    if use_existing_cache:
        replay_prefill = load_replay_cache(
            cache_path,
            agent.replay_buffer,
            replay_signature,
            config=training_day_config,
            minimum_steps=config.training.replay_prefill_steps,
            batch_size=agent.batch_size,
        )
        print(
            json.dumps(
                {"phase": "replay_prefill_cache_loaded", **replay_prefill},
                ensure_ascii=False,
            )
        )
    else:
        replay_prefill = prepopulate_replay(
            env,
            agent,
            minimum_steps=config.training.replay_prefill_steps,
            seed=config.seed,
            expert_fraction=(
                config.training.replay_prefill_expert_fraction
            ),
            expert_trajectory_path=(
                config.training.replay_prefill_expert_trajectory_path
            ),
            expert_episodes=(
                config.training.replay_prefill_expert_episodes
            ),
            perturbed_expert_episodes=(
                config.training.replay_prefill_perturbed_expert_episodes
            ),
            random_episodes=(
                config.training.replay_prefill_random_episodes
            ),
            perturbation_decisions=(
                config.training.replay_prefill_perturbation_decisions
            ),
        )
        replay_prefill["source"] = "generated"
        if cache_path is not None and config.training.replay_prefill_steps > 0:
            cache_details = save_replay_cache(
                cache_path,
                agent.replay_buffer,
                replay_signature,
                replay_prefill,
            )
            replay_prefill.update(cache_details)
            print(
                json.dumps(
                    {"phase": "replay_prefill_cache_saved", **cache_details},
                    ensure_ascii=False,
                )
            )
    print(
        json.dumps(
            {"phase": "replay_prefill_complete", **replay_prefill},
            ensure_ascii=False,
        )
    )
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_log = CsvStreamWriter(output_dir / "train.csv")
    episode_rows: List[Dict[str, object]] = []
    periodic_evaluation_rows: List[Dict[str, object]] = []
    evaluation_config = daily_reset_episode_config(config)
    checkpoint_stem = "masked_han_dtqn"

    for episode in range(config.training.episodes):
        step_index = 0
        losses: List[float] = []
        daily_rows: List[Dict[str, object]] = []
        episode_requests: List[Dict[str, object]] = []
        for day_index in range(config.training.evaluation_days):
            environment_episode = (
                episode * config.training.evaluation_days + day_index
            )
            observation, reset_info = env.reset(environment_episode)
            agent.begin_episode(
                observation["node_features"],
                observation["action_mask"],
                training=True,
                node_mask=observation["node_mask"],
            )
            done = False
            day_step_index = 0
            switch_actions = 0
            day_step_rows: List[Dict[str, object]] = []
            while not done:
                epsilon = agent.epsilon
                action = agent.act(
                    observation["action_mask"], explore=True
                )
                next_observation, reward, terminated, truncated, info = (
                    env.step(action)
                )
                done = bool(terminated or truncated)
                state_interval_s = float(
                    observation["node_features"][0, 0]
                )
                agent.observe(
                    next_observation["node_features"],
                    action,
                    reward,
                    done,
                    action_mask=next_observation["action_mask"],
                    node_mask=next_observation["node_mask"],
                )
                step_losses: List[float] = []
                if agent.steps % config.training.learning_interval_steps == 0:
                    for _ in range(
                        config.training.gradient_updates_per_learning_step
                    ):
                        update_loss = agent.learn()
                        if update_loss is not None:
                            step_losses.append(update_loss)
                            losses.append(update_loss)
                loss = float(np.mean(step_losses)) if step_losses else None
                row = {
                    **info,
                    "episode": episode,
                    "day": day_index + 1,
                    "environment_episode": environment_episode,
                    "step": step_index,
                    "day_step": day_step_index,
                    "epsilon": epsilon,
                    "reward": reward,
                    "loss": loss,
                    "state_interval_s": state_interval_s,
                    "state_interval_change": float(observation["node_features"][0, 1]),
                }
                day_step_rows.append(row)
                if bool(info["action_applied"]) and action != 0:
                    switch_actions += 1
                observation = next_observation
                step_index += 1
                day_step_index += 1

            agent.end_episode(training=True)
            settlement = env.finalize_episode()
            day_records = settlement["request_records"]
            episode_requests.extend(day_records)
            daily_row = episode_metrics(
                day_index,
                environment_episode,
                reset_info,
                settlement,
                float(settlement["elapsed_time_s"]),
                switch_actions,
                day_step_rows,
                training_day_config.initial_instances,
                list(training_day_config.instance_types),
            )
            daily_rows.append(daily_row)

        episode_row = {
            "episode": episode,
            **aggregate_daily_episode_metrics(
                daily_rows,
                episode_requests,
                list(config.instance_types),
            ),
            "mean_loss": float(np.mean(losses)) if losses else None,
        }
        episode_rows.append(episode_row)
        episode_log.write(
            {
                "episode": episode,
                "episode_reward": episode_row["episode_reward"],
                "total_cost_usd": episode_row["total_cost_usd"],
                "p99_ttft_s": episode_row["p99_ttft_s"],
            }
        )
        episode_log.flush()
        print(json.dumps(episode_row, ensure_ascii=False))
        if (episode + 1) % config.training.checkpoint_interval_episodes == 0:
            agent.save(checkpoint_dir / f"{checkpoint_stem}_latest.pt")
            agent.save(
                checkpoint_dir
                / f"{checkpoint_stem}_ep{episode + 1:04d}.pt"
            )
        if ((episode + 1) % config.training.evaluation_interval_episodes == 0
                or episode + 1 == config.training.episodes):
            evaluation_summary, _ = run_greedy_evaluation(
                config=evaluation_config,
                agent=agent,
                episodes=config.training.evaluation_days,
                start_episode=0,
                write_detailed_logs=False,
            )
            from evaluate.core.evaluation_selection import update_evaluation_control
            control = getattr(agent, "evaluation_control", {})
            old_best = control.get("best_checkpoint")
            if old_best and not Path(old_best).is_file():
                control.pop("best_key", None)
            best_updated, lr_reduced = update_evaluation_control(
                agent, evaluation_summary, episode + 1, config.reward.sla_s, config.training)
            if best_updated:
                best_path = checkpoint_dir / f"{checkpoint_stem}_best.pt"
                agent.evaluation_control["best_checkpoint"] = str(best_path.resolve())
                agent.save(best_path)
                                                                                  
            agent.save(checkpoint_dir / f"{checkpoint_stem}_latest.pt")
            evaluation_row: Dict[str, object] = {
                "best_updated": best_updated,
                "learning_rate": agent.optimizer.param_groups[0]["lr"],
                "lr_reduced": lr_reduced,
                "training_episode": episode + 1,
                "episode_reward": evaluation_summary["episode_reward"],
                "total_cost_usd": evaluation_summary["total_cost_usd"],
                "p99_ttft_s": evaluation_summary["p99_ttft_s"],
            }
            periodic_evaluation_rows.append(evaluation_row)
            print(json.dumps(evaluation_row, ensure_ascii=False))

    episode_log.close()
    agent.save(checkpoint_dir / f"{checkpoint_stem}.pt")
    agent.save(checkpoint_dir / f"{checkpoint_stem}_latest.pt")
    best_checkpoint = getattr(agent, "evaluation_control", {}).get("best_checkpoint")
    if best_checkpoint and Path(best_checkpoint).is_file():
        agent.load(Path(best_checkpoint), load_optimizer=False)

        from shutil import copy2
        copy2(best_checkpoint, checkpoint_dir / f"{checkpoint_stem}.pt")
    final_test_dir = output_dir / "evaluation_original"
    final_test_summary, _ = run_greedy_evaluation(
        config=evaluation_config,
        agent=agent,
        episodes=config.training.evaluation_days,
        start_episode=0,
        output_dir=final_test_dir,
        write_detailed_logs=True,
        write_request_log=True,
        write_extended_step_log=True,
    )
    summary = {
        "seed": config.seed,
        "device": device,
        "agent_family": config.training.agent_family,
        "graph_encoder": agent.graph_encoder,
        "action_masking": config.training.action_masking,
        "torch_num_threads": config.training.torch_num_threads,
        "resume_checkpoint": (
            str(resume_checkpoint) if resume_checkpoint is not None else None
        ),
        "episodes": config.training.episodes,
        "replay_prefill_requested_steps": replay_prefill["requested_steps"],
        "replay_prefill_actual_steps": replay_prefill["actual_steps"],
        "replay_prefill_episodes": replay_prefill["episodes"],
        "replay_prefill_expert_fraction_requested": replay_prefill.get(
            "expert_fraction_requested", 0.0
        ),
        "replay_prefill_expert_fraction_actual": replay_prefill.get(
            "expert_fraction_actual", 0.0
        ),
        "replay_prefill_expert_episodes": replay_prefill.get(
            "expert_episodes", 0
        ),
        "replay_prefill_perturbed_expert_episodes": replay_prefill.get(
            "perturbed_expert_episodes", 0
        ),
        "replay_prefill_random_episodes": replay_prefill.get(
            "random_episodes", replay_prefill["episodes"]
        ),
        "replay_prefill_expert_mask_fallbacks": replay_prefill.get(
            "expert_mask_fallbacks", 0
        ),
        "replay_sampleable_after_prefill": replay_prefill["sampleable"],
        "replay_prefill_source": replay_prefill.get("source"),
        "replay_prefill_cache_path": replay_prefill.get("cache_path"),
        "replay_prefill_cache_config_sha256": replay_prefill.get(
            "cache_config_sha256"
        ),
        "training_days_per_episode": config.training.evaluation_days,
        "training_day_duration_s": training_day_config.duration_s,
        "reset_environment_at_day_end": True,
        "instance_type_count": len(config.instance_types),
        "instance_types": list(config.instance_types),
        "state_shape": [env.node_count, env.node_feature_dim],
        "instance_slot_counts": list(env.instance_slot_counts),
        "node_mask_side_channel": True,
        "action_count": env.action_count,
        "action_labels": list(env.action_labels()),
        "mean_episode_reward": float(np.mean([row["episode_reward"] for row in episode_rows])),
        "mean_completion_ratio": float(np.mean([row["completion_ratio"] for row in episode_rows])),
        "mean_cost_usd": float(np.mean([row["total_cost_usd"] for row in episode_rows])),
        "periodic_evaluations": len(periodic_evaluation_rows),
        "periodic_evaluation_interval_episodes": (
            config.training.evaluation_interval_episodes
        ),
        "periodic_evaluation_trace_days": config.training.evaluation_days,
        "periodic_evaluation_day_duration_s": (
            evaluation_config.duration_s
        ),
        "periodic_evaluation_group_duration_s": (
            evaluation_config.duration_s * config.training.evaluation_days
        ),
        "selected_checkpoint": best_checkpoint,
        "best_evaluation_episode": getattr(agent, "evaluation_control", {}).get("best_episode"),
        "final_test_episode_reward": final_test_summary["episode_reward"],
        "final_test_total_cost_usd": final_test_summary["total_cost_usd"],
        "final_test_p99_ttft_s": final_test_summary["p99_ttft_s"],
        "final_test_group_csv": str(
            final_test_dir / "evaluation_group.csv"
        ),
        "final_test_step_csv": str(final_test_dir / "evaluation_steps.csv"),
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
    }
    return summary
