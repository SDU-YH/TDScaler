from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from environment.config import ExperimentConfig
from environment.reporting import json_ready
from environment.rewarding import calculate_step_reward

CACHE_FORMAT = "tgrl_dtqn_replay_prefill_relabel_v4"
TRAJECTORY_EXPERT = 0
TRAJECTORY_PERTURBED_EXPERT = 1
TRAJECTORY_RANDOM = 2

class ReplayCacheError(ValueError):
    pass

def ensure_relabel_storage(replay_buffer: Any) -> None:
    shape = (
        int(replay_buffer.max_size),
        int(replay_buffer.max_episode_steps),
        1,
    )
    specifications = {
        "reward_step_cost_usd": (np.float64, 0.0),
        "reward_step_duration_s": (np.float64, 0.0),
        "reward_p99_ttft_s": (np.float64, 0.0),
        "reward_has_ttft_signal": (np.bool_, False),
        "reward_action_applied": (np.bool_, False),
        "reward_interval_load": (np.float64, 0.0),
        "reward_max_pending_ttft_age_s": (np.float64, np.nan),
    }
    for name, (dtype, fill) in specifications.items():
        value = getattr(replay_buffer, name, None)
        if value is None or value.shape != shape:
            setattr(replay_buffer, name, np.full(shape, fill, dtype=dtype))
    episode_shape = (int(replay_buffer.max_size),)
    if getattr(replay_buffer, "trajectory_kinds", None) is None:
        replay_buffer.trajectory_kinds = np.full(
            episode_shape, TRAJECTORY_RANDOM, dtype=np.uint8
        )
    if getattr(replay_buffer, "perturbation_counts", None) is None:
        replay_buffer.perturbation_counts = np.zeros(
            episode_shape, dtype=np.int16
        )

def record_relabel_step(
    replay_buffer: Any,
    *,
    step_cost_usd: float,
    step_duration_s: float,
    p99_ttft_s: float,
    has_ttft_signal: bool,
    action_applied: bool,
    interval_load: float,
    max_pending_ttft_age_s: float = float("nan"),
) -> None:
    ensure_relabel_storage(replay_buffer)
    episode = int(replay_buffer.pos[0]) % int(replay_buffer.max_size)
    step = int(replay_buffer.pos[1])
    if not 0 <= step < int(replay_buffer.max_episode_steps):
        raise ReplayCacheError("reward component step is outside replay episode")
    replay_buffer.reward_step_cost_usd[episode, step, 0] = step_cost_usd
    replay_buffer.reward_step_duration_s[episode, step, 0] = step_duration_s
    replay_buffer.reward_p99_ttft_s[episode, step, 0] = p99_ttft_s
    replay_buffer.reward_has_ttft_signal[episode, step, 0] = has_ttft_signal
    replay_buffer.reward_action_applied[episode, step, 0] = action_applied
    replay_buffer.reward_interval_load[episode, step, 0] = interval_load
    replay_buffer.reward_max_pending_ttft_age_s[episode, step, 0] = max_pending_ttft_age_s

def set_replay_episode_kind(
    replay_buffer: Any, kind: int, perturbation_count: int = 0
) -> None:
    ensure_relabel_storage(replay_buffer)
    if int(replay_buffer.pos[0]) <= 0 or int(replay_buffer.pos[1]) != 0:
        raise ReplayCacheError("no completed replay episode is available to label")
    episode = (int(replay_buffer.pos[0]) - 1) % int(replay_buffer.max_size)
    replay_buffer.trajectory_kinds[episode] = int(kind)
    replay_buffer.perturbation_counts[episode] = int(perturbation_count)

def _file_identity(path: Path | None) -> Dict[str, object] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "sha256": digest.hexdigest(),
    }

def build_replay_signature(
    config: ExperimentConfig,
    *,
    node_count: int,
    feature_dim: int,
    action_count: int,
    instance_slot_counts: tuple[int, ...],
    combined_observation_dim: int,
    max_episode_steps: int,
) -> Dict[str, object]:
    workload = json_ready(asdict(config.workload))
    payload: Dict[str, object] = {
        "cache_format": CACHE_FORMAT,
        "seed": int(config.seed),
        "duration_s": float(config.duration_s),
        "scaling_interval_s": float(config.scaling_interval_s),
        "adaptive_interval": asdict(config.adaptive_interval),
        "adaptive_interval_load_source": (
            "none_random" if config.adaptive_interval.mode == "random"
            else "new_arrival_prompt_tokens"
        ),
        "load_balancer": config.load_balancer,
        "active_model": config.active_model,
        "initial_instances": dict(config.initial_instances),
        "workload": workload,

        "state_reward_context": {
            "sla_s": float(config.reward.sla_s),
            "critical_ttft_s": float(config.reward.critical_ttft_s),
        },
        "models": {
            name: json_ready(asdict(spec)) for name, spec in config.models.items()
        },
        "instance_types": {
            name: json_ready(asdict(spec))
            for name, spec in config.instance_types.items()
        },
        "source_files": {
            "trace": _file_identity(config.workload.trace_path),
            "dataset": _file_identity(config.workload.dataset_path),
            "performance_profile": _file_identity(config.performance_profile_path),
            "expert_prefill_trajectory": _file_identity(
                config.training.replay_prefill_expert_trajectory_path
            ),
        },
        "prefill_policy": {
            "expert_fraction": float(
                config.training.replay_prefill_expert_fraction
            ),
            "expert_episodes": int(
                config.training.replay_prefill_expert_episodes
            ),
            "perturbed_expert_episodes": int(
                config.training.replay_prefill_perturbed_expert_episodes
            ),
            "random_episodes": int(
                config.training.replay_prefill_random_episodes
            ),
            "perturbation_decisions": int(
                config.training.replay_prefill_perturbation_decisions
            ),
        },
        "replay_layout": {
            "node_count": int(node_count),
            "feature_dim": int(feature_dim),
            "action_count": int(action_count),
            "instance_slot_counts": list(instance_slot_counts),
            "combined_observation_dim": int(combined_observation_dim),
            "max_episode_steps": int(max_episode_steps),
        },
    }
    if config.training.replay_prefill_expert_interval_s is not None:
        payload["prefill_policy"]["expert_source_interval_s"] = float(
            config.training.replay_prefill_expert_interval_s
        )
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    reward_parameters = json_ready(asdict(config.reward))
    reward_canonical = json.dumps(
        reward_parameters,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "payload": payload,

        "policy_action_masking": bool(config.training.action_masking),
        "policy_action_mask_input": (
            True
        ),
        "reward_sha256": hashlib.sha256(reward_canonical).hexdigest(),
        "reward_parameters": reward_parameters,
    }

def save_replay_cache(
    path: Path,
    replay_buffer: Any,
    signature: Mapping[str, object],
    prefill_summary: Mapping[str, object],
) -> Dict[str, object]:
    resolved = path.expanduser().resolve()
    completed_episodes = int(replay_buffer.pos[0])
    if int(replay_buffer.pos[1]) != 0:
        raise ReplayCacheError("cannot cache a partially written replay episode")
    if completed_episodes <= 0:
        raise ReplayCacheError("cannot cache an empty replay buffer")
    if completed_episodes > int(replay_buffer.max_size):
        raise ReplayCacheError(
            "prefill wrapped around replay capacity; increase replay_capacity"
        )

    ensure_relabel_storage(replay_buffer)
    episode_lengths = np.asarray(
        replay_buffer.episode_lengths[:completed_episodes], dtype=np.int32
    )
    max_steps = int(replay_buffer.max_episode_steps)
    if np.any(episode_lengths <= 0) or np.any(episode_lengths > max_steps):
        raise ReplayCacheError(
            "cannot save replay cache with invalid episode lengths"
        )
    transition_count = int(episode_lengths.sum())
    expected_transitions = int(
        prefill_summary.get("actual_steps", transition_count)
    )
    if transition_count != expected_transitions:
        raise ReplayCacheError(
            "cannot save replay cache with an inconsistent transition count"
        )
    metadata = {
        "cache_format": CACHE_FORMAT,
        "signature": dict(signature),
        "episode_count": completed_episodes,
        "transition_count": transition_count,
        "requested_steps": int(prefill_summary["requested_steps"]),
        "prefill_composition": {
            key: json_ready(prefill_summary[key])
            for key in (
                "expert_fraction_requested",
                "expert_fraction_actual",
                "expert_episodes",
                "perturbed_expert_episodes",
                "random_episodes",
                "expert_transitions",
                "perturbed_expert_transitions",
                "random_transitions",
                "expert_mask_fallbacks",
                "perturbation_decisions_requested",
                "perturbation_actions_actual",
                "expert_trajectory_path",
            )
            if key in prefill_summary
        },
        "reward_relabel_capable": True,
    }
    metadata_json = json.dumps(
        metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )

    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=resolved.parent,
            prefix=f".{resolved.name}.",
            suffix=".npz",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        np.savez_compressed(
            temporary_path,
            metadata_json=np.asarray(metadata_json),
            obss=replay_buffer.obss[:completed_episodes],
            actions=replay_buffer.actions[:completed_episodes],
            rewards=replay_buffer.rewards[:completed_episodes],
            dones=replay_buffer.dones[:completed_episodes],
            episode_lengths=episode_lengths,
            step_cost_usd=(
                replay_buffer.reward_step_cost_usd[:completed_episodes]
            ),
            max_pending_ttft_age_s=(
                replay_buffer.reward_max_pending_ttft_age_s[:completed_episodes]
            ),
            step_duration_s=(
                replay_buffer.reward_step_duration_s[:completed_episodes]
            ),
            p99_ttft_s=replay_buffer.reward_p99_ttft_s[:completed_episodes],
            has_ttft_signal=(
                replay_buffer.reward_has_ttft_signal[:completed_episodes]
            ),
            action_applied=(
                replay_buffer.reward_action_applied[:completed_episodes]
            ),
            interval_load=(
                replay_buffer.reward_interval_load[:completed_episodes]
            ),
            trajectory_kinds=(
                replay_buffer.trajectory_kinds[:completed_episodes]
            ),
            perturbation_counts=(
                replay_buffer.perturbation_counts[:completed_episodes]
            ),
        )
        os.replace(temporary_path, resolved)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return {
        "cache_path": str(resolved),
        "cache_config_sha256": str(signature["sha256"]),
        "cache_reward_sha256": str(signature["reward_sha256"]),
        "reward_relabel_capable": True,
        "cache_bytes": int(resolved.stat().st_size),
        "actual_steps": transition_count,
        "episodes": completed_episodes,
    }

def _require_shape(name: str, value: np.ndarray, expected: tuple[int, ...]) -> None:
    if value.shape != expected:
        raise ReplayCacheError(
            f"replay cache {name} shape mismatch: {value.shape} != {expected}"
        )

def load_replay_cache(
    path: Path,
    replay_buffer: Any,
    expected_signature: Mapping[str, object],
    *,
    config: ExperimentConfig,
    minimum_steps: int,
    batch_size: int,
) -> Dict[str, object]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"replay prefill cache does not exist: {resolved}")
    if list(replay_buffer.pos) != [0, 0]:
        raise ReplayCacheError("replay cache can only load into a fresh buffer")

    try:
        with np.load(resolved, allow_pickle=False) as archive:
            required = {
                "metadata_json",
                "obss",
                "actions",
                "rewards",
                "dones",
                "episode_lengths",
                "step_cost_usd",
                "step_duration_s",
                "p99_ttft_s",
                "has_ttft_signal",
                "action_applied",
                "interval_load",
                "trajectory_kinds",
                "perturbation_counts",
            }
            missing = required - set(archive.files)
            if missing:
                raise ReplayCacheError(
                    f"replay cache is missing arrays: {sorted(missing)}"
                )
            metadata = json.loads(str(archive["metadata_json"].item()))
            obss = np.asarray(archive["obss"]).copy()
            actions = np.asarray(archive["actions"]).copy()
            rewards = np.asarray(archive["rewards"]).copy()
            dones = np.asarray(archive["dones"]).copy()
            episode_lengths = np.asarray(
                archive["episode_lengths"], dtype=np.int32
            ).copy()
            step_cost_usd = np.asarray(
                archive["step_cost_usd"], dtype=np.float64
            ).copy()
            step_duration_s = np.asarray(
                archive["step_duration_s"], dtype=np.float64
            ).copy()
            p99_ttft_s = np.asarray(
                archive["p99_ttft_s"], dtype=np.float64
            ).copy()
            has_ttft_signal = np.asarray(
                archive["has_ttft_signal"], dtype=np.bool_
            ).copy()
            action_applied = np.asarray(
                archive["action_applied"], dtype=np.bool_
            ).copy()
            interval_load = np.asarray(
                archive["interval_load"], dtype=np.float64
            ).copy()
            max_pending_age = (
                np.asarray(archive["max_pending_ttft_age_s"], dtype=np.float64).copy()
                if "max_pending_ttft_age_s" in archive.files
                else np.full_like(step_duration_s, np.nan)
            )
            trajectory_kinds = np.asarray(
                archive["trajectory_kinds"], dtype=np.uint8
            ).copy()
            perturbation_counts = np.asarray(
                archive["perturbation_counts"], dtype=np.int16
            ).copy()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, ReplayCacheError):
            raise
        raise ReplayCacheError(f"failed to read replay cache {resolved}: {error}") from error

    if metadata.get("cache_format") != CACHE_FORMAT:
        raise ReplayCacheError(
            f"unsupported replay cache format: {metadata.get('cache_format')!r}"
        )
    cached_signature = metadata.get("signature", {})
    cached_sha256 = cached_signature.get("sha256")
    if any(key in cached_signature.get("payload", {})
           for key in ("graph_schema", "heterogeneity_level")):
        cached_payload = dict(cached_signature["payload"])
        cached_payload.pop("graph_schema", None)
        cached_payload.pop("heterogeneity_level", None)
        canonical = json.dumps(
            cached_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        cached_sha256 = hashlib.sha256(canonical).hexdigest()
    if cached_sha256 != expected_signature.get("sha256"):
        raise ReplayCacheError(
            "replay cache configuration fingerprint differs from the current "
            "environment; use a different cache path or --rebuild-replay-prefill-cache"
        )
    cached_action_masking = bool(
        cached_signature.get("policy_action_masking", True)
    )
    cached_action_mask_input = bool(
        cached_signature.get("policy_action_mask_input", True)
    )
    requested_action_masking = bool(
        expected_signature.get(
            "policy_action_masking", config.training.action_masking
        )
    )
    requested_action_mask_input = bool(
        expected_signature.get(
            "policy_action_mask_input", (
                True
            )
        )
    )
    if (
        requested_action_mask_input
        and requested_action_masking
        and (not cached_action_mask_input or not cached_action_masking)
    ):
        raise ReplayCacheError(
            "an unmasked replay cache cannot reconstruct environment action "
            "masks; rebuild the cache with action masking enabled"
        )

    episode_count = int(metadata.get("episode_count", -1))
    if episode_count <= 0 or episode_count > int(replay_buffer.max_size):
        raise ReplayCacheError(
            "replay cache episode count does not fit the current replay capacity"
        )
    max_steps = int(replay_buffer.max_episode_steps)
    obs_shape = tuple(replay_buffer.obss.shape[2:])
    _require_shape("obss", obss, (episode_count, max_steps + 1, *obs_shape))
    _require_shape("actions", actions, (episode_count, max_steps + 1, 1))
    _require_shape("rewards", rewards, (episode_count, max_steps, 1))
    _require_shape("dones", dones, (episode_count, max_steps, 1))
    _require_shape("episode_lengths", episode_lengths, (episode_count,))
    reward_signal_shape = (episode_count, max_steps, 1)
    _require_shape("step_cost_usd", step_cost_usd, reward_signal_shape)
    _require_shape("step_duration_s", step_duration_s, reward_signal_shape)
    _require_shape("p99_ttft_s", p99_ttft_s, reward_signal_shape)
    _require_shape("has_ttft_signal", has_ttft_signal, reward_signal_shape)
    _require_shape("action_applied", action_applied, reward_signal_shape)
    _require_shape("interval_load", interval_load, reward_signal_shape)
    _require_shape("max_pending_ttft_age_s", max_pending_age, reward_signal_shape)
    for episode, length in enumerate(episode_lengths):
        values = max_pending_age[episode, :int(length), 0]
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ReplayCacheError(
                "sla_cost requires observed max_pending_ttft_age_s for every "
                "transition; rebuild the cache with the current generator"
            )
    _require_shape("trajectory_kinds", trajectory_kinds, (episode_count,))
    _require_shape(
        "perturbation_counts", perturbation_counts, (episode_count,)
    )
    if np.any(episode_lengths <= 0) or np.any(episode_lengths > max_steps):
        raise ReplayCacheError("replay cache contains invalid episode lengths")
    transition_count = int(episode_lengths.sum())
    if transition_count != int(metadata.get("transition_count", -1)):
        raise ReplayCacheError("replay cache transition count is inconsistent")
    if transition_count < int(minimum_steps):
        raise ReplayCacheError(
            f"replay cache has {transition_count} steps, fewer than requested "
            f"{int(minimum_steps)}"
        )
    if (
        not np.isfinite(obss).all()
        or not np.isfinite(rewards).all()
        or not np.isfinite(step_cost_usd).all()
        or not np.isfinite(step_duration_s).all()
        or not np.isfinite(p99_ttft_s).all()
        or not np.isfinite(interval_load).all()
    ):
        raise ReplayCacheError(
            "replay cache contains non-finite observations, rewards, or reward signals"
        )
    if np.any(step_cost_usd < 0.0) or np.any(step_duration_s < 0.0):
        raise ReplayCacheError("replay cache contains negative reward signals")

    action_masks_rewritten = False
    if requested_action_mask_input and not requested_action_masking:
        action_count = int(
            expected_signature["payload"]["replay_layout"]["action_count"]
        )
        obss[..., -action_count:] = 1.0
        action_masks_rewritten = (
            cached_action_mask_input and cached_action_masking
        )

    replay_buffer.obss.fill(replay_buffer.obs_mask)
    replay_buffer.actions.fill(0)
    replay_buffer.rewards.fill(0.0)
    replay_buffer.dones.fill(True)
    replay_buffer.episode_lengths.fill(0)
    replay_buffer.obss[:episode_count] = obss.astype(
        replay_buffer.obss.dtype, copy=False
    )
    replay_buffer.actions[:episode_count] = actions.astype(
        replay_buffer.actions.dtype, copy=False
    )
    replay_buffer.dones[:episode_count] = dones.astype(
        replay_buffer.dones.dtype, copy=False
    )
    replay_buffer.episode_lengths[:episode_count] = episode_lengths.astype(
        replay_buffer.episode_lengths.dtype, copy=False
    )
    ensure_relabel_storage(replay_buffer)
    replay_buffer.reward_step_cost_usd.fill(0.0)
    replay_buffer.reward_step_duration_s.fill(0.0)
    replay_buffer.reward_p99_ttft_s.fill(0.0)
    replay_buffer.reward_has_ttft_signal.fill(False)
    replay_buffer.reward_action_applied.fill(False)
    replay_buffer.reward_interval_load.fill(0.0)
    replay_buffer.reward_max_pending_ttft_age_s.fill(np.nan)
    replay_buffer.trajectory_kinds.fill(TRAJECTORY_RANDOM)
    replay_buffer.perturbation_counts.fill(0)
    replay_buffer.reward_step_cost_usd[:episode_count] = step_cost_usd
    replay_buffer.reward_step_duration_s[:episode_count] = step_duration_s
    replay_buffer.reward_p99_ttft_s[:episode_count] = p99_ttft_s
    replay_buffer.reward_has_ttft_signal[:episode_count] = has_ttft_signal
    replay_buffer.reward_action_applied[:episode_count] = action_applied
    replay_buffer.reward_interval_load[:episode_count] = interval_load
    replay_buffer.reward_max_pending_ttft_age_s[:episode_count] = max_pending_age
    replay_buffer.trajectory_kinds[:episode_count] = trajectory_kinds
    replay_buffer.perturbation_counts[:episode_count] = perturbation_counts

    for episode in range(episode_count):
        elapsed = 0.0
        for step in range(int(episode_lengths[episode])):
            elapsed += float(step_duration_s[episode, step, 0])
            relabelled_reward, _ = calculate_step_reward(
                config,
                p99_ttft_s=float(p99_ttft_s[episode, step, 0]),
                has_ttft_signal=bool(has_ttft_signal[episode, step, 0]),
                natural_episode_end=(
                    step == int(episode_lengths[episode]) - 1
                    and elapsed >= config.duration_s - 1e-7
                ),
                step_cost_usd=float(step_cost_usd[episode, step, 0]),
                step_duration_s=float(step_duration_s[episode, step, 0]),
                max_pending_ttft_age_s=float(max_pending_age[episode, step, 0]),
                remaining_duration_s=max(0.0, config.duration_s - elapsed),
            )
            replay_buffer.rewards[episode, step, 0] = relabelled_reward
    replay_buffer.pos = [episode_count, 0]

    cached_reward_sha256 = str(cached_signature.get("reward_sha256", ""))
    active_reward_sha256 = str(expected_signature.get("reward_sha256", ""))
    relabelled = cached_reward_sha256 != active_reward_sha256
    sampleable = bool(replay_buffer.can_sample(int(batch_size)))

    result = {
        "source": "cache",
        "cache_path": str(resolved),
        "cache_config_sha256": str(expected_signature["sha256"]),
        "cached_reward_sha256": cached_reward_sha256,
        "active_reward_sha256": active_reward_sha256,
        "reward_relabelled": relabelled,
        "reward_relabel_capable": True,
        "cached_action_masking": cached_action_masking,
        "requested_action_masking": requested_action_masking,
        "action_masks_rewritten": action_masks_rewritten,
        "cache_bytes": int(resolved.stat().st_size),
        "requested_steps": int(minimum_steps),
        "actual_steps": transition_count,
        "episodes": episode_count,
        "sampleable": sampleable,
    }
    composition = metadata.get("prefill_composition", {})
    if isinstance(composition, dict):
        result.update(composition)
    return result
