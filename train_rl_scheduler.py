#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "torch is not installed. Activate the simulator venv and install requirements-simulator.txt first."
    ) from exc

from simulator_core import (
    ActionSpaceConfig,
    DecodeRouterPolicy,
    DistributionAwareScheduler,
    add_dynamic_workload_args,
    dynamic_workload_config_from_args,
    LLMSingleNodeSimulator,
    PhaseMode,
    QueueClass,
    PrefillRouterPolicy,
    Request,
    SchedulerAction,
    SimulatorConfig,
    generate_synthetic_requests,
    load_action_space_config,
    load_trace,
)
from simulator_core.action_space import (
    action_key,
    allowed_admission_profiles,
    allowed_budget_profiles,
    allowed_new_admission_limits,
    allowed_risky_admission_caps,
    allowed_shadow_quantiles,
    build_action_index,
    build_action_table,
    phase_label_from_preference,
    project_action_to_stable_space,
)
from build_workload_splits import generate_bucket_coverage_requests
from projection_analysis import parse_reasons


CURRENT_ACTION_SPACE = ActionSpaceConfig()
HISTORY_CRITIC_FEATURES_PER_STEP = 12
PHASE_CHOICES = list(CURRENT_ACTION_SPACE.phase_choices)
ADMISSION_CHOICES = list(CURRENT_ACTION_SPACE.admission_choices)
NEW_ADMISSION_LIMIT_CHOICES = list(CURRENT_ACTION_SPACE.new_admission_limit_choices)
QUANTILE_CHOICES = list(CURRENT_ACTION_SPACE.quantile_choices)
BUDGET_PROFILE_CHOICES = list(CURRENT_ACTION_SPACE.budget_profile_choices)
RISKY_ADMISSION_CAP_CHOICES = list(CURRENT_ACTION_SPACE.risky_admission_cap_choices)
SAFETY_FILTER_CHOICES = [False, True]
ACTION_TABLE = build_action_table(CURRENT_ACTION_SPACE)
ACTION_INDEX: Dict[Tuple[str, str, int, float, str, int], int] = build_action_index(
    CURRENT_ACTION_SPACE,
    ACTION_TABLE,
)
RecoveryJointProfile = Tuple[int, int, int, int, int]
DECISION_GATE_CHOICES = (False, True)
LIVENESS_COST_NAMES = (
    "deadlock_cost",
    "unfinished_cost",
    "prefill_liveness_violation",
    "first_token_backlog_violation",
    "unsafe_projection_cost",
)
REWARD_COMPONENT_NAMES = (
    "reward_completion",
    "reward_admission",
    "reward_latency",
    "reward_kv_safety",
    "reward_projection_penalty",
    "reward_liveness_penalty",
    "reward_queue_penalty",
    "reward_deadlock_penalty",
    "reward_other",
)
REQUEST_ID_LIST_KEYS = (
    "new_admission_request_ids",
    "prefill_progress_request_ids",
    "prefill_complete_request_ids",
    "decode_progress_request_ids",
    "first_token_request_ids",
    "complete_request_ids",
    "evicted_request_ids",
    "recomputed_request_ids",
)
DEFAULT_ACTOR_PROJECTION_DEBIT_REASON_WEIGHTS = (
    "kv_recovery_budget_guard=2.0,"
    "kv_recovery_phase_guard=2.0,"
    "kv_recovery_admission_guard=2.0,"
    "new_admission_limit_guard=1.5,"
    "prefill_queue_admission_guard=1.5,"
    "admission_profile_guard=1.0,"
    "first_token_risky_cap=1.0,"
    "prefill_liveness_risky_cap_floor=1.0,"
    "shadow_quantile_guard=0.75"
)


def set_current_action_space(action_space: ActionSpaceConfig) -> None:
    global CURRENT_ACTION_SPACE
    global PHASE_CHOICES
    global ADMISSION_CHOICES
    global NEW_ADMISSION_LIMIT_CHOICES
    global QUANTILE_CHOICES
    global BUDGET_PROFILE_CHOICES
    global RISKY_ADMISSION_CAP_CHOICES
    global ACTION_TABLE
    global ACTION_INDEX

    CURRENT_ACTION_SPACE = ActionSpaceConfig.from_payload(action_space)
    PHASE_CHOICES = list(CURRENT_ACTION_SPACE.phase_choices)
    ADMISSION_CHOICES = list(CURRENT_ACTION_SPACE.admission_choices)
    NEW_ADMISSION_LIMIT_CHOICES = list(CURRENT_ACTION_SPACE.new_admission_limit_choices)
    QUANTILE_CHOICES = list(CURRENT_ACTION_SPACE.quantile_choices)
    BUDGET_PROFILE_CHOICES = list(CURRENT_ACTION_SPACE.budget_profile_choices)
    RISKY_ADMISSION_CAP_CHOICES = list(CURRENT_ACTION_SPACE.risky_admission_cap_choices)
    ACTION_TABLE = build_action_table(CURRENT_ACTION_SPACE)
    ACTION_INDEX = build_action_index(CURRENT_ACTION_SPACE, ACTION_TABLE)


def default_new_admission_limit_choice() -> int:
    positive_choices = [int(choice) for choice in NEW_ADMISSION_LIMIT_CHOICES if int(choice) > 0]
    return min(positive_choices) if positive_choices else int(NEW_ADMISSION_LIMIT_CHOICES[0])


def default_shadow_quantile_choice() -> float:
    return float(QUANTILE_CHOICES[0])


def default_risky_admission_cap_choice() -> int:
    return int(RISKY_ADMISSION_CAP_CHOICES[0])


def _prefill_limit_value(value: Optional[int]) -> int:
    return int(value) if value is not None else default_new_admission_limit_choice()


def _risky_cap_value(value: Optional[int]) -> int:
    return int(value) if value is not None else default_risky_admission_cap_choice()


def nearest_quantile_choice(value: float) -> float:
    return min(QUANTILE_CHOICES, key=lambda candidate: (abs(float(candidate) - float(value)), float(candidate)))


@dataclass(slots=True)
class RewardConfig:
    profile: str = "hybrid-stability"
    first_token_reward: float = 3.5
    completion_reward: float = 0.4
    new_admission_reward: float = 0.2
    duration_penalty: float = 0.025
    admission_wait_penalty: float = 0.030
    ttft_penalty: float = 0.020
    e2e_penalty: float = 0.001
    pending_admission_penalty: float = 0.060
    pending_admission_tail_penalty: float = 0.080
    first_token_pending_penalty: float = 0.040
    first_token_pending_tail_penalty: float = 0.050
    kv_penalty: float = 1.200
    kv_penalty_start_util: float = 0.920
    prefill_padding_penalty: float = 0.010
    eviction_penalty: float = 1.500
    recompute_token_penalty: float = 0.004
    wasted_kv_penalty: float = 0.002
    terminal_all_complete_bonus: float = 6.0
    terminal_early_stop_penalty: float = 16.0
    terminal_unfinished_request_penalty: float = 3.0
    terminal_deadlock_recovery_penalty: float = 0.5
    terminal_evicted_request_penalty: float = 0.4
    terminal_recompute_total_penalty: float = 0.025
    terminal_wasted_kv_total_penalty: float = 0.010
    terminal_external_fragmentation_penalty: float = 0.12
    terminal_internal_fragmentation_penalty: float = 0.04
    terminal_ttft_p95_penalty: float = 0.35
    terminal_ttft_p99_penalty: float = 0.15
    terminal_ttft_cvar95_penalty: float = 0.55
    terminal_ttft_cvar99_penalty: float = 0.20
    terminal_ttft_max_penalty: float = 0.12
    terminal_e2e_p95_penalty: float = 0.08
    terminal_admission_wait_cvar95_penalty: float = 0.18
    terminal_peak_kv_target_min_util: float = 0.90
    terminal_peak_kv_target_max_util: float = 0.95
    terminal_peak_kv_underutil_penalty: float = 0.0
    terminal_peak_kv_overutil_penalty: float = 0.0
    high_kv_fast_admission_reward: float = 0.10
    high_kv_normal_admission_reward: float = 0.04
    high_kv_new_admission_penalty: float = 0.65
    high_kv_risky_admission_penalty: float = 0.18
    high_kv_fast_first_token_reward: float = 0.18
    high_kv_decode_token_reward: float = 0.030
    high_kv_zero_admission_reward: float = 0.45
    high_kv_decode_drain_reward: float = 0.40
    high_kv_prefill_priority_penalty: float = 0.45
    high_kv_prefill_work_penalty: float = 0.050
    projected_kv_admission_penalty: float = 2.50
    projected_shadow_admission_penalty: float = 3.00
    projected_kv_admission_penalty_start_util: float = 0.88
    projected_shadow_admission_penalty_start_util: float = 0.82
    high_kv_fast_pending_tail_penalty: float = 0.040
    high_kv_normal_pending_tail_penalty: float = 0.020
    high_kv_admission_stall_penalty: float = 0.080
    short_admission_stall_penalty: float = 0.120
    first_token_pending_age_quadratic_penalty: float = 0.015
    risky_first_token_backlog_penalty: float = 0.180
    action_safety_projection_penalty: float = 0.0
    deadlock_cost_penalty: float = 0.0
    unfinished_cost_penalty: float = 0.0
    prefill_liveness_violation_penalty: float = 0.0
    first_token_backlog_violation_penalty: float = 0.0
    unsafe_projection_cost_penalty: float = 0.0
    actor_projection_debit_scale: float = 0.0
    actor_projection_debit_reason_weights: str = DEFAULT_ACTOR_PROJECTION_DEBIT_REASON_WEIGHTS


def reward_profile_defaults(profile: str) -> RewardConfig:
    if profile == "ttft-focused":
        return RewardConfig(
            profile=profile,
            first_token_reward=4.0,
            completion_reward=0.3,
            new_admission_reward=0.2,
            duration_penalty=0.025,
            admission_wait_penalty=0.035,
            ttft_penalty=0.030,
            e2e_penalty=0.0005,
            pending_admission_penalty=0.060,
            pending_admission_tail_penalty=0.090,
            first_token_pending_penalty=0.060,
            first_token_pending_tail_penalty=0.070,
            kv_penalty=0.900,
            kv_penalty_start_util=0.920,
            prefill_padding_penalty=0.010,
            eviction_penalty=1.000,
            recompute_token_penalty=0.002,
            wasted_kv_penalty=0.001,
            terminal_all_complete_bonus=4.0,
            terminal_early_stop_penalty=8.0,
            terminal_unfinished_request_penalty=1.5,
            terminal_deadlock_recovery_penalty=0.25,
            terminal_evicted_request_penalty=0.15,
            terminal_recompute_total_penalty=0.008,
            terminal_wasted_kv_total_penalty=0.004,
            terminal_external_fragmentation_penalty=0.04,
            terminal_internal_fragmentation_penalty=0.01,
            terminal_ttft_p95_penalty=0.45,
            terminal_ttft_p99_penalty=0.35,
            terminal_ttft_cvar95_penalty=0.60,
            terminal_ttft_cvar99_penalty=0.45,
            terminal_ttft_max_penalty=0.30,
            terminal_e2e_p95_penalty=0.04,
            terminal_admission_wait_cvar95_penalty=0.14,
            terminal_peak_kv_target_min_util=0.90,
            terminal_peak_kv_target_max_util=0.95,
            terminal_peak_kv_underutil_penalty=0.0,
            terminal_peak_kv_overutil_penalty=0.0,
            high_kv_fast_admission_reward=0.14,
            high_kv_normal_admission_reward=0.05,
            high_kv_risky_admission_penalty=0.20,
            high_kv_fast_first_token_reward=0.22,
            high_kv_fast_pending_tail_penalty=0.050,
            high_kv_normal_pending_tail_penalty=0.020,
            high_kv_admission_stall_penalty=0.090,
            short_admission_stall_penalty=0.140,
            first_token_pending_age_quadratic_penalty=0.030,
            risky_first_token_backlog_penalty=0.260,
        )
    if profile == "stability-strict":
        return RewardConfig(
            profile=profile,
            first_token_reward=3.0,
            completion_reward=0.5,
            new_admission_reward=0.15,
            duration_penalty=0.020,
            admission_wait_penalty=0.025,
            ttft_penalty=0.015,
            e2e_penalty=0.0015,
            pending_admission_penalty=0.070,
            pending_admission_tail_penalty=0.090,
            first_token_pending_penalty=0.035,
            first_token_pending_tail_penalty=0.045,
            kv_penalty=1.500,
            kv_penalty_start_util=0.920,
            prefill_padding_penalty=0.012,
            eviction_penalty=2.200,
            recompute_token_penalty=0.006,
            wasted_kv_penalty=0.003,
            terminal_all_complete_bonus=8.0,
            terminal_early_stop_penalty=24.0,
            terminal_unfinished_request_penalty=4.0,
            terminal_deadlock_recovery_penalty=0.8,
            terminal_evicted_request_penalty=0.7,
            terminal_recompute_total_penalty=0.040,
            terminal_wasted_kv_total_penalty=0.015,
            terminal_external_fragmentation_penalty=0.16,
            terminal_internal_fragmentation_penalty=0.05,
            terminal_ttft_p95_penalty=0.25,
            terminal_ttft_p99_penalty=0.12,
            terminal_ttft_cvar95_penalty=0.40,
            terminal_ttft_cvar99_penalty=0.18,
            terminal_ttft_max_penalty=0.10,
            terminal_e2e_p95_penalty=0.12,
            terminal_admission_wait_cvar95_penalty=0.22,
            terminal_peak_kv_target_min_util=0.90,
            terminal_peak_kv_target_max_util=0.95,
            terminal_peak_kv_underutil_penalty=0.0,
            terminal_peak_kv_overutil_penalty=0.0,
            high_kv_fast_admission_reward=0.06,
            high_kv_normal_admission_reward=0.02,
            high_kv_risky_admission_penalty=0.22,
            high_kv_fast_first_token_reward=0.10,
            high_kv_fast_pending_tail_penalty=0.030,
            high_kv_normal_pending_tail_penalty=0.015,
            high_kv_admission_stall_penalty=0.050,
            short_admission_stall_penalty=0.100,
            first_token_pending_age_quadratic_penalty=0.012,
            risky_first_token_backlog_penalty=0.220,
        )
    return RewardConfig(profile="hybrid-stability")


def _safe_metric(summary: Dict[str, object], key: str, metric: str) -> float:
    payload = summary.get(key, {})
    if not isinstance(payload, dict):
        return 0.0
    value = payload.get(metric)
    if value in (None, ""):
        return 0.0
    return float(value)


def _terminal_peak_kv_band_penalty(summary: Dict[str, object], reward_config: RewardConfig) -> Dict[str, float]:
    capacity = float(summary.get("kv_capacity_tokens", 0.0) or 0.0)
    peak = float(summary.get("peak_kv_tokens", 0.0) or 0.0)
    if capacity <= 0.0:
        return {"util": 0.0, "underutil_penalty": 0.0, "overutil_penalty": 0.0}
    peak_util = peak / capacity
    min_target = min(
        float(reward_config.terminal_peak_kv_target_min_util),
        float(reward_config.terminal_peak_kv_target_max_util),
    )
    max_target = max(
        float(reward_config.terminal_peak_kv_target_min_util),
        float(reward_config.terminal_peak_kv_target_max_util),
    )
    underutil_penalty = reward_config.terminal_peak_kv_underutil_penalty * max(0.0, min_target - peak_util) * 100.0
    overutil_penalty = reward_config.terminal_peak_kv_overutil_penalty * max(0.0, peak_util - max_target) * 100.0
    return {
        "util": peak_util,
        "underutil_penalty": underutil_penalty,
        "overutil_penalty": overutil_penalty,
    }


def compute_terminal_objective(summary: Dict[str, object], reward_config: RewardConfig) -> float:
    split = compute_terminal_objective_split(summary, reward_config)
    return split["total"]


def zero_costs() -> Dict[str, float]:
    return {name: 0.0 for name in LIVENESS_COST_NAMES}


def compute_terminal_costs(summary: Dict[str, object]) -> Dict[str, float]:
    costs = zero_costs()
    termination_reason = str(summary.get("termination_reason") or "")
    deadlock_like = termination_reason.startswith("deadlock") or float(summary.get("deadlock_recoveries_total", 0.0) or 0.0) > 0.0
    costs["deadlock_cost"] = 1.0 if deadlock_like else 0.0
    if "punishable_unfinished_requests" in summary:
        costs["unfinished_cost"] = float(summary.get("punishable_unfinished_requests", 0.0) or 0.0)
    elif termination_reason == "simulation_duration_elapsed":
        costs["unfinished_cost"] = float(summary.get("aged_unfinished_requests", 0.0) or 0.0)
    else:
        costs["unfinished_cost"] = float(summary.get("unfinished_requests", 0.0) or 0.0)
    return costs


def compute_step_costs(step_result: Dict[str, float], config: SimulatorConfig) -> Dict[str, float]:
    costs = zero_costs()
    live_decode_after = float(step_result.get("active_decode_after", 0.0))
    pending_after = float(step_result.get("pending_unadmitted_after", 0.0))
    new_admissions = float(step_result.get("new_admissions", 0.0))
    prefill_completed = float(step_result.get("prefill_completed_requests", 0.0))
    if live_decode_after <= 0.0 and pending_after > 0.0 and new_admissions <= 0.0 and prefill_completed <= 0.0:
        costs["prefill_liveness_violation"] = 1.0

    first_token_pending = float(step_result.get("first_token_pending_after", 0.0))
    first_token_threshold = max(
        1.0,
        math.ceil(max(config.decode_parallelism or config.max_decode_batch, 1) * config.action_safety_first_token_load_threshold),
    )
    risky_cap = float(step_result.get("requested_scheduler_risky_admission_cap", step_result.get("scheduler_risky_admission_cap", 0.0)))
    if first_token_pending >= first_token_threshold and risky_cap > float(config.action_safety_first_token_max_risky_admissions):
        costs["first_token_backlog_violation"] = 1.0

    if bool(step_result.get("action_safety_triggered", 0.0)):
        costs["unsafe_projection_cost"] = 1.0
    return costs


def add_costs(left: Dict[str, float], right: Dict[str, float]) -> Dict[str, float]:
    merged = dict(left)
    for name in LIVENESS_COST_NAMES:
        merged[name] = float(merged.get(name, 0.0)) + float(right.get(name, 0.0))
    return merged


def counterfactual_router_cost_attribution(step_result: Dict[str, float], config: SimulatorConfig) -> Dict[str, float]:
    """One-step counterfactual attribution using requested-vs-executed router heads.

    The simulator does not snapshot and replay every state here. Instead this keeps
    one router fixed and asks whether replacing the other router's requested heads
    with the shield/expert heads would have removed the local violation signal.
    """
    costs = compute_step_costs(step_result, config)
    prefill_changed = any(
        step_result.get(f"scheduler_{key}") != step_result.get(f"requested_scheduler_{key}")
        for key in ("admission_profile", "new_admission_limit", "shadow_quantile", "risky_admission_cap")
    )
    decode_changed = (
        step_result.get("phase_mode") != step_result.get("requested_phase_mode")
        or step_result.get("scheduler_budget_profile") != step_result.get("requested_scheduler_budget_profile")
    )
    prefill_delta = 0.0
    decode_delta = 0.0
    if costs["prefill_liveness_violation"] > 0.0:
        prefill_delta += costs["prefill_liveness_violation"] if prefill_changed or not decode_changed else 0.5
        decode_delta += 0.5 * costs["prefill_liveness_violation"] if decode_changed else 0.0
    if costs["first_token_backlog_violation"] > 0.0:
        prefill_delta += 0.7 * costs["first_token_backlog_violation"] if prefill_changed else 0.3
        decode_delta += 0.7 * costs["first_token_backlog_violation"] if decode_changed else 0.3
    if costs["unsafe_projection_cost"] > 0.0:
        prefill_delta += 0.5 * costs["unsafe_projection_cost"] if prefill_changed else 0.0
        decode_delta += 0.5 * costs["unsafe_projection_cost"] if decode_changed else 0.0
    return {
        "counterfactual_prefill_cost_delta": prefill_delta,
        "counterfactual_decode_cost_delta": decode_delta,
        "counterfactual_prefill_responsibility": prefill_delta / max(prefill_delta + decode_delta, 1e-9),
        "counterfactual_decode_responsibility": decode_delta / max(prefill_delta + decode_delta, 1e-9),
    }


DEADLOCK_CLASS_DESCRIPTIONS = {
    "A": "prefill_over_conservative_idle_pending",
    "B": "decode_over_conservative_backlog_not_drained",
    "C": "prefill_preempts_decode_starvation",
    "D": "decode_preempts_prefill_starvation",
    "E": "persistent_unsafe_projection_boundary",
    "F": "workload_burst_or_physical_capacity",
    "G": "trace_horizon_duration_censored",
    "U": "unclassified_terminal_failure",
}


def _float_from_mapping(row: Dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _last_float(rows: Sequence[Dict[str, object]], key: str, default: float = 0.0) -> float:
    for row in reversed(rows):
        if key in row:
            return _float_from_mapping(row, key, default)
    return default


def _sum_float(rows: Sequence[Dict[str, object]], key: str) -> float:
    return sum(_float_from_mapping(row, key, 0.0) for row in rows)


def _max_float(rows: Sequence[Dict[str, object]], key: str, default: float = 0.0) -> float:
    if not rows:
        return default
    return max((_float_from_mapping(row, key, default) for row in rows), default=default)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def classify_deadlock_case(
    summary: Dict[str, object],
    recent_steps: Sequence[Dict[str, object]],
    *,
    kv_capacity_tokens: float = 0.0,
) -> List[str]:
    """Classify early termination windows into the todo's A-G buckets."""
    termination_reason = str(summary.get("termination_reason") or "")
    horizon_unfinished = float(summary.get("horizon_unfinished_requests", 0.0) or 0.0)
    punishable_unfinished = float(summary.get("punishable_unfinished_requests", 0.0) or 0.0)
    if (
        termination_reason == "simulation_duration_elapsed"
        and horizon_unfinished > 0.0
        and punishable_unfinished <= 0.0
    ):
        return ["G"]

    details = summary.get("termination_details", {})
    detail_map = details if isinstance(details, dict) else {}
    pending_unadmitted = _float_from_mapping(
        detail_map,
        "pending_unadmitted",
        _last_float(recent_steps, "pending_unadmitted_count", 0.0),
    )
    started_prefill = _float_from_mapping(
        detail_map,
        "started_prefill_waiting",
        _last_float(recent_steps, "started_prefill_waiting_count", 0.0),
    )
    active_decode = _float_from_mapping(
        detail_map,
        "active_decode",
        _last_float(recent_steps, "decode_queue_length", 0.0),
    )
    current_kv = _float_from_mapping(
        detail_map,
        "current_kv_tokens",
        _last_float(recent_steps, "kv_tokens", 0.0),
    )
    capacity = max(float(kv_capacity_tokens or 0.0), _last_float(recent_steps, "kv_capacity_tokens", 0.0), 1.0)
    kv_util = current_kv / capacity
    first_token_backlog = max(
        _last_float(recent_steps, "first_token_backlog", 0.0),
        _max_float(recent_steps, "first_token_pending_after", 0.0),
    )
    new_admissions = _sum_float(recent_steps, "new_admissions")
    prefill_tokens = _sum_float(recent_steps, "prefill_tokens")
    decode_tokens = _sum_float(recent_steps, "decode_tokens")
    first_tokens = _sum_float(recent_steps, "first_tokens_emitted")
    projection_count = sum(1 for row in recent_steps if bool(row.get("action_safety_projection_triggered", False)))
    projection_fraction = projection_count / max(len(recent_steps), 1)

    classes: List[str] = []
    if pending_unadmitted > 0.0 and active_decode <= 0.0 and started_prefill <= 0.0 and kv_util < 0.80:
        classes.append("A")
    if (active_decode > 0.0 or first_token_backlog > 0.0) and (decode_tokens <= 0.0 or first_tokens <= 0.0):
        classes.append("B")
    if (active_decode > 0.0 or first_token_backlog > 0.0) and (
        new_admissions > 0.0 or prefill_tokens > max(decode_tokens * 1.5, 0.0)
    ):
        classes.append("C")
    if pending_unadmitted > 0.0 and decode_tokens > 0.0 and new_admissions <= 0.0 and prefill_tokens <= 0.0:
        classes.append("D")
    if projection_count >= max(3, int(math.ceil(len(recent_steps) * 0.50))) or projection_fraction >= 0.50:
        classes.append("E")
    if kv_util >= 0.95 or (
        pending_unadmitted > 0.0
        and (active_decode > 0.0 or first_token_backlog > 0.0)
        and kv_util >= 0.85
    ):
        classes.append("F")
    if not classes and (bool(summary.get("terminated_early", False)) or termination_reason.startswith("deadlock")):
        classes.append("U")
    return classes


def compute_terminal_objective_split(summary: Dict[str, object], reward_config: RewardConfig) -> Dict[str, float]:
    prefill_reward = 0.0
    decode_reward = 0.0
    reward = 0.0
    terminal_costs = compute_terminal_costs(summary)
    punishable_unfinished = float(terminal_costs["unfinished_cost"])
    deadlock_cost = float(terminal_costs["deadlock_cost"])
    terminal_failure = bool(summary.get("terminated_early", False)) and (
        punishable_unfinished > 0.0 or deadlock_cost > 0.0
    )
    if terminal_failure:
        early_stop_penalty = reward_config.terminal_early_stop_penalty
        unfinished_penalty = reward_config.terminal_unfinished_request_penalty * punishable_unfinished
        prefill_reward -= 0.5 * early_stop_penalty + 0.6 * unfinished_penalty
        decode_reward -= 0.5 * early_stop_penalty + 0.4 * unfinished_penalty
        completion_component = 0.0
        deadlock_component = -early_stop_penalty
        unfinished_component = -unfinished_penalty
    elif not bool(summary.get("terminated_early", False)):
        all_complete_bonus = reward_config.terminal_all_complete_bonus
        prefill_reward += 0.35 * all_complete_bonus
        decode_reward += 0.65 * all_complete_bonus
        completion_component = all_complete_bonus
        deadlock_component = 0.0
        unfinished_component = 0.0
    else:
        completion_component = 0.0
        deadlock_component = 0.0
        unfinished_component = 0.0

    deadlock_penalty = reward_config.terminal_deadlock_recovery_penalty * float(
        summary.get("deadlock_recoveries_total", 0.0)
    )
    evicted_penalty = reward_config.terminal_evicted_request_penalty * float(summary.get("evicted_requests", 0.0))
    recompute_penalty = reward_config.terminal_recompute_total_penalty * (
        float(summary.get("recompute_tokens_total", 0.0)) / 1024.0
    )
    wasted_kv_penalty = reward_config.terminal_wasted_kv_total_penalty * (
        float(summary.get("wasted_kv_tokens_total", 0.0)) / 1024.0
    )
    external_frag_penalty = reward_config.terminal_external_fragmentation_penalty * (
        float(summary.get("peak_allocator_external_fragmentation_tokens", 0.0)) / 512.0
    )
    internal_frag_penalty = reward_config.terminal_internal_fragmentation_penalty * (
        float(summary.get("peak_allocator_internal_fragmentation_tokens", 0.0)) / 512.0
    )
    ttft_p95_penalty = reward_config.terminal_ttft_p95_penalty * (_safe_metric(summary, "ttft_ms", "p95") / 1000.0)
    ttft_p99_penalty = reward_config.terminal_ttft_p99_penalty * (_safe_metric(summary, "ttft_ms", "p99") / 1000.0)
    ttft_cvar95_penalty = reward_config.terminal_ttft_cvar95_penalty * (
        _safe_metric(summary, "ttft_ms", "cvar95") / 1000.0
    )
    ttft_cvar99_penalty = reward_config.terminal_ttft_cvar99_penalty * (
        _safe_metric(summary, "ttft_ms", "cvar99") / 1000.0
    )
    ttft_max_penalty = reward_config.terminal_ttft_max_penalty * (_safe_metric(summary, "ttft_ms", "max") / 1000.0)
    e2e_p95_penalty = reward_config.terminal_e2e_p95_penalty * (_safe_metric(summary, "e2e_ms", "p95") / 1000.0)
    admission_wait_penalty = reward_config.terminal_admission_wait_cvar95_penalty * (
        _safe_metric(summary, "admission_wait_ms", "cvar95") / 1000.0
    )
    peak_kv_band_penalty = _terminal_peak_kv_band_penalty(summary, reward_config)
    prefill_reward -= (
        deadlock_penalty
        + reward_config.deadlock_cost_penalty * terminal_costs["deadlock_cost"]
        + 0.6 * reward_config.unfinished_cost_penalty * terminal_costs["unfinished_cost"]
        + 0.6 * evicted_penalty
        + 0.7 * recompute_penalty
        + 0.7 * wasted_kv_penalty
        + external_frag_penalty
        + internal_frag_penalty
        + admission_wait_penalty
        + peak_kv_band_penalty["underutil_penalty"]
    )
    decode_reward -= (
        0.4 * reward_config.unfinished_cost_penalty * terminal_costs["unfinished_cost"]
        + 0.4 * evicted_penalty
        + 0.3 * recompute_penalty
        + 0.3 * wasted_kv_penalty
        + ttft_p95_penalty
        + ttft_p99_penalty
        + ttft_cvar95_penalty
        + ttft_cvar99_penalty
        + ttft_max_penalty
        + e2e_p95_penalty
        + peak_kv_band_penalty["overutil_penalty"]
    )
    reward = prefill_reward + decode_reward
    latency_component = -(
        ttft_p95_penalty
        + ttft_p99_penalty
        + ttft_cvar95_penalty
        + ttft_cvar99_penalty
        + ttft_max_penalty
        + e2e_p95_penalty
        + admission_wait_penalty
    )
    kv_safety_component = -(
        evicted_penalty
        + recompute_penalty
        + wasted_kv_penalty
        + external_frag_penalty
        + internal_frag_penalty
        + peak_kv_band_penalty["underutil_penalty"]
        + peak_kv_band_penalty["overutil_penalty"]
    )
    deadlock_component -= deadlock_penalty + reward_config.deadlock_cost_penalty * terminal_costs["deadlock_cost"]
    liveness_component = -reward_config.unfinished_cost_penalty * terminal_costs["unfinished_cost"]
    component_sum = (
        completion_component
        + latency_component
        + kv_safety_component
        + deadlock_component
        + liveness_component
        + unfinished_component
    )
    return {
        "prefill_reward": prefill_reward,
        "decode_reward": decode_reward,
        "total": reward,
        "total_reward": reward,
        "reward_completion": completion_component,
        "reward_admission": 0.0,
        "reward_latency": latency_component,
        "reward_kv_safety": kv_safety_component,
        "reward_projection_penalty": 0.0,
        "reward_liveness_penalty": liveness_component + unfinished_component,
        "reward_queue_penalty": 0.0,
        "reward_deadlock_penalty": deadlock_component,
        "reward_other": reward - component_sum,
    }


def parse_actor_projection_debit_reason_weights(spec: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    if not spec:
        return weights
    for raw_item in spec.replace(";", ",").split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" in item:
            reason, value = item.split("=", 1)
        elif ":" in item:
            reason, value = item.split(":", 1)
        else:
            raise ValueError(f"expected reason=weight in actor projection debit override: {item!r}")
        reason = reason.strip()
        if not reason:
            raise ValueError(f"empty reason in actor projection debit override: {item!r}")
        weights[reason] = max(float(value), 0.0)
    return weights


def actor_projection_debit_for_step(step_result: Dict[str, float], reward_config: RewardConfig) -> float:
    if not bool(step_result.get("action_safety_triggered", 0.0)):
        return 0.0
    scale = max(float(reward_config.actor_projection_debit_scale), 0.0)
    if scale <= 0.0:
        return 0.0
    reasons = parse_reasons(step_result.get("action_safety_reasons", ""))
    if not reasons:
        return scale
    weights = parse_actor_projection_debit_reason_weights(reward_config.actor_projection_debit_reason_weights)
    return scale * sum(float(weights.get(reason, 1.0)) for reason in reasons)


def projection_debit_responsibility(step_result: Dict[str, float]) -> Tuple[float, float]:
    prefill_responsibility = float(step_result.get("counterfactual_prefill_responsibility", 0.0) or 0.0)
    decode_responsibility = float(step_result.get("counterfactual_decode_responsibility", 0.0) or 0.0)
    total = prefill_responsibility + decode_responsibility
    if total > 0.0:
        return prefill_responsibility / total, decode_responsibility / total

    prefill_changed = any(
        step_result.get(f"scheduler_{key}") != step_result.get(f"requested_scheduler_{key}")
        for key in ("admission_profile", "new_admission_limit", "shadow_quantile", "risky_admission_cap")
    )
    decode_changed = (
        step_result.get("phase_mode") != step_result.get("requested_phase_mode")
        or step_result.get("scheduler_budget_profile") != step_result.get("requested_scheduler_budget_profile")
    )
    if prefill_changed and decode_changed:
        return 0.5, 0.5
    if prefill_changed:
        return 1.0, 0.0
    if decode_changed:
        return 0.0, 1.0
    return 0.5, 0.5


def compute_step_reward_split(step_result: Dict[str, float], config: SimulatorConfig, reward_config: RewardConfig) -> Dict[str, float]:
    kv_util = step_result["kv_after"] / max(config.kv_capacity_tokens, 1)
    kv_before_util = float(step_result.get("kv_before", step_result["kv_after"])) / max(config.kv_capacity_tokens, 1)
    kv_pressure = 0.0
    if reward_config.kv_penalty_start_util < 1.0:
        kv_pressure = max(0.0, max(kv_util, kv_before_util) - reward_config.kv_penalty_start_util) / max(
            1.0 - reward_config.kv_penalty_start_util,
            1e-6,
        )
    kv_pressure = min(kv_pressure, 1.0)
    prefill_work = (
        float(step_result.get("prefill_tokens", 0.0))
        + float(step_result.get("prefill_padding_tokens", 0.0))
    )
    decode_work = float(step_result.get("decode_tokens", 0.0))
    total_work = prefill_work + decode_work
    if total_work <= 0:
        prefill_share = 0.5
        decode_share = 0.5
    else:
        prefill_share = prefill_work / total_work
        decode_share = decode_work / total_work
    duration_penalty = reward_config.duration_penalty * float(step_result["duration_ms"])
    evictions_penalty = reward_config.eviction_penalty * float(step_result.get("evictions", 0.0))
    recompute_penalty = reward_config.recompute_token_penalty * (
        float(step_result.get("recompute_tokens_added", 0.0)) / 64.0
    )
    wasted_penalty = reward_config.wasted_kv_penalty * (float(step_result.get("reclaimed_kv_tokens", 0.0)) / 64.0)
    fast_admissions = float(step_result.get("new_admissions_fast", 0.0))
    normal_admissions = float(step_result.get("new_admissions_normal", 0.0))
    risky_admissions = float(step_result.get("new_admissions_risky", 0.0))
    fast_first_tokens = float(step_result.get("first_tokens_fast", 0.0))
    new_admissions = float(step_result.get("new_admissions", 0.0))
    phase_mode = str(step_result.get("phase_mode", "mixed"))
    budget_profile = str(step_result.get("scheduler_budget_profile", "balanced"))
    new_admission_limit = float(step_result.get("scheduler_new_admission_limit", 1.0))
    active_decode_after = float(step_result.get("active_decode_after", 0.0))
    projected_real_util = float(step_result.get("projected_real_kv_util_after_new_admissions", kv_util))
    projected_shadow_util = float(step_result.get("projected_shadow_kv_util_after_new_admissions", kv_util))
    projected_real_pressure = 0.0
    if reward_config.projected_kv_admission_penalty_start_util < 1.0:
        projected_real_pressure = max(
            0.0,
            projected_real_util - reward_config.projected_kv_admission_penalty_start_util,
        ) / max(1.0 - reward_config.projected_kv_admission_penalty_start_util, 1e-6)
    projected_shadow_pressure = 0.0
    if reward_config.projected_shadow_admission_penalty_start_util < 1.0:
        projected_shadow_pressure = max(
            0.0,
            projected_shadow_util - reward_config.projected_shadow_admission_penalty_start_util,
        ) / max(1.0 - reward_config.projected_shadow_admission_penalty_start_util, 1e-6)
    projected_real_pressure = min(projected_real_pressure, 1.0)
    projected_shadow_pressure = min(projected_shadow_pressure, 1.0)
    admission_pressure = max(kv_pressure, projected_real_pressure, projected_shadow_pressure)
    fast_pending_tail_s = float(step_result.get("oldest_pending_admission_fast_after_ms", 0.0)) / 100.0
    normal_pending_tail_s = float(step_result.get("oldest_pending_admission_normal_after_ms", 0.0)) / 100.0
    pending_short_after = (
        float(step_result.get("pending_unadmitted_fast_after", 0.0))
        + float(step_result.get("pending_unadmitted_normal_after", 0.0))
    )
    short_pending_tail_pressure = min(
        max(
            float(step_result.get("oldest_pending_admission_fast_after_ms", 0.0)),
            float(step_result.get("oldest_pending_admission_normal_after_ms", 0.0)),
        )
        / 1000.0,
        1.0,
    )
    admission_stalled = (
        pending_short_after > 0.0
        and float(step_result.get("scheduler_new_admission_limit", 1.0)) <= 0.0
        and float(step_result.get("first_token_pending_after", 0.0)) <= 0.0
    )
    short_admission_stall_penalty = (
        reward_config.short_admission_stall_penalty
        * (0.25 + 0.75 * short_pending_tail_pressure)
        if admission_stalled
        else 0.0
    )
    high_kv_prefill_shaping = kv_pressure * (
        reward_config.high_kv_fast_admission_reward * fast_admissions
        + reward_config.high_kv_normal_admission_reward * normal_admissions
        - reward_config.high_kv_new_admission_penalty * new_admissions
        - reward_config.high_kv_risky_admission_penalty * risky_admissions
        - reward_config.high_kv_fast_pending_tail_penalty * fast_pending_tail_s
        - reward_config.high_kv_normal_pending_tail_penalty * normal_pending_tail_s
        - (reward_config.high_kv_admission_stall_penalty if admission_stalled else 0.0)
    )
    high_kv_control_shaping = admission_pressure * (
        (
            reward_config.high_kv_zero_admission_reward
            if active_decode_after > 0.0 and new_admissions <= 0.0 and new_admission_limit <= 0.0
            else 0.0
        )
        - reward_config.high_kv_prefill_priority_penalty
        * (1.0 if phase_mode == PhaseMode.PREFILL_PRIORITY.value else 0.0)
        - reward_config.high_kv_prefill_work_penalty * (prefill_work / 64.0)
    )
    projected_admission_penalty = new_admissions * (
        reward_config.projected_kv_admission_penalty * projected_real_pressure
        + reward_config.projected_shadow_admission_penalty * projected_shadow_pressure
    )
    high_kv_decode_shaping = (
        kv_pressure
        * (
            reward_config.high_kv_fast_first_token_reward * fast_first_tokens
            + reward_config.high_kv_decode_token_reward * (decode_work / 64.0)
            + (
                reward_config.high_kv_decode_drain_reward
                if active_decode_after > 0.0
                and (
                    phase_mode == PhaseMode.DECODE_PRIORITY.value
                    or budget_profile in {"decode-drain", "decode-strict"}
                )
                else 0.0
            )
        )
    )
    first_token_pending_after = float(step_result.get("first_token_pending_after", 0.0))
    first_token_pending_age_s = float(step_result.get("oldest_first_token_pending_after_ms", 0.0)) / 1000.0
    first_token_age_quadratic_penalty = (
        reward_config.first_token_pending_age_quadratic_penalty
        * first_token_pending_after
        * first_token_pending_age_s
        * first_token_pending_age_s
    )
    risky_first_token_backlog_penalty = 0.0
    if risky_admissions > 0.0 and first_token_pending_after > 0.0:
        backlog_fraction = min(first_token_pending_after / max(config.decode_parallelism or config.max_decode_batch, 1), 1.0)
        risky_first_token_backlog_penalty = (
            reward_config.risky_first_token_backlog_penalty
            * risky_admissions
            * backlog_fraction
            * (1.0 + min(first_token_pending_age_s, 10.0))
        )
    action_safety_projection_penalty = (
        reward_config.action_safety_projection_penalty
        if bool(step_result.get("action_safety_triggered", 0.0))
        else 0.0
    )
    step_costs = compute_step_costs(step_result, config)

    prefill_reward = (
        reward_config.new_admission_reward * float(step_result.get("new_admissions", 0.0))
        + high_kv_prefill_shaping
        + high_kv_control_shaping
        - projected_admission_penalty
        - short_admission_stall_penalty
        - risky_first_token_backlog_penalty
        - 0.55 * action_safety_projection_penalty
        - reward_config.prefill_liveness_violation_penalty * step_costs["prefill_liveness_violation"]
        - 0.55 * reward_config.unsafe_projection_cost_penalty * step_costs["unsafe_projection_cost"]
        - reward_config.admission_wait_penalty * (float(step_result.get("admission_wait_sum_ms", 0.0)) / 100.0)
        - reward_config.pending_admission_penalty * float(step_result.get("pending_unadmitted_after", 0.0))
        - reward_config.pending_admission_tail_penalty
        * (float(step_result.get("oldest_pending_admission_after_ms", 0.0)) / 100.0)
        - reward_config.prefill_padding_penalty * (float(step_result.get("prefill_padding_tokens", 0.0)) / 64.0)
        - prefill_share * duration_penalty
        - prefill_share * evictions_penalty
        - 0.7 * recompute_penalty
        - 0.7 * wasted_penalty
    )
    decode_reward = (
        reward_config.first_token_reward * float(step_result.get("first_tokens_emitted", 0.0))
        + high_kv_decode_shaping
        + reward_config.completion_reward * float(step_result.get("completed_requests", 0.0))
        - reward_config.ttft_penalty * (float(step_result.get("ttft_sum_ms", 0.0)) / 100.0)
        - reward_config.e2e_penalty * (float(step_result.get("e2e_sum_ms", 0.0)) / 100.0)
        - reward_config.first_token_pending_penalty * first_token_pending_after
        - reward_config.first_token_pending_tail_penalty
        * (float(step_result.get("oldest_first_token_pending_after_ms", 0.0)) / 100.0)
        - first_token_age_quadratic_penalty
        - reward_config.first_token_backlog_violation_penalty * step_costs["first_token_backlog_violation"]
        - reward_config.kv_penalty * max(0.0, kv_util - reward_config.kv_penalty_start_util)
        - 0.45 * action_safety_projection_penalty
        - 0.45 * reward_config.unsafe_projection_cost_penalty * step_costs["unsafe_projection_cost"]
        - decode_share * duration_penalty
        - decode_share * evictions_penalty
        - 0.3 * recompute_penalty
        - 0.3 * wasted_penalty
    )
    executed_prefill_reward = prefill_reward
    executed_decode_reward = decode_reward
    actor_projection_debit = actor_projection_debit_for_step(step_result, reward_config)
    if actor_projection_debit > 0.0:
        prefill_debit_share, decode_debit_share = projection_debit_responsibility(step_result)
        prefill_reward -= actor_projection_debit * prefill_debit_share
        decode_reward -= actor_projection_debit * decode_debit_share
    completion_component = (
        reward_config.first_token_reward * float(step_result.get("first_tokens_emitted", 0.0))
        + reward_config.completion_reward * float(step_result.get("completed_requests", 0.0))
    )
    admission_component = reward_config.new_admission_reward * float(step_result.get("new_admissions", 0.0))
    latency_component = -(
        duration_penalty
        + reward_config.ttft_penalty * (float(step_result.get("ttft_sum_ms", 0.0)) / 100.0)
        + reward_config.e2e_penalty * (float(step_result.get("e2e_sum_ms", 0.0)) / 100.0)
    )
    queue_component = -(
        reward_config.admission_wait_penalty * (float(step_result.get("admission_wait_sum_ms", 0.0)) / 100.0)
        + reward_config.pending_admission_penalty * float(step_result.get("pending_unadmitted_after", 0.0))
        + reward_config.pending_admission_tail_penalty
        * (float(step_result.get("oldest_pending_admission_after_ms", 0.0)) / 100.0)
        + reward_config.first_token_pending_penalty * first_token_pending_after
        + reward_config.first_token_pending_tail_penalty
        * (float(step_result.get("oldest_first_token_pending_after_ms", 0.0)) / 100.0)
        + first_token_age_quadratic_penalty
        + short_admission_stall_penalty
    )
    kv_safety_component = (
        high_kv_prefill_shaping
        + high_kv_control_shaping
        + high_kv_decode_shaping
        - projected_admission_penalty
        - reward_config.kv_penalty * max(0.0, kv_util - reward_config.kv_penalty_start_util)
        - reward_config.prefill_padding_penalty * (float(step_result.get("prefill_padding_tokens", 0.0)) / 64.0)
        - evictions_penalty
        - recompute_penalty
        - wasted_penalty
    )
    projection_penalty_component = -(
        action_safety_projection_penalty
        + reward_config.unsafe_projection_cost_penalty * step_costs["unsafe_projection_cost"]
        + actor_projection_debit
    )
    liveness_penalty_component = -(
        reward_config.prefill_liveness_violation_penalty * step_costs["prefill_liveness_violation"]
        + reward_config.first_token_backlog_violation_penalty * step_costs["first_token_backlog_violation"]
        + risky_first_token_backlog_penalty
    )
    component_sum = (
        completion_component
        + admission_component
        + latency_component
        + kv_safety_component
        + projection_penalty_component
        + liveness_penalty_component
        + queue_component
    )
    total_reward = prefill_reward + decode_reward
    executed_reward = executed_prefill_reward + executed_decode_reward
    return {
        "prefill_reward": prefill_reward,
        "decode_reward": decode_reward,
        "total_reward": total_reward,
        "executed_reward": executed_reward,
        "actor_projection_debit": actor_projection_debit,
        "actor_reward": total_reward,
        "reward_completion": completion_component,
        "reward_admission": admission_component,
        "reward_latency": latency_component,
        "reward_kv_safety": kv_safety_component,
        "reward_projection_penalty": projection_penalty_component,
        "reward_liveness_penalty": liveness_penalty_component,
        "reward_queue_penalty": queue_component,
        "reward_deadlock_penalty": 0.0,
        "reward_other": total_reward - component_sum,
    }


def feedback_event_type(step_result: Dict[str, float]) -> str | None:
    if float(step_result.get("completed_requests", 0.0)) > 0.0:
        return "decode-complete"
    if float(step_result.get("first_tokens_emitted", 0.0)) > 0.0:
        return "decode-first-token"
    if float(step_result.get("prefill_completed_requests", 0.0)) > 0.0:
        return "prefill-complete"
    if float(step_result.get("evictions", 0.0)) > 0.0 or float(step_result.get("recompute_tokens_added", 0.0)) > 0.0:
        return "kv-update"
    if float(step_result.get("allocation_failures", 0.0)) > 0.0:
        return "allocation-failure"
    return None


def parse_request_id_list(value: object) -> List[int]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        raw_items = str(value).replace(";", ",").split(",")
    request_ids: List[int] = []
    for item in raw_items:
        if item is None or item == "":
            continue
        try:
            request_ids.append(int(float(str(item).strip())))
        except ValueError:
            continue
    return request_ids


def join_request_id_list(values: Iterable[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def union_request_id_values(*values: object) -> str:
    request_ids: set[int] = set()
    for value in values:
        request_ids.update(parse_request_id_list(value))
    return join_request_id_list(sorted(request_ids))


def merge_feedback_step_result(
    aggregate: Dict[str, float] | None,
    step_result: Dict[str, float],
) -> Dict[str, float]:
    if aggregate is None:
        merged = dict(step_result)
        event_type = feedback_event_type(step_result)
        if event_type is not None:
            merged["feedback_event_type"] = event_type
        prefill_cf = float(merged.get("counterfactual_prefill_cost_delta", 0.0) or 0.0)
        decode_cf = float(merged.get("counterfactual_decode_cost_delta", 0.0) or 0.0)
        total_cf = prefill_cf + decode_cf
        if total_cf > 0.0:
            merged["counterfactual_prefill_responsibility"] = prefill_cf / total_cf
            merged["counterfactual_decode_responsibility"] = decode_cf / total_cf
        return merged

    sum_keys = {
        "duration_ms",
        "prefill_tokens",
        "prefill_padding_tokens",
        "padded_prefill_tokens",
        "decode_tokens",
        "first_tokens_emitted",
        "completed_requests",
        "prefill_completed_requests",
        "completed_output_tokens",
        "new_admissions",
        "new_admissions_fast",
        "new_admissions_normal",
        "new_admissions_risky",
        "resumed_prefills",
        "evictions",
        "reclaimed_kv_tokens",
        "recompute_tokens_added",
        "allocation_failures",
        "actual_prefill_progress_tokens",
        "actual_decode_progress_tokens",
        "projected_new_admission_prompt_tokens",
        "projected_new_admission_decode_shadow_tokens",
        "projected_new_admission_shadow_tokens",
        "first_tokens_fast",
        "first_tokens_normal",
        "first_tokens_risky",
        "admission_wait_sum_ms",
        "ttft_sum_ms",
        "e2e_sum_ms",
        "counterfactual_prefill_cost_delta",
        "counterfactual_decode_cost_delta",
    }
    max_keys = {
        "kv_peak",
        "oldest_pending_admission_after_ms",
        "oldest_pending_admission_fast_after_ms",
        "oldest_pending_admission_normal_after_ms",
        "oldest_pending_admission_risky_after_ms",
        "oldest_first_token_pending_after_ms",
        "projected_real_kv_util_after_new_admissions",
        "projected_shadow_kv_util_after_new_admissions",
    }
    latest_keys = {
        "phase_mode",
        "kv_after",
        "logical_kv_after",
        "active_decode_after",
        "prefill_waiting_after",
        "pending_unadmitted_after",
        "pending_unadmitted_fast_after",
        "pending_unadmitted_normal_after",
        "pending_unadmitted_risky_after",
        "mean_pending_admission_after_ms",
        "first_token_pending_after",
        "scheduler_admission_profile",
        "requested_scheduler_admission_profile",
        "scheduler_new_admission_limit",
        "requested_scheduler_new_admission_limit",
        "scheduler_shadow_quantile",
        "requested_scheduler_shadow_quantile",
        "scheduler_budget_profile",
        "requested_scheduler_budget_profile",
        "scheduler_risky_admission_cap",
        "requested_scheduler_risky_admission_cap",
        "scheduler_prefill_safety_filter_enabled",
        "scheduler_decode_safety_filter_enabled",
        "requested_scheduler_prefill_safety_filter_enabled",
        "requested_scheduler_decode_safety_filter_enabled",
        "requested_phase_mode",
    }
    first_keys = {"kv_before"}

    merged = dict(aggregate)
    for key, value in step_result.items():
        if key in sum_keys:
            merged[key] = float(merged.get(key, 0.0)) + float(value)
        elif key in REQUEST_ID_LIST_KEYS:
            merged[key] = union_request_id_values(merged.get(key, ""), value)
        elif key in max_keys:
            merged[key] = max(float(merged.get(key, 0.0)), float(value))
        elif key in latest_keys:
            merged[key] = value
        elif key in first_keys:
            merged.setdefault(key, value)
        elif key == "action_safety_triggered":
            merged[key] = float(merged.get(key, 0.0)) or float(value)
        elif key == "action_safety_reasons":
            reasons = [item for item in str(merged.get(key, "")).split(",") if item]
            reasons.extend(item for item in str(value).split(",") if item)
            merged[key] = ",".join(sorted(set(reasons)))
        else:
            merged[key] = value

    event_type = feedback_event_type(step_result)
    if event_type is not None:
        merged["feedback_event_type"] = event_type
    else:
        merged.setdefault("feedback_event_type", aggregate.get("feedback_event_type"))
    prefill_cf = float(merged.get("counterfactual_prefill_cost_delta", 0.0) or 0.0)
    decode_cf = float(merged.get("counterfactual_decode_cost_delta", 0.0) or 0.0)
    total_cf = prefill_cf + decode_cf
    if total_cf > 0.0:
        merged["counterfactual_prefill_responsibility"] = prefill_cf / total_cf
        merged["counterfactual_decode_responsibility"] = decode_cf / total_cf
    return merged


@dataclass(slots=True)
class TrainingConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.98
    gae_lambda: float = 0.95
    ppo_clip: float = 0.20
    ppo_epochs: int = 4
    entropy_bonus: float = 0.010
    value_coef: float = 0.50
    grad_clip: float = 1.0
    hidden_dim: int = 192
    hidden_depth: int = 3
    advantage_normalization: bool = True
    observation_normalization: bool = True
    observation_momentum: float = 0.95
    feedback_mode: str = "event"
    exploration_temperature: float = 0.85
    exploration_top_k: int = 3
    exploration_greedy_prob: float = 0.15
    unsafe_state_temperature: float = 0.35
    unsafe_state_top_k: int = 1
    unsafe_state_greedy_prob: float = 1.0
    unsafe_state_heuristic_fallback: bool = True
    value_normalization: bool = True
    shared_critic: bool = False
    no_prefill_to_decode_condition: bool = False
    no_prev_decode_to_prefill: bool = False
    router_boundary_mode: str = "current"
    router_update_role: str = "joint"
    serialized_joint_critic: bool = True
    autonomous_decision_timing: bool = True
    phased_curriculum: bool = False
    curriculum_warmup_fraction: float = 0.25
    curriculum_mid_fraction: float = 0.35
    projection_imitation_coef: float = 0.0
    targeted_projection_bc_coef: float = 0.0
    targeted_projection_bc_reasons: str = (
        "live_decode_prefill_priority_guard,"
        "phase_decode_backlog_guard,"
        "kv_recovery_phase_guard,"
        "kv_recovery_admission_guard,"
        "kv_recovery_risky_guard,"
        "kv_recovery_budget_guard,"
        "kv_drain_phase_guard,"
        "kv_drain_admission_guard,"
        "kv_drain_risky_guard,"
        "kv_drain_budget_guard,"
        "budget_decode_guard,"
        "budget_kv_guard,"
        "admission_profile_guard,"
        "new_admission_limit_guard,"
        "shadow_quantile_guard,"
        "prefill_queue_guard,"
        "prefill_queue_admission_guard,"
        "prefill_liveness_phase_guard,"
        "prefill_liveness_budget_guard,"
        "prefill_liveness_admission_guard,"
        "prefill_liveness_risky_cap_floor,"
        "short_admission_budget_floor,"
        "short_admission_phase_floor,"
        "short_admission_limit_floor,"
        "decode_guard_risky_cap,"
        "first_token_risky_cap"
    )
    targeted_projection_bc_reason_coefs: str = ""
    targeted_projection_bc_group_by_reason: bool = True
    targeted_projection_bc_min_weight: float = 0.25
    late_update_start_fraction: float = 1.10
    late_learning_rate_scale: float = 1.0
    late_freeze_prefill_trunk: bool = False
    late_freeze_admission_heads: bool = False
    teacher_stage_a_fraction: float = 0.15
    teacher_stage_b_fraction: float = 0.70
    stage_a_imitation_coef: float = 1.0
    stage_c_learning_rate_scale: float = 0.20
    stage_c_freeze_admission_heads: bool = False
    sequential_router_update: bool = False
    first_valid_freeze_actor: bool = False
    anchor_kl_coef: float = 0.0
    strong_action_mask: bool = True
    recovery_joint_profile_enabled: bool = False
    recovery_joint_profile_min_kv_util: float = 0.65
    history_critic_window: int = 0


def split_router_training_configs(config: TrainingConfig) -> Tuple[TrainingConfig, TrainingConfig]:
    """Pin split async router agents to the heads they own."""
    return (
        replace(config, router_update_role="prefill"),
        replace(config, router_update_role="decode"),
    )


TARGETED_PROJECTION_REASON_HEADS: Dict[str, Tuple[int, ...]] = {
    "live_decode_prefill_priority_guard": (0,),
    "phase_decode_backlog_guard": (0,),
    "kv_recovery_phase_guard": (0,),
    "kv_recovery_admission_guard": (2,),
    "kv_recovery_risky_guard": (5,),
    "kv_recovery_budget_guard": (4,),
    "kv_drain_phase_guard": (0,),
    "kv_drain_admission_guard": (2,),
    "kv_drain_risky_guard": (5,),
    "kv_drain_budget_guard": (4,),
    "budget_decode_guard": (4,),
    "budget_kv_guard": (4,),
    "admission_profile_guard": (1,),
    "new_admission_limit_guard": (2,),
    "shadow_quantile_guard": (3,),
    "prefill_queue_guard": (1,),
    "prefill_queue_admission_guard": (2,),
    "prefill_liveness_phase_guard": (0,),
    "prefill_liveness_budget_guard": (4,),
    "prefill_liveness_admission_guard": (2,),
    "prefill_liveness_risky_cap_floor": (5,),
    "short_admission_budget_floor": (4,),
    "short_admission_phase_floor": (0,),
    "short_admission_limit_floor": (2,),
    "decode_guard_risky_cap": (5,),
    "first_token_risky_cap": (5,),
}

PREFILL_TARGETED_PROJECTION_HEADS = {1, 2, 3, 5, 6}
DECODE_TARGETED_PROJECTION_HEADS = {0, 4, 7}


def targeted_projection_head_indices(
    reasons: Iterable[str],
    configured_reasons: Iterable[str],
) -> Tuple[int, ...]:
    return tuple(head_index for head_index, _reason in targeted_projection_head_reason_pairs(reasons, configured_reasons))


def targeted_projection_head_reason_pairs(
    reasons: Iterable[str],
    configured_reasons: Iterable[str],
) -> Tuple[Tuple[int, str], ...]:
    active_reasons = set(reasons)
    pairs: List[Tuple[int, str]] = []
    seen_heads: set[int] = set()
    for reason in configured_reasons:
        if reason not in active_reasons:
            continue
        for head_index in TARGETED_PROJECTION_REASON_HEADS.get(reason, ()):
            if head_index not in seen_heads:
                pairs.append((head_index, reason))
                seen_heads.add(head_index)
    return tuple(pairs)


def parse_targeted_projection_reason_coefs(spec: str) -> Dict[str, float]:
    coefs: Dict[str, float] = {}
    if not spec:
        return coefs
    for raw_item in spec.replace(";", ",").split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" in item:
            reason, value = item.split("=", 1)
        elif ":" in item:
            reason, value = item.split(":", 1)
        else:
            raise ValueError(f"expected reason=coef in targeted projection BC override: {item!r}")
        reason = reason.strip()
        if not reason:
            raise ValueError(f"empty reason in targeted projection BC override: {item!r}")
        coefs[reason] = max(float(value), 0.0)
    return coefs


def grouped_targeted_projection_bc_loss(
    grouped_losses: Dict[Tuple[str, int], Sequence[torch.Tensor]],
    reason_coefs: Dict[str, float],
) -> Optional[torch.Tensor]:
    terms: List[torch.Tensor] = []
    for (reason, _head_index), losses in grouped_losses.items():
        if not losses:
            continue
        reason_coef = max(float(reason_coefs.get(reason, 1.0)), 0.0)
        terms.append(reason_coef * torch.stack(list(losses)).mean())
    if not terms:
        return None
    return torch.stack(terms).mean()


def targeted_projection_bc_error_weight(
    head_index: int,
    prefill_responsibility: float,
    decode_responsibility: float,
    min_weight: float = 0.25,
) -> float:
    """Weight guard-imitation loss toward the router attributed to the error."""
    min_weight = min(max(float(min_weight), 0.0), 1.0)
    prefill = min(max(float(prefill_responsibility), 0.0), 1.0)
    decode = min(max(float(decode_responsibility), 0.0), 1.0)
    if prefill + decode <= 0.0:
        return 1.0
    if head_index in PREFILL_TARGETED_PROJECTION_HEADS:
        responsibility = prefill
    elif head_index in DECODE_TARGETED_PROJECTION_HEADS:
        responsibility = decode
    else:
        responsibility = max(prefill, decode)
    return min_weight + (1.0 - min_weight) * min(max(float(responsibility), 0.0), 1.0)


def action_to_dict(action: SchedulerAction) -> Dict[str, object]:
    return {
        "phase_preference": action.phase_preference.value if action.phase_preference is not None else None,
        "admission_profile": action.admission_profile,
        "new_admission_limit": action.new_admission_limit,
        "shadow_quantile": action.shadow_quantile,
        "budget_profile": action.budget_profile,
        "risky_admission_cap": action.risky_admission_cap,
        "prefill_safety_filter_enabled": action.prefill_safety_filter_enabled,
        "decode_safety_filter_enabled": action.decode_safety_filter_enabled,
    }


def allowed_indices(values: Sequence[object], allowed_values: Sequence[object]) -> List[int]:
    allowed = set(allowed_values)
    indices = [index for index, value in enumerate(values) if value in allowed]
    return indices or list(range(len(values)))


def masked_logits(logits: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    mask_value = torch.finfo(logits.dtype).min
    masked = torch.full_like(logits, mask_value)
    masked[list(indices)] = logits[list(indices)]
    return masked


def mask_indices(
    mask: Optional[Dict[str, Sequence[int]]],
    key: str,
    default_count: int,
    base_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    base = list(base_indices) if base_indices is not None else list(range(default_count))
    if mask is None or key not in mask:
        return base or list(range(default_count))
    allowed = set(int(index) for index in mask.get(key, ()))
    filtered = [index for index in base if index in allowed]
    return filtered or base or list(range(default_count))


def categorical_entropy_from_probs(probs: Sequence[float]) -> float:
    values = np.asarray(probs, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 0.0
    return float(-np.sum(values * np.log(values)))


def categorical_kl_from_logits(
    logits: torch.Tensor,
    anchor_logits: torch.Tensor,
    allowed: Optional[Sequence[int]] = None,
) -> float:
    if allowed is None:
        allowed = list(range(int(logits.shape[-1])))
    current_dist = torch.distributions.Categorical(logits=masked_logits(logits, allowed))
    anchor_dist = torch.distributions.Categorical(logits=masked_logits(anchor_logits, allowed))
    return float(torch.distributions.kl_divergence(current_dist, anchor_dist).detach().cpu().item())


def choose_index_from_logits(
    logits: torch.Tensor,
    greedy: bool,
    *,
    temperature: float = 1.0,
    greedy_probability: float = 0.0,
    top_k: int = 0,
) -> Tuple[int, np.ndarray]:
    working_logits = logits
    if not greedy:
        if top_k > 0 and top_k < int(logits.shape[-1]):
            _, top_indices = torch.topk(logits, k=top_k)
            working_logits = masked_logits(logits, top_indices.tolist())
        safe_temperature = max(float(temperature), 0.05)
        working_logits = working_logits / safe_temperature
    probs_tensor = torch.softmax(working_logits, dim=-1)
    if greedy:
        index = int(torch.argmax(probs_tensor).item())
    else:
        if greedy_probability > 0.0 and float(torch.rand(1).item()) < float(greedy_probability):
            index = int(torch.argmax(probs_tensor).item())
        else:
            index = int(torch.distributions.Categorical(probs=probs_tensor).sample().item())
    return index, probs_tensor.detach().cpu().numpy()


def build_recovery_joint_profiles() -> Tuple[RecoveryJointProfile, ...]:
    profiles: List[RecoveryJointProfile] = []
    for phase_index, phase_label in enumerate(PHASE_CHOICES):
        for admission_index, admission_label in enumerate(ADMISSION_CHOICES):
            if admission_label not in allowed_admission_profiles(phase_label, ADMISSION_CHOICES):
                continue
            for budget_profile_index, budget_profile in enumerate(BUDGET_PROFILE_CHOICES):
                if budget_profile not in allowed_budget_profiles(
                    phase_label,
                    admission_label,
                    BUDGET_PROFILE_CHOICES,
                ):
                    continue
                allowed_limits = allowed_new_admission_limits(
                    phase_label,
                    admission_label,
                    budget_profile,
                    NEW_ADMISSION_LIMIT_CHOICES,
                )
                allowed_risky_caps = allowed_risky_admission_caps(
                    phase_label,
                    admission_label,
                    budget_profile,
                    RISKY_ADMISSION_CAP_CHOICES,
                )
                for admission_limit_index, admission_limit in enumerate(NEW_ADMISSION_LIMIT_CHOICES):
                    if int(admission_limit) not in allowed_limits:
                        continue
                    for risky_index, risky_cap in enumerate(RISKY_ADMISSION_CAP_CHOICES):
                        if int(risky_cap) not in allowed_risky_caps:
                            continue
                        profiles.append(
                            (
                                int(phase_index),
                                int(admission_index),
                                int(admission_limit_index),
                                int(budget_profile_index),
                                int(risky_index),
                            )
                        )
    if not profiles:
        raise ValueError("recovery joint profile action space is empty")
    return tuple(profiles)


def recovery_joint_profile_from_heads(
    heads: Tuple[int, int, int, int, int, int, int, int],
) -> RecoveryJointProfile:
    return (
        int(heads[0]),
        int(heads[1]),
        int(heads[2]),
        int(heads[4]),
        int(heads[5]),
    )


def recovery_joint_profile_index(profile: RecoveryJointProfile) -> int:
    profile_map = {
        candidate: index
        for index, candidate in enumerate(build_recovery_joint_profiles())
    }
    if profile not in profile_map:
        raise ValueError(f"illegal recovery joint profile: {profile}")
    return profile_map[profile]


def recovery_joint_profile_allowed_indices(
    action_mask: Optional[Dict[str, Sequence[int]]] = None,
) -> List[int]:
    profiles = build_recovery_joint_profiles()
    phase_allowed = set(mask_indices(action_mask, "phase", len(PHASE_CHOICES)))
    admission_allowed = set(mask_indices(action_mask, "admission", len(ADMISSION_CHOICES)))
    limit_allowed = set(mask_indices(action_mask, "admission_limit", len(NEW_ADMISSION_LIMIT_CHOICES)))
    budget_allowed = set(mask_indices(action_mask, "budget_profile", len(BUDGET_PROFILE_CHOICES)))
    risky_allowed = set(mask_indices(action_mask, "risky_admission_cap", len(RISKY_ADMISSION_CAP_CHOICES)))
    allowed = [
        index
        for index, profile in enumerate(profiles)
        if profile[0] in phase_allowed
        and profile[1] in admission_allowed
        and profile[2] in limit_allowed
        and profile[3] in budget_allowed
        and profile[4] in risky_allowed
    ]
    return allowed or list(range(len(profiles)))


def project_action_head_indices(
    phase_index: int,
    admission_index: int,
    admission_limit_index: int,
    quantile_index: int,
    budget_profile_index: int,
    risky_admission_cap_index: int,
    prefill_safety_filter_index: int = 1,
    decode_safety_filter_index: int = 1,
) -> Tuple[int, int, int, int, int, int, int, int]:
    raw_action = SchedulerAction(
        phase_preference=None if PHASE_CHOICES[phase_index] == "heuristic" else PhaseMode(PHASE_CHOICES[phase_index]),
        admission_profile=ADMISSION_CHOICES[admission_index],
        new_admission_limit=NEW_ADMISSION_LIMIT_CHOICES[admission_limit_index],
        shadow_quantile=QUANTILE_CHOICES[quantile_index],
        budget_profile=BUDGET_PROFILE_CHOICES[budget_profile_index],
        risky_admission_cap=RISKY_ADMISSION_CAP_CHOICES[risky_admission_cap_index],
        prefill_safety_filter_enabled=SAFETY_FILTER_CHOICES[prefill_safety_filter_index],
        decode_safety_filter_enabled=SAFETY_FILTER_CHOICES[decode_safety_filter_index],
    )
    projected = project_action_to_stable_space(
        raw_action,
        admission_choices=ADMISSION_CHOICES,
        new_admission_limit_choices=NEW_ADMISSION_LIMIT_CHOICES,
        quantile_choices=QUANTILE_CHOICES,
        budget_profile_choices=BUDGET_PROFILE_CHOICES,
        risky_admission_cap_choices=RISKY_ADMISSION_CAP_CHOICES,
    )
    projected_phase_label = phase_label_from_preference(projected.phase_preference)
    return (
        PHASE_CHOICES.index(projected_phase_label),
        ADMISSION_CHOICES.index(str(projected.admission_profile)),
        NEW_ADMISSION_LIMIT_CHOICES.index(_prefill_limit_value(projected.new_admission_limit)),
        QUANTILE_CHOICES.index(nearest_quantile_choice(float(projected.shadow_quantile or default_shadow_quantile_choice()))),
        BUDGET_PROFILE_CHOICES.index(str(projected.budget_profile)),
        RISKY_ADMISSION_CAP_CHOICES.index(
            _risky_cap_value(projected.risky_admission_cap)
        ),
        SAFETY_FILTER_CHOICES.index(
            bool(projected.prefill_safety_filter_enabled)
            if projected.prefill_safety_filter_enabled is not None
            else True
        ),
        SAFETY_FILTER_CHOICES.index(
            bool(projected.decode_safety_filter_enabled)
            if projected.decode_safety_filter_enabled is not None
            else True
        ),
    )


def masked_action_head_probs(
    outputs: Dict[str, torch.Tensor],
    row_index: int,
    *,
    greedy: bool,
    temperature: float = 1.0,
    greedy_probability: float = 0.0,
    top_k: int = 0,
) -> Tuple[Tuple[int, int, int, int, int, int, int, int], Dict[str, np.ndarray]]:
    phase_logits = outputs["phase_logits"][row_index]
    phase_index, phase_probs = choose_index_from_logits(
        phase_logits,
        greedy=greedy,
        temperature=temperature,
        greedy_probability=greedy_probability,
        top_k=top_k,
    )
    phase_label = PHASE_CHOICES[phase_index]

    admission_allowed = allowed_indices(
        ADMISSION_CHOICES,
        allowed_admission_profiles(phase_label, ADMISSION_CHOICES),
    )
    admission_logits = masked_logits(outputs["admission_logits"][row_index], admission_allowed)
    admission_index, admission_probs = choose_index_from_logits(
        admission_logits,
        greedy=greedy,
        temperature=temperature,
        greedy_probability=greedy_probability,
        top_k=top_k,
    )
    admission_label = ADMISSION_CHOICES[admission_index]

    budget_allowed = allowed_indices(
        BUDGET_PROFILE_CHOICES,
        allowed_budget_profiles(phase_label, admission_label, BUDGET_PROFILE_CHOICES),
    )
    budget_logits = masked_logits(outputs["budget_profile_logits"][row_index], budget_allowed)
    budget_profile_index, budget_profile_probs = choose_index_from_logits(
        budget_logits,
        greedy=greedy,
        temperature=temperature,
        greedy_probability=greedy_probability,
        top_k=top_k,
    )
    budget_profile = BUDGET_PROFILE_CHOICES[budget_profile_index]

    admission_limit_allowed = allowed_indices(
        NEW_ADMISSION_LIMIT_CHOICES,
        allowed_new_admission_limits(
            phase_label,
            admission_label,
            budget_profile,
            NEW_ADMISSION_LIMIT_CHOICES,
        ),
    )
    admission_limit_logits = masked_logits(outputs["admission_limit_logits"][row_index], admission_limit_allowed)
    admission_limit_index, admission_limit_probs = choose_index_from_logits(
        admission_limit_logits,
        greedy=greedy,
        temperature=temperature,
        greedy_probability=greedy_probability,
        top_k=top_k,
    )

    quantile_allowed = allowed_indices(
        QUANTILE_CHOICES,
        allowed_shadow_quantiles(
            phase_label,
            admission_label,
            budget_profile,
            QUANTILE_CHOICES,
        ),
    )
    quantile_logits = masked_logits(outputs["quantile_logits"][row_index], quantile_allowed)
    quantile_index, quantile_probs = choose_index_from_logits(
        quantile_logits,
        greedy=greedy,
        temperature=temperature,
        greedy_probability=greedy_probability,
        top_k=top_k,
    )

    risky_cap_allowed = allowed_indices(
        RISKY_ADMISSION_CAP_CHOICES,
        allowed_risky_admission_caps(
            phase_label,
            admission_label,
            budget_profile,
            RISKY_ADMISSION_CAP_CHOICES,
        ),
    )
    risky_logits = masked_logits(outputs["risky_admission_cap_logits"][row_index], risky_cap_allowed)
    risky_admission_cap_index, risky_admission_cap_probs = choose_index_from_logits(
        risky_logits,
        greedy=greedy,
        temperature=temperature,
        greedy_probability=greedy_probability,
        top_k=top_k,
    )
    prefill_safety_filter_index, prefill_safety_filter_probs = choose_index_from_logits(
        outputs["prefill_safety_filter_logits"][row_index],
        greedy=greedy,
        temperature=temperature,
        greedy_probability=greedy_probability,
        top_k=top_k,
    )
    decode_safety_filter_index, decode_safety_filter_probs = choose_index_from_logits(
        outputs["decode_safety_filter_logits"][row_index],
        greedy=greedy,
        temperature=temperature,
        greedy_probability=greedy_probability,
        top_k=top_k,
    )

    projected_heads = project_action_head_indices(
        phase_index,
        admission_index,
        admission_limit_index,
        quantile_index,
        budget_profile_index,
        risky_admission_cap_index,
        prefill_safety_filter_index,
        decode_safety_filter_index,
    )
    return projected_heads, {
        "phase_probs": phase_probs,
        "admission_probs": admission_probs,
        "admission_limit_probs": admission_limit_probs,
        "quantile_probs": quantile_probs,
        "budget_profile_probs": budget_profile_probs,
        "risky_admission_cap_probs": risky_admission_cap_probs,
        "prefill_safety_filter_probs": prefill_safety_filter_probs,
        "decode_safety_filter_probs": decode_safety_filter_probs,
        "phase_logits": phase_logits,
        "admission_logits": admission_logits,
        "admission_limit_logits": admission_limit_logits,
        "quantile_logits": quantile_logits,
        "budget_profile_logits": budget_logits,
        "risky_admission_cap_logits": risky_logits,
        "prefill_safety_filter_logits": outputs["prefill_safety_filter_logits"][row_index],
        "decode_safety_filter_logits": outputs["decode_safety_filter_logits"][row_index],
    }


def action_from_step_result(step_result: Dict[str, float]) -> SchedulerAction:
    requested_phase = step_result.get("phase_mode")
    phase_label = requested_phase if requested_phase in PHASE_CHOICES else "heuristic"
    return SchedulerAction(
        phase_preference=None if phase_label == "heuristic" else PhaseMode(phase_label),
        admission_profile=str(step_result.get("scheduler_admission_profile", "balanced")),
        new_admission_limit=int(step_result.get("scheduler_new_admission_limit", default_new_admission_limit_choice())),
        shadow_quantile=float(step_result.get("scheduler_shadow_quantile", default_shadow_quantile_choice())),
        budget_profile=str(step_result.get("scheduler_budget_profile", "balanced")),
        risky_admission_cap=int(step_result.get("scheduler_risky_admission_cap", default_risky_admission_cap_choice())),
        prefill_safety_filter_enabled=bool(
            step_result.get("scheduler_prefill_safety_filter_enabled", 1.0)
        ),
        decode_safety_filter_enabled=bool(
            step_result.get("scheduler_decode_safety_filter_enabled", 1.0)
        ),
    )


def requested_action_from_step_result(step_result: Dict[str, float]) -> SchedulerAction:
    phase_label = str(step_result.get("requested_phase_mode", "heuristic"))
    return SchedulerAction(
        phase_preference=None if phase_label == "heuristic" else PhaseMode(phase_label),
        admission_profile=str(step_result.get("requested_scheduler_admission_profile", "balanced")),
        new_admission_limit=int(
            step_result.get("requested_scheduler_new_admission_limit", default_new_admission_limit_choice())
        ),
        shadow_quantile=float(step_result.get("requested_scheduler_shadow_quantile", default_shadow_quantile_choice())),
        budget_profile=str(step_result.get("requested_scheduler_budget_profile", "balanced")),
        risky_admission_cap=int(
            step_result.get("requested_scheduler_risky_admission_cap", default_risky_admission_cap_choice())
        ),
        prefill_safety_filter_enabled=bool(
            step_result.get("requested_scheduler_prefill_safety_filter_enabled", 1.0)
        ),
        decode_safety_filter_enabled=bool(
            step_result.get("requested_scheduler_decode_safety_filter_enabled", 1.0)
        ),
    )


def action_index_from_action(action: SchedulerAction) -> int:
    projected = project_action_to_stable_space(
        action,
        admission_choices=ADMISSION_CHOICES,
        new_admission_limit_choices=NEW_ADMISSION_LIMIT_CHOICES,
        quantile_choices=QUANTILE_CHOICES,
        budget_profile_choices=BUDGET_PROFILE_CHOICES,
        risky_admission_cap_choices=RISKY_ADMISSION_CAP_CHOICES,
    )
    return ACTION_INDEX[action_key(projected, CURRENT_ACTION_SPACE)]


def action_heads_from_action(action: SchedulerAction) -> Tuple[int, int, int, int, int, int, int, int]:
    projected = project_action_to_stable_space(
        action,
        admission_choices=ADMISSION_CHOICES,
        new_admission_limit_choices=NEW_ADMISSION_LIMIT_CHOICES,
        quantile_choices=QUANTILE_CHOICES,
        budget_profile_choices=BUDGET_PROFILE_CHOICES,
        risky_admission_cap_choices=RISKY_ADMISSION_CAP_CHOICES,
    )
    phase_label = phase_label_from_preference(projected.phase_preference)
    return (
        PHASE_CHOICES.index(phase_label),
        ADMISSION_CHOICES.index(str(projected.admission_profile)),
        NEW_ADMISSION_LIMIT_CHOICES.index(_prefill_limit_value(projected.new_admission_limit)),
        QUANTILE_CHOICES.index(nearest_quantile_choice(float(projected.shadow_quantile or default_shadow_quantile_choice()))),
        BUDGET_PROFILE_CHOICES.index(str(projected.budget_profile)),
        RISKY_ADMISSION_CAP_CHOICES.index(_risky_cap_value(projected.risky_admission_cap)),
        SAFETY_FILTER_CHOICES.index(
            bool(projected.prefill_safety_filter_enabled)
            if projected.prefill_safety_filter_enabled is not None
            else True
        ),
        SAFETY_FILTER_CHOICES.index(
            bool(projected.decode_safety_filter_enabled)
            if projected.decode_safety_filter_enabled is not None
            else True
        ),
    )


def prefill_policy_to_dict(policy: PrefillRouterPolicy) -> Dict[str, object]:
    return {
        "admission_profile": policy.admission_profile,
        "new_admission_limit": policy.new_admission_limit,
        "shadow_quantile": policy.shadow_quantile,
        "risky_admission_cap": policy.risky_admission_cap,
        "safety_filter_enabled": policy.safety_filter_enabled,
    }


def decode_policy_to_dict(policy: DecodeRouterPolicy) -> Dict[str, object]:
    return {
        "phase_preference": policy.phase_preference.value if policy.phase_preference is not None else None,
        "budget_profile": policy.budget_profile,
        "safety_filter_enabled": policy.safety_filter_enabled,
    }


ACTION_MASK_HEAD_KEYS = {
    "phase": 0,
    "admission": 1,
    "admission_limit": 2,
    "quantile": 3,
    "budget_profile": 4,
    "risky_admission_cap": 5,
    "prefill_safety_filter": 6,
    "decode_safety_filter": 7,
}
PREFILL_ACTION_MASK_KEYS = (
    "admission",
    "admission_limit",
    "quantile",
    "risky_admission_cap",
    "prefill_safety_filter",
)
DECODE_ACTION_MASK_KEYS = ("phase", "budget_profile", "decode_safety_filter")


def normalized_action_mask(mask: Optional[Dict[str, Sequence[int]]]) -> Dict[str, List[int]]:
    return {
        str(key): sorted({int(index) for index in value})
        for key, value in dict(mask or {}).items()
        if value is not None
    }


def stable_json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def action_mask_hash(mask: Optional[Dict[str, Sequence[int]]], keys: Sequence[str]) -> str:
    normalized = normalized_action_mask(mask)
    subset = {key: normalized[key] for key in keys if key in normalized}
    if not subset:
        return ""
    return hashlib.sha256(stable_json_dumps(subset).encode("utf-8")).hexdigest()[:16]


def router_action_mask_valid(
    action_heads: Sequence[int],
    mask: Optional[Dict[str, Sequence[int]]],
    keys: Sequence[str],
) -> bool:
    normalized = normalized_action_mask(mask)
    for key in keys:
        allowed = normalized.get(key)
        if not allowed:
            continue
        head_index = ACTION_MASK_HEAD_KEYS[key]
        if head_index >= len(action_heads) or int(action_heads[head_index]) not in allowed:
            return False
    return True


def action_snapshot_fields(
    requested_action: SchedulerAction,
    executed_action: SchedulerAction,
    action_mask: Optional[Dict[str, Sequence[int]]],
    *,
    prefill_policy_logprob: float = 0.0,
    decode_policy_logprob: float = 0.0,
    prefill_policy_entropy: float = 0.0,
    decode_policy_entropy: float = 0.0,
    safety_projection_applied: bool = False,
    safety_projection_reasons: object = "",
) -> Dict[str, object]:
    requested_heads = action_heads_from_action(requested_action)
    executed_heads = action_heads_from_action(executed_action)
    prefill_requested_mask_valid = router_action_mask_valid(requested_heads, action_mask, PREFILL_ACTION_MASK_KEYS)
    decode_requested_mask_valid = router_action_mask_valid(requested_heads, action_mask, DECODE_ACTION_MASK_KEYS)
    prefill_executed_mask_valid = router_action_mask_valid(executed_heads, action_mask, PREFILL_ACTION_MASK_KEYS)
    decode_executed_mask_valid = router_action_mask_valid(executed_heads, action_mask, DECODE_ACTION_MASK_KEYS)
    return {
        "prefill_requested_action": prefill_policy_to_dict(requested_action.prefill_policy()),
        "prefill_executed_action": prefill_policy_to_dict(executed_action.prefill_policy()),
        "decode_requested_action": decode_policy_to_dict(requested_action.decode_policy()),
        "decode_executed_action": decode_policy_to_dict(executed_action.decode_policy()),
        "prefill_action_mask_hash": action_mask_hash(action_mask, PREFILL_ACTION_MASK_KEYS),
        "decode_action_mask_hash": action_mask_hash(action_mask, DECODE_ACTION_MASK_KEYS),
        "prefill_mask_valid": prefill_requested_mask_valid,
        "decode_mask_valid": decode_requested_mask_valid,
        "prefill_requested_mask_valid": prefill_requested_mask_valid,
        "decode_requested_mask_valid": decode_requested_mask_valid,
        "prefill_executed_mask_valid": prefill_executed_mask_valid,
        "decode_executed_mask_valid": decode_executed_mask_valid,
        "action_support_pass": bool(prefill_requested_mask_valid and decode_requested_mask_valid),
        "executed_action_support_pass": bool(prefill_executed_mask_valid and decode_executed_mask_valid),
        "safety_projection_applied": bool(safety_projection_applied),
        "safety_projection_reasons": str(safety_projection_reasons or ""),
        "router_head_values": {
            "requested": [int(value) for value in requested_heads],
            "executed": [int(value) for value in executed_heads],
        },
        "prefill_policy_logprob": round(float(prefill_policy_logprob), 6),
        "decode_policy_logprob": round(float(decode_policy_logprob), 6),
        "prefill_policy_entropy": round(float(prefill_policy_entropy), 6),
        "decode_policy_entropy": round(float(decode_policy_entropy), 6),
    }


def action_index_from_heads(
    phase_index: int,
    admission_index: int,
    admission_limit_index: int,
    quantile_index: int,
    budget_profile_index: int,
    risky_admission_cap_index: int,
    prefill_safety_filter_index: int = 1,
    decode_safety_filter_index: int = 1,
) -> int:
    return ACTION_INDEX[
        (
            PHASE_CHOICES[phase_index],
            ADMISSION_CHOICES[admission_index],
            NEW_ADMISSION_LIMIT_CHOICES[admission_limit_index],
            QUANTILE_CHOICES[quantile_index],
            BUDGET_PROFILE_CHOICES[budget_profile_index],
            RISKY_ADMISSION_CAP_CHOICES[risky_admission_cap_index],
        )
    ]


def router_policies_from_heads(
    phase_index: int,
    admission_index: int,
    admission_limit_index: int,
    quantile_index: int,
    budget_profile_index: int,
    risky_admission_cap_index: int,
    prefill_safety_filter_index: int = 1,
    decode_safety_filter_index: int = 1,
) -> Tuple[PrefillRouterPolicy, DecodeRouterPolicy]:
    prefill_policy = PrefillRouterPolicy(
        admission_profile=ADMISSION_CHOICES[admission_index],
        new_admission_limit=NEW_ADMISSION_LIMIT_CHOICES[admission_limit_index],
        shadow_quantile=QUANTILE_CHOICES[quantile_index],
        risky_admission_cap=RISKY_ADMISSION_CAP_CHOICES[risky_admission_cap_index],
        safety_filter_enabled=SAFETY_FILTER_CHOICES[prefill_safety_filter_index],
    )
    phase_label = PHASE_CHOICES[phase_index]
    decode_policy = DecodeRouterPolicy(
        phase_preference=None if phase_label == "heuristic" else PhaseMode(phase_label),
        budget_profile=BUDGET_PROFILE_CHOICES[budget_profile_index],
        safety_filter_enabled=SAFETY_FILTER_CHOICES[decode_safety_filter_index],
    )
    return prefill_policy, decode_policy


def prefill_policy_condition_vector(policy: PrefillRouterPolicy) -> List[float]:
    admission_value = (
        float(ADMISSION_CHOICES.index(policy.admission_profile) + 1) / float(max(len(ADMISSION_CHOICES), 1))
        if policy.admission_profile in ADMISSION_CHOICES
        else 0.0
    )
    limit_max = max(NEW_ADMISSION_LIMIT_CHOICES or [1])
    risky_max = max(RISKY_ADMISSION_CAP_CHOICES or [1])
    return [
        admission_value,
        min(float(_prefill_limit_value(policy.new_admission_limit)) / max(float(limit_max), 1.0), 1.0),
        min(max(float(policy.shadow_quantile or 0.0), 0.0), 1.0),
        min(float(_risky_cap_value(policy.risky_admission_cap)) / max(float(risky_max), 1.0), 1.0),
        1.0 if policy.safety_filter_enabled is not False else 0.0,
    ]


def decode_observation_conditioned_on_prefill(
    decode_observation: Sequence[float],
    prefill_policy: PrefillRouterPolicy,
) -> List[float]:
    values = [float(value) for value in decode_observation]
    condition = prefill_policy_condition_vector(prefill_policy)
    if len(values) >= len(condition):
        values[-len(condition):] = condition
        return values
    return values + condition


def action_factor_marginals(
    phase_probs: Sequence[float],
    admission_probs: Sequence[float],
    admission_limit_probs: Sequence[float],
    quantile_probs: Sequence[float],
    budget_profile_probs: Sequence[float],
    risky_admission_cap_probs: Sequence[float],
    prefill_safety_filter_probs: Sequence[float],
    decode_safety_filter_probs: Sequence[float],
) -> Dict[str, List[float]]:
    return {
        "phase": [float(value) for value in phase_probs],
        "admission": [float(value) for value in admission_probs],
        "admission_limit": [float(value) for value in admission_limit_probs],
        "quantile": [float(value) for value in quantile_probs],
        "budget_profile": [float(value) for value in budget_profile_probs],
        "risky_admission_cap": [float(value) for value in risky_admission_cap_probs],
        "prefill_safety_filter": [float(value) for value in prefill_safety_filter_probs],
        "decode_safety_filter": [float(value) for value in decode_safety_filter_probs],
    }


def router_policy_marginals(
    phase_probs: Sequence[float],
    admission_probs: Sequence[float],
    admission_limit_probs: Sequence[float],
    quantile_probs: Sequence[float],
    budget_profile_probs: Sequence[float],
    risky_admission_cap_probs: Sequence[float],
    prefill_safety_filter_probs: Sequence[float],
    decode_safety_filter_probs: Sequence[float],
) -> Dict[str, Dict[str, List[float]]]:
    return {
        "prefill": {
            "admission": [float(value) for value in admission_probs],
            "admission_limit": [float(value) for value in admission_limit_probs],
            "quantile": [float(value) for value in quantile_probs],
            "risky_admission_cap": [float(value) for value in risky_admission_cap_probs],
            "safety_filter": [float(value) for value in prefill_safety_filter_probs],
        },
        "decode": {
            "phase": [float(value) for value in phase_probs],
            "budget_profile": [float(value) for value in budget_profile_probs],
            "safety_filter": [float(value) for value in decode_safety_filter_probs],
        },
    }


def prefill_policy_joint_probabilities(
    admission_probs: Sequence[float],
    admission_limit_probs: Sequence[float],
    quantile_probs: Sequence[float],
    risky_admission_cap_probs: Sequence[float],
    prefill_safety_filter_probs: Sequence[float],
) -> List[float]:
    probs: List[float] = []
    for admission_label in ADMISSION_CHOICES:
        for admission_limit in NEW_ADMISSION_LIMIT_CHOICES:
            for quantile in QUANTILE_CHOICES:
                for risky_admission_cap in RISKY_ADMISSION_CAP_CHOICES:
                    for safety_filter in SAFETY_FILTER_CHOICES:
                        probs.append(
                            float(admission_probs[ADMISSION_CHOICES.index(admission_label)])
                            * float(admission_limit_probs[NEW_ADMISSION_LIMIT_CHOICES.index(admission_limit)])
                            * float(quantile_probs[QUANTILE_CHOICES.index(quantile)])
                            * float(risky_admission_cap_probs[RISKY_ADMISSION_CAP_CHOICES.index(risky_admission_cap)])
                            * float(prefill_safety_filter_probs[SAFETY_FILTER_CHOICES.index(safety_filter)])
                        )
    return probs


def decode_policy_joint_probabilities(
    phase_probs: Sequence[float],
    budget_profile_probs: Sequence[float],
    decode_safety_filter_probs: Sequence[float],
) -> List[float]:
    probs: List[float] = []
    for phase_label in PHASE_CHOICES:
        for budget_profile in BUDGET_PROFILE_CHOICES:
            for safety_filter in SAFETY_FILTER_CHOICES:
                probs.append(
                    float(phase_probs[PHASE_CHOICES.index(phase_label)])
                    * float(budget_profile_probs[BUDGET_PROFILE_CHOICES.index(budget_profile)])
                    * float(decode_safety_filter_probs[SAFETY_FILTER_CHOICES.index(safety_filter)])
                )
    return probs


def joint_action_probabilities(
    phase_probs: Sequence[float],
    admission_probs: Sequence[float],
    admission_limit_probs: Sequence[float],
    quantile_probs: Sequence[float],
    budget_profile_probs: Sequence[float],
    risky_admission_cap_probs: Sequence[float],
    prefill_safety_filter_probs: Sequence[float],
    decode_safety_filter_probs: Sequence[float],
) -> List[float]:
    probs: List[float] = []
    for action in ACTION_TABLE:
        phase_label = action.phase_preference.value if action.phase_preference is not None else "heuristic"
        probs.append(
            float(phase_probs[PHASE_CHOICES.index(phase_label)])
            * float(admission_probs[ADMISSION_CHOICES.index(action.admission_profile)])
            * float(
                admission_limit_probs[
                    NEW_ADMISSION_LIMIT_CHOICES.index(_prefill_limit_value(action.new_admission_limit))
                ]
            )
            * float(
                quantile_probs[
                    QUANTILE_CHOICES.index(nearest_quantile_choice(float(action.shadow_quantile or default_shadow_quantile_choice())))
                ]
            )
            * float(budget_profile_probs[BUDGET_PROFILE_CHOICES.index(action.budget_profile)])
            * float(
                risky_admission_cap_probs[
                    RISKY_ADMISSION_CAP_CHOICES.index(_risky_cap_value(action.risky_admission_cap))
                ]
            )
            * float(
                prefill_safety_filter_probs[
                    SAFETY_FILTER_CHOICES.index(
                        bool(action.prefill_safety_filter_enabled)
                        if action.prefill_safety_filter_enabled is not None
                        else True
                    )
                ]
            )
            * float(
                decode_safety_filter_probs[
                    SAFETY_FILTER_CHOICES.index(
                        bool(action.decode_safety_filter_enabled)
                        if action.decode_safety_filter_enabled is not None
                        else True
                    )
                ]
            )
        )
    return probs


class ObservationNormalizer:
    def __init__(self, dimension: int, momentum: float = 0.95):
        self.dimension = dimension
        self.momentum = momentum
        self.count = 0
        self.mean = np.zeros(dimension, dtype=np.float64)
        self.var = np.ones(dimension, dtype=np.float64)

    def update(self, observation: Sequence[float]) -> None:
        obs = np.asarray(observation, dtype=np.float64)
        if self.count == 0:
            self.mean = obs.copy()
            self.var = np.ones_like(obs)
            self.count = 1
            return
        delta = obs - self.mean
        self.mean = self.momentum * self.mean + (1.0 - self.momentum) * obs
        self.var = self.momentum * self.var + (1.0 - self.momentum) * np.square(delta)
        self.count += 1

    def normalize(self, observation: Sequence[float]) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float64)
        scale = np.sqrt(np.maximum(self.var, 1e-6))
        return (obs - self.mean) / scale

    def state_dict(self) -> Dict[str, object]:
        return {
            "dimension": self.dimension,
            "momentum": self.momentum,
            "count": self.count,
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
        }

    @classmethod
    def from_state_dict(cls, payload: Dict[str, object]) -> "ObservationNormalizer":
        obj = cls(int(payload["dimension"]), float(payload["momentum"]))
        obj.count = int(payload["count"])
        obj.mean = np.asarray(payload["mean"], dtype=np.float64)
        obj.var = np.asarray(payload["var"], dtype=np.float64)
        return obj


class ReturnNormalizer:
    def __init__(self, momentum: float = 0.95):
        self.momentum = momentum
        self.count = 0
        self.mean = 0.0
        self.var = 1.0

    def update(self, values: Sequence[float]) -> None:
        if not values:
            return
        batch = np.asarray(values, dtype=np.float64)
        batch_mean = float(batch.mean())
        batch_var = float(batch.var()) if batch.size > 1 else 1.0
        if self.count == 0:
            self.mean = batch_mean
            self.var = max(batch_var, 1e-6)
            self.count = int(batch.size)
            return
        delta = batch_mean - self.mean
        self.mean = self.momentum * self.mean + (1.0 - self.momentum) * batch_mean
        self.var = self.momentum * self.var + (1.0 - self.momentum) * (batch_var + delta * delta)
        self.count += int(batch.size)

    def normalize_tensor(self, values: torch.Tensor) -> torch.Tensor:
        scale = math.sqrt(max(float(self.var), 1e-6))
        return (values - float(self.mean)) / scale

    def denormalize_tensor(self, values: torch.Tensor) -> torch.Tensor:
        scale = math.sqrt(max(float(self.var), 1e-6))
        return values * scale + float(self.mean)

    def state_dict(self) -> Dict[str, object]:
        return {
            "momentum": self.momentum,
            "count": self.count,
            "mean": self.mean,
            "var": self.var,
        }

    @classmethod
    def from_state_dict(cls, payload: Dict[str, object]) -> "ReturnNormalizer":
        obj = cls(float(payload.get("momentum", 0.95)))
        obj.count = int(payload.get("count", 0))
        obj.mean = float(payload.get("mean", 0.0))
        obj.var = float(payload.get("var", 1.0))
        return obj


class SequentialMultiAgentRolloutBuffer:
    def __init__(self) -> None:
        self.raw_prefill_observations: List[np.ndarray] = []
        self.raw_decode_observations: List[np.ndarray] = []
        self.prefill_observations: List[np.ndarray] = []
        self.decode_observations: List[np.ndarray] = []
        self.central_observations: List[np.ndarray] = []
        self.action_heads: List[Tuple[int, int, int, int, int, int, int, int]] = []
        self.requested_action_heads: List[Tuple[int, int, int, int, int, int, int, int]] = []
        self.projection_action_heads: List[Tuple[int, int, int, int, int, int, int, int]] = []
        self.projection_masks: List[float] = []
        self.projection_reason_sets: List[Tuple[str, ...]] = []
        self.projection_prefill_responsibilities: List[float] = []
        self.projection_decode_responsibilities: List[float] = []
        self.sample_routers: List[str] = []
        self.action_masks: List[Dict[str, Tuple[int, ...]]] = []
        self.prefill_decision_gate_indices: List[int] = []
        self.decode_decision_gate_indices: List[int] = []
        self.old_prefill_log_probs: List[float] = []
        self.old_decode_log_probs: List[float] = []
        self.prefill_policy_entropies: List[float] = []
        self.decode_policy_entropies: List[float] = []
        self.prefill_rewards: List[float] = []
        self.decode_rewards: List[float] = []
        self.rewards: List[float] = []
        self.transition_rows: List[Dict[str, object]] = []
        self.step_events: List[Dict[str, object]] = []

    def append_sample(self, sample_detail: Dict[str, object]) -> None:
        self.raw_prefill_observations.append(
            np.asarray(sample_detail.get("prefill_observation", []), dtype=np.float64)
        )
        self.raw_decode_observations.append(
            np.asarray(sample_detail.get("decode_observation", []), dtype=np.float64)
        )
        self.prefill_observations.append(np.asarray(sample_detail["normalized_prefill_observation"], dtype=np.float64))
        self.decode_observations.append(np.asarray(sample_detail["normalized_decode_observation"], dtype=np.float64))
        self.central_observations.append(np.asarray(sample_detail["normalized_central_observation"], dtype=np.float64))
        self.action_heads.append(tuple(sample_detail["action_heads"]))  # type: ignore[arg-type]
        self.requested_action_heads.append(
            tuple(sample_detail.get("requested_action_heads_for_training", sample_detail["action_heads"]))  # type: ignore[arg-type]
        )
        self.projection_action_heads.append(tuple(sample_detail["action_heads"]))  # type: ignore[arg-type]
        self.projection_masks.append(1.0 if bool(sample_detail.get("training_action_rewritten_by_safety", False)) else 0.0)
        self.projection_reason_sets.append(
            tuple(parse_reasons(sample_detail.get("action_safety_reasons_for_training", "")))
        )
        self.projection_prefill_responsibilities.append(
            float(sample_detail.get("counterfactual_prefill_responsibility", 0.0) or 0.0)
        )
        self.projection_decode_responsibilities.append(
            float(sample_detail.get("counterfactual_decode_responsibility", 0.0) or 0.0)
        )
        self.sample_routers.append(str(sample_detail.get("sample_router", "joint") or "joint"))
        self.action_masks.append(
            {
                str(key): tuple(int(index) for index in value)  # type: ignore[union-attr]
                for key, value in dict(sample_detail.get("action_mask") or {}).items()
            }
        )
        self.prefill_decision_gate_indices.append(int(sample_detail.get("prefill_decision_gate_index", 1)))
        self.decode_decision_gate_indices.append(int(sample_detail.get("decode_decision_gate_index", 1)))
        self.old_prefill_log_probs.append(float(sample_detail["old_prefill_log_prob"]))
        self.old_decode_log_probs.append(float(sample_detail["old_decode_log_prob"]))
        self.prefill_policy_entropies.append(float(sample_detail.get("prefill_policy_entropy", 0.0) or 0.0))
        self.decode_policy_entropies.append(float(sample_detail.get("decode_policy_entropy", 0.0) or 0.0))

    def set_last_projection(
        self,
        action_heads: Tuple[int, int, int, int, int, int, int, int],
        projected: bool,
        reasons: object = "",
        prefill_responsibility: float = 0.0,
        decode_responsibility: float = 0.0,
    ) -> None:
        if not self.projection_action_heads:
            return
        self.projection_action_heads[-1] = action_heads
        self.projection_masks[-1] = 1.0 if projected else 0.0
        if self.projection_reason_sets:
            self.projection_reason_sets[-1] = tuple(parse_reasons(reasons)) if projected else ()
        if self.projection_prefill_responsibilities:
            self.projection_prefill_responsibilities[-1] = float(prefill_responsibility) if projected else 0.0
        if self.projection_decode_responsibilities:
            self.projection_decode_responsibilities[-1] = float(decode_responsibility) if projected else 0.0

    def discard_last_sample(self) -> None:
        if self.raw_prefill_observations:
            self.raw_prefill_observations.pop()
        if self.raw_decode_observations:
            self.raw_decode_observations.pop()
        if self.prefill_observations:
            self.prefill_observations.pop()
        if self.decode_observations:
            self.decode_observations.pop()
        if self.central_observations:
            self.central_observations.pop()
        if self.action_heads:
            self.action_heads.pop()
        if self.requested_action_heads:
            self.requested_action_heads.pop()
        if self.projection_action_heads:
            self.projection_action_heads.pop()
        if self.projection_masks:
            self.projection_masks.pop()
        if self.projection_reason_sets:
            self.projection_reason_sets.pop()
        if self.projection_prefill_responsibilities:
            self.projection_prefill_responsibilities.pop()
        if self.projection_decode_responsibilities:
            self.projection_decode_responsibilities.pop()
        if self.sample_routers:
            self.sample_routers.pop()
        if self.action_masks:
            self.action_masks.pop()
        if self.prefill_decision_gate_indices:
            self.prefill_decision_gate_indices.pop()
        if self.decode_decision_gate_indices:
            self.decode_decision_gate_indices.pop()
        if self.old_prefill_log_probs:
            self.old_prefill_log_probs.pop()
        if self.old_decode_log_probs:
            self.old_decode_log_probs.pop()
        if self.prefill_policy_entropies:
            self.prefill_policy_entropies.pop()
        if self.decode_policy_entropies:
            self.decode_policy_entropies.pop()

    def append_reward(self, prefill_reward: float, decode_reward: float) -> None:
        self.prefill_rewards.append(prefill_reward)
        self.decode_rewards.append(decode_reward)
        self.rewards.append(prefill_reward + decode_reward)

    def add_terminal_reward(
        self,
        prefill_reward: float,
        decode_reward: float,
        summary: Optional[Dict[str, object]] = None,
    ) -> None:
        total_reward = prefill_reward + decode_reward
        if self.rewards:
            self.prefill_rewards[-1] += prefill_reward
            self.decode_rewards[-1] += decode_reward
            self.rewards[-1] += total_reward
        if self.transition_rows:
            self.transition_rows[-1]["terminal_reward"] = round(total_reward, 6)
            self.transition_rows[-1]["terminal_prefill_reward"] = round(prefill_reward, 6)
            self.transition_rows[-1]["terminal_decode_reward"] = round(decode_reward, 6)
            if summary is not None:
                self.transition_rows[-1]["terminal_summary"] = {
                    "completed": summary.get("completed"),
                    "terminated_early": summary.get("terminated_early"),
                    "unfinished_requests": summary.get("unfinished_requests"),
                    "termination_reason": summary.get("termination_reason"),
                }
        if self.step_events:
            self.step_events[-1]["terminal_reward"] = round(total_reward, 6)
            self.step_events[-1]["terminal_prefill_reward"] = round(prefill_reward, 6)
            self.step_events[-1]["terminal_decode_reward"] = round(decode_reward, 6)

    def __len__(self) -> int:
        return len(self.rewards)


class FactorizedActorCritic(nn.Module):
    def __init__(
        self,
        prefill_observation_dim: int,
        decode_observation_dim: int,
        hidden_dim: int,
        hidden_depth: int,
    ):
        super().__init__()
        self.prefill_trunk = self._build_trunk(prefill_observation_dim, hidden_dim, hidden_depth)
        self.decode_trunk = self._build_trunk(decode_observation_dim, hidden_dim, hidden_depth)
        self.admission_head = nn.Linear(hidden_dim, len(ADMISSION_CHOICES))
        self.admission_limit_head = nn.Linear(hidden_dim, len(NEW_ADMISSION_LIMIT_CHOICES))
        self.quantile_head = nn.Linear(hidden_dim, len(QUANTILE_CHOICES))
        self.risky_admission_cap_head = nn.Linear(hidden_dim, len(RISKY_ADMISSION_CAP_CHOICES))
        self.prefill_safety_filter_head = nn.Linear(hidden_dim, len(SAFETY_FILTER_CHOICES))
        self.prefill_decision_gate_head = nn.Linear(hidden_dim, len(DECISION_GATE_CHOICES))
        self.prefill_value_head = nn.Linear(hidden_dim, 1)
        self.phase_head = nn.Linear(hidden_dim, len(PHASE_CHOICES))
        self.budget_profile_head = nn.Linear(hidden_dim, len(BUDGET_PROFILE_CHOICES))
        self.decode_safety_filter_head = nn.Linear(hidden_dim, len(SAFETY_FILTER_CHOICES))
        self.decode_decision_gate_head = nn.Linear(hidden_dim, len(DECISION_GATE_CHOICES))
        self.decode_value_head = nn.Linear(hidden_dim, 1)
        self.recovery_joint_profiles = build_recovery_joint_profiles()
        self.recovery_joint_profile_head = nn.Linear(hidden_dim * 2, len(self.recovery_joint_profiles))
        self._initialize_safety_heads()

    @staticmethod
    def _build_trunk(input_dim: int, hidden_dim: int, hidden_depth: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        current_dim = input_dim
        for _ in range(hidden_depth):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            current_dim = hidden_dim
        return nn.Sequential(*layers)

    def _initialize_safety_heads(self) -> None:
        for head in (self.prefill_safety_filter_head, self.decode_safety_filter_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            head.bias.data[SAFETY_FILTER_CHOICES.index(True)] = 0.75
        for head in (self.prefill_decision_gate_head, self.decode_decision_gate_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            head.bias.data[DECISION_GATE_CHOICES.index(True)] = 1.0

    def forward(
        self,
        prefill_observation: torch.Tensor,
        decode_observation: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        prefill_hidden = self.prefill_trunk(prefill_observation)
        decode_hidden = self.decode_trunk(decode_observation)
        return self._heads(prefill_hidden, decode_hidden)

    def prefill_outputs(self, prefill_observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        prefill_hidden = self.prefill_trunk(prefill_observation)
        return {
            "_prefill_hidden": prefill_hidden,
            "admission_logits": self.admission_head(prefill_hidden),
            "admission_limit_logits": self.admission_limit_head(prefill_hidden),
            "quantile_logits": self.quantile_head(prefill_hidden),
            "risky_admission_cap_logits": self.risky_admission_cap_head(prefill_hidden),
            "prefill_safety_filter_logits": self.prefill_safety_filter_head(prefill_hidden),
            "prefill_decision_gate_logits": self.prefill_decision_gate_head(prefill_hidden),
            "prefill_value": self.prefill_value_head(prefill_hidden).squeeze(-1),
        }

    def decode_outputs(self, decode_observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        decode_hidden = self.decode_trunk(decode_observation)
        return {
            "_decode_hidden": decode_hidden,
            "phase_logits": self.phase_head(decode_hidden),
            "budget_profile_logits": self.budget_profile_head(decode_hidden),
            "decode_safety_filter_logits": self.decode_safety_filter_head(decode_hidden),
            "decode_decision_gate_logits": self.decode_decision_gate_head(decode_hidden),
            "decode_value": self.decode_value_head(decode_hidden).squeeze(-1),
        }

    def _heads(
        self,
        prefill_hidden: torch.Tensor,
        decode_hidden: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "phase_logits": self.phase_head(decode_hidden),
            "admission_logits": self.admission_head(prefill_hidden),
            "admission_limit_logits": self.admission_limit_head(prefill_hidden),
            "quantile_logits": self.quantile_head(prefill_hidden),
            "budget_profile_logits": self.budget_profile_head(decode_hidden),
            "risky_admission_cap_logits": self.risky_admission_cap_head(prefill_hidden),
            "recovery_joint_profile_logits": self.recovery_joint_profile_head(
                torch.cat((prefill_hidden, decode_hidden), dim=-1)
            ),
            "prefill_safety_filter_logits": self.prefill_safety_filter_head(prefill_hidden),
            "decode_safety_filter_logits": self.decode_safety_filter_head(decode_hidden),
            "prefill_decision_gate_logits": self.prefill_decision_gate_head(prefill_hidden),
            "decode_decision_gate_logits": self.decode_decision_gate_head(decode_hidden),
            "prefill_value": self.prefill_value_head(prefill_hidden).squeeze(-1),
            "decode_value": self.decode_value_head(decode_hidden).squeeze(-1),
        }


class CentralizedCritic(nn.Module):
    def __init__(
        self,
        central_observation_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        shared_critic: bool,
    ):
        super().__init__()
        self.shared_critic = shared_critic
        self.trunk = FactorizedActorCritic._build_trunk(central_observation_dim, hidden_dim, hidden_depth)
        self.prefill_value_head = nn.Linear(hidden_dim, 1)
        self.decode_value_head = self.prefill_value_head if shared_critic else nn.Linear(hidden_dim, 1)

    def forward(self, central_observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.trunk(central_observation)
        prefill_value = self.prefill_value_head(hidden).squeeze(-1)
        decode_value = self.decode_value_head(hidden).squeeze(-1)
        return {
            "prefill_value": prefill_value,
            "decode_value": decode_value,
        }


class TorchFactorizedAgent:
    def __init__(
        self,
        prefill_observation_dim: int,
        decode_observation_dim: int,
        rng_seed: int,
        config: TrainingConfig,
        device: str = "cpu",
        central_observation_dim: Optional[int] = None,
    ):
        self.prefill_observation_dim = prefill_observation_dim
        self.decode_observation_dim = decode_observation_dim
        self.central_observation_dim = central_observation_dim or (
            prefill_observation_dim + decode_observation_dim
        )
        self.config = config
        self.device = torch.device(device)
        torch.manual_seed(rng_seed)
        self.model = FactorizedActorCritic(
            prefill_observation_dim=prefill_observation_dim,
            decode_observation_dim=decode_observation_dim,
            hidden_dim=config.hidden_dim,
            hidden_depth=config.hidden_depth,
        ).to(self.device)
        self.central_critic = CentralizedCritic(
            central_observation_dim=self.central_observation_dim,
            hidden_dim=config.hidden_dim,
            hidden_depth=config.hidden_depth,
            shared_critic=config.shared_critic,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.central_critic.parameters()),
            lr=config.learning_rate,
        )
        self.prefill_normalizer = ObservationNormalizer(prefill_observation_dim, momentum=config.observation_momentum)
        self.decode_normalizer = ObservationNormalizer(decode_observation_dim, momentum=config.observation_momentum)
        self.central_normalizer = ObservationNormalizer(self.central_observation_dim, momentum=config.observation_momentum)
        self.prefill_return_normalizer = ReturnNormalizer(momentum=config.observation_momentum)
        self.decode_return_normalizer = ReturnNormalizer(momentum=config.observation_momentum)
        self._late_update_active = False
        self.training_stage = "stage_b_ppo"
        self._policy_loss_scale = 1.0
        self._value_loss_scale = 1.0
        self._stage_imitation_coef = 0.0
        self._actor_updates_frozen = False
        self._observation_normalizers_frozen = not bool(config.observation_normalization)
        self._anchor_model: Optional[FactorizedActorCritic] = None
        self.reset_episode()

    @staticmethod
    def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = enabled

    def _hard_freeze_actor_parameters(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False
            parameter.grad = None
            self.optimizer.state.pop(parameter, None)

    def configure_update(self, progress_fraction: float) -> Dict[str, object]:
        stage_a_end = min(max(float(self.config.teacher_stage_a_fraction), 0.0), 1.0)
        stage_b_end = min(max(stage_a_end + float(self.config.teacher_stage_b_fraction), stage_a_end), 1.0)
        if progress_fraction < stage_a_end:
            stage = "stage_a_imitation"
        elif progress_fraction < stage_b_end:
            stage = "stage_b_ppo"
        else:
            stage = "stage_c_finetune"
        self.training_stage = stage
        self._policy_loss_scale = 0.0 if stage == "stage_a_imitation" else 1.0
        self._value_loss_scale = 0.0 if stage == "stage_a_imitation" else 1.0
        self._stage_imitation_coef = float(self.config.stage_a_imitation_coef) if stage == "stage_a_imitation" else 0.0
        late_start = float(self.config.late_update_start_fraction)
        late_active = (late_start <= 1.0 and progress_fraction >= max(late_start, 0.0)) or stage == "stage_c_finetune"
        lr_scale = float(self.config.late_learning_rate_scale) if late_active else 1.0
        if stage == "stage_c_finetune":
            lr_scale = min(lr_scale, float(self.config.stage_c_learning_rate_scale))
        lr = float(self.config.learning_rate) * max(lr_scale, 0.0)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        freeze_prefill_trunk = bool(self.config.late_freeze_prefill_trunk and late_active)
        freeze_admission_heads = bool(
            (self.config.late_freeze_admission_heads and late_active)
            or (self.config.stage_c_freeze_admission_heads and stage == "stage_c_finetune")
        )
        self._set_requires_grad(self.model.prefill_trunk, not freeze_prefill_trunk)
        for head in (
            self.model.admission_head,
            self.model.admission_limit_head,
            self.model.quantile_head,
            self.model.risky_admission_cap_head,
            self.model.recovery_joint_profile_head,
        ):
            self._set_requires_grad(head, not freeze_admission_heads)
        if self._actor_updates_frozen:
            self._hard_freeze_actor_parameters()
        self._late_update_active = late_active
        return {
            "late_update_active": late_active,
            "optimizer_lr": lr,
            "late_freeze_prefill_trunk": freeze_prefill_trunk,
            "late_freeze_admission_heads": freeze_admission_heads,
            "training_stage": stage,
            "policy_loss_scale": self._policy_loss_scale,
            "value_loss_scale": self._value_loss_scale,
            "stage_imitation_coef": self._stage_imitation_coef,
        }

    def clone_for_eval(self) -> "TorchFactorizedAgent":
        cloned = TorchFactorizedAgent(
            self.prefill_observation_dim,
            self.decode_observation_dim,
            0,
            self.config,
            device=str(self.device),
            central_observation_dim=self.central_observation_dim,
        )
        cloned.model.load_state_dict(self.model.state_dict())
        cloned.central_critic.load_state_dict(self.central_critic.state_dict())
        if self.config.observation_normalization:
            cloned.prefill_normalizer = ObservationNormalizer.from_state_dict(self.prefill_normalizer.state_dict())
            cloned.decode_normalizer = ObservationNormalizer.from_state_dict(self.decode_normalizer.state_dict())
            cloned.central_normalizer = ObservationNormalizer.from_state_dict(self.central_normalizer.state_dict())
        cloned.prefill_return_normalizer = self.prefill_return_normalizer
        cloned.decode_return_normalizer = self.decode_return_normalizer
        return cloned

    def load_checkpoint_payload(self, policy_payload: Dict[str, object], *, strict: bool = False) -> None:
        model_state = policy_payload["state_dict"]  # type: ignore[assignment]
        if strict:
            self.model.load_state_dict(model_state, strict=True)  # type: ignore[arg-type]
        else:
            current_model_state = self.model.state_dict()
            compatible_model_state = {
                key: value
                for key, value in model_state.items()  # type: ignore[union-attr]
                if key in current_model_state and current_model_state[key].shape == value.shape
            }
            current_model_state.update(compatible_model_state)
            self.model.load_state_dict(current_model_state, strict=True)
        central_state = policy_payload.get("central_critic_state_dict")
        if central_state is not None:
            current_central_state = self.central_critic.state_dict()
            compatible_central_state = {
                key: value
                for key, value in central_state.items()  # type: ignore[union-attr]
                if key in current_central_state and current_central_state[key].shape == value.shape
            }
            current_central_state.update(compatible_central_state)
            self.central_critic.load_state_dict(current_central_state, strict=True)
        if self.config.observation_normalization and "prefill_normalizer" in policy_payload:
            self.prefill_normalizer = ObservationNormalizer.from_state_dict(
                policy_payload["prefill_normalizer"]  # type: ignore[arg-type]
            )
        if self.config.observation_normalization and "decode_normalizer" in policy_payload:
            self.decode_normalizer = ObservationNormalizer.from_state_dict(
                policy_payload["decode_normalizer"]  # type: ignore[arg-type]
            )
        if self.config.observation_normalization and "central_normalizer" in policy_payload:
            central_normalizer_payload = policy_payload["central_normalizer"]  # type: ignore[assignment]
            if int(central_normalizer_payload.get("dimension", -1)) == self.central_observation_dim:  # type: ignore[union-attr]
                self.central_normalizer = ObservationNormalizer.from_state_dict(
                    central_normalizer_payload  # type: ignore[arg-type]
                )
        if "prefill_return_normalizer" in policy_payload:
            self.prefill_return_normalizer = ReturnNormalizer.from_state_dict(
                policy_payload["prefill_return_normalizer"]  # type: ignore[arg-type]
            )
        if "decode_return_normalizer" in policy_payload:
            self.decode_return_normalizer = ReturnNormalizer.from_state_dict(
                policy_payload["decode_return_normalizer"]  # type: ignore[arg-type]
            )

    def freeze_actor_updates(self) -> None:
        self._actor_updates_frozen = True
        self._hard_freeze_actor_parameters()

    def actor_updates_frozen(self) -> bool:
        return self._actor_updates_frozen

    def freeze_observation_normalizers(self) -> None:
        self._observation_normalizers_frozen = True

    def observation_normalizers_frozen(self) -> bool:
        return self._observation_normalizers_frozen

    def set_anchor_policy_from_current(self) -> None:
        anchor_model = copy.deepcopy(self.model).to(self.device)
        anchor_model.eval()
        for parameter in anchor_model.parameters():
            parameter.requires_grad = False
        self._anchor_model = anchor_model

    def reset_episode(self) -> None:
        self.rollout = SequentialMultiAgentRolloutBuffer()

    @property
    def prefill_observations(self) -> List[np.ndarray]:
        return self.rollout.prefill_observations

    @property
    def decode_observations(self) -> List[np.ndarray]:
        return self.rollout.decode_observations

    @property
    def central_observations(self) -> List[np.ndarray]:
        return self.rollout.central_observations

    @property
    def action_heads(self) -> List[Tuple[int, int, int, int, int, int, int, int]]:
        return self.rollout.action_heads

    @property
    def prefill_rewards(self) -> List[float]:
        return self.rollout.prefill_rewards

    @property
    def decode_rewards(self) -> List[float]:
        return self.rollout.decode_rewards

    @property
    def rewards(self) -> List[float]:
        return self.rollout.rewards

    @property
    def transition_rows(self) -> List[Dict[str, object]]:
        return self.rollout.transition_rows

    @property
    def step_events(self) -> List[Dict[str, object]]:
        return self.rollout.step_events

    def discard_last_sample(self) -> None:
        self.rollout.discard_last_sample()

    def _store_sample_detail(self, sample_detail: Dict[str, object]) -> None:
        self.rollout.append_sample(sample_detail)

    def commit_sample_detail(self, sample_detail: Dict[str, object]) -> None:
        self._store_sample_detail(sample_detail)

    def _prepare_prefill_observation(
        self,
        observation: Sequence[float],
        update_normalizer: bool,
    ) -> np.ndarray:
        if not self.config.observation_normalization:
            return np.asarray(observation, dtype=np.float64)
        if update_normalizer and not self._observation_normalizers_frozen:
            self.prefill_normalizer.update(observation)
        return self.prefill_normalizer.normalize(observation)

    def _prepare_decode_observation(
        self,
        observation: Sequence[float],
        update_normalizer: bool,
    ) -> np.ndarray:
        if not self.config.observation_normalization:
            return np.asarray(observation, dtype=np.float64)
        if update_normalizer and not self._observation_normalizers_frozen:
            self.decode_normalizer.update(observation)
        return self.decode_normalizer.normalize(observation)

    def _prepare_central_observation(
        self,
        observation: Sequence[float],
        update_normalizer: bool,
    ) -> np.ndarray:
        if not self.config.observation_normalization:
            return np.asarray(observation, dtype=np.float64)
        if update_normalizer and not self._observation_normalizers_frozen:
            self.central_normalizer.update(observation)
        return self.central_normalizer.normalize(observation)

    def _log_probs_for_heads(
        self,
        prefill_outputs: Dict[str, torch.Tensor],
        decode_outputs: Dict[str, torch.Tensor],
        heads: Tuple[int, int, int, int, int, int, int, int],
        action_mask: Optional[Dict[str, Sequence[int]]] = None,
        prefill_decision_gate_index: int = 1,
        decode_decision_gate_index: int = 1,
    ) -> Tuple[float, float]:
        (
            phase_index,
            admission_index,
            admission_limit_index,
            quantile_index,
            budget_profile_index,
            risky_index,
            prefill_safety_filter_index,
            decode_safety_filter_index,
        ) = heads
        phase_label = PHASE_CHOICES[phase_index]
        admission_label = ADMISSION_CHOICES[admission_index]
        budget_allowed = allowed_indices(
            BUDGET_PROFILE_CHOICES,
            allowed_budget_profiles(phase_label, admission_label, BUDGET_PROFILE_CHOICES),
        )
        admission_dist = torch.distributions.Categorical(
            logits=masked_logits(
                prefill_outputs["admission_logits"][0],
                mask_indices(action_mask, "admission", len(ADMISSION_CHOICES)),
            )
        )
        admission_limit_dist = torch.distributions.Categorical(
            logits=masked_logits(
                prefill_outputs["admission_limit_logits"][0],
                mask_indices(action_mask, "admission_limit", len(NEW_ADMISSION_LIMIT_CHOICES)),
            )
        )
        quantile_dist = torch.distributions.Categorical(logits=prefill_outputs["quantile_logits"][0])
        risky_dist = torch.distributions.Categorical(
            logits=masked_logits(
                prefill_outputs["risky_admission_cap_logits"][0],
                mask_indices(action_mask, "risky_admission_cap", len(RISKY_ADMISSION_CAP_CHOICES)),
            )
        )
        prefill_safety_filter_dist = torch.distributions.Categorical(
            logits=prefill_outputs["prefill_safety_filter_logits"][0]
        )
        prefill_decision_gate_dist = torch.distributions.Categorical(
            logits=prefill_outputs["prefill_decision_gate_logits"][0]
        )
        phase_dist = torch.distributions.Categorical(
            logits=masked_logits(
                decode_outputs["phase_logits"][0],
                mask_indices(action_mask, "phase", len(PHASE_CHOICES)),
            )
        )
        budget_allowed = mask_indices(action_mask, "budget_profile", len(BUDGET_PROFILE_CHOICES), budget_allowed)
        budget_profile_dist = torch.distributions.Categorical(
            logits=masked_logits(decode_outputs["budget_profile_logits"][0], budget_allowed)
        )
        decode_safety_filter_dist = torch.distributions.Categorical(
            logits=decode_outputs["decode_safety_filter_logits"][0]
        )
        decode_decision_gate_dist = torch.distributions.Categorical(
            logits=decode_outputs["decode_decision_gate_logits"][0]
        )
        prefill_log_prob = (
            admission_dist.log_prob(torch.tensor(admission_index, dtype=torch.int64, device=self.device))
            + admission_limit_dist.log_prob(torch.tensor(admission_limit_index, dtype=torch.int64, device=self.device))
            + quantile_dist.log_prob(torch.tensor(quantile_index, dtype=torch.int64, device=self.device))
            + risky_dist.log_prob(torch.tensor(risky_index, dtype=torch.int64, device=self.device))
            + prefill_safety_filter_dist.log_prob(
                torch.tensor(prefill_safety_filter_index, dtype=torch.int64, device=self.device)
            )
            + prefill_decision_gate_dist.log_prob(
                torch.tensor(prefill_decision_gate_index, dtype=torch.int64, device=self.device)
            )
        )
        decode_log_prob = (
            phase_dist.log_prob(torch.tensor(phase_index, dtype=torch.int64, device=self.device))
            + budget_profile_dist.log_prob(torch.tensor(budget_profile_index, dtype=torch.int64, device=self.device))
            + decode_safety_filter_dist.log_prob(
                torch.tensor(decode_safety_filter_index, dtype=torch.int64, device=self.device)
            )
            + decode_decision_gate_dist.log_prob(
                torch.tensor(decode_decision_gate_index, dtype=torch.int64, device=self.device)
            )
        )
        return (
            float(prefill_log_prob.detach().cpu().item()),
            float(decode_log_prob.detach().cpu().item()),
        )

    def log_probs_for_normalized_sample(
        self,
        normalized_prefill_observation: Sequence[float],
        normalized_decode_observation: Sequence[float],
        heads: Tuple[int, int, int, int, int, int, int, int],
        action_mask: Optional[Dict[str, Sequence[int]]] = None,
        prefill_decision_gate_index: int = 1,
        decode_decision_gate_index: int = 1,
    ) -> Tuple[float, float]:
        prefill_tensor = torch.tensor(
            normalized_prefill_observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        decode_tensor = torch.tensor(
            normalized_decode_observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            prefill_outputs = self.model.prefill_outputs(prefill_tensor)
            decode_outputs = self.model.decode_outputs(decode_tensor)
        return self._log_probs_for_heads(
            prefill_outputs,
            decode_outputs,
            heads,
            action_mask,
            prefill_decision_gate_index=prefill_decision_gate_index,
            decode_decision_gate_index=decode_decision_gate_index,
        )

    def sample_action(
        self,
        prefill_observation: Sequence[float],
        decode_observation: Sequence[float],
        central_observation: Optional[Sequence[float]] = None,
        greedy: bool = False,
        capture_policy: bool = False,
        sampling_overrides: Optional[Dict[str, float]] = None,
    ) -> Tuple[int, Optional[Dict[str, np.ndarray]]]:
        sample_detail = self.sample_action_detail(
            prefill_observation,
            decode_observation,
            central_observation,
            greedy=greedy,
            capture_policy=capture_policy,
            sampling_overrides=sampling_overrides,
        )
        if not greedy:
            self._store_sample_detail(sample_detail)
        policy_snapshot = sample_detail.get("policy_snapshot")
        return int(sample_detail["action_index"]), policy_snapshot

    def _recovery_joint_profile_active(
        self,
        prefill_observation: Sequence[float],
        decode_observation: Sequence[float],
        *,
        greedy: bool,
    ) -> bool:
        if not bool(self.config.recovery_joint_profile_enabled):
            return False
        if not greedy and float(self.config.learning_rate) > 0.0:
            return False
        prefill_values = [float(value) for value in prefill_observation]
        decode_values = [float(value) for value in decode_observation]
        kv_util = max(
            prefill_values[1] if len(prefill_values) > 1 else 0.0,
            decode_values[1] if len(decode_values) > 1 else 0.0,
        )
        shadow_util = max(
            prefill_values[2] if len(prefill_values) > 2 else 0.0,
            decode_values[2] if len(decode_values) > 2 else 0.0,
        )
        first_token_pending = max(
            prefill_values[21] if len(prefill_values) > 21 else 0.0,
            decode_values[4] if len(decode_values) > 4 else 0.0,
        )
        pending_unadmitted = max(
            prefill_values[4] if len(prefill_values) > 4 else 0.0,
            decode_values[12] if len(decode_values) > 12 else 0.0,
        )
        started_prefill_remaining = max(
            prefill_values[13] if len(prefill_values) > 13 else 0.0,
            decode_values[10] if len(decode_values) > 10 else 0.0,
        )
        decode_load = max(
            prefill_values[20] if len(prefill_values) > 20 else 0.0,
            decode_values[3] if len(decode_values) > 3 else 0.0,
        )
        min_kv_util = max(float(self.config.recovery_joint_profile_min_kv_util), 0.0)
        return (
            kv_util >= min_kv_util
            or shadow_util >= min_kv_util
            or (first_token_pending > 0.0 and pending_unadmitted > 0.0)
            or (started_prefill_remaining > 0.0 and decode_load > 0.0)
        )

    def sample_action_detail(
        self,
        prefill_observation: Sequence[float],
        decode_observation: Sequence[float],
        central_observation: Optional[Sequence[float]] = None,
        greedy: bool = False,
        capture_policy: bool = False,
        sampling_overrides: Optional[Dict[str, float]] = None,
        forced_action: Optional[SchedulerAction] = None,
        action_mask: Optional[Dict[str, Sequence[int]]] = None,
    ) -> Dict[str, object]:
        normalized_prefill = self._prepare_prefill_observation(prefill_observation, update_normalizer=not greedy)
        prefill_tensor = torch.tensor(normalized_prefill, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            prefill_outputs = self.model.prefill_outputs(prefill_tensor)
        effective_sampling = sampling_overrides or {}

        temperature = float(effective_sampling.get("temperature", self.config.exploration_temperature))
        greedy_probability = float(
            effective_sampling.get("greedy_probability", self.config.exploration_greedy_prob)
        )
        top_k = int(effective_sampling.get("top_k", self.config.exploration_top_k))

        admission_allowed = mask_indices(action_mask, "admission", len(ADMISSION_CHOICES))
        admission_index, admission_probs = choose_index_from_logits(
            masked_logits(prefill_outputs["admission_logits"][0], admission_allowed),
            greedy=greedy,
            temperature=temperature,
            greedy_probability=greedy_probability,
            top_k=top_k,
        )
        admission_limit_allowed = mask_indices(action_mask, "admission_limit", len(NEW_ADMISSION_LIMIT_CHOICES))
        admission_limit_index, admission_limit_probs = choose_index_from_logits(
            masked_logits(prefill_outputs["admission_limit_logits"][0], admission_limit_allowed),
            greedy=greedy,
            temperature=temperature,
            greedy_probability=greedy_probability,
            top_k=top_k,
        )
        quantile_index, quantile_probs = choose_index_from_logits(
            prefill_outputs["quantile_logits"][0],
            greedy=greedy,
            temperature=temperature,
            greedy_probability=greedy_probability,
            top_k=top_k,
        )
        risky_allowed = mask_indices(action_mask, "risky_admission_cap", len(RISKY_ADMISSION_CAP_CHOICES))
        risky_admission_cap_index, risky_admission_cap_probs = choose_index_from_logits(
            masked_logits(prefill_outputs["risky_admission_cap_logits"][0], risky_allowed),
            greedy=greedy,
            temperature=temperature,
            greedy_probability=greedy_probability,
            top_k=top_k,
        )
        prefill_safety_filter_index, prefill_safety_filter_probs = choose_index_from_logits(
            prefill_outputs["prefill_safety_filter_logits"][0],
            greedy=greedy,
            temperature=temperature,
            greedy_probability=greedy_probability,
            top_k=top_k,
        )
        prefill_decision_gate_index, prefill_decision_gate_probs = choose_index_from_logits(
            prefill_outputs["prefill_decision_gate_logits"][0],
            greedy=greedy,
            temperature=temperature,
            greedy_probability=greedy_probability,
            top_k=top_k,
        )

        if forced_action is None:
            sampled_prefill_policy = PrefillRouterPolicy(
                admission_profile=ADMISSION_CHOICES[admission_index],
                new_admission_limit=NEW_ADMISSION_LIMIT_CHOICES[admission_limit_index],
                shadow_quantile=QUANTILE_CHOICES[quantile_index],
                risky_admission_cap=RISKY_ADMISSION_CAP_CHOICES[risky_admission_cap_index],
                safety_filter_enabled=SAFETY_FILTER_CHOICES[prefill_safety_filter_index],
            )
        else:
            forced_heads = action_heads_from_action(forced_action)
            (
                _forced_phase_index,
                admission_index,
                admission_limit_index,
                quantile_index,
                _forced_budget_profile_index,
                risky_admission_cap_index,
                prefill_safety_filter_index,
                _forced_decode_safety_filter_index,
            ) = forced_heads
            sampled_prefill_policy = forced_action.prefill_policy()
            prefill_decision_gate_index = DECISION_GATE_CHOICES.index(True)

        conditioned_decode_observation = decode_observation_conditioned_on_prefill(
            decode_observation,
            sampled_prefill_policy,
        )
        if (
            self.config.no_prefill_to_decode_condition
            or str(getattr(self.config, "router_boundary_mode", "current") or "current") == "strict_env_only"
        ):
            conditioned_decode_observation = list(decode_observation)
        normalized_decode = self._prepare_decode_observation(
            conditioned_decode_observation,
            update_normalizer=not greedy,
        )
        if central_observation is None:
            central_observation = [float(value) for value in prefill_observation] + [
                float(value) for value in decode_observation
            ]
        if self.config.serialized_joint_critic:
            central_values = [float(value) for value in central_observation]
            base_dim = len(prefill_observation) + len(decode_observation)
            history_tail = central_values[base_dim:] if len(central_values) > base_dim else []
            central_observation = [float(value) for value in prefill_observation] + [
                float(value) for value in conditioned_decode_observation
            ] + history_tail
        normalized_central = self._prepare_central_observation(central_observation, update_normalizer=not greedy)
        decode_tensor = torch.tensor(normalized_decode, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            decode_outputs = self.model.decode_outputs(decode_tensor)

        phase_allowed = mask_indices(action_mask, "phase", len(PHASE_CHOICES))
        phase_index, phase_probs = choose_index_from_logits(
            masked_logits(decode_outputs["phase_logits"][0], phase_allowed),
            greedy=greedy,
            temperature=temperature,
            greedy_probability=greedy_probability,
            top_k=top_k,
        )
        phase_label = PHASE_CHOICES[phase_index]
        budget_allowed = allowed_indices(
            BUDGET_PROFILE_CHOICES,
            allowed_budget_profiles(phase_label, ADMISSION_CHOICES[admission_index], BUDGET_PROFILE_CHOICES),
        )
        budget_allowed = mask_indices(action_mask, "budget_profile", len(BUDGET_PROFILE_CHOICES), budget_allowed)
        budget_logits = masked_logits(decode_outputs["budget_profile_logits"][0], budget_allowed)
        budget_profile_index, budget_profile_probs = choose_index_from_logits(
            budget_logits,
            greedy=greedy,
            temperature=temperature,
            greedy_probability=greedy_probability,
            top_k=top_k,
        )
        decode_safety_filter_index, decode_safety_filter_probs = choose_index_from_logits(
            decode_outputs["decode_safety_filter_logits"][0],
            greedy=greedy,
            temperature=temperature,
            greedy_probability=greedy_probability,
            top_k=top_k,
        )
        decode_decision_gate_index, decode_decision_gate_probs = choose_index_from_logits(
            decode_outputs["decode_decision_gate_logits"][0],
            greedy=greedy,
            temperature=temperature,
            greedy_probability=greedy_probability,
            top_k=top_k,
        )
        recovery_joint_profile_index_value = -1
        recovery_joint_profile_active = False
        recovery_joint_profile_probs = np.zeros(len(build_recovery_joint_profiles()), dtype=np.float64)
        if forced_action is None and self._recovery_joint_profile_active(
            prefill_observation,
            conditioned_decode_observation,
            greedy=greedy,
        ):
            prefill_hidden = prefill_outputs["_prefill_hidden"]
            decode_hidden = decode_outputs["_decode_hidden"]
            joint_logits_tensor = self.model.recovery_joint_profile_head(
                torch.cat((prefill_hidden, decode_hidden), dim=-1)
            )
            joint_allowed = recovery_joint_profile_allowed_indices(action_mask)
            joint_logits = masked_logits(
                joint_logits_tensor[0],
                joint_allowed,
            )
            recovery_joint_profile_index_value, recovery_joint_profile_probs = choose_index_from_logits(
                joint_logits,
                greedy=greedy,
                temperature=temperature,
                greedy_probability=greedy_probability,
                top_k=top_k,
            )
            (
                phase_index,
                admission_index,
                admission_limit_index,
                budget_profile_index,
                risky_admission_cap_index,
            ) = build_recovery_joint_profiles()[recovery_joint_profile_index_value]
            recovery_joint_profile_active = True

        if forced_action is None:
            action_heads = project_action_head_indices(
                phase_index,
                admission_index,
                admission_limit_index,
                quantile_index,
                budget_profile_index,
                risky_admission_cap_index,
                prefill_safety_filter_index,
                decode_safety_filter_index,
            )
            action_index = action_index_from_heads(*action_heads)
            prefill_policy, decode_policy = router_policies_from_heads(*action_heads)
            sampled_action = SchedulerAction.from_router_policies(prefill_policy, decode_policy)
        else:
            action_heads = action_heads_from_action(forced_action)
            action_index = action_index_from_action(forced_action)
            (
                phase_index,
                admission_index,
                admission_limit_index,
                quantile_index,
                budget_profile_index,
                risky_admission_cap_index,
                prefill_safety_filter_index,
                decode_safety_filter_index,
            ) = action_heads
            sampled_action = forced_action
            decode_decision_gate_index = DECISION_GATE_CHOICES.index(True)

        old_prefill_log_prob, old_decode_log_prob = self._log_probs_for_heads(
            prefill_outputs,
            decode_outputs,
            action_heads,
            action_mask,
            prefill_decision_gate_index=prefill_decision_gate_index,
            decode_decision_gate_index=decode_decision_gate_index,
        )
        prefill_policy_entropy = sum(
            categorical_entropy_from_probs(values)
            for values in (
                admission_probs,
                admission_limit_probs,
                quantile_probs,
                risky_admission_cap_probs,
                prefill_safety_filter_probs,
                prefill_decision_gate_probs,
            )
        )
        decode_policy_entropy = sum(
            categorical_entropy_from_probs(values)
            for values in (
                phase_probs,
                budget_profile_probs,
                decode_safety_filter_probs,
                decode_decision_gate_probs,
            )
        )
        central_tensor = torch.tensor(normalized_central, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            critic_outputs = self.central_critic(central_tensor)
        prefill_critic_value = float(critic_outputs["prefill_value"][0].detach().cpu().item())
        decode_critic_value = float(critic_outputs["decode_value"][0].detach().cpu().item())
        prefill_anchor_kl = 0.0
        decode_anchor_kl = 0.0
        if self._anchor_model is not None:
            with torch.no_grad():
                anchor_outputs = self._anchor_model(prefill_tensor, decode_tensor)
            prefill_anchor_kl = (
                categorical_kl_from_logits(
                    prefill_outputs["admission_logits"][0],
                    anchor_outputs["admission_logits"][0],
                    admission_allowed,
                )
                + categorical_kl_from_logits(
                    prefill_outputs["admission_limit_logits"][0],
                    anchor_outputs["admission_limit_logits"][0],
                    admission_limit_allowed,
                )
                + categorical_kl_from_logits(
                    prefill_outputs["quantile_logits"][0],
                    anchor_outputs["quantile_logits"][0],
                )
                + categorical_kl_from_logits(
                    prefill_outputs["risky_admission_cap_logits"][0],
                    anchor_outputs["risky_admission_cap_logits"][0],
                    risky_allowed,
                )
                + categorical_kl_from_logits(
                    prefill_outputs["prefill_safety_filter_logits"][0],
                    anchor_outputs["prefill_safety_filter_logits"][0],
                )
                + categorical_kl_from_logits(
                    prefill_outputs["prefill_decision_gate_logits"][0],
                    anchor_outputs["prefill_decision_gate_logits"][0],
                )
            )
            decode_anchor_kl = (
                categorical_kl_from_logits(
                    decode_outputs["phase_logits"][0],
                    anchor_outputs["phase_logits"][0],
                    phase_allowed,
                )
                + categorical_kl_from_logits(
                    decode_outputs["budget_profile_logits"][0],
                    anchor_outputs["budget_profile_logits"][0],
                    budget_allowed,
                )
                + categorical_kl_from_logits(
                    decode_outputs["decode_safety_filter_logits"][0],
                    anchor_outputs["decode_safety_filter_logits"][0],
                )
                + categorical_kl_from_logits(
                    decode_outputs["decode_decision_gate_logits"][0],
                    anchor_outputs["decode_decision_gate_logits"][0],
                )
            )
        detail: Dict[str, object] = {
            "action_index": action_index,
            "action_heads": action_heads,
            "scheduler_action": sampled_action,
            "prefill_observation": [float(value) for value in prefill_observation],
            "decode_observation": [float(value) for value in conditioned_decode_observation],
            "conditioned_decode_observation": conditioned_decode_observation,
            "normalized_prefill_observation": normalized_prefill,
            "normalized_decode_observation": normalized_decode,
            "normalized_central_observation": normalized_central,
            "old_prefill_log_prob": old_prefill_log_prob,
            "old_decode_log_prob": old_decode_log_prob,
            "prefill_decision_gate_index": prefill_decision_gate_index,
            "decode_decision_gate_index": decode_decision_gate_index,
            "prefill_decision_gate": bool(DECISION_GATE_CHOICES[prefill_decision_gate_index]),
            "decode_decision_gate": bool(DECISION_GATE_CHOICES[decode_decision_gate_index]),
            "recovery_joint_profile_active": recovery_joint_profile_active,
            "recovery_joint_profile_index": recovery_joint_profile_index_value,
            "prefill_policy_entropy": prefill_policy_entropy,
            "decode_policy_entropy": decode_policy_entropy,
            "policy_entropy": prefill_policy_entropy + decode_policy_entropy,
            "prefill_critic_value": prefill_critic_value,
            "decode_critic_value": decode_critic_value,
            "critic_value": prefill_critic_value + decode_critic_value,
            "prefill_anchor_kl_to_teacher": prefill_anchor_kl,
            "decode_anchor_kl_to_teacher": decode_anchor_kl,
            "anchor_kl_to_teacher": prefill_anchor_kl + decode_anchor_kl,
            "action_mask": {
                str(key): [int(index) for index in value]
                for key, value in dict(action_mask or {}).items()
            },
            "sampled_prefill_policy": prefill_policy_to_dict(sampled_action.prefill_policy()),
            "sampled_decode_policy": decode_policy_to_dict(sampled_action.decode_policy()),
        }
        if capture_policy:
            detail["policy_snapshot"] = {
                "normalized_prefill_observation": normalized_prefill,
                "normalized_decode_observation": normalized_decode,
                "normalized_central_observation": normalized_central,
                "phase_probs": phase_probs,
                "admission_probs": admission_probs,
                "admission_limit_probs": admission_limit_probs,
                "quantile_probs": quantile_probs,
                "budget_profile_probs": budget_profile_probs,
                "risky_admission_cap_probs": risky_admission_cap_probs,
                "recovery_joint_profile_probs": recovery_joint_profile_probs,
                "prefill_safety_filter_probs": prefill_safety_filter_probs,
                "decode_safety_filter_probs": decode_safety_filter_probs,
                "prefill_decision_gate_probs": prefill_decision_gate_probs,
                "decode_decision_gate_probs": decode_decision_gate_probs,
            }
        else:
            detail["policy_snapshot"] = None
        detail["sampling_overrides"] = {
            "temperature": temperature,
            "greedy_probability": greedy_probability,
            "top_k": top_k,
        }
        detail["forced_action"] = action_to_dict(forced_action) if forced_action is not None else None
        return detail

    def sample_joint_action_detail(
        self,
        prefill_observation: Sequence[float],
        decode_observation: Sequence[float],
        central_observation: Optional[Sequence[float]] = None,
        greedy: bool = False,
        capture_policy: bool = False,
        sampling_overrides: Optional[Dict[str, float]] = None,
        forced_action: Optional[SchedulerAction] = None,
        action_mask: Optional[Dict[str, Sequence[int]]] = None,
    ) -> Dict[str, object]:
        return self.sample_action_detail(
            prefill_observation,
            decode_observation,
            central_observation,
            greedy=greedy,
            capture_policy=capture_policy,
            sampling_overrides=sampling_overrides,
            forced_action=forced_action,
            action_mask=action_mask,
        )

    def sample_prefill_action_detail(
        self,
        prefill_observation: Sequence[float],
        decode_observation: Sequence[float],
        central_observation: Optional[Sequence[float]] = None,
        *,
        base_decode_policy: Optional[DecodeRouterPolicy] = None,
        greedy: bool = False,
        capture_policy: bool = False,
        sampling_overrides: Optional[Dict[str, float]] = None,
        forced_action: Optional[SchedulerAction] = None,
        action_mask: Optional[Dict[str, Sequence[int]]] = None,
    ) -> Dict[str, object]:
        normalized_prefill = self._prepare_prefill_observation(prefill_observation, update_normalizer=not greedy)
        prefill_tensor = torch.tensor(normalized_prefill, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            prefill_outputs = self.model.prefill_outputs(prefill_tensor)
        effective_sampling = sampling_overrides or {}
        temperature = float(effective_sampling.get("temperature", self.config.exploration_temperature))
        greedy_probability = float(
            effective_sampling.get("greedy_probability", self.config.exploration_greedy_prob)
        )
        top_k = int(effective_sampling.get("top_k", self.config.exploration_top_k))

        admission_allowed = mask_indices(action_mask, "admission", len(ADMISSION_CHOICES))
        admission_index, admission_probs = choose_index_from_logits(
            masked_logits(prefill_outputs["admission_logits"][0], admission_allowed),
            greedy=greedy,
            temperature=temperature,
            greedy_probability=greedy_probability,
            top_k=top_k,
        )
        admission_limit_allowed = mask_indices(action_mask, "admission_limit", len(NEW_ADMISSION_LIMIT_CHOICES))
        admission_limit_index, admission_limit_probs = choose_index_from_logits(
            masked_logits(prefill_outputs["admission_limit_logits"][0], admission_limit_allowed),
            greedy=greedy,
            temperature=temperature,
            greedy_probability=greedy_probability,
            top_k=top_k,
        )
        quantile_index, quantile_probs = choose_index_from_logits(
            prefill_outputs["quantile_logits"][0],
            greedy=greedy,
            temperature=temperature,
            greedy_probability=greedy_probability,
            top_k=top_k,
        )
        risky_allowed = mask_indices(action_mask, "risky_admission_cap", len(RISKY_ADMISSION_CAP_CHOICES))
        risky_admission_cap_index, risky_admission_cap_probs = choose_index_from_logits(
            masked_logits(prefill_outputs["risky_admission_cap_logits"][0], risky_allowed),
            greedy=greedy,
            temperature=temperature,
            greedy_probability=greedy_probability,
            top_k=top_k,
        )
        prefill_safety_filter_index, prefill_safety_filter_probs = choose_index_from_logits(
            prefill_outputs["prefill_safety_filter_logits"][0],
            greedy=greedy,
            temperature=temperature,
            greedy_probability=greedy_probability,
            top_k=top_k,
        )
        prefill_decision_gate_index, prefill_decision_gate_probs = choose_index_from_logits(
            prefill_outputs["prefill_decision_gate_logits"][0],
            greedy=greedy,
            temperature=temperature,
            greedy_probability=greedy_probability,
            top_k=top_k,
        )
        if forced_action is None:
            sampled_prefill_policy = PrefillRouterPolicy(
                admission_profile=ADMISSION_CHOICES[admission_index],
                new_admission_limit=NEW_ADMISSION_LIMIT_CHOICES[admission_limit_index],
                shadow_quantile=QUANTILE_CHOICES[quantile_index],
                risky_admission_cap=RISKY_ADMISSION_CAP_CHOICES[risky_admission_cap_index],
                safety_filter_enabled=SAFETY_FILTER_CHOICES[prefill_safety_filter_index],
            )
            decode_policy = base_decode_policy or DecodeRouterPolicy()
        else:
            sampled_prefill_policy = forced_action.prefill_policy()
            decode_policy = forced_action.decode_policy()
            forced_heads = action_heads_from_action(forced_action)
            (
                _forced_phase_index,
                admission_index,
                admission_limit_index,
                quantile_index,
                _forced_budget_profile_index,
                risky_admission_cap_index,
                prefill_safety_filter_index,
                _forced_decode_safety_filter_index,
            ) = forced_heads
            prefill_decision_gate_index = DECISION_GATE_CHOICES.index(True)
        conditioned_decode_observation = list(decode_observation)
        if (
            not self.config.no_prefill_to_decode_condition
            and str(getattr(self.config, "router_boundary_mode", "current") or "current") != "strict_env_only"
        ):
            conditioned_decode_observation = decode_observation_conditioned_on_prefill(
                decode_observation,
                sampled_prefill_policy,
            )
        normalized_decode = self._prepare_decode_observation(
            conditioned_decode_observation,
            update_normalizer=not greedy,
        )
        if central_observation is None:
            central_observation = [float(value) for value in prefill_observation] + [
                float(value) for value in conditioned_decode_observation
            ]
        if self.config.serialized_joint_critic:
            central_values = [float(value) for value in central_observation]
            base_dim = len(prefill_observation) + len(conditioned_decode_observation)
            history_tail = central_values[base_dim:] if len(central_values) > base_dim else []
            central_observation = [float(value) for value in prefill_observation] + [
                float(value) for value in conditioned_decode_observation
            ] + history_tail
        normalized_central = self._prepare_central_observation(central_observation, update_normalizer=not greedy)
        decode_tensor = torch.tensor(normalized_decode, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            decode_outputs = self.model.decode_outputs(decode_tensor)
        phase_label = decode_policy.phase_preference.value if decode_policy.phase_preference is not None else "heuristic"
        phase_index = PHASE_CHOICES.index(phase_label)
        budget_profile_index = BUDGET_PROFILE_CHOICES.index(decode_policy.budget_profile)
        decode_safety_filter_index = SAFETY_FILTER_CHOICES.index(
            bool(decode_policy.safety_filter_enabled) if decode_policy.safety_filter_enabled is not None else True
        )
        decode_decision_gate_index = DECISION_GATE_CHOICES.index(True)
        phase_allowed = mask_indices(action_mask, "phase", len(PHASE_CHOICES))
        budget_allowed = allowed_indices(
            BUDGET_PROFILE_CHOICES,
            allowed_budget_profiles(phase_label, sampled_prefill_policy.admission_profile, BUDGET_PROFILE_CHOICES),
        )
        budget_allowed = mask_indices(action_mask, "budget_profile", len(BUDGET_PROFILE_CHOICES), budget_allowed)
        phase_probs = torch.softmax(
            masked_logits(decode_outputs["phase_logits"][0], phase_allowed),
            dim=-1,
        ).detach().cpu().numpy()
        budget_profile_probs = torch.softmax(
            masked_logits(decode_outputs["budget_profile_logits"][0], budget_allowed),
            dim=-1,
        ).detach().cpu().numpy()
        decode_safety_filter_probs = torch.softmax(
            decode_outputs["decode_safety_filter_logits"][0],
            dim=-1,
        ).detach().cpu().numpy()
        decode_decision_gate_probs = torch.softmax(
            decode_outputs["decode_decision_gate_logits"][0],
            dim=-1,
        ).detach().cpu().numpy()
        sampled_action = SchedulerAction.from_router_policies(sampled_prefill_policy, decode_policy)
        action_heads = action_heads_from_action(sampled_action)
        action_index = action_index_from_action(sampled_action)
        old_prefill_log_prob, old_decode_log_prob = self._log_probs_for_heads(
            prefill_outputs,
            decode_outputs,
            action_heads,
            action_mask,
            prefill_decision_gate_index=prefill_decision_gate_index,
            decode_decision_gate_index=decode_decision_gate_index,
        )
        prefill_policy_entropy = sum(
            categorical_entropy_from_probs(values)
            for values in (
                admission_probs,
                admission_limit_probs,
                quantile_probs,
                risky_admission_cap_probs,
                prefill_safety_filter_probs,
                prefill_decision_gate_probs,
            )
        )
        decode_policy_entropy = sum(
            categorical_entropy_from_probs(values)
            for values in (
                phase_probs,
                budget_profile_probs,
                decode_safety_filter_probs,
                decode_decision_gate_probs,
            )
        )
        central_tensor = torch.tensor(normalized_central, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            critic_outputs = self.central_critic(central_tensor)
        prefill_critic_value = float(critic_outputs["prefill_value"][0].detach().cpu().item())
        decode_critic_value = float(critic_outputs["decode_value"][0].detach().cpu().item())
        prefill_anchor_kl = 0.0
        decode_anchor_kl = 0.0
        if self._anchor_model is not None:
            with torch.no_grad():
                anchor_outputs = self._anchor_model(prefill_tensor, decode_tensor)
            prefill_anchor_kl = (
                categorical_kl_from_logits(
                    prefill_outputs["admission_logits"][0],
                    anchor_outputs["admission_logits"][0],
                    admission_allowed,
                )
                + categorical_kl_from_logits(
                    prefill_outputs["admission_limit_logits"][0],
                    anchor_outputs["admission_limit_logits"][0],
                    admission_limit_allowed,
                )
                + categorical_kl_from_logits(
                    prefill_outputs["quantile_logits"][0],
                    anchor_outputs["quantile_logits"][0],
                )
                + categorical_kl_from_logits(
                    prefill_outputs["risky_admission_cap_logits"][0],
                    anchor_outputs["risky_admission_cap_logits"][0],
                    risky_allowed,
                )
                + categorical_kl_from_logits(
                    prefill_outputs["prefill_safety_filter_logits"][0],
                    anchor_outputs["prefill_safety_filter_logits"][0],
                )
                + categorical_kl_from_logits(
                    prefill_outputs["prefill_decision_gate_logits"][0],
                    anchor_outputs["prefill_decision_gate_logits"][0],
                )
            )
            decode_anchor_kl = (
                categorical_kl_from_logits(
                    decode_outputs["phase_logits"][0],
                    anchor_outputs["phase_logits"][0],
                    phase_allowed,
                )
                + categorical_kl_from_logits(
                    decode_outputs["budget_profile_logits"][0],
                    anchor_outputs["budget_profile_logits"][0],
                    budget_allowed,
                )
                + categorical_kl_from_logits(
                    decode_outputs["decode_safety_filter_logits"][0],
                    anchor_outputs["decode_safety_filter_logits"][0],
                )
                + categorical_kl_from_logits(
                    decode_outputs["decode_decision_gate_logits"][0],
                    anchor_outputs["decode_decision_gate_logits"][0],
                )
            )
        detail: Dict[str, object] = {
            "action_index": action_index,
            "action_heads": action_heads,
            "scheduler_action": sampled_action,
            "prefill_observation": [float(value) for value in prefill_observation],
            "decode_observation": [float(value) for value in conditioned_decode_observation],
            "conditioned_decode_observation": conditioned_decode_observation,
            "normalized_prefill_observation": normalized_prefill,
            "normalized_decode_observation": normalized_decode,
            "normalized_central_observation": normalized_central,
            "old_prefill_log_prob": old_prefill_log_prob,
            "old_decode_log_prob": old_decode_log_prob,
            "prefill_decision_gate_index": prefill_decision_gate_index,
            "decode_decision_gate_index": decode_decision_gate_index,
            "prefill_decision_gate": bool(DECISION_GATE_CHOICES[prefill_decision_gate_index]),
            "decode_decision_gate": bool(DECISION_GATE_CHOICES[decode_decision_gate_index]),
            "recovery_joint_profile_active": False,
            "recovery_joint_profile_index": -1,
            "prefill_policy_entropy": prefill_policy_entropy,
            "decode_policy_entropy": decode_policy_entropy,
            "policy_entropy": prefill_policy_entropy + decode_policy_entropy,
            "prefill_critic_value": prefill_critic_value,
            "decode_critic_value": decode_critic_value,
            "critic_value": prefill_critic_value + decode_critic_value,
            "prefill_anchor_kl_to_teacher": prefill_anchor_kl,
            "decode_anchor_kl_to_teacher": decode_anchor_kl,
            "anchor_kl_to_teacher": prefill_anchor_kl + decode_anchor_kl,
            "action_mask": {
                str(key): [int(index) for index in value]
                for key, value in dict(action_mask or {}).items()
            },
            "sampled_prefill_policy": prefill_policy_to_dict(sampled_action.prefill_policy()),
            "sampled_decode_policy": decode_policy_to_dict(sampled_action.decode_policy()),
        }
        if capture_policy:
            detail["policy_snapshot"] = {
                "normalized_prefill_observation": normalized_prefill,
                "normalized_decode_observation": normalized_decode,
                "normalized_central_observation": normalized_central,
                "phase_probs": phase_probs,
                "admission_probs": admission_probs,
                "admission_limit_probs": admission_limit_probs,
                "quantile_probs": quantile_probs,
                "budget_profile_probs": budget_profile_probs,
                "risky_admission_cap_probs": risky_admission_cap_probs,
                "recovery_joint_profile_probs": np.zeros(len(build_recovery_joint_profiles()), dtype=np.float64),
                "prefill_safety_filter_probs": prefill_safety_filter_probs,
                "decode_safety_filter_probs": decode_safety_filter_probs,
                "prefill_decision_gate_probs": prefill_decision_gate_probs,
                "decode_decision_gate_probs": decode_decision_gate_probs,
            }
        else:
            detail["policy_snapshot"] = None
        detail["sampling_overrides"] = {
            "temperature": temperature,
            "greedy_probability": greedy_probability,
            "top_k": top_k,
        }
        detail["forced_action"] = action_to_dict(forced_action) if forced_action is not None else None
        return detail

    def sample_decode_action_detail(
        self,
        prefill_observation: Sequence[float],
        decode_observation: Sequence[float],
        central_observation: Optional[Sequence[float]] = None,
        *,
        prefill_policy: Optional[PrefillRouterPolicy] = None,
        greedy: bool = False,
        capture_policy: bool = False,
        sampling_overrides: Optional[Dict[str, float]] = None,
        forced_action: Optional[SchedulerAction] = None,
        action_mask: Optional[Dict[str, Sequence[int]]] = None,
    ) -> Dict[str, object]:
        normalized_prefill = self._prepare_prefill_observation(prefill_observation, update_normalizer=not greedy)
        prefill_tensor = torch.tensor(normalized_prefill, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            prefill_outputs = self.model.prefill_outputs(prefill_tensor)
        sampled_prefill_policy = forced_action.prefill_policy() if forced_action is not None else (prefill_policy or PrefillRouterPolicy())
        conditioned_decode_observation = [float(value) for value in decode_observation]
        normalized_decode = self._prepare_decode_observation(
            conditioned_decode_observation,
            update_normalizer=not greedy,
        )
        if central_observation is None:
            central_observation = [float(value) for value in prefill_observation] + [
                float(value) for value in conditioned_decode_observation
            ]
        if self.config.serialized_joint_critic:
            central_values = [float(value) for value in central_observation]
            base_dim = len(prefill_observation) + len(conditioned_decode_observation)
            history_tail = central_values[base_dim:] if len(central_values) > base_dim else []
            central_observation = [float(value) for value in prefill_observation] + [
                float(value) for value in conditioned_decode_observation
            ] + history_tail
        normalized_central = self._prepare_central_observation(central_observation, update_normalizer=not greedy)
        decode_tensor = torch.tensor(normalized_decode, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            decode_outputs = self.model.decode_outputs(decode_tensor)
        effective_sampling = sampling_overrides or {}
        temperature = float(effective_sampling.get("temperature", self.config.exploration_temperature))
        greedy_probability = float(
            effective_sampling.get("greedy_probability", self.config.exploration_greedy_prob)
        )
        top_k = int(effective_sampling.get("top_k", self.config.exploration_top_k))
        phase_allowed = mask_indices(action_mask, "phase", len(PHASE_CHOICES))
        phase_index, phase_probs = choose_index_from_logits(
            masked_logits(decode_outputs["phase_logits"][0], phase_allowed),
            greedy=greedy,
            temperature=temperature,
            greedy_probability=greedy_probability,
            top_k=top_k,
        )
        phase_label = PHASE_CHOICES[phase_index]
        budget_allowed = allowed_indices(
            BUDGET_PROFILE_CHOICES,
            allowed_budget_profiles(phase_label, sampled_prefill_policy.admission_profile, BUDGET_PROFILE_CHOICES),
        )
        budget_allowed = mask_indices(action_mask, "budget_profile", len(BUDGET_PROFILE_CHOICES), budget_allowed)
        budget_profile_index, budget_profile_probs = choose_index_from_logits(
            masked_logits(decode_outputs["budget_profile_logits"][0], budget_allowed),
            greedy=greedy,
            temperature=temperature,
            greedy_probability=greedy_probability,
            top_k=top_k,
        )
        decode_safety_filter_index, decode_safety_filter_probs = choose_index_from_logits(
            decode_outputs["decode_safety_filter_logits"][0],
            greedy=greedy,
            temperature=temperature,
            greedy_probability=greedy_probability,
            top_k=top_k,
        )
        decode_decision_gate_index, decode_decision_gate_probs = choose_index_from_logits(
            decode_outputs["decode_decision_gate_logits"][0],
            greedy=greedy,
            temperature=temperature,
            greedy_probability=greedy_probability,
            top_k=top_k,
        )
        if forced_action is not None:
            (
                phase_index,
                _admission_index,
                _admission_limit_index,
                _quantile_index,
                budget_profile_index,
                _risky_index,
                _prefill_safety_filter_index,
                decode_safety_filter_index,
            ) = action_heads_from_action(forced_action)
            decode_decision_gate_index = DECISION_GATE_CHOICES.index(True)
        decode_policy = DecodeRouterPolicy(
            phase_preference=None if PHASE_CHOICES[phase_index] == "heuristic" else PhaseMode(PHASE_CHOICES[phase_index]),
            budget_profile=BUDGET_PROFILE_CHOICES[budget_profile_index],
            safety_filter_enabled=SAFETY_FILTER_CHOICES[decode_safety_filter_index],
        )
        sampled_action = SchedulerAction.from_router_policies(sampled_prefill_policy, decode_policy)
        action_heads = action_heads_from_action(sampled_action)
        action_index = action_index_from_action(sampled_action)
        old_prefill_log_prob, old_decode_log_prob = self._log_probs_for_heads(
            prefill_outputs,
            decode_outputs,
            action_heads,
            action_mask,
            prefill_decision_gate_index=DECISION_GATE_CHOICES.index(True),
            decode_decision_gate_index=decode_decision_gate_index,
        )
        phase_probs_full = torch.softmax(
            masked_logits(decode_outputs["phase_logits"][0], phase_allowed),
            dim=-1,
        ).detach().cpu().numpy()
        budget_profile_probs_full = torch.softmax(
            masked_logits(decode_outputs["budget_profile_logits"][0], budget_allowed),
            dim=-1,
        ).detach().cpu().numpy()
        decode_safety_filter_probs_full = torch.softmax(
            decode_outputs["decode_safety_filter_logits"][0],
            dim=-1,
        ).detach().cpu().numpy()
        decode_decision_gate_probs_full = torch.softmax(
            decode_outputs["decode_decision_gate_logits"][0],
            dim=-1,
        ).detach().cpu().numpy()
        prefill_policy_entropy = 0.0
        decode_policy_entropy = sum(
            categorical_entropy_from_probs(values)
            for values in (
                phase_probs_full,
                budget_profile_probs_full,
                decode_safety_filter_probs_full,
                decode_decision_gate_probs_full,
            )
        )
        central_tensor = torch.tensor(normalized_central, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            critic_outputs = self.central_critic(central_tensor)
        prefill_critic_value = float(critic_outputs["prefill_value"][0].detach().cpu().item())
        decode_critic_value = float(critic_outputs["decode_value"][0].detach().cpu().item())
        prefill_anchor_kl = 0.0
        decode_anchor_kl = 0.0
        if self._anchor_model is not None:
            with torch.no_grad():
                anchor_outputs = self._anchor_model(prefill_tensor, decode_tensor)
            decode_anchor_kl = (
                categorical_kl_from_logits(
                    decode_outputs["phase_logits"][0],
                    anchor_outputs["phase_logits"][0],
                    phase_allowed,
                )
                + categorical_kl_from_logits(
                    decode_outputs["budget_profile_logits"][0],
                    anchor_outputs["budget_profile_logits"][0],
                    budget_allowed,
                )
                + categorical_kl_from_logits(
                    decode_outputs["decode_safety_filter_logits"][0],
                    anchor_outputs["decode_safety_filter_logits"][0],
                )
                + categorical_kl_from_logits(
                    decode_outputs["decode_decision_gate_logits"][0],
                    anchor_outputs["decode_decision_gate_logits"][0],
                )
            )
        detail = {
            "action_index": action_index,
            "action_heads": action_heads,
            "scheduler_action": sampled_action,
            "prefill_observation": [float(value) for value in prefill_observation],
            "decode_observation": [float(value) for value in conditioned_decode_observation],
            "conditioned_decode_observation": conditioned_decode_observation,
            "normalized_prefill_observation": normalized_prefill,
            "normalized_decode_observation": normalized_decode,
            "normalized_central_observation": normalized_central,
            "old_prefill_log_prob": old_prefill_log_prob,
            "old_decode_log_prob": old_decode_log_prob,
            "prefill_decision_gate_index": DECISION_GATE_CHOICES.index(True),
            "decode_decision_gate_index": decode_decision_gate_index,
            "prefill_decision_gate": True,
            "decode_decision_gate": bool(DECISION_GATE_CHOICES[decode_decision_gate_index]),
            "recovery_joint_profile_active": False,
            "recovery_joint_profile_index": -1,
            "prefill_policy_entropy": prefill_policy_entropy,
            "decode_policy_entropy": decode_policy_entropy,
            "policy_entropy": prefill_policy_entropy + decode_policy_entropy,
            "prefill_critic_value": prefill_critic_value,
            "decode_critic_value": decode_critic_value,
            "critic_value": prefill_critic_value + decode_critic_value,
            "prefill_anchor_kl_to_teacher": prefill_anchor_kl,
            "decode_anchor_kl_to_teacher": decode_anchor_kl,
            "anchor_kl_to_teacher": prefill_anchor_kl + decode_anchor_kl,
            "action_mask": {
                str(key): [int(index) for index in value]
                for key, value in dict(action_mask or {}).items()
            },
            "sampled_prefill_policy": prefill_policy_to_dict(sampled_action.prefill_policy()),
            "sampled_decode_policy": decode_policy_to_dict(sampled_action.decode_policy()),
        }
        if capture_policy:
            detail["policy_snapshot"] = {
                "normalized_prefill_observation": normalized_prefill,
                "normalized_decode_observation": normalized_decode,
                "normalized_central_observation": normalized_central,
                "phase_probs": phase_probs_full,
                "admission_probs": np.asarray([], dtype=np.float64),
                "admission_limit_probs": np.asarray([], dtype=np.float64),
                "quantile_probs": np.asarray([], dtype=np.float64),
                "budget_profile_probs": budget_profile_probs_full,
                "risky_admission_cap_probs": np.asarray([], dtype=np.float64),
                "recovery_joint_profile_probs": np.zeros(len(build_recovery_joint_profiles()), dtype=np.float64),
                "prefill_safety_filter_probs": np.asarray([], dtype=np.float64),
                "decode_safety_filter_probs": decode_safety_filter_probs_full,
                "prefill_decision_gate_probs": np.asarray([], dtype=np.float64),
                "decode_decision_gate_probs": decode_decision_gate_probs_full,
            }
        else:
            detail["policy_snapshot"] = None
        detail["sampling_overrides"] = {
            "temperature": temperature,
            "greedy_probability": greedy_probability,
            "top_k": top_k,
        }
        detail["forced_action"] = action_to_dict(forced_action) if forced_action is not None else None
        return detail

    def record_reward(self, prefill_reward: float, decode_reward: float) -> None:
        self.prefill_rewards.append(prefill_reward)
        self.decode_rewards.append(decode_reward)
        self.rewards.append(prefill_reward + decode_reward)

    def apply_terminal_reward(
        self,
        prefill_reward: float,
        decode_reward: float,
        summary: Optional[Dict[str, object]] = None,
    ) -> None:
        total_reward = prefill_reward + decode_reward
        if self.rewards:
            self.prefill_rewards[-1] += prefill_reward
            self.decode_rewards[-1] += decode_reward
            self.rewards[-1] += total_reward
        else:
            self.prefill_rewards.append(prefill_reward)
            self.decode_rewards.append(decode_reward)
            self.rewards.append(total_reward)
        if self.transition_rows:
            self.transition_rows[-1]["reward"] = round(float(self.transition_rows[-1]["reward"]) + total_reward, 6)
            self.transition_rows[-1]["prefill_reward"] = round(
                float(self.transition_rows[-1]["prefill_reward"]) + prefill_reward,
                6,
            )
            self.transition_rows[-1]["decode_reward"] = round(
                float(self.transition_rows[-1]["decode_reward"]) + decode_reward,
                6,
            )
            self.transition_rows[-1]["terminal_reward"] = round(total_reward, 6)
            self.transition_rows[-1]["terminal_prefill_reward"] = round(prefill_reward, 6)
            self.transition_rows[-1]["terminal_decode_reward"] = round(decode_reward, 6)
            if summary is not None:
                self.transition_rows[-1]["terminal_summary"] = {
                    "terminated_early": bool(summary.get("terminated_early", False)),
                    "termination_reason": summary.get("termination_reason"),
                    "unfinished_requests": int(summary.get("unfinished_requests", 0)),
                    "evictions_total": int(summary.get("evictions_total", 0)),
                    "recompute_tokens_total": int(summary.get("recompute_tokens_total", 0)),
                }
        if self.step_events:
            self.step_events[-1]["reward"] = round(float(self.step_events[-1]["reward"]) + total_reward, 6)
            self.step_events[-1]["prefill_reward"] = round(
                float(self.step_events[-1]["prefill_reward"]) + prefill_reward,
                6,
            )
            self.step_events[-1]["decode_reward"] = round(
                float(self.step_events[-1]["decode_reward"]) + decode_reward,
                6,
            )
            self.step_events[-1]["terminal_reward"] = round(total_reward, 6)
            self.step_events[-1]["terminal_prefill_reward"] = round(prefill_reward, 6)
            self.step_events[-1]["terminal_decode_reward"] = round(decode_reward, 6)
            if summary is not None:
                self.step_events[-1]["termination_reason"] = summary.get("termination_reason")

    def record_transition(
        self,
        prefill_reward: float,
        decode_reward: float,
        action_index: int,
        phase_probs: np.ndarray,
        admission_probs: np.ndarray,
        admission_limit_probs: np.ndarray,
        quantile_probs: np.ndarray,
        budget_profile_probs: np.ndarray,
        risky_admission_cap_probs: np.ndarray,
        prefill_safety_filter_probs: np.ndarray,
        decode_safety_filter_probs: np.ndarray,
        prefill_observation: Sequence[float],
        decode_observation: Sequence[float],
        normalized_prefill_observation: Sequence[float],
        normalized_decode_observation: Sequence[float],
        step_result: Dict[str, float],
    ) -> None:
        total_reward = prefill_reward + decode_reward
        self.record_reward(prefill_reward, decode_reward)
        requested_action = ACTION_TABLE[action_index]
        action = action_from_step_result(step_result)
        requested_from_step = requested_action_from_step_result(step_result)
        effective_action_index = action_index_from_action(action)
        action_safety_triggered = bool(step_result.get("action_safety_triggered", 0.0))
        step_costs = {name: float(step_result.get(name, 0.0) or 0.0) for name in LIVENESS_COST_NAMES}
        attribution = {
            key: float(step_result.get(key, 0.0) or 0.0)
            for key in (
                "counterfactual_prefill_cost_delta",
                "counterfactual_decode_cost_delta",
                "counterfactual_prefill_responsibility",
                "counterfactual_decode_responsibility",
            )
        }
        self.rollout.set_last_projection(
            action_heads_from_action(action),
            action_safety_triggered,
            step_result.get("action_safety_reasons", ""),
            prefill_responsibility=attribution["counterfactual_prefill_responsibility"],
            decode_responsibility=attribution["counterfactual_decode_responsibility"],
        )
        prefill_policy = action.prefill_policy()
        decode_policy = action.decode_policy()
        joint_probs = joint_action_probabilities(
            phase_probs,
            admission_probs,
            admission_limit_probs,
            quantile_probs,
            budget_profile_probs,
            risky_admission_cap_probs,
            prefill_safety_filter_probs,
            decode_safety_filter_probs,
        )
        prefill_joint_probs = prefill_policy_joint_probabilities(
            admission_probs,
            admission_limit_probs,
            quantile_probs,
            risky_admission_cap_probs,
            prefill_safety_filter_probs,
        )
        decode_joint_probs = decode_policy_joint_probabilities(
            phase_probs,
            budget_profile_probs,
            decode_safety_filter_probs,
        )
        marginals = action_factor_marginals(
            phase_probs,
            admission_probs,
            admission_limit_probs,
            quantile_probs,
            budget_profile_probs,
            risky_admission_cap_probs,
            prefill_safety_filter_probs,
            decode_safety_filter_probs,
        )
        router_marginals = router_policy_marginals(
            phase_probs,
            admission_probs,
            admission_limit_probs,
            quantile_probs,
            budget_profile_probs,
            risky_admission_cap_probs,
            prefill_safety_filter_probs,
            decode_safety_filter_probs,
        )
        prefill_policy_probability = (
            float(admission_probs[ADMISSION_CHOICES.index(prefill_policy.admission_profile)])
            * float(
                admission_limit_probs[
                    NEW_ADMISSION_LIMIT_CHOICES.index(
                        _prefill_limit_value(prefill_policy.new_admission_limit)
                    )
                ]
            )
            * float(
                quantile_probs[
                    QUANTILE_CHOICES.index(
                        nearest_quantile_choice(float(prefill_policy.shadow_quantile or default_shadow_quantile_choice()))
                    )
                ]
            )
            * float(
                risky_admission_cap_probs[
                    RISKY_ADMISSION_CAP_CHOICES.index(
                        _risky_cap_value(prefill_policy.risky_admission_cap)
                    )
                ]
            )
            * float(
                prefill_safety_filter_probs[
                    SAFETY_FILTER_CHOICES.index(
                        bool(prefill_policy.safety_filter_enabled)
                        if prefill_policy.safety_filter_enabled is not None
                        else True
                    )
                ]
            )
        )
        phase_label = decode_policy.phase_preference.value if decode_policy.phase_preference is not None else "heuristic"
        decode_policy_probability = (
            float(phase_probs[PHASE_CHOICES.index(phase_label)])
            * float(budget_profile_probs[BUDGET_PROFILE_CHOICES.index(decode_policy.budget_profile)])
            * float(
                decode_safety_filter_probs[
                    SAFETY_FILTER_CHOICES.index(
                        bool(decode_policy.safety_filter_enabled)
                        if decode_policy.safety_filter_enabled is not None
                        else True
                    )
                ]
            )
        )
        action_mask = self.rollout.action_masks[-1] if self.rollout.action_masks else {}
        prefill_policy_logprob = (
            self.rollout.old_prefill_log_probs[-1]
            if self.rollout.old_prefill_log_probs
            else math.log(max(float(prefill_policy_probability), 1e-12))
        )
        decode_policy_logprob = (
            self.rollout.old_decode_log_probs[-1]
            if self.rollout.old_decode_log_probs
            else math.log(max(float(decode_policy_probability), 1e-12))
        )
        prefill_policy_entropy = (
            self.rollout.prefill_policy_entropies[-1] if self.rollout.prefill_policy_entropies else 0.0
        )
        decode_policy_entropy = (
            self.rollout.decode_policy_entropies[-1] if self.rollout.decode_policy_entropies else 0.0
        )
        action_snapshot = action_snapshot_fields(
            requested_from_step,
            action,
            action_mask,
            prefill_policy_logprob=prefill_policy_logprob,
            decode_policy_logprob=decode_policy_logprob,
            prefill_policy_entropy=prefill_policy_entropy,
            decode_policy_entropy=decode_policy_entropy,
            safety_projection_applied=action_safety_triggered,
            safety_projection_reasons=step_result.get("action_safety_reasons", ""),
        )
        self.transition_rows.append(
            {
                "reward": round(total_reward, 6),
                "prefill_reward": round(prefill_reward, 6),
                "decode_reward": round(decode_reward, 6),
                "executed_reward": round(float(step_result.get("executed_reward", total_reward) or 0.0), 6),
                "actor_projection_debit": round(float(step_result.get("actor_projection_debit", 0.0) or 0.0), 6),
                "actor_reward": round(float(step_result.get("actor_reward", total_reward) or 0.0), 6),
                **{
                    name: round(float(step_result.get(name, 0.0) or 0.0), 6)
                    for name in REWARD_COMPONENT_NAMES
                },
                "action_index": effective_action_index,
                "action": action_to_dict(action),
                "requested_action_index": action_index,
                "requested_action": action_to_dict(requested_action),
                "requested_action_from_step": action_to_dict(requested_from_step),
                "prefill_policy": prefill_policy_to_dict(prefill_policy),
                "decode_policy": decode_policy_to_dict(decode_policy),
                **action_snapshot,
                "action_safety_triggered": action_safety_triggered,
                "action_safety_reasons": str(step_result.get("action_safety_reasons", "")),
                "action_mask": action_mask,
                **{key: round(float(value), 6) for key, value in step_costs.items()},
                **{key: round(float(value), 6) for key, value in attribution.items()},
                "phase_mode": step_result["phase_mode"],
                "duration_ms": round(step_result["duration_ms"], 6),
                "prefill_tokens": step_result["prefill_tokens"],
                "prefill_padding_tokens": step_result.get("prefill_padding_tokens", 0.0),
                "decode_tokens": step_result["decode_tokens"],
                **{key: str(step_result.get(key, "")) for key in REQUEST_ID_LIST_KEYS},
                "prefill_completed_requests": step_result.get("prefill_completed_requests", 0.0),
                "completed_output_tokens": step_result.get("completed_output_tokens", 0.0),
                "new_admissions": step_result.get("new_admissions", 0.0),
                "new_admissions_fast": step_result.get("new_admissions_fast", 0.0),
                "new_admissions_normal": step_result.get("new_admissions_normal", 0.0),
                "new_admissions_risky": step_result.get("new_admissions_risky", 0.0),
                "admission_wait_sum_ms": round(step_result.get("admission_wait_sum_ms", 0.0), 6),
                "first_tokens_emitted": step_result["first_tokens_emitted"],
                "first_tokens_fast": step_result.get("first_tokens_fast", 0.0),
                "first_tokens_normal": step_result.get("first_tokens_normal", 0.0),
                "first_tokens_risky": step_result.get("first_tokens_risky", 0.0),
                "completed_requests": step_result["completed_requests"],
                "feedback_event_type": step_result.get("feedback_event_type"),
                "kv_after": step_result["kv_after"],
                "prefill_waiting_after": step_result["prefill_waiting_after"],
                "pending_unadmitted_after": step_result.get("pending_unadmitted_after", 0.0),
                "pending_unadmitted_fast_after": step_result.get("pending_unadmitted_fast_after", 0.0),
                "pending_unadmitted_normal_after": step_result.get("pending_unadmitted_normal_after", 0.0),
                "pending_unadmitted_risky_after": step_result.get("pending_unadmitted_risky_after", 0.0),
                "oldest_pending_admission_after_ms": round(
                    step_result.get("oldest_pending_admission_after_ms", 0.0),
                    6,
                ),
                "oldest_pending_admission_fast_after_ms": round(
                    step_result.get("oldest_pending_admission_fast_after_ms", 0.0),
                    6,
                ),
                "oldest_pending_admission_normal_after_ms": round(
                    step_result.get("oldest_pending_admission_normal_after_ms", 0.0),
                    6,
                ),
                "oldest_pending_admission_risky_after_ms": round(
                    step_result.get("oldest_pending_admission_risky_after_ms", 0.0),
                    6,
                ),
                "first_token_pending_after": step_result.get("first_token_pending_after", 0.0),
                "observation": [
                    round(float(value), 6)
                    for value in list(prefill_observation) + list(decode_observation)
                ],
                "normalized_observation": [
                    round(float(value), 6)
                    for value in list(normalized_prefill_observation) + list(normalized_decode_observation)
                ],
                "prefill_observation": [round(float(value), 6) for value in prefill_observation],
                "decode_observation": [round(float(value), 6) for value in decode_observation],
                "normalized_prefill_observation": [
                    round(float(value), 6) for value in normalized_prefill_observation
                ],
                "normalized_decode_observation": [
                    round(float(value), 6) for value in normalized_decode_observation
                ],
                "action_probability": round(float(joint_probs[action_index]), 6),
                "prefill_policy_probability": round(float(prefill_policy_probability), 6),
                "decode_policy_probability": round(float(decode_policy_probability), 6),
            }
        )
        if not action_safety_triggered:
            self.transition_rows[-1].update(
                {
                    "teacher_action_probabilities": [round(float(value), 8) for value in joint_probs],
                    "teacher_prefill_policy_probabilities": [round(float(value), 8) for value in prefill_joint_probs],
                    "teacher_decode_policy_probabilities": [round(float(value), 8) for value in decode_joint_probs],
                    "teacher_phase_probabilities": [round(float(value), 8) for value in marginals["phase"]],
                    "teacher_admission_probabilities": [round(float(value), 8) for value in marginals["admission"]],
                    "teacher_admission_limit_probabilities": [
                        round(float(value), 8) for value in marginals["admission_limit"]
                    ],
                    "teacher_quantile_probabilities": [round(float(value), 8) for value in marginals["quantile"]],
                    "teacher_budget_profile_probabilities": [
                        round(float(value), 8) for value in marginals["budget_profile"]
                    ],
                    "teacher_risky_admission_cap_probabilities": [
                        round(float(value), 8) for value in marginals["risky_admission_cap"]
                    ],
                    "teacher_prefill_safety_filter_probabilities": [
                        round(float(value), 8) for value in marginals["prefill_safety_filter"]
                    ],
                    "teacher_decode_safety_filter_probabilities": [
                        round(float(value), 8) for value in marginals["decode_safety_filter"]
                    ],
                    "teacher_prefill_router_marginals": {
                        key: [round(float(value), 8) for value in values]
                        for key, values in router_marginals["prefill"].items()
                    },
                    "teacher_decode_router_marginals": {
                        key: [round(float(value), 8) for value in values]
                        for key, values in router_marginals["decode"].items()
                    },
                }
            )

    def record_step_event(
        self,
        prefill_reward: float,
        decode_reward: float,
        action_index: int,
        step_result: Dict[str, float],
    ) -> None:
        if (
            step_result.get("new_admissions", 0.0) <= 0.0
            and step_result.get("prefill_completed_requests", 0.0) <= 0.0
            and step_result.get("first_tokens_emitted", 0.0) <= 0.0
            and step_result.get("completed_requests", 0.0) <= 0.0
            and step_result.get("evictions", 0.0) <= 0.0
            and step_result.get("recompute_tokens_added", 0.0) <= 0.0
        ):
            return
        action = action_from_step_result(step_result)
        requested_action = ACTION_TABLE[action_index]
        requested_from_step = requested_action_from_step_result(step_result)
        total_reward = prefill_reward + decode_reward
        step_costs = {name: float(step_result.get(name, 0.0) or 0.0) for name in LIVENESS_COST_NAMES}
        attribution = {
            key: float(step_result.get(key, 0.0) or 0.0)
            for key in (
                "counterfactual_prefill_cost_delta",
                "counterfactual_decode_cost_delta",
                "counterfactual_prefill_responsibility",
                "counterfactual_decode_responsibility",
            )
        }
        action_mask = self.rollout.action_masks[-1] if self.rollout.action_masks else {}
        action_snapshot = action_snapshot_fields(
            requested_from_step,
            action,
            action_mask,
            prefill_policy_logprob=self.rollout.old_prefill_log_probs[-1]
            if self.rollout.old_prefill_log_probs
            else 0.0,
            decode_policy_logprob=self.rollout.old_decode_log_probs[-1]
            if self.rollout.old_decode_log_probs
            else 0.0,
            prefill_policy_entropy=self.rollout.prefill_policy_entropies[-1]
            if self.rollout.prefill_policy_entropies
            else 0.0,
            decode_policy_entropy=self.rollout.decode_policy_entropies[-1]
            if self.rollout.decode_policy_entropies
            else 0.0,
            safety_projection_applied=bool(step_result.get("action_safety_triggered", 0.0)),
            safety_projection_reasons=step_result.get("action_safety_reasons", ""),
        )
        self.step_events.append(
            {
                "action_index": action_index_from_action(action),
                "action": action_to_dict(action),
                "requested_action_index": action_index,
                "requested_action": action_to_dict(requested_action),
                "requested_action_from_step": action_to_dict(requested_from_step),
                "prefill_policy": prefill_policy_to_dict(action.prefill_policy()),
                "decode_policy": decode_policy_to_dict(action.decode_policy()),
                **action_snapshot,
                "action_mask": action_mask,
                "action_safety_triggered": bool(step_result.get("action_safety_triggered", 0.0)),
                "action_safety_reasons": str(step_result.get("action_safety_reasons", "")),
                **{key: round(float(value), 6) for key, value in step_costs.items()},
                **{key: round(float(value), 6) for key, value in attribution.items()},
                "reward": round(total_reward, 6),
                "prefill_reward": round(prefill_reward, 6),
                "decode_reward": round(decode_reward, 6),
                "executed_reward": round(float(step_result.get("executed_reward", total_reward) or 0.0), 6),
                "actor_projection_debit": round(float(step_result.get("actor_projection_debit", 0.0) or 0.0), 6),
                "actor_reward": round(float(step_result.get("actor_reward", total_reward) or 0.0), 6),
                **{
                    name: round(float(step_result.get(name, 0.0) or 0.0), 6)
                    for name in REWARD_COMPONENT_NAMES
                },
                "phase_mode": step_result["phase_mode"],
                "duration_ms": round(step_result["duration_ms"], 6),
                **{key: str(step_result.get(key, "")) for key in REQUEST_ID_LIST_KEYS},
                "new_admissions": step_result.get("new_admissions", 0.0),
                "new_admissions_fast": step_result.get("new_admissions_fast", 0.0),
                "new_admissions_normal": step_result.get("new_admissions_normal", 0.0),
                "new_admissions_risky": step_result.get("new_admissions_risky", 0.0),
                "first_tokens_emitted": step_result.get("first_tokens_emitted", 0.0),
                "first_tokens_fast": step_result.get("first_tokens_fast", 0.0),
                "first_tokens_normal": step_result.get("first_tokens_normal", 0.0),
                "first_tokens_risky": step_result.get("first_tokens_risky", 0.0),
                "completed_requests": step_result.get("completed_requests", 0.0),
                "prefill_completed_requests": step_result.get("prefill_completed_requests", 0.0),
                "completed_output_tokens": step_result.get("completed_output_tokens", 0.0),
                "feedback_event_type": step_result.get("feedback_event_type"),
                "admission_wait_sum_ms": round(step_result.get("admission_wait_sum_ms", 0.0), 6),
                "ttft_sum_ms": round(step_result.get("ttft_sum_ms", 0.0), 6),
                "e2e_sum_ms": round(step_result.get("e2e_sum_ms", 0.0), 6),
                "pending_unadmitted_after": step_result.get("pending_unadmitted_after", 0.0),
                "pending_unadmitted_fast_after": step_result.get("pending_unadmitted_fast_after", 0.0),
                "pending_unadmitted_normal_after": step_result.get("pending_unadmitted_normal_after", 0.0),
                "pending_unadmitted_risky_after": step_result.get("pending_unadmitted_risky_after", 0.0),
                "first_token_pending_after": step_result.get("first_token_pending_after", 0.0),
                "kv_after": step_result.get("kv_after", 0.0),
            }
        )

    def finish_episode(self) -> Dict[str, float]:
        if not self.rewards:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "mean_return": 0.0,
                "prefill_policy_loss": 0.0,
                "decode_policy_loss": 0.0,
                "prefill_value_loss": 0.0,
                "decode_value_loss": 0.0,
                "projection_imitation_loss": 0.0,
                "targeted_projection_bc_loss": 0.0,
                "anchor_kl_loss": 0.0,
                "targeted_projection_bc_rows": 0.0,
                "targeted_projection_bc_heads": 0.0,
                "targeted_projection_bc_groups": 0.0,
                "prefill_mean_return": 0.0,
                "decode_mean_return": 0.0,
            }
        if (
            not self.prefill_observations
            or not self.decode_observations
            or not self.central_observations
            or not self.action_heads
        ):
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "mean_return": 0.0,
                "prefill_policy_loss": 0.0,
                "decode_policy_loss": 0.0,
                "prefill_value_loss": 0.0,
                "decode_value_loss": 0.0,
                "projection_imitation_loss": 0.0,
                "targeted_projection_bc_loss": 0.0,
                "anchor_kl_loss": 0.0,
                "targeted_projection_bc_rows": 0.0,
                "targeted_projection_bc_heads": 0.0,
                "targeted_projection_bc_groups": 0.0,
                "prefill_mean_return": 0.0,
                "decode_mean_return": 0.0,
            }

        sample_count = min(
            len(self.rewards),
            len(self.prefill_observations),
            len(self.decode_observations),
            len(self.central_observations),
            len(self.action_heads),
            len(self.rollout.requested_action_heads),
            len(self.rollout.projection_action_heads),
            len(self.rollout.projection_masks),
            len(self.rollout.projection_reason_sets),
            len(self.rollout.projection_prefill_responsibilities),
            len(self.rollout.projection_decode_responsibilities),
            len(self.rollout.sample_routers),
            len(self.rollout.action_masks),
            len(self.rollout.prefill_decision_gate_indices),
            len(self.rollout.decode_decision_gate_indices),
        )
        if sample_count <= 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "mean_return": 0.0,
                "prefill_policy_loss": 0.0,
                "decode_policy_loss": 0.0,
                "prefill_value_loss": 0.0,
                "decode_value_loss": 0.0,
                "projection_imitation_loss": 0.0,
                "targeted_projection_bc_loss": 0.0,
                "anchor_kl_loss": 0.0,
                "targeted_projection_bc_rows": 0.0,
                "targeted_projection_bc_heads": 0.0,
                "targeted_projection_bc_groups": 0.0,
                "prefill_mean_return": 0.0,
                "decode_mean_return": 0.0,
            }

        prefill_observations_tensor = torch.tensor(
            np.asarray(self.prefill_observations[:sample_count]),
            dtype=torch.float32,
            device=self.device,
        )
        decode_observations_tensor = torch.tensor(
            np.asarray(self.decode_observations[:sample_count]),
            dtype=torch.float32,
            device=self.device,
        )
        central_observations_tensor = torch.tensor(
            np.asarray(self.central_observations[:sample_count]),
            dtype=torch.float32,
            device=self.device,
        )
        old_prefill_log_probs_tensor = torch.tensor(
            self.rollout.old_prefill_log_probs[:sample_count],
            dtype=torch.float32,
            device=self.device,
        )
        old_decode_log_probs_tensor = torch.tensor(
            self.rollout.old_decode_log_probs[:sample_count],
            dtype=torch.float32,
            device=self.device,
        )
        projection_mask_tensor = torch.tensor(
            self.rollout.projection_masks[:sample_count],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            old_values = self.central_critic(central_observations_tensor)
            prefill_old_values = (
                self.prefill_return_normalizer.denormalize_tensor(old_values["prefill_value"])
                if self.config.value_normalization
                else old_values["prefill_value"]
            )
            decode_old_values = (
                self.decode_return_normalizer.denormalize_tensor(old_values["decode_value"])
                if self.config.value_normalization
                else old_values["decode_value"]
            )

        def compute_gae(rewards: Sequence[float], values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            advantages: List[float] = []
            running_advantage = 0.0
            value_rows = values.detach().cpu().numpy().tolist()
            for index in reversed(range(sample_count)):
                next_value = value_rows[index + 1] if index + 1 < sample_count else 0.0
                delta = float(rewards[index]) + self.config.gamma * float(next_value) - float(value_rows[index])
                running_advantage = delta + self.config.gamma * self.config.gae_lambda * running_advantage
                advantages.append(running_advantage)
            advantages = list(reversed(advantages))
            advantage_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
            return_tensor = advantage_tensor + values.detach()
            return advantage_tensor, return_tensor

        prefill_advantages, prefill_returns_tensor = compute_gae(
            self.prefill_rewards[:sample_count],
            prefill_old_values,
        )
        decode_advantages, decode_returns_tensor = compute_gae(
            self.decode_rewards[:sample_count],
            decode_old_values,
        )
        if self.config.advantage_normalization and prefill_advantages.numel() > 1:
            prefill_advantages = (
                prefill_advantages - prefill_advantages.mean()
            ) / (prefill_advantages.std(unbiased=False) + 1e-8)
        if self.config.advantage_normalization and decode_advantages.numel() > 1:
            decode_advantages = (
                decode_advantages - decode_advantages.mean()
            ) / (decode_advantages.std(unbiased=False) + 1e-8)

        self.prefill_return_normalizer.update(prefill_returns_tensor.detach().cpu().numpy().tolist())
        self.decode_return_normalizer.update(decode_returns_tensor.detach().cpu().numpy().tolist())
        prefill_value_targets = (
            self.prefill_return_normalizer.normalize_tensor(prefill_returns_tensor)
            if self.config.value_normalization
            else prefill_returns_tensor
        )
        decode_value_targets = (
            self.decode_return_normalizer.normalize_tensor(decode_returns_tensor)
            if self.config.value_normalization
            else decode_returns_tensor
        )

        def policy_terms(
            outputs: Dict[str, torch.Tensor],
            heads_sequence: Sequence[Tuple[int, int, int, int, int, int, int, int]],
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            prefill_log_probs: List[torch.Tensor] = []
            decode_log_probs: List[torch.Tensor] = []
            prefill_entropies: List[torch.Tensor] = []
            decode_entropies: List[torch.Tensor] = []
            for row_index, heads in enumerate(heads_sequence[:sample_count]):
                action_mask = self.rollout.action_masks[row_index] if row_index < len(self.rollout.action_masks) else None
                (
                    phase_index,
                    admission_index,
                    admission_limit_index,
                    quantile_index,
                    budget_profile_index,
                    risky_index,
                    prefill_safety_filter_index,
                    decode_safety_filter_index,
                ) = heads
                phase_label = PHASE_CHOICES[phase_index]
                admission_label = ADMISSION_CHOICES[admission_index]

                phase_dist = torch.distributions.Categorical(
                    logits=masked_logits(
                        outputs["phase_logits"][row_index],
                        mask_indices(action_mask, "phase", len(PHASE_CHOICES)),
                    )
                )
                admission_dist = torch.distributions.Categorical(
                    logits=masked_logits(
                        outputs["admission_logits"][row_index],
                        mask_indices(action_mask, "admission", len(ADMISSION_CHOICES)),
                    )
                )
                budget_allowed = allowed_indices(
                    BUDGET_PROFILE_CHOICES,
                    allowed_budget_profiles(phase_label, admission_label, BUDGET_PROFILE_CHOICES),
                )
                budget_allowed = mask_indices(action_mask, "budget_profile", len(BUDGET_PROFILE_CHOICES), budget_allowed)
                budget_profile_dist = torch.distributions.Categorical(
                    logits=masked_logits(outputs["budget_profile_logits"][row_index], budget_allowed)
                )
                admission_limit_dist = torch.distributions.Categorical(
                    logits=masked_logits(
                        outputs["admission_limit_logits"][row_index],
                        mask_indices(action_mask, "admission_limit", len(NEW_ADMISSION_LIMIT_CHOICES)),
                    )
                )
                quantile_dist = torch.distributions.Categorical(logits=outputs["quantile_logits"][row_index])
                risky_dist = torch.distributions.Categorical(
                    logits=masked_logits(
                        outputs["risky_admission_cap_logits"][row_index],
                        mask_indices(action_mask, "risky_admission_cap", len(RISKY_ADMISSION_CAP_CHOICES)),
                    )
                )
                prefill_safety_filter_dist = torch.distributions.Categorical(
                    logits=outputs["prefill_safety_filter_logits"][row_index]
                )
                decode_safety_filter_dist = torch.distributions.Categorical(
                    logits=outputs["decode_safety_filter_logits"][row_index]
                )
                prefill_decision_gate_dist = torch.distributions.Categorical(
                    logits=outputs["prefill_decision_gate_logits"][row_index]
                )
                decode_decision_gate_dist = torch.distributions.Categorical(
                    logits=outputs["decode_decision_gate_logits"][row_index]
                )
                prefill_decision_gate_index = int(self.rollout.prefill_decision_gate_indices[row_index])
                decode_decision_gate_index = int(self.rollout.decode_decision_gate_indices[row_index])
                prefill_log_probs.append(
                    admission_dist.log_prob(torch.tensor(admission_index, dtype=torch.int64, device=self.device))
                    + admission_limit_dist.log_prob(
                        torch.tensor(admission_limit_index, dtype=torch.int64, device=self.device)
                    )
                    + quantile_dist.log_prob(torch.tensor(quantile_index, dtype=torch.int64, device=self.device))
                    + risky_dist.log_prob(torch.tensor(risky_index, dtype=torch.int64, device=self.device))
                    + prefill_safety_filter_dist.log_prob(
                        torch.tensor(prefill_safety_filter_index, dtype=torch.int64, device=self.device)
                    )
                    + prefill_decision_gate_dist.log_prob(
                        torch.tensor(prefill_decision_gate_index, dtype=torch.int64, device=self.device)
                    )
                )
                decode_log_probs.append(
                    phase_dist.log_prob(torch.tensor(phase_index, dtype=torch.int64, device=self.device))
                    + budget_profile_dist.log_prob(
                        torch.tensor(budget_profile_index, dtype=torch.int64, device=self.device)
                    )
                    + decode_safety_filter_dist.log_prob(
                        torch.tensor(decode_safety_filter_index, dtype=torch.int64, device=self.device)
                    )
                    + decode_decision_gate_dist.log_prob(
                        torch.tensor(decode_decision_gate_index, dtype=torch.int64, device=self.device)
                    )
                )
                prefill_entropies.append(
                    admission_dist.entropy()
                    + admission_limit_dist.entropy()
                    + quantile_dist.entropy()
                    + risky_dist.entropy()
                    + prefill_safety_filter_dist.entropy()
                    + prefill_decision_gate_dist.entropy()
                )
                decode_entropies.append(
                    phase_dist.entropy()
                    + budget_profile_dist.entropy()
                    + decode_safety_filter_dist.entropy()
                    + decode_decision_gate_dist.entropy()
                )
            return (
                torch.stack(prefill_log_probs),
                torch.stack(decode_log_probs),
                torch.stack(prefill_entropies),
                torch.stack(decode_entropies),
            )

        configured_target_reasons = tuple(parse_reasons(self.config.targeted_projection_bc_reasons))
        targeted_reason_coefs = parse_targeted_projection_reason_coefs(
            self.config.targeted_projection_bc_reason_coefs
        )
        targeted_head_logits = {
            0: "phase_logits",
            1: "admission_logits",
            2: "admission_limit_logits",
            3: "quantile_logits",
            4: "budget_profile_logits",
            5: "risky_admission_cap_logits",
            6: "prefill_safety_filter_logits",
            7: "decode_safety_filter_logits",
        }

        def targeted_projection_bc_terms(
            outputs: Dict[str, torch.Tensor],
            router: Optional[str] = None,
        ) -> Tuple[torch.Tensor, int, int, int]:
            losses: List[torch.Tensor] = []
            grouped_losses: Dict[Tuple[str, int], List[torch.Tensor]] = {}
            matched_rows = 0
            selected_heads = 0
            for row_index in range(sample_count):
                if float(self.rollout.projection_masks[row_index]) <= 0.0:
                    continue
                row_router = (
                    self.rollout.sample_routers[row_index]
                    if row_index < len(self.rollout.sample_routers)
                    else "joint"
                )
                if router in {"prefill", "decode"} and row_router in {"prefill", "decode"} and row_router != router:
                    continue
                target_head_reasons = targeted_projection_head_reason_pairs(
                    self.rollout.projection_reason_sets[row_index],
                    configured_target_reasons,
                )
                if not target_head_reasons:
                    continue
                requested_heads = self.rollout.requested_action_heads[row_index]
                executed_heads = self.rollout.projection_action_heads[row_index]
                row_selected = False
                for head_index, reason in target_head_reasons:
                    effective_router = router
                    if effective_router is None and row_router in {"prefill", "decode"}:
                        effective_router = row_router
                    if effective_router == "prefill" and head_index not in PREFILL_TARGETED_PROJECTION_HEADS:
                        continue
                    if effective_router == "decode" and head_index not in DECODE_TARGETED_PROJECTION_HEADS:
                        continue
                    if head_index >= len(requested_heads) or head_index >= len(executed_heads):
                        continue
                    if requested_heads[head_index] == executed_heads[head_index]:
                        continue
                    logits_key = targeted_head_logits.get(head_index)
                    if logits_key is None:
                        continue
                    logits = outputs[logits_key][row_index].unsqueeze(0)
                    target = torch.tensor([executed_heads[head_index]], dtype=torch.int64, device=self.device)
                    loss = torch.nn.functional.cross_entropy(logits, target)
                    responsibility_weight = targeted_projection_bc_error_weight(
                        head_index,
                        self.rollout.projection_prefill_responsibilities[row_index],
                        self.rollout.projection_decode_responsibilities[row_index],
                        min_weight=self.config.targeted_projection_bc_min_weight,
                    )
                    loss = loss * responsibility_weight
                    reason_coef = max(float(targeted_reason_coefs.get(reason, 1.0)), 0.0)
                    if self.config.targeted_projection_bc_group_by_reason:
                        grouped_losses.setdefault((reason, head_index), []).append(loss)
                    else:
                        losses.append(reason_coef * loss)
                    selected_heads += 1
                    row_selected = True
                if row_selected:
                    matched_rows += 1
            zero = outputs["admission_logits"].sum() * 0.0
            if self.config.targeted_projection_bc_group_by_reason:
                grouped_loss = grouped_targeted_projection_bc_loss(grouped_losses, targeted_reason_coefs)
                if grouped_loss is None:
                    return zero, 0, 0, 0
                return grouped_loss, matched_rows, selected_heads, len(grouped_losses)
            if not losses:
                return zero, 0, 0, 0
            return torch.stack(losses).mean(), matched_rows, selected_heads, 0

        def categorical_kl(
            logits: torch.Tensor,
            anchor_logits: torch.Tensor,
            allowed: Optional[Sequence[int]] = None,
        ) -> torch.Tensor:
            if allowed is None:
                allowed = list(range(int(logits.shape[-1])))
            current_dist = torch.distributions.Categorical(logits=masked_logits(logits, allowed))
            anchor_dist = torch.distributions.Categorical(logits=masked_logits(anchor_logits, allowed))
            return torch.distributions.kl_divergence(current_dist, anchor_dist)

        def anchor_kl_terms(outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            zero = outputs["admission_logits"].sum() * 0.0
            if self._anchor_model is None or float(self.config.anchor_kl_coef) <= 0.0:
                return zero, zero, zero
            with torch.no_grad():
                anchor_outputs = self._anchor_model(prefill_observations_tensor, decode_observations_tensor)
            prefill_kls: List[torch.Tensor] = []
            decode_kls: List[torch.Tensor] = []
            for row_index, heads in enumerate(self.action_heads[:sample_count]):
                action_mask = self.rollout.action_masks[row_index] if row_index < len(self.rollout.action_masks) else None
                (
                    phase_index,
                    admission_index,
                    _admission_limit_index,
                    _quantile_index,
                    _budget_profile_index,
                    _risky_index,
                    _prefill_safety_filter_index,
                    _decode_safety_filter_index,
                ) = heads
                phase_label = PHASE_CHOICES[phase_index]
                admission_label = ADMISSION_CHOICES[admission_index]
                budget_allowed = allowed_indices(
                    BUDGET_PROFILE_CHOICES,
                    allowed_budget_profiles(phase_label, admission_label, BUDGET_PROFILE_CHOICES),
                )
                budget_allowed = mask_indices(
                    action_mask,
                    "budget_profile",
                    len(BUDGET_PROFILE_CHOICES),
                    budget_allowed,
                )
                prefill_kls.append(
                    categorical_kl(
                        outputs["admission_logits"][row_index],
                        anchor_outputs["admission_logits"][row_index],
                        mask_indices(action_mask, "admission", len(ADMISSION_CHOICES)),
                    )
                    + categorical_kl(
                        outputs["admission_limit_logits"][row_index],
                        anchor_outputs["admission_limit_logits"][row_index],
                        mask_indices(action_mask, "admission_limit", len(NEW_ADMISSION_LIMIT_CHOICES)),
                    )
                    + categorical_kl(
                        outputs["quantile_logits"][row_index],
                        anchor_outputs["quantile_logits"][row_index],
                    )
                    + categorical_kl(
                        outputs["risky_admission_cap_logits"][row_index],
                        anchor_outputs["risky_admission_cap_logits"][row_index],
                        mask_indices(action_mask, "risky_admission_cap", len(RISKY_ADMISSION_CAP_CHOICES)),
                    )
                    + categorical_kl(
                        outputs["prefill_safety_filter_logits"][row_index],
                        anchor_outputs["prefill_safety_filter_logits"][row_index],
                    )
                    + categorical_kl(
                        outputs["prefill_decision_gate_logits"][row_index],
                        anchor_outputs["prefill_decision_gate_logits"][row_index],
                    )
                )
                decode_kls.append(
                    categorical_kl(
                        outputs["phase_logits"][row_index],
                        anchor_outputs["phase_logits"][row_index],
                        mask_indices(action_mask, "phase", len(PHASE_CHOICES)),
                    )
                    + categorical_kl(
                        outputs["budget_profile_logits"][row_index],
                        anchor_outputs["budget_profile_logits"][row_index],
                        budget_allowed,
                    )
                    + categorical_kl(
                        outputs["decode_safety_filter_logits"][row_index],
                        anchor_outputs["decode_safety_filter_logits"][row_index],
                    )
                    + categorical_kl(
                        outputs["decode_decision_gate_logits"][row_index],
                        anchor_outputs["decode_decision_gate_logits"][row_index],
                    )
                )
            prefill_kl = torch.stack(prefill_kls).mean() if prefill_kls else zero
            decode_kl = torch.stack(decode_kls).mean() if decode_kls else zero
            return prefill_kl + decode_kl, prefill_kl, decode_kl

        last_stats: Dict[str, torch.Tensor] = {}
        last_targeted_bc_rows = 0
        last_targeted_bc_heads = 0
        last_targeted_bc_groups = 0

        def compute_update_terms() -> Tuple[Dict[str, torch.Tensor], int, int, int]:
            outputs = self.model(prefill_observations_tensor, decode_observations_tensor)
            critic_outputs = self.central_critic(central_observations_tensor)
            (
                prefill_log_prob_tensor,
                decode_log_prob_tensor,
                prefill_entropy_tensor,
                decode_entropy_tensor,
            ) = policy_terms(outputs, self.action_heads)
            (
                projection_prefill_log_prob_tensor,
                projection_decode_log_prob_tensor,
                _projection_prefill_entropy_tensor,
                _projection_decode_entropy_tensor,
            ) = policy_terms(outputs, self.rollout.projection_action_heads)
            prefill_ratio = torch.exp(prefill_log_prob_tensor - old_prefill_log_probs_tensor)
            decode_ratio = torch.exp(decode_log_prob_tensor - old_decode_log_probs_tensor)
            clip = float(self.config.ppo_clip)
            prefill_policy_loss = -torch.min(
                prefill_ratio * prefill_advantages,
                torch.clamp(prefill_ratio, 1.0 - clip, 1.0 + clip) * prefill_advantages,
            ).mean()
            decode_policy_loss = -torch.min(
                decode_ratio * decode_advantages,
                torch.clamp(decode_ratio, 1.0 - clip, 1.0 + clip) * decode_advantages,
            ).mean()
            prefill_value_loss = torch.nn.functional.mse_loss(
                critic_outputs["prefill_value"],
                prefill_value_targets,
            )
            decode_value_loss = torch.nn.functional.mse_loss(
                critic_outputs["decode_value"],
                decode_value_targets,
            )
            policy_loss = prefill_policy_loss + decode_policy_loss
            value_loss = prefill_value_loss + decode_value_loss
            prefill_entropy_bonus = prefill_entropy_tensor.mean()
            decode_entropy_bonus = decode_entropy_tensor.mean()
            entropy_bonus = prefill_entropy_bonus + decode_entropy_bonus
            projection_count = projection_mask_tensor.sum().clamp_min(1.0)
            prefill_projection_imitation_loss = -(
                projection_mask_tensor * projection_prefill_log_prob_tensor
            ).sum() / projection_count
            decode_projection_imitation_loss = -(
                projection_mask_tensor * projection_decode_log_prob_tensor
            ).sum() / projection_count
            has_projection = 1.0 if projection_mask_tensor.sum().detach().cpu().item() > 0 else 0.0
            prefill_projection_imitation_loss = prefill_projection_imitation_loss * has_projection
            decode_projection_imitation_loss = decode_projection_imitation_loss * has_projection
            projection_imitation_loss = prefill_projection_imitation_loss + decode_projection_imitation_loss
            (
                targeted_projection_bc_loss,
                targeted_bc_rows,
                targeted_bc_heads,
                targeted_bc_groups,
            ) = targeted_projection_bc_terms(outputs)
            prefill_targeted_projection_bc_loss, _, _, _ = targeted_projection_bc_terms(outputs, router="prefill")
            decode_targeted_projection_bc_loss, _, _, _ = targeted_projection_bc_terms(outputs, router="decode")
            anchor_kl_loss, prefill_anchor_kl_loss, decode_anchor_kl_loss = anchor_kl_terms(outputs)
            return (
                {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "prefill_policy_loss": prefill_policy_loss,
                    "decode_policy_loss": decode_policy_loss,
                    "prefill_value_loss": prefill_value_loss,
                    "decode_value_loss": decode_value_loss,
                    "entropy_bonus": entropy_bonus,
                    "prefill_entropy_bonus": prefill_entropy_bonus,
                    "decode_entropy_bonus": decode_entropy_bonus,
                    "projection_imitation_loss": projection_imitation_loss,
                    "prefill_projection_imitation_loss": prefill_projection_imitation_loss,
                    "decode_projection_imitation_loss": decode_projection_imitation_loss,
                    "targeted_projection_bc_loss": targeted_projection_bc_loss,
                    "prefill_targeted_projection_bc_loss": prefill_targeted_projection_bc_loss,
                    "decode_targeted_projection_bc_loss": decode_targeted_projection_bc_loss,
                    "anchor_kl_loss": anchor_kl_loss,
                    "prefill_anchor_kl_loss": prefill_anchor_kl_loss,
                    "decode_anchor_kl_loss": decode_anchor_kl_loss,
                },
                targeted_bc_rows,
                targeted_bc_heads,
                targeted_bc_groups,
            )

        router_update_role = str(getattr(self.config, "router_update_role", "joint") or "joint")

        def select_router_terms(terms: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            if router_update_role == "prefill":
                return {
                    **terms,
                    "policy_loss": terms["prefill_policy_loss"],
                    "value_loss": terms["prefill_value_loss"],
                    "entropy_bonus": terms["prefill_entropy_bonus"],
                    "projection_imitation_loss": terms["prefill_projection_imitation_loss"],
                    "targeted_projection_bc_loss": terms["prefill_targeted_projection_bc_loss"],
                    "anchor_kl_loss": terms["prefill_anchor_kl_loss"],
                }
            if router_update_role == "decode":
                return {
                    **terms,
                    "policy_loss": terms["decode_policy_loss"],
                    "value_loss": terms["decode_value_loss"],
                    "entropy_bonus": terms["decode_entropy_bonus"],
                    "projection_imitation_loss": terms["decode_projection_imitation_loss"],
                    "targeted_projection_bc_loss": terms["decode_targeted_projection_bc_loss"],
                    "anchor_kl_loss": terms["decode_anchor_kl_loss"],
                }
            return terms

        def optimizer_step(loss: torch.Tensor) -> None:
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.central_critic.parameters()),
                self.config.grad_clip,
            )
            self.optimizer.step()

        actor_update_scale = 0.0 if self._actor_updates_frozen else 1.0
        policy_loss_scale = self._policy_loss_scale * actor_update_scale
        entropy_bonus = float(self.config.entropy_bonus) * actor_update_scale
        imitation_coef = (self.config.projection_imitation_coef + self._stage_imitation_coef) * actor_update_scale
        targeted_projection_bc_coef = float(self.config.targeted_projection_bc_coef) * actor_update_scale
        anchor_kl_coef = float(self.config.anchor_kl_coef) * actor_update_scale
        for _ in range(max(1, int(self.config.ppo_epochs))):
            if self.config.sequential_router_update:
                prefill_terms, _targeted_bc_rows, _targeted_bc_heads, _targeted_bc_groups = compute_update_terms()
                prefill_loss = (
                    policy_loss_scale * prefill_terms["prefill_policy_loss"]
                    + self._value_loss_scale * self.config.value_coef * prefill_terms["prefill_value_loss"]
                    - entropy_bonus * prefill_terms["prefill_entropy_bonus"]
                    + imitation_coef * prefill_terms["prefill_projection_imitation_loss"]
                    + targeted_projection_bc_coef * prefill_terms["prefill_targeted_projection_bc_loss"]
                    + anchor_kl_coef * prefill_terms["prefill_anchor_kl_loss"]
                )
                optimizer_step(prefill_loss)

                decode_terms, _targeted_bc_rows, _targeted_bc_heads, _targeted_bc_groups = compute_update_terms()
                decode_loss = (
                    policy_loss_scale * decode_terms["decode_policy_loss"]
                    + self._value_loss_scale * self.config.value_coef * decode_terms["decode_value_loss"]
                    - entropy_bonus * decode_terms["decode_entropy_bonus"]
                    + imitation_coef * decode_terms["decode_projection_imitation_loss"]
                    + targeted_projection_bc_coef * decode_terms["decode_targeted_projection_bc_loss"]
                    + anchor_kl_coef * decode_terms["decode_anchor_kl_loss"]
                )
                optimizer_step(decode_loss)
                terms, targeted_bc_rows, targeted_bc_heads, targeted_bc_groups = compute_update_terms()
                terms = select_router_terms(terms)
            else:
                terms, targeted_bc_rows, targeted_bc_heads, targeted_bc_groups = compute_update_terms()
                terms = select_router_terms(terms)
                loss = (
                    policy_loss_scale * terms["policy_loss"]
                    + self._value_loss_scale * self.config.value_coef * terms["value_loss"]
                    - entropy_bonus * terms["entropy_bonus"]
                    + imitation_coef * terms["projection_imitation_loss"]
                    + targeted_projection_bc_coef * terms["targeted_projection_bc_loss"]
                    + anchor_kl_coef * terms["anchor_kl_loss"]
                )
                optimizer_step(loss)
            last_stats = {
                "policy_loss": terms["policy_loss"].detach(),
                "value_loss": terms["value_loss"].detach(),
                "prefill_policy_loss": terms["prefill_policy_loss"].detach(),
                "decode_policy_loss": terms["decode_policy_loss"].detach(),
                "prefill_value_loss": terms["prefill_value_loss"].detach(),
                "decode_value_loss": terms["decode_value_loss"].detach(),
                "projection_imitation_loss": terms["projection_imitation_loss"].detach(),
                "targeted_projection_bc_loss": terms["targeted_projection_bc_loss"].detach(),
                "anchor_kl_loss": terms["anchor_kl_loss"].detach(),
            }
            last_targeted_bc_rows = targeted_bc_rows
            last_targeted_bc_heads = targeted_bc_heads
            last_targeted_bc_groups = targeted_bc_groups

        return {
            "policy_loss": float(last_stats["policy_loss"].cpu().item()),
            "value_loss": float(last_stats["value_loss"].cpu().item()),
            "mean_return": float((prefill_returns_tensor + decode_returns_tensor).mean().detach().cpu().item()),
            "prefill_policy_loss": float(last_stats["prefill_policy_loss"].cpu().item()),
            "decode_policy_loss": float(last_stats["decode_policy_loss"].cpu().item()),
            "prefill_value_loss": float(last_stats["prefill_value_loss"].cpu().item()),
            "decode_value_loss": float(last_stats["decode_value_loss"].cpu().item()),
            "projection_imitation_loss": float(last_stats["projection_imitation_loss"].cpu().item()),
            "targeted_projection_bc_loss": float(last_stats["targeted_projection_bc_loss"].cpu().item()),
            "anchor_kl_loss": float(last_stats["anchor_kl_loss"].cpu().item()),
            "targeted_projection_bc_rows": float(last_targeted_bc_rows),
            "targeted_projection_bc_heads": float(last_targeted_bc_heads),
            "targeted_projection_bc_groups": float(last_targeted_bc_groups),
            "prefill_mean_return": float(prefill_returns_tensor.mean().detach().cpu().item()),
            "decode_mean_return": float(decode_returns_tensor.mean().detach().cpu().item()),
        }

    def checkpoint(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "model_type": "torch-factorized-actor-critic-split-router-reward",
            "training_config": asdict(self.config),
            "prefill_observation_dim": self.prefill_observation_dim,
            "decode_observation_dim": self.decode_observation_dim,
            "central_observation_dim": self.central_observation_dim,
            "prefill_return_normalizer": self.prefill_return_normalizer.state_dict(),
            "decode_return_normalizer": self.decode_return_normalizer.state_dict(),
            "state_dict": {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()},
            "central_critic_state_dict": {
                key: value.detach().cpu().clone() for key, value in self.central_critic.state_dict().items()
            },
            "prefill_router_policy_heads": [
                "admission",
                "admission_limit",
                "quantile",
                "risky_admission_cap",
                "safety_filter",
                "decision_gate",
            ],
            "decode_router_policy_heads": [
                "phase",
                "budget_profile",
                "safety_filter",
                "decision_gate",
            ],
            "action_space": asdict(CURRENT_ACTION_SPACE),
            "phase_choices": PHASE_CHOICES,
            "admission_choices": ADMISSION_CHOICES,
            "new_admission_limit_choices": NEW_ADMISSION_LIMIT_CHOICES,
            "quantile_choices": QUANTILE_CHOICES,
            "budget_profile_choices": BUDGET_PROFILE_CHOICES,
            "risky_admission_cap_choices": RISKY_ADMISSION_CAP_CHOICES,
            "recovery_joint_profiles": [list(profile) for profile in build_recovery_joint_profiles()],
            "decision_gate_choices": list(DECISION_GATE_CHOICES),
        }
        if self.config.observation_normalization:
            payload.update(
                {
                    "prefill_normalizer": self.prefill_normalizer.state_dict(),
                    "decode_normalizer": self.decode_normalizer.state_dict(),
                    "central_normalizer": self.central_normalizer.state_dict(),
                }
            )
        return payload


def combined_router_policy_checkpoint(
    prefill_checkpoint: Dict[str, object],
    decode_checkpoint: Dict[str, object],
) -> Dict[str, object]:
    combined = copy.deepcopy(prefill_checkpoint)
    combined_state = combined.get("state_dict")
    decode_state = decode_checkpoint.get("state_dict")
    if isinstance(combined_state, dict) and isinstance(decode_state, dict):
        decode_prefixes = (
            "decode_trunk.",
            "phase_head.",
            "budget_profile_head.",
            "recovery_joint_profile_head.",
            "decode_safety_filter_head.",
            "decode_decision_gate_head.",
            "decode_value_head.",
        )
        for key, value in decode_state.items():
            if any(str(key).startswith(prefix) for prefix in decode_prefixes):
                combined_state[key] = value
    for key in ("decode_normalizer", "decode_return_normalizer"):
        if key in decode_checkpoint:
            combined[key] = copy.deepcopy(decode_checkpoint[key])
    combined["separate_router_policies"] = True
    return combined


class RLDistributionAwareScheduler(DistributionAwareScheduler):
    def __init__(
        self,
        config: SimulatorConfig,
        agent: TorchFactorizedAgent,
        reward_config: RewardConfig,
        decode_agent: Optional[TorchFactorizedAgent] = None,
        greedy: bool = False,
        capture_trajectory: bool = False,
        capture_sparse_events: bool = False,
        feedback_mode: str = "event",
        allow_heuristic_fallback: bool = True,
        deadlock_case_window: int = 20,
    ):
        super().__init__(config)
        self.agent = agent
        self.prefill_agent = agent
        self.decode_agent = decode_agent or agent
        self.use_async_router_decisions = decode_agent is not None
        self.reward_config = reward_config
        self.greedy = greedy
        self.capture_trajectory = capture_trajectory
        self.capture_sparse_events = capture_sparse_events
        self.feedback_mode = feedback_mode
        self.allow_heuristic_fallback = allow_heuristic_fallback
        self.last_action_index: Optional[int] = None
        self.last_policy_snapshot: Optional[Dict[str, np.ndarray]] = None
        self.last_prefill_observation: Optional[List[float]] = None
        self.last_decode_observation: Optional[List[float]] = None
        self.pending_sample_detail: Optional[Dict[str, object]] = None
        self.pending_prefill_observation: Optional[List[float]] = None
        self.pending_decode_observation: Optional[List[float]] = None
        self.pending_feedback_step_result: Optional[Dict[str, float]] = None
        self.pending_prefill_reward: float = 0.0
        self.pending_decode_reward: float = 0.0
        self.pending_prefill_sample_detail: Optional[Dict[str, object]] = None
        self.pending_decode_sample_detail: Optional[Dict[str, object]] = None
        self.pending_async_prefill_observation: Optional[List[float]] = None
        self.pending_async_decode_observation: Optional[List[float]] = None
        self.pending_async_prefill_step_result: Optional[Dict[str, float]] = None
        self.pending_async_decode_step_result: Optional[Dict[str, float]] = None
        self.pending_async_prefill_reward: float = 0.0
        self.pending_async_decode_reward: float = 0.0
        self.feedback_step_count = 0
        self.action_safety_trigger_count = 0
        self.prefill_gate_checks = 0
        self.prefill_gate_acts = 0
        self.decode_gate_checks = 0
        self.decode_gate_acts = 0
        self.cost_sums: Dict[str, float] = zero_costs()
        self.reward_component_sums: Dict[str, float] = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        self.executed_reward_sum = 0.0
        self.actor_projection_debit_sum = 0.0
        self.critic_history: List[List[float]] = []
        self.projection_corrections: List[Dict[str, object]] = []
        self.deadlock_case_window = max(0, int(deadlock_case_window))
        self.deadlock_recent_steps: List[Dict[str, object]] = []
        self._last_deadlock_schedule_row: Optional[Dict[str, object]] = None
        self.diagnostic_step_rows = 0
        self.diagnostic_projection_trigger_count = 0
        self.diagnostic_projection_reason_counter: Counter[str] = Counter()
        self.diagnostic_fallback_count = 0
        self.diagnostic_kv_util_values: List[float] = []
        self.diagnostic_unfinished_values: List[float] = []
        self.diagnostic_decode_starvation_steps = 0
        self.diagnostic_prefill_starvation_steps = 0
        self.diagnostic_policy_entropy_sum = 0.0
        self.diagnostic_policy_entropy_count = 0
        self.diagnostic_anchor_kl_sum = 0.0
        self.diagnostic_anchor_kl_count = 0

    def safety_projection_fraction(self) -> float:
        return self.action_safety_trigger_count / max(self.feedback_step_count, 1)

    def decision_gate_fractions(self) -> Dict[str, float]:
        return {
            "prefill_decision_act_fraction": self.prefill_gate_acts / max(self.prefill_gate_checks, 1),
            "decode_decision_act_fraction": self.decode_gate_acts / max(self.decode_gate_checks, 1),
            "prefill_decision_checks": float(self.prefill_gate_checks),
            "decode_decision_checks": float(self.decode_gate_checks),
        }

    def normalized_costs(self) -> Dict[str, float]:
        denominator = max(float(self.feedback_step_count), 1.0)
        return {
            name: (
                float(self.cost_sums.get(name, 0.0)) / denominator
                if name not in {"deadlock_cost", "unfinished_cost"}
                else float(self.cost_sums.get(name, 0.0))
            )
            for name in LIVENESS_COST_NAMES
        }

    @staticmethod
    def _round_value(value: object, digits: int = 6) -> object:
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, (int, float)):
            number = float(value)
            if not math.isfinite(number):
                return ""
            return round(number, digits)
        return value

    def _sample_details_for_mining(self) -> List[Tuple[str, Dict[str, object]]]:
        if self.pending_sample_detail is not None:
            return [("joint", self.pending_sample_detail)]
        rows: List[Tuple[str, Dict[str, object]]] = []
        if self.pending_prefill_sample_detail is not None:
            rows.append(("prefill", self.pending_prefill_sample_detail))
        if self.pending_decode_sample_detail is not None:
            rows.append(("decode", self.pending_decode_sample_detail))
        return rows

    def _pending_policy_diagnostics_for_mining(self) -> Dict[str, object]:
        details = self._sample_details_for_mining()
        if not details:
            return {
                "fallback_used": False,
                "fallback_routers": [],
                "policy_entropy": 0.0,
                "prefill_policy_entropy": 0.0,
                "decode_policy_entropy": 0.0,
                "critic_value": 0.0,
                "prefill_critic_value": 0.0,
                "decode_critic_value": 0.0,
                "anchor_kl_to_teacher": 0.0,
                "prefill_anchor_kl_to_teacher": 0.0,
                "decode_anchor_kl_to_teacher": 0.0,
            }

        fallback_routers = [
            router
            for router, detail in details
            if bool(detail.get("used_heuristic_fallback", False))
        ]
        if len(details) == 1 and details[0][0] == "joint":
            detail = details[0][1]
            return {
                "fallback_used": bool(fallback_routers),
                "fallback_routers": fallback_routers,
                "policy_entropy": float(detail.get("policy_entropy", 0.0) or 0.0),
                "prefill_policy_entropy": float(detail.get("prefill_policy_entropy", 0.0) or 0.0),
                "decode_policy_entropy": float(detail.get("decode_policy_entropy", 0.0) or 0.0),
                "critic_value": float(detail.get("critic_value", 0.0) or 0.0),
                "prefill_critic_value": float(detail.get("prefill_critic_value", 0.0) or 0.0),
                "decode_critic_value": float(detail.get("decode_critic_value", 0.0) or 0.0),
                "anchor_kl_to_teacher": float(detail.get("anchor_kl_to_teacher", 0.0) or 0.0),
                "prefill_anchor_kl_to_teacher": float(detail.get("prefill_anchor_kl_to_teacher", 0.0) or 0.0),
                "decode_anchor_kl_to_teacher": float(detail.get("decode_anchor_kl_to_teacher", 0.0) or 0.0),
            }

        prefill_detail = next((detail for router, detail in details if router == "prefill"), {})
        decode_detail = next((detail for router, detail in details if router == "decode"), {})
        prefill_entropy = float(prefill_detail.get("prefill_policy_entropy", 0.0) or 0.0)
        decode_entropy = float(decode_detail.get("decode_policy_entropy", 0.0) or 0.0)
        prefill_value = float(prefill_detail.get("prefill_critic_value", 0.0) or 0.0)
        decode_value = float(decode_detail.get("decode_critic_value", 0.0) or 0.0)
        prefill_kl = float(prefill_detail.get("prefill_anchor_kl_to_teacher", 0.0) or 0.0)
        decode_kl = float(decode_detail.get("decode_anchor_kl_to_teacher", 0.0) or 0.0)
        return {
            "fallback_used": bool(fallback_routers),
            "fallback_routers": fallback_routers,
            "policy_entropy": prefill_entropy + decode_entropy,
            "prefill_policy_entropy": prefill_entropy,
            "decode_policy_entropy": decode_entropy,
            "critic_value": prefill_value + decode_value,
            "prefill_critic_value": prefill_value,
            "decode_critic_value": decode_value,
            "anchor_kl_to_teacher": prefill_kl + decode_kl,
            "prefill_anchor_kl_to_teacher": prefill_kl,
            "decode_anchor_kl_to_teacher": decode_kl,
        }

    def _build_deadlock_schedule_row(self, engine: LLMSingleNodeSimulator, decision: object) -> Dict[str, object]:
        state = self._observation_state(engine)
        pending_unadmitted_by_queue = {
            queue: [
                request
                for request in engine.prefill_queues[queue]
                if request.batch_admitted_at_ms is None
            ]
            for queue in QueueClass
        }
        pending_unadmitted = [
            request
            for queue in QueueClass
            for request in pending_unadmitted_by_queue[queue]
        ]
        started_prefill = [
            request
            for queue in QueueClass
            for request in engine.prefill_queues[queue]
            if request.batch_admitted_at_ms is not None
        ]
        live_decodes = [request for request in engine.active_decode if not request.is_complete]
        first_token_backlog = [request for request in live_decodes if request.generated_tokens == 0]
        unfinished_requests = [request for request in engine.requests if not request.is_complete]
        executed_action = getattr(decision, "scheduler_action", None)
        requested_action = getattr(decision, "requested_scheduler_action", None)
        action_safety_reasons = ",".join(str(reason) for reason in getattr(decision, "action_safety_reasons", ()))
        diagnostics = self._pending_policy_diagnostics_for_mining()
        row: Dict[str, object] = {
            "step_id": int(engine.simulation_steps) + 1,
            "time_ms": round(float(engine.now_ms), 6),
            "event_type": "scheduled",
            "prefill_queue_length": len(pending_unadmitted) + len(started_prefill),
            "prefill_queue_length_before": len(pending_unadmitted) + len(started_prefill),
            "pending_unadmitted_count": len(pending_unadmitted),
            "pending_unadmitted_count_before": len(pending_unadmitted),
            "pending_unadmitted_fast_count": len(pending_unadmitted_by_queue[QueueClass.FAST]),
            "pending_unadmitted_normal_count": len(pending_unadmitted_by_queue[QueueClass.NORMAL]),
            "pending_unadmitted_risky_count": len(pending_unadmitted_by_queue[QueueClass.RISKY]),
            "started_prefill_waiting_count": len(started_prefill),
            "started_prefill_waiting_count_before": len(started_prefill),
            "decode_queue_length": len(live_decodes),
            "decode_queue_length_before": len(live_decodes),
            "first_token_backlog": len(first_token_backlog),
            "first_token_backlog_before": len(first_token_backlog),
            "unfinished_request_count": len(unfinished_requests),
            "kv_tokens": float(engine.current_kv_tokens),
            "logical_kv_tokens": float(engine.current_logical_kv_tokens),
            "kv_capacity_tokens": float(self.config.kv_capacity_tokens),
            "kv_util": round(float(engine.current_kv_tokens) / max(float(self.config.kv_capacity_tokens), 1.0), 6),
            "shadow_util": round(float(state.get("shadow_util", 0.0) or 0.0), 6),
            "prefix_cache_physical_tokens": float(engine.prefix_cache_physical_tokens),
            "prefix_cache_entries": len(engine.prefix_cache),
            "prefix_cache_util": round(float(state.get("prefix_cache_util", 0.0) or 0.0), 6),
            "action_safety_projection_triggered": bool(getattr(decision, "action_safety_triggered", False)),
            "action_safety_projection_reason": action_safety_reasons,
            "fallback_used": bool(diagnostics["fallback_used"]),
            "fallback_routers": diagnostics["fallback_routers"],
            "requested_action": action_to_dict(requested_action) if requested_action is not None else None,
            "executed_action": action_to_dict(executed_action) if executed_action is not None else None,
            "prefill_action": prefill_policy_to_dict(executed_action.prefill_policy())
            if executed_action is not None
            else None,
            "decode_action": decode_policy_to_dict(executed_action.decode_policy())
            if executed_action is not None
            else None,
            "requested_prefill_action": prefill_policy_to_dict(requested_action.prefill_policy())
            if requested_action is not None
            else None,
            "requested_decode_action": decode_policy_to_dict(requested_action.decode_policy())
            if requested_action is not None
            else None,
        }
        for key, value in diagnostics.items():
            if key in {"fallback_used", "fallback_routers"}:
                continue
            row[key] = self._round_value(value)
        return row

    def _step_result_mining_fields(self, step_result: Dict[str, float]) -> Dict[str, object]:
        fields: Dict[str, object] = {
            "event_type": step_result.get("feedback_event_type") or "step",
            "phase_mode": step_result.get("phase_mode"),
            "duration_ms": step_result.get("duration_ms", 0.0),
            "prefill_tokens": step_result.get("prefill_tokens", 0.0),
            "decode_tokens": step_result.get("decode_tokens", 0.0),
            "new_admissions": step_result.get("new_admissions", 0.0),
            "first_tokens_emitted": step_result.get("first_tokens_emitted", 0.0),
            "completed_requests": step_result.get("completed_requests", 0.0),
            "prefill_completed_requests": step_result.get("prefill_completed_requests", 0.0),
            "kv_tokens": step_result.get("kv_after", 0.0),
            "logical_kv_tokens": step_result.get("logical_kv_after", 0.0),
            "kv_util": float(step_result.get("kv_after", 0.0)) / max(float(self.config.kv_capacity_tokens), 1.0),
            "prefill_queue_length": step_result.get("prefill_waiting_after", 0.0),
            "pending_unadmitted_count": step_result.get("pending_unadmitted_after", 0.0),
            "pending_unadmitted_fast_count": step_result.get("pending_unadmitted_fast_after", 0.0),
            "pending_unadmitted_normal_count": step_result.get("pending_unadmitted_normal_after", 0.0),
            "pending_unadmitted_risky_count": step_result.get("pending_unadmitted_risky_after", 0.0),
            "decode_queue_length": step_result.get("active_decode_after", 0.0),
            "first_token_backlog": step_result.get("first_token_pending_after", 0.0),
            "oldest_pending_admission_after_ms": step_result.get("oldest_pending_admission_after_ms", 0.0),
            "oldest_first_token_pending_after_ms": step_result.get("oldest_first_token_pending_after_ms", 0.0),
            "action_safety_projection_triggered": bool(step_result.get("action_safety_triggered", 0.0)),
            "action_safety_projection_reason": str(step_result.get("action_safety_reasons", "")),
            "prefill_liveness_violation": step_result.get("prefill_liveness_violation", 0.0),
            "first_token_backlog_violation": step_result.get("first_token_backlog_violation", 0.0),
            "unsafe_projection_cost": step_result.get("unsafe_projection_cost", 0.0),
            **{key: str(step_result.get(key, "")) for key in REQUEST_ID_LIST_KEYS},
            "requested_action": action_to_dict(requested_action_from_step_result(step_result)),
            "executed_action": action_to_dict(action_from_step_result(step_result)),
            "requested_prefill_action": prefill_policy_to_dict(requested_action_from_step_result(step_result).prefill_policy()),
            "executed_prefill_action": prefill_policy_to_dict(action_from_step_result(step_result).prefill_policy()),
            "requested_decode_action": decode_policy_to_dict(requested_action_from_step_result(step_result).decode_policy()),
            "executed_decode_action": decode_policy_to_dict(action_from_step_result(step_result).decode_policy()),
            "counterfactual_prefill_responsibility": step_result.get(
                "counterfactual_prefill_responsibility",
                0.0,
            ),
            "counterfactual_decode_responsibility": step_result.get(
                "counterfactual_decode_responsibility",
                0.0,
            ),
            "counterfactual_prefill_cost_delta": step_result.get("counterfactual_prefill_cost_delta", 0.0),
            "counterfactual_decode_cost_delta": step_result.get("counterfactual_decode_cost_delta", 0.0),
        }
        return {key: self._round_value(value) for key, value in fields.items()}

    def _append_deadlock_mining_row(self, row: Dict[str, object]) -> None:
        if self.deadlock_case_window <= 0:
            return
        self.deadlock_recent_steps.append(row)
        if len(self.deadlock_recent_steps) > self.deadlock_case_window:
            self.deadlock_recent_steps = self.deadlock_recent_steps[-self.deadlock_case_window :]

    def _record_diagnostic_row(self, row: Dict[str, object]) -> None:
        self.diagnostic_step_rows += 1
        reasons = parse_reasons(row.get("action_safety_projection_reason", ""))
        if bool(row.get("action_safety_projection_triggered", False)) or reasons:
            self.diagnostic_projection_trigger_count += 1
        for reason in reasons:
            self.diagnostic_projection_reason_counter[str(reason)] += 1
        if bool(row.get("fallback_used", False)):
            self.diagnostic_fallback_count += 1

        self.diagnostic_kv_util_values.append(_float_from_mapping(row, "kv_util", 0.0))
        self.diagnostic_unfinished_values.append(_float_from_mapping(row, "unfinished_request_count", 0.0))

        decode_queue = _float_from_mapping(row, "decode_queue_length", 0.0)
        first_token_backlog = _float_from_mapping(row, "first_token_backlog", 0.0)
        decode_tokens = _float_from_mapping(row, "decode_tokens", 0.0)
        first_tokens_emitted = _float_from_mapping(row, "first_tokens_emitted", 0.0)
        completed_requests = _float_from_mapping(row, "completed_requests", 0.0)
        if (decode_queue > 0.0 or first_token_backlog > 0.0) and (
            decode_tokens <= 0.0 and first_tokens_emitted <= 0.0 and completed_requests <= 0.0
        ):
            self.diagnostic_decode_starvation_steps += 1

        pending_unadmitted = _float_from_mapping(row, "pending_unadmitted_count", 0.0)
        new_admissions = _float_from_mapping(row, "new_admissions", 0.0)
        prefill_tokens = _float_from_mapping(row, "prefill_tokens", 0.0)
        prefill_completed = _float_from_mapping(row, "prefill_completed_requests", 0.0)
        if pending_unadmitted > 0.0 and new_admissions <= 0.0 and prefill_tokens <= 0.0 and prefill_completed <= 0.0:
            self.diagnostic_prefill_starvation_steps += 1

        if "policy_entropy" in row:
            self.diagnostic_policy_entropy_sum += _float_from_mapping(row, "policy_entropy", 0.0)
            self.diagnostic_policy_entropy_count += 1
        if "anchor_kl_to_teacher" in row:
            self.diagnostic_anchor_kl_sum += _float_from_mapping(row, "anchor_kl_to_teacher", 0.0)
            self.diagnostic_anchor_kl_count += 1

    def diagnostic_metrics(self) -> Dict[str, object]:
        denominator = max(float(self.diagnostic_step_rows), 1.0)
        reason_histogram = {
            reason: int(count)
            for reason, count in sorted(self.diagnostic_projection_reason_counter.items())
        }
        risky_floor_count = int(reason_histogram.get("prefill_liveness_risky_cap_floor", 0))
        metrics: Dict[str, object] = {
            "diagnostic_step_rows": int(self.diagnostic_step_rows),
            "projection_reason_histogram": json.dumps(reason_histogram, sort_keys=True, ensure_ascii=True),
            "projection_count": int(self.diagnostic_projection_trigger_count),
            "projection_rate": round(float(self.diagnostic_projection_trigger_count) / denominator, 6),
            "prefill_liveness_risky_cap_floor_count": risky_floor_count,
            "prefill_liveness_risky_cap_floor_rate": round(float(risky_floor_count) / denominator, 6),
            "fallback_count": int(self.diagnostic_fallback_count),
            "fallback_rate": round(float(self.diagnostic_fallback_count) / denominator, 6),
            "unfinished_mean": round(_mean(self.diagnostic_unfinished_values), 6),
            "unfinished_p95": round(_percentile(self.diagnostic_unfinished_values, 95.0), 6),
            "unfinished_max": round(max(self.diagnostic_unfinished_values, default=0.0), 6),
            "kv_util_mean": round(_mean(self.diagnostic_kv_util_values), 6),
            "kv_util_p95": round(_percentile(self.diagnostic_kv_util_values, 95.0), 6),
            "decode_starvation_steps": int(self.diagnostic_decode_starvation_steps),
            "prefill_starvation_steps": int(self.diagnostic_prefill_starvation_steps),
            "policy_entropy_mean": round(
                self.diagnostic_policy_entropy_sum / max(float(self.diagnostic_policy_entropy_count), 1.0),
                6,
            ),
            "kl_to_anchor_teacher_mean": round(
                self.diagnostic_anchor_kl_sum / max(float(self.diagnostic_anchor_kl_count), 1.0),
                6,
            ),
        }
        for reason, count in reason_histogram.items():
            column = "".join(ch if ch.isalnum() else "_" for ch in reason).strip("_").lower()
            metrics[f"projection_reason_{column}_count"] = count
        return metrics

    def schedule(self, engine: LLMSingleNodeSimulator):  # type: ignore[override]
        decision = super().schedule(engine)
        self._last_deadlock_schedule_row = self._build_deadlock_schedule_row(engine, decision)
        return decision

    def build_deadlock_case(
        self,
        summary: Dict[str, object],
        *,
        episode: int,
        seed: int,
        train_trace: str,
        curriculum_phase: str,
    ) -> Optional[Dict[str, object]]:
        if self.deadlock_case_window <= 0:
            return None
        if not bool(summary.get("terminated_early", False)):
            return None
        recent_steps = list(self.deadlock_recent_steps[-self.deadlock_case_window :])
        classes = classify_deadlock_case(
            summary,
            recent_steps,
            kv_capacity_tokens=float(self.config.kv_capacity_tokens),
        )
        details = summary.get("termination_details", {})
        detail_map = details if isinstance(details, dict) else {}
        pending_unadmitted = _float_from_mapping(
            detail_map,
            "pending_unadmitted",
            _last_float(recent_steps, "pending_unadmitted_count", 0.0),
        )
        started_prefill = _float_from_mapping(
            detail_map,
            "started_prefill_waiting",
            _last_float(recent_steps, "started_prefill_waiting_count", 0.0),
        )
        active_decode = _float_from_mapping(
            detail_map,
            "active_decode",
            _last_float(recent_steps, "decode_queue_length", 0.0),
        )
        current_kv = _float_from_mapping(
            detail_map,
            "current_kv_tokens",
            _last_float(recent_steps, "kv_tokens", 0.0),
        )
        action_reasons = sorted(
            {
                reason
                for row in recent_steps
                for reason in parse_reasons(str(row.get("action_safety_projection_reason", "")))
            }
        )
        return {
            "episode": episode,
            "seed": seed,
            "train_trace": train_trace,
            "curriculum_phase": curriculum_phase,
            "step_id": int(summary.get("simulation_steps", 0) or 0),
            "termination_reason": str(summary.get("termination_reason") or ""),
            "termination_details": detail_map,
            "classification": classes,
            "classification_labels": [DEADLOCK_CLASS_DESCRIPTIONS.get(code, code) for code in classes],
            "prefill_queue_length": pending_unadmitted + started_prefill,
            "pending_unadmitted_count": pending_unadmitted,
            "pending_unadmitted_fast_count": _last_float(recent_steps, "pending_unadmitted_fast_count", 0.0),
            "pending_unadmitted_normal_count": _last_float(recent_steps, "pending_unadmitted_normal_count", 0.0),
            "pending_unadmitted_risky_count": _last_float(recent_steps, "pending_unadmitted_risky_count", 0.0),
            "decode_queue_length": active_decode,
            "first_token_backlog": _last_float(recent_steps, "first_token_backlog", 0.0),
            "kv_pressure": {
                "kv_tokens": current_kv,
                "kv_capacity_tokens": float(self.config.kv_capacity_tokens),
                "kv_util": current_kv / max(float(self.config.kv_capacity_tokens), 1.0),
                "shadow_util": _last_float(recent_steps, "shadow_util", 0.0),
                "prefix_cache_physical_tokens": _float_from_mapping(
                    detail_map,
                    "prefix_cache_physical_tokens",
                    _last_float(recent_steps, "prefix_cache_physical_tokens", 0.0),
                ),
                "prefix_cache_entries": _float_from_mapping(
                    detail_map,
                    "prefix_cache_entries",
                    _last_float(recent_steps, "prefix_cache_entries", 0.0),
                ),
            },
            "action_safety_projection_reasons": action_reasons,
            "fallback_used": any(bool(row.get("fallback_used", False)) for row in recent_steps),
            "unfinished_request_count": int(summary.get("unfinished_requests", 0) or 0),
            "unfinished_split": {
                "horizon": int(summary.get("horizon_unfinished_requests", 0) or 0),
                "aged": int(summary.get("aged_unfinished_requests", 0) or 0),
                "liveness": int(summary.get("liveness_unfinished_requests", 0) or 0),
                "punishable": int(summary.get("punishable_unfinished_requests", 0) or 0),
            },
            "last_prefill_actions": [row.get("prefill_action") for row in recent_steps],
            "last_decode_actions": [row.get("decode_action") for row in recent_steps],
            "critic_value": _last_float(recent_steps, "critic_value", 0.0),
            "policy_entropy": _last_float(recent_steps, "policy_entropy", 0.0),
            "kl_to_teacher": _last_float(recent_steps, "anchor_kl_to_teacher", 0.0),
            "policy_entropy_mean": round(_sum_float(recent_steps, "policy_entropy") / max(len(recent_steps), 1), 6),
            "critic_value_mean": round(_sum_float(recent_steps, "critic_value") / max(len(recent_steps), 1), 6),
            "kl_to_teacher_mean": round(_sum_float(recent_steps, "anchor_kl_to_teacher") / max(len(recent_steps), 1), 6),
            "counterfactual_prefill_responsibility": _last_float(
                recent_steps,
                "counterfactual_prefill_responsibility",
                0.0,
            ),
            "counterfactual_decode_responsibility": _last_float(
                recent_steps,
                "counterfactual_decode_responsibility",
                0.0,
            ),
            "recent_steps": recent_steps,
        }

    def _unsafe_state(self, engine: LLMSingleNodeSimulator) -> bool:
        state = self._observation_state(engine)
        kv_util = float(state["kv_util"])
        first_token_pending_load = float(state["first_token_pending_load"])
        pending_unadmitted_load = float(state["pending_unadmitted_load"])
        recent_risky_fraction = float(state.get("recent_q8_risky_fraction", 0.0))
        first_token_pending_count = len(
            [request for request in engine.active_decode if not request.is_complete and request.generated_tokens == 0]
        )
        first_token_backlog_threshold = max(
            2,
            int(math.ceil(self._resolved_decode_parallelism() * self.config.action_safety_first_token_load_threshold)),
        )
        return (
            kv_util >= self.config.action_safety_kv_util_threshold
            or (
                first_token_pending_count >= first_token_backlog_threshold
                and first_token_pending_load >= self.config.action_safety_first_token_load_threshold
            )
            or pending_unadmitted_load >= self.config.action_safety_pending_prompt_load_threshold
            or recent_risky_fraction >= 0.45
        )

    def _sampling_overrides(self, engine: LLMSingleNodeSimulator) -> Dict[str, float]:
        unsafe_state = self._unsafe_state(engine)
        if unsafe_state:
            return {
                "temperature": self.agent.config.unsafe_state_temperature,
                "greedy_probability": self.agent.config.unsafe_state_greedy_prob,
                "top_k": float(self.agent.config.unsafe_state_top_k),
            }
        return {
            "temperature": self.agent.config.exploration_temperature,
            "greedy_probability": self.agent.config.exploration_greedy_prob,
            "top_k": float(self.agent.config.exploration_top_k),
        }

    def _action_mask(self, engine: LLMSingleNodeSimulator) -> Dict[str, List[int]]:
        if not self.agent.config.strong_action_mask:
            return {}
        state = self._observation_state(engine)
        live_decode_count = len([request for request in engine.active_decode if not request.is_complete])
        first_token_pending_count = len(
            [request for request in engine.active_decode if not request.is_complete and request.generated_tokens == 0]
        )
        pending_unadmitted_count = sum(
            1
            for queue in QueueClass
            for request in engine.prefill_queues[queue]
            if request.batch_admitted_at_ms is None
        )
        pending_short_count = sum(
            1
            for queue in (QueueClass.FAST, QueueClass.NORMAL)
            for request in engine.prefill_queues[queue]
            if request.batch_admitted_at_ms is None
        )
        pending_risky_count = sum(
            1
            for request in engine.prefill_queues[QueueClass.RISKY]
            if request.batch_admitted_at_ms is None
        )
        started_prefill_count = sum(
            1
            for queue in QueueClass
            for request in engine.prefill_queues[queue]
            if request.batch_admitted_at_ms is not None and not request.is_complete
        )
        mask: Dict[str, List[int]] = {}

        def restrict(key: str, allowed_values: Sequence[int]) -> None:
            allowed = [int(value) for value in allowed_values]
            if key in mask:
                allowed_set = set(allowed)
                intersection = [value for value in mask[key] if value in allowed_set]
                mask[key] = intersection or allowed
            else:
                mask[key] = allowed

        kv_util = float(state["kv_util"])
        first_token_load = float(state["first_token_pending_load"])
        pending_unadmitted_load = float(state["pending_unadmitted_load"])
        decode_load = live_decode_count / max(float(self._resolved_decode_parallelism()), 1.0)
        severe_kv_pressure = kv_util >= min(max(float(self.config.action_safety_kv_util_threshold), 0.0), 1.0)
        first_token_backlog_threshold = max(
            2,
            int(math.ceil(self._resolved_decode_parallelism() * self.config.action_safety_first_token_load_threshold)),
        )
        severe_decode_pressure = (
            first_token_pending_count >= first_token_backlog_threshold
            and first_token_load >= self.config.action_safety_first_token_load_threshold
        )
        drain_required = live_decode_count > 0 and (
            kv_util >= min(max(float(self.config.action_safety_drain_kv_util_threshold), 0.0), 1.0)
            or (
                first_token_load >= max(0.25, self.config.action_safety_first_token_load_threshold)
                if self.config.action_safety_drain_on_first_token_pending
                else severe_decode_pressure
            )
        )
        severe_prefill_pressure = (
            pending_unadmitted_load >= self.config.action_safety_pending_prompt_load_threshold
            or pending_unadmitted_count > self._resolved_prefill_parallelism()
        )
        needs_prefill_liveness = (
            live_decode_count == 0
            and pending_unadmitted_count > 0
            and started_prefill_count == 0
        )

        if live_decode_count == 0 and pending_unadmitted_count > 0:
            restrict("admission_limit", [
                index for index, value in enumerate(NEW_ADMISSION_LIMIT_CHOICES) if int(value) > 0
            ])
            restrict("phase", [
                index for index, value in enumerate(PHASE_CHOICES) if value != PhaseMode.DECODE_PRIORITY.value
            ])
            restrict("budget_profile", [
                index for index, value in enumerate(BUDGET_PROFILE_CHOICES) if value != "decode-drain"
            ])
        elif live_decode_count == 0:
            restrict("phase", [
                index for index, value in enumerate(PHASE_CHOICES) if value != PhaseMode.DECODE_PRIORITY.value
            ])
            restrict("budget_profile", [
                index for index, value in enumerate(BUDGET_PROFILE_CHOICES) if value != "decode-drain"
            ])

        if pending_unadmitted_count == 0:
            restrict("admission_limit", [
                index for index, value in enumerate(NEW_ADMISSION_LIMIT_CHOICES) if int(value) == 0
            ])
            restrict("risky_admission_cap", [
                index for index, value in enumerate(RISKY_ADMISSION_CAP_CHOICES) if int(value) == 0
            ])

        if severe_kv_pressure or severe_decode_pressure:
            max_new_admissions = max(1, int(self.config.action_safety_max_new_admissions))
            restrict("admission_limit", [
                index for index, value in enumerate(NEW_ADMISSION_LIMIT_CHOICES) if int(value) <= max_new_admissions
            ])
            restrict("admission", [
                index
                for index, value in enumerate(ADMISSION_CHOICES)
                if value not in {"prefill-drain", "tail-relief"}
            ])

        if severe_prefill_pressure:
            max_new_admissions = max(1, int(self.config.action_safety_max_new_admissions))
            restrict("admission_limit", [
                index for index, value in enumerate(NEW_ADMISSION_LIMIT_CHOICES) if int(value) <= max_new_admissions
            ])

        if drain_required:
            restrict("admission_limit", [
                index for index, value in enumerate(NEW_ADMISSION_LIMIT_CHOICES) if int(value) == 0
            ])
            restrict("risky_admission_cap", [
                index for index, value in enumerate(RISKY_ADMISSION_CAP_CHOICES) if int(value) == 0
            ])
            if PhaseMode.DECODE_PRIORITY.value in PHASE_CHOICES:
                restrict("phase", [PHASE_CHOICES.index(PhaseMode.DECODE_PRIORITY.value)])
            if "decode-drain" in BUDGET_PROFILE_CHOICES:
                restrict("budget_profile", [BUDGET_PROFILE_CHOICES.index("decode-drain")])

        if severe_decode_pressure or first_token_pending_count > 0:
            max_risky_cap = max(0, int(self.config.action_safety_first_token_max_risky_admissions))
            restrict("risky_admission_cap", [
                index for index, value in enumerate(RISKY_ADMISSION_CAP_CHOICES) if int(value) <= max_risky_cap
            ])
            if not drain_required:
                restrict("budget_profile", [
                    index
                    for index, value in enumerate(BUDGET_PROFILE_CHOICES)
                    if value not in {"prefill-heavy", "decode-strict"}
                ])

        if first_token_pending_count <= 0 and decode_load >= float(self.config.action_safety_decode_guard_load_threshold):
            max_risky_cap = max(0, int(self.config.action_safety_decode_guard_max_risky_admissions))
            restrict("risky_admission_cap", [
                index for index, value in enumerate(RISKY_ADMISSION_CAP_CHOICES) if int(value) <= max_risky_cap
            ])

        if (severe_prefill_pressure or needs_prefill_liveness) and live_decode_count == 0:
            restrict("admission_limit", [
                index for index, value in enumerate(NEW_ADMISSION_LIMIT_CHOICES) if int(value) >= 1
            ])
            restrict("phase", [
                index for index, value in enumerate(PHASE_CHOICES) if value != PhaseMode.DECODE_PRIORITY.value
            ])
            restrict("budget_profile", [
                index for index, value in enumerate(BUDGET_PROFILE_CHOICES) if value != "decode-drain"
            ])
            if pending_short_count <= 0 and pending_risky_count > 0:
                restrict("risky_admission_cap", [
                    index for index, value in enumerate(RISKY_ADMISSION_CAP_CHOICES) if int(value) >= 1
                ])

        needs_short_admission_floor = pending_short_count > 0 and not drain_required
        if needs_short_admission_floor:
            short_admission_floor = min(
                max(1, pending_short_count),
                max(1, int(self.config.action_safety_max_new_admissions)),
                self._resolved_prefill_parallelism(),
            )
            restrict("admission_limit", [
                index for index, value in enumerate(NEW_ADMISSION_LIMIT_CHOICES) if int(value) >= short_admission_floor
            ])
            restrict("phase", [
                index for index, value in enumerate(PHASE_CHOICES) if value != PhaseMode.DECODE_PRIORITY.value
            ])
            restrict("budget_profile", [
                index for index, value in enumerate(BUDGET_PROFILE_CHOICES) if value != "decode-drain"
            ])
        return {key: value for key, value in mask.items() if value}

    def _heuristic_fallback_action(self, engine: LLMSingleNodeSimulator) -> SchedulerAction:
        state = self._observation_state(engine)
        live_decode_count = len([request for request in engine.active_decode if not request.is_complete])
        kv_util = float(state["kv_util"])
        shadow_util = float(state["shadow_util"])
        started_prefill_remaining = float(state["started_prefill_remaining"])
        pending_short_load = float(state["pending_fast_load"]) + float(state["pending_normal_load"])
        drain_required = live_decode_count > 0 and (
            kv_util >= max(self.config.action_safety_kv_util_threshold, self.config.kv_pressure_threshold, 0.90)
            or shadow_util >= 0.95
            or started_prefill_remaining > 0.0
        )
        if drain_required:
            return SchedulerAction(
                phase_preference=PhaseMode.DECODE_PRIORITY,
                admission_profile="balanced",
                new_admission_limit=0,
                shadow_quantile=min(self.config.kv_reservation_quantile, 0.90),
                budget_profile="decode-drain",
                risky_admission_cap=0,
            )
        if live_decode_count > 0 and pending_short_load > 0.0:
            return SchedulerAction(
                phase_preference=PhaseMode.MIXED,
                admission_profile="ttft-greedy",
                new_admission_limit=1,
                shadow_quantile=min(self.config.kv_reservation_quantile, 0.90),
                budget_profile="decode-heavy",
                risky_admission_cap=0,
            )
        return SchedulerAction(
            phase_preference=self._heuristic_phase_mode(engine),
            admission_profile="balanced",
            new_admission_limit=self._resolved_prefill_parallelism(),
            shadow_quantile=self.config.kv_reservation_quantile,
            budget_profile="balanced",
            risky_admission_cap=self._resolved_prefill_parallelism(),
        )

    def _prefill_actor_observation(self, engine: LLMSingleNodeSimulator) -> List[float]:
        observation = self.build_prefill_router_observation(engine)
        boundary_mode = str(getattr(self.agent.config, "router_boundary_mode", "current") or "current")
        if (self.agent.config.no_prev_decode_to_prefill or boundary_mode == "strict_env_only") and len(observation) >= 26:
            # Indices 22:26 are last decode phase/budget and last decode event counters.
            observation[22:26] = [0.0, 0.0, 0.0, 0.0]
        return observation

    def _central_observation(
        self,
        prefill_observation: Sequence[float],
        decode_observation: Sequence[float],
    ) -> List[float]:
        central = [float(value) for value in prefill_observation] + [
            float(value) for value in decode_observation
        ]
        central.extend(self._critic_history_vector())
        return central

    def _critic_history_vector(self) -> List[float]:
        window = max(0, int(self.agent.config.history_critic_window))
        if window <= 0:
            return []
        rows = self.critic_history[-window:]
        padding_rows = window - len(rows)
        vector: List[float] = []
        for _ in range(padding_rows):
            vector.extend([0.0] * HISTORY_CRITIC_FEATURES_PER_STEP)
        for row in rows:
            vector.extend(row)
        return vector

    def _step_history_features(self, step_result: Dict[str, float]) -> List[float]:
        capacity = max(float(self.config.kv_capacity_tokens), 1.0)
        decode_parallelism = max(float(self._resolved_decode_parallelism()), 1.0)
        prefill_parallelism = max(float(self._resolved_prefill_parallelism()), 1.0)
        token_budget = max(float(self.config.token_budget_per_step), 1.0)
        duration_scale = max(float(self.config.decode_ms_per_token * self.config.max_decode_batch), 1.0)
        return [
            min(float(step_result.get("kv_before", 0.0)) / capacity, 2.0),
            min(float(step_result.get("kv_after", 0.0)) / capacity, 2.0),
            min(float(step_result.get("logical_kv_after", step_result.get("kv_after", 0.0))) / capacity, 2.0),
            min(float(step_result.get("prefill_tokens", 0.0)) / token_budget, 4.0),
            min(float(step_result.get("decode_tokens", 0.0)) / token_budget, 4.0),
            min(float(step_result.get("new_admissions", 0.0)) / prefill_parallelism, 4.0),
            min(float(step_result.get("active_decode_after", 0.0)) / decode_parallelism, 4.0),
            min(float(step_result.get("first_token_pending_after", 0.0)) / decode_parallelism, 4.0),
            min(float(step_result.get("pending_unadmitted_after", 0.0)) / prefill_parallelism, 8.0),
            min(float(step_result.get("completed_requests", 0.0)) / decode_parallelism, 4.0),
            1.0 if bool(step_result.get("action_safety_triggered", 0.0)) else 0.0,
            min(float(step_result.get("duration_ms", 0.0)) / duration_scale, 8.0),
        ]

    def choose_action(self, engine: LLMSingleNodeSimulator) -> Optional[SchedulerAction]:
        if self.feedback_mode == "event" and self.pending_sample_detail is not None:
            return self.pending_sample_detail["scheduler_action"]  # type: ignore[return-value]
        prefill_observation = self._prefill_actor_observation(engine)
        decode_observation = self.build_decode_router_observation(engine)
        central_observation = self._central_observation(prefill_observation, decode_observation)
        unsafe_state = self._unsafe_state(engine)
        action_mask = self._action_mask(engine)
        force_imitation = not self.greedy and self.agent.training_stage == "stage_a_imitation"
        if (
            not self.greedy
            and (unsafe_state or force_imitation)
            and self.agent.config.unsafe_state_heuristic_fallback
            and self.allow_heuristic_fallback
        ):
            heuristic_action = self._heuristic_fallback_action(engine)
            sample_detail = self.agent.sample_action_detail(
                prefill_observation,
                decode_observation,
                central_observation,
                greedy=self.greedy,
                capture_policy=self.capture_trajectory,
                sampling_overrides=self._sampling_overrides(engine),
                forced_action=heuristic_action,
                action_mask=action_mask,
            )
            sample_detail["used_heuristic_fallback"] = True
            if force_imitation:
                sample_detail["training_action_rewritten_by_safety"] = True
                sample_detail["action_safety_reasons_for_training"] = "stage_a_heuristic"
            if self.feedback_mode == "event":
                self.pending_sample_detail = sample_detail
                self.pending_prefill_observation = prefill_observation
                self.pending_decode_observation = list(sample_detail.get("conditioned_decode_observation", decode_observation))
                self.pending_feedback_step_result = None
                self.pending_prefill_reward = 0.0
                self.pending_decode_reward = 0.0
                return heuristic_action
            self.agent._store_sample_detail(sample_detail)
            self.last_action_index = int(sample_detail["action_index"])
            self.last_policy_snapshot = sample_detail.get("policy_snapshot")
            self.last_prefill_observation = prefill_observation
            self.last_decode_observation = list(sample_detail.get("conditioned_decode_observation", decode_observation))
            return heuristic_action
        sampling_overrides = None if self.greedy else self._sampling_overrides(engine)
        if self.feedback_mode == "event":
            sample_detail = self.agent.sample_action_detail(
                prefill_observation,
                decode_observation,
                central_observation,
                greedy=self.greedy,
                capture_policy=self.capture_trajectory,
                sampling_overrides=sampling_overrides,
                action_mask=action_mask,
            )
            self.pending_sample_detail = sample_detail
            self.pending_prefill_observation = prefill_observation
            self.pending_decode_observation = list(sample_detail.get("conditioned_decode_observation", decode_observation))
            self.pending_feedback_step_result = None
            self.pending_prefill_reward = 0.0
            self.pending_decode_reward = 0.0
            return sample_detail["scheduler_action"]  # type: ignore[return-value]

        sample_detail = self.agent.sample_action_detail(
            prefill_observation,
            decode_observation,
            central_observation,
            greedy=self.greedy,
            capture_policy=self.capture_trajectory,
            sampling_overrides=sampling_overrides,
            action_mask=action_mask,
        )
        if not self.greedy:
            self.agent._store_sample_detail(sample_detail)
        action_index = int(sample_detail["action_index"])
        policy_snapshot = sample_detail.get("policy_snapshot")
        self.last_action_index = action_index
        self.last_policy_snapshot = policy_snapshot
        self.last_prefill_observation = prefill_observation
        self.last_decode_observation = list(sample_detail.get("conditioned_decode_observation", decode_observation))
        return sample_detail["scheduler_action"]  # type: ignore[return-value]

    def _sample_router_action_detail(
        self,
        agent: TorchFactorizedAgent,
        engine: LLMSingleNodeSimulator,
        *,
        prefill_policy: Optional[PrefillRouterPolicy] = None,
    ) -> Dict[str, object]:
        prefill_observation = self._prefill_actor_observation(engine)
        decode_observation = self.build_decode_router_observation(engine, prefill_policy)
        central_observation = self._central_observation(prefill_observation, decode_observation)
        action_mask = self._action_mask(engine)
        sampling_overrides = None if self.greedy else self._sampling_overrides(engine)
        if agent is self.prefill_agent:
            sample_detail = agent.sample_prefill_action_detail(
                prefill_observation,
                decode_observation,
                central_observation,
                base_decode_policy=self.last_decode_policy,
                greedy=self.greedy,
                capture_policy=self.capture_trajectory,
                sampling_overrides=sampling_overrides,
                action_mask=action_mask,
            )
        elif agent is self.decode_agent:
            sample_detail = agent.sample_decode_action_detail(
                prefill_observation,
                decode_observation,
                central_observation,
                prefill_policy=prefill_policy,
                greedy=self.greedy,
                capture_policy=self.capture_trajectory,
                sampling_overrides=sampling_overrides,
                action_mask=action_mask,
            )
        else:
            sample_detail = agent.sample_joint_action_detail(
                prefill_observation,
                decode_observation,
                central_observation,
                greedy=self.greedy,
                capture_policy=self.capture_trajectory,
                sampling_overrides=sampling_overrides,
                action_mask=action_mask,
            )
        sample_detail["prefill_observation"] = list(prefill_observation)
        sample_detail["decode_observation"] = list(
            sample_detail.get("conditioned_decode_observation", decode_observation)
        )
        return sample_detail

    def _rewrite_sample_detail_action_for_router_hold(
        self,
        sample_detail: Dict[str, object],
        agent: TorchFactorizedAgent,
        action: SchedulerAction,
    ) -> Dict[str, object]:
        rewritten = dict(sample_detail)
        action_heads = action_heads_from_action(action)
        rewritten["action_index"] = action_index_from_action(action)
        rewritten["action_heads"] = action_heads
        rewritten["scheduler_action"] = action
        old_prefill_log_prob, old_decode_log_prob = agent.log_probs_for_normalized_sample(
            rewritten["normalized_prefill_observation"],  # type: ignore[arg-type]
            rewritten["normalized_decode_observation"],  # type: ignore[arg-type]
            action_heads,
            action_mask=rewritten.get("action_mask"),  # type: ignore[arg-type]
            prefill_decision_gate_index=int(rewritten.get("prefill_decision_gate_index", 1)),
            decode_decision_gate_index=int(rewritten.get("decode_decision_gate_index", 1)),
        )
        rewritten["old_prefill_log_prob"] = old_prefill_log_prob
        rewritten["old_decode_log_prob"] = old_decode_log_prob
        return rewritten

    def choose_prefill_policy(self, engine: LLMSingleNodeSimulator) -> Optional[PrefillRouterPolicy]:
        if self.pending_prefill_sample_detail is not None:
            if self.pending_async_prefill_step_result is not None:
                self._finalize_async_router_sample("prefill", flush_event_type="next-prefill-decision")
            else:
                action = self.pending_prefill_sample_detail["scheduler_action"]
                return action.prefill_policy()  # type: ignore[union-attr]
        if self.pending_prefill_sample_detail is not None:
            action = self.pending_prefill_sample_detail["scheduler_action"]
            return action.prefill_policy()  # type: ignore[union-attr]
        sample_detail = self._sample_router_action_detail(self.prefill_agent, engine)
        self.prefill_gate_checks += 1
        prefill_gate = bool(sample_detail.get("prefill_decision_gate", True))
        if prefill_gate:
            self.prefill_gate_acts += 1
        if not prefill_gate:
            hold_action = SchedulerAction.from_router_policies(
                self.last_prefill_policy or PrefillRouterPolicy(new_admission_limit=0, risky_admission_cap=0),
                self.last_decode_policy or DecodeRouterPolicy(),
            )
            sample_detail = self._rewrite_sample_detail_action_for_router_hold(
                sample_detail,
                self.prefill_agent,
                hold_action,
            )
        self.pending_prefill_sample_detail = sample_detail
        self.pending_async_prefill_observation = list(sample_detail.get("prefill_observation", []))
        action = sample_detail["scheduler_action"]
        if not prefill_gate:
            return None
        return action.prefill_policy()  # type: ignore[union-attr]

    def choose_decode_policy(
        self,
        engine: LLMSingleNodeSimulator,
        prefill_policy: Optional[PrefillRouterPolicy] = None,
    ) -> Optional[DecodeRouterPolicy]:
        if self.pending_decode_sample_detail is not None:
            if self.pending_async_decode_step_result is not None:
                self._finalize_async_router_sample("decode", flush_event_type="next-decode-decision")
            else:
                action = self.pending_decode_sample_detail["scheduler_action"]
                return action.decode_policy()  # type: ignore[union-attr]
        if self.pending_decode_sample_detail is not None:
            action = self.pending_decode_sample_detail["scheduler_action"]
            return action.decode_policy()  # type: ignore[union-attr]
        sample_detail = self._sample_router_action_detail(
            self.decode_agent,
            engine,
            prefill_policy=prefill_policy,
        )
        self.decode_gate_checks += 1
        decode_gate = bool(sample_detail.get("decode_decision_gate", True))
        if decode_gate:
            self.decode_gate_acts += 1
        if not decode_gate:
            hold_action = SchedulerAction.from_router_policies(
                self.last_prefill_policy or prefill_policy or PrefillRouterPolicy(),
                self.last_decode_policy or DecodeRouterPolicy(),
            )
            sample_detail = self._rewrite_sample_detail_action_for_router_hold(
                sample_detail,
                self.decode_agent,
                hold_action,
            )
        self.pending_decode_sample_detail = sample_detail
        self.pending_async_decode_observation = list(sample_detail.get("decode_observation", []))
        action = sample_detail["scheduler_action"]
        if not decode_gate:
            return None
        return action.decode_policy()  # type: ignore[union-attr]

    def _finalize_async_router_sample(
        self,
        router: str,
        *,
        flush_event_type: Optional[str] = None,
    ) -> None:
        if router == "prefill":
            agent = self.prefill_agent
            sample_detail = self.pending_prefill_sample_detail
            aggregated_step_result = self.pending_async_prefill_step_result
            shared_reward = self.pending_async_prefill_reward
            prefill_reward = shared_reward
            decode_reward = 0.0
        elif router == "decode":
            agent = self.decode_agent
            sample_detail = self.pending_decode_sample_detail
            aggregated_step_result = self.pending_async_decode_step_result
            shared_reward = self.pending_async_decode_reward
            prefill_reward = 0.0
            decode_reward = shared_reward
        else:
            raise ValueError(f"unknown async router: {router}")

        if sample_detail is None:
            return
        if aggregated_step_result is None:
            self._clear_async_router_sample(router)
            return
        if flush_event_type is not None:
            aggregated_step_result.setdefault("feedback_event_type", flush_event_type)

        original_action_index = int(sample_detail["action_index"])
        training_sample_detail = self._training_sample_detail_for_step_result(
            sample_detail,
            aggregated_step_result,
            agent=agent,
            router=router,
        )
        agent.commit_sample_detail(training_sample_detail)
        policy_snapshot = sample_detail.get("policy_snapshot")
        if self.capture_trajectory and policy_snapshot is not None:
            agent.record_transition(
                prefill_reward=prefill_reward,
                decode_reward=decode_reward,
                action_index=original_action_index,
                phase_probs=policy_snapshot["phase_probs"],
                admission_probs=policy_snapshot["admission_probs"],
                admission_limit_probs=policy_snapshot["admission_limit_probs"],
                quantile_probs=policy_snapshot["quantile_probs"],
                budget_profile_probs=policy_snapshot["budget_profile_probs"],
                risky_admission_cap_probs=policy_snapshot["risky_admission_cap_probs"],
                prefill_safety_filter_probs=policy_snapshot["prefill_safety_filter_probs"],
                decode_safety_filter_probs=policy_snapshot["decode_safety_filter_probs"],
                prefill_observation=sample_detail.get("prefill_observation", []),  # type: ignore[arg-type]
                decode_observation=sample_detail.get("decode_observation", []),  # type: ignore[arg-type]
                normalized_prefill_observation=policy_snapshot["normalized_prefill_observation"],
                normalized_decode_observation=policy_snapshot["normalized_decode_observation"],
                step_result=aggregated_step_result,
            )
            if self.capture_sparse_events:
                agent.record_step_event(
                    prefill_reward=prefill_reward,
                    decode_reward=decode_reward,
                    action_index=original_action_index,
                    step_result=aggregated_step_result,
                )
        else:
            agent.record_reward(prefill_reward, decode_reward)
            if self.capture_sparse_events:
                agent.record_step_event(
                    prefill_reward=prefill_reward,
                    decode_reward=decode_reward,
                    action_index=original_action_index,
                    step_result=aggregated_step_result,
                )
        self._clear_async_router_sample(router)

    def _clear_async_router_sample(self, router: str) -> None:
        if router == "prefill":
            self.pending_prefill_sample_detail = None
            self.pending_async_prefill_observation = None
            self.pending_async_prefill_step_result = None
            self.pending_async_prefill_reward = 0.0
            return
        if router == "decode":
            self.pending_decode_sample_detail = None
            self.pending_async_decode_observation = None
            self.pending_async_decode_step_result = None
            self.pending_async_decode_reward = 0.0
            return
        raise ValueError(f"unknown async router: {router}")

    def _finalize_event_feedback(self, force: bool = False) -> None:
        if self.pending_sample_detail is None:
            return
        aggregated_step_result = self.pending_feedback_step_result
        if aggregated_step_result is None:
            if force:
                self.pending_sample_detail = None
                self.pending_prefill_observation = None
                self.pending_decode_observation = None
            return
        if force and aggregated_step_result.get("feedback_event_type") is None:
            aggregated_step_result["feedback_event_type"] = "terminal-flush"
        if not force and feedback_event_type(aggregated_step_result) is None:
            return

        original_action_index = int(self.pending_sample_detail["action_index"])
        training_sample_detail = self._training_sample_detail_for_step_result(
            self.pending_sample_detail,
            aggregated_step_result,
        )
        self.agent.commit_sample_detail(training_sample_detail)
        policy_snapshot = self.pending_sample_detail.get("policy_snapshot")
        if self.capture_trajectory and policy_snapshot is not None:
            self.agent.record_transition(
                prefill_reward=self.pending_prefill_reward,
                decode_reward=self.pending_decode_reward,
                action_index=original_action_index,
                phase_probs=policy_snapshot["phase_probs"],
                admission_probs=policy_snapshot["admission_probs"],
                admission_limit_probs=policy_snapshot["admission_limit_probs"],
                quantile_probs=policy_snapshot["quantile_probs"],
                budget_profile_probs=policy_snapshot["budget_profile_probs"],
                risky_admission_cap_probs=policy_snapshot["risky_admission_cap_probs"],
                prefill_safety_filter_probs=policy_snapshot["prefill_safety_filter_probs"],
                decode_safety_filter_probs=policy_snapshot["decode_safety_filter_probs"],
                prefill_observation=self.pending_prefill_observation or [],
                decode_observation=self.pending_decode_observation or [],
                normalized_prefill_observation=policy_snapshot["normalized_prefill_observation"],
                normalized_decode_observation=policy_snapshot["normalized_decode_observation"],
                step_result=aggregated_step_result,
            )
            if self.capture_sparse_events:
                self.agent.record_step_event(
                    prefill_reward=self.pending_prefill_reward,
                    decode_reward=self.pending_decode_reward,
                    action_index=original_action_index,
                    step_result=aggregated_step_result,
                )
        else:
            self.agent.record_reward(
                self.pending_prefill_reward,
                self.pending_decode_reward,
            )
            if self.capture_sparse_events:
                self.agent.record_step_event(
                    prefill_reward=self.pending_prefill_reward,
                    decode_reward=self.pending_decode_reward,
                    action_index=original_action_index,
                    step_result=aggregated_step_result,
                )
        self.pending_sample_detail = None
        self.pending_prefill_observation = None
        self.pending_decode_observation = None
        self.pending_feedback_step_result = None
        self.pending_prefill_reward = 0.0
        self.pending_decode_reward = 0.0

    def _training_sample_detail_for_step_result(
        self,
        sample_detail: Dict[str, object],
        step_result: Dict[str, float],
        *,
        agent: Optional[TorchFactorizedAgent] = None,
        router: str = "joint",
    ) -> Dict[str, object]:
        attribution = {
            "counterfactual_prefill_cost_delta": float(step_result.get("counterfactual_prefill_cost_delta", 0.0) or 0.0),
            "counterfactual_decode_cost_delta": float(step_result.get("counterfactual_decode_cost_delta", 0.0) or 0.0),
            "counterfactual_prefill_responsibility": float(
                step_result.get("counterfactual_prefill_responsibility", 0.0) or 0.0
            ),
            "counterfactual_decode_responsibility": float(
                step_result.get("counterfactual_decode_responsibility", 0.0) or 0.0
            ),
        }
        if not bool(step_result.get("action_safety_triggered", 0.0)):
            if router == "joint":
                return sample_detail
            tagged = dict(sample_detail)
            tagged["sample_router"] = router
            tagged.update(attribution)
            return tagged
        active_agent = agent or self.agent
        executed_action = action_from_step_result(step_result)
        rewritten = dict(sample_detail)
        rewritten["sample_router"] = router
        rewritten.update(attribution)
        rewritten["requested_action_index_for_training"] = sample_detail["action_index"]
        rewritten["requested_action_heads_for_training"] = sample_detail["action_heads"]
        rewritten["action_index"] = action_index_from_action(executed_action)
        executed_action_heads = action_heads_from_action(executed_action)
        rewritten["action_heads"] = executed_action_heads
        rewritten_mask = {
            str(key): [int(index) for index in value]  # type: ignore[union-attr]
            for key, value in dict(rewritten.get("action_mask") or {}).items()
        }
        for head_index, mask_key in (
            (0, "phase"),
            (1, "admission"),
            (2, "admission_limit"),
            (4, "budget_profile"),
            (5, "risky_admission_cap"),
        ):
            if mask_key not in rewritten_mask:
                continue
            executed_index = int(executed_action_heads[head_index])
            if executed_index not in rewritten_mask[mask_key]:
                rewritten_mask[mask_key].append(executed_index)
                rewritten_mask[mask_key].sort()
        rewritten["action_mask"] = rewritten_mask
        old_prefill_log_prob, old_decode_log_prob = active_agent.log_probs_for_normalized_sample(
            rewritten["normalized_prefill_observation"],  # type: ignore[arg-type]
            rewritten["normalized_decode_observation"],  # type: ignore[arg-type]
            executed_action_heads,
            action_mask=rewritten_mask,
            prefill_decision_gate_index=int(rewritten.get("prefill_decision_gate_index", 1)),
            decode_decision_gate_index=int(rewritten.get("decode_decision_gate_index", 1)),
        )
        rewritten["old_prefill_log_prob"] = old_prefill_log_prob
        rewritten["old_decode_log_prob"] = old_decode_log_prob
        rewritten["training_action_rewritten_by_safety"] = True
        rewritten["action_safety_reasons_for_training"] = str(step_result.get("action_safety_reasons", ""))
        self.projection_corrections.append(
            {
                "router": router,
                "requested_action_index": sample_detail["action_index"],
                "requested_action_heads": list(sample_detail["action_heads"]),  # type: ignore[arg-type]
                "requested_action": action_to_dict(sample_detail["scheduler_action"]),  # type: ignore[arg-type]
                "executed_action_index": rewritten["action_index"],
                "executed_action_heads": list(executed_action_heads),
                "executed_action": action_to_dict(executed_action),
                "action_safety_reasons": str(step_result.get("action_safety_reasons", "")),
                "normalized_prefill_observation": [
                    round(float(value), 6)
                    for value in rewritten["normalized_prefill_observation"]  # type: ignore[index]
                ],
                "normalized_decode_observation": [
                    round(float(value), 6)
                    for value in rewritten["normalized_decode_observation"]  # type: ignore[index]
                ],
                "normalized_central_observation": [
                    round(float(value), 6)
                    for value in rewritten["normalized_central_observation"]  # type: ignore[index]
                ],
                "kv_after": float(step_result.get("kv_after", 0.0)),
                "logical_kv_after": float(step_result.get("logical_kv_after", 0.0)),
                "kv_capacity_tokens": float(self.config.kv_capacity_tokens),
                "kv_util": float(step_result.get("kv_after", 0.0)) / max(float(self.config.kv_capacity_tokens), 1.0),
                "active_decode_after": float(step_result.get("active_decode_after", 0.0)),
                "pending_unadmitted_after": float(step_result.get("pending_unadmitted_after", 0.0)),
                "first_token_pending_after": float(step_result.get("first_token_pending_after", 0.0)),
                "new_admissions": float(step_result.get("new_admissions", 0.0)),
                "decode_tokens": float(step_result.get("decode_tokens", 0.0)),
                "prefill_tokens": float(step_result.get("prefill_tokens", 0.0)),
                "executed_reward": float(step_result.get("executed_reward", 0.0)),
                "actor_projection_debit": float(step_result.get("actor_projection_debit", 0.0)),
                "actor_reward": float(step_result.get("actor_reward", 0.0)),
                **{
                    name: round(float(step_result.get(name, 0.0) or 0.0), 6)
                    for name in REWARD_COMPONENT_NAMES
                },
                **{key: round(value, 6) for key, value in attribution.items()},
            }
        )
        return rewritten

    def on_step_result(self, step_result: Dict[str, float]) -> None:
        step_costs = compute_step_costs(step_result, self.config)
        attribution = counterfactual_router_cost_attribution(step_result, self.config)
        step_result.update(step_costs)
        step_result.update(attribution)
        mining_row = dict(self._last_deadlock_schedule_row or {})
        mining_row.update(self._step_result_mining_fields(step_result))
        self._append_deadlock_mining_row(mining_row)
        self._record_diagnostic_row(mining_row)
        self._last_deadlock_schedule_row = None
        self.last_step_result = dict(step_result)
        self.feedback_step_count += 1
        self.cost_sums = add_costs(self.cost_sums, step_costs)
        if bool(step_result.get("action_safety_triggered", 0.0)):
            self.action_safety_trigger_count += 1
        if int(self.agent.config.history_critic_window) > 0:
            self.critic_history.append(self._step_history_features(step_result))
            history_limit = max(1, int(self.agent.config.history_critic_window))
            if len(self.critic_history) > history_limit:
                self.critic_history = self.critic_history[-history_limit:]
        if bool(getattr(self, "use_async_router_decisions", False)) and (
            self.pending_prefill_sample_detail is not None or self.pending_decode_sample_detail is not None
        ):
            reward_split = self.compute_reward_split(step_result)
            shared_reward = reward_split["total_reward"]
            if self.pending_prefill_sample_detail is not None:
                self.pending_async_prefill_reward += shared_reward
                self.pending_async_prefill_step_result = merge_feedback_step_result(
                    self.pending_async_prefill_step_result,
                    step_result,
                )
            if self.pending_decode_sample_detail is not None:
                self.pending_async_decode_reward += shared_reward
                self.pending_async_decode_step_result = merge_feedback_step_result(
                    self.pending_async_decode_step_result,
                    step_result,
                )
            return
        if self.feedback_mode == "event":
            if self.pending_sample_detail is None:
                return
            reward_split = self.compute_reward_split(step_result)
            self.pending_prefill_reward += reward_split["prefill_reward"]
            self.pending_decode_reward += reward_split["decode_reward"]
            self.pending_feedback_step_result = merge_feedback_step_result(self.pending_feedback_step_result, step_result)
            if feedback_event_type(step_result) is not None:
                self._finalize_event_feedback(force=False)
            return

        if (
            self.last_action_index is None
            or self.last_prefill_observation is None
            or self.last_decode_observation is None
        ):
            return
        reward_split = self.compute_reward_split(step_result)
        if self.capture_trajectory and self.last_policy_snapshot is not None:
            self.agent.record_transition(
                prefill_reward=reward_split["prefill_reward"],
                decode_reward=reward_split["decode_reward"],
                action_index=self.last_action_index,
                phase_probs=self.last_policy_snapshot["phase_probs"],
                admission_probs=self.last_policy_snapshot["admission_probs"],
                admission_limit_probs=self.last_policy_snapshot["admission_limit_probs"],
                quantile_probs=self.last_policy_snapshot["quantile_probs"],
                budget_profile_probs=self.last_policy_snapshot["budget_profile_probs"],
                risky_admission_cap_probs=self.last_policy_snapshot["risky_admission_cap_probs"],
                prefill_safety_filter_probs=self.last_policy_snapshot["prefill_safety_filter_probs"],
                decode_safety_filter_probs=self.last_policy_snapshot["decode_safety_filter_probs"],
                prefill_observation=self.last_prefill_observation,
                decode_observation=self.last_decode_observation,
                normalized_prefill_observation=self.last_policy_snapshot["normalized_prefill_observation"],
                normalized_decode_observation=self.last_policy_snapshot["normalized_decode_observation"],
                step_result=step_result,
            )
            if self.capture_sparse_events:
                self.agent.record_step_event(
                    prefill_reward=reward_split["prefill_reward"],
                    decode_reward=reward_split["decode_reward"],
                    action_index=self.last_action_index,
                    step_result=step_result,
                )
            return
        self.agent.record_reward(
            reward_split["prefill_reward"],
            reward_split["decode_reward"],
        )
        if self.capture_sparse_events:
            self.agent.record_step_event(
                prefill_reward=reward_split["prefill_reward"],
                decode_reward=reward_split["decode_reward"],
                action_index=self.last_action_index,
                step_result=step_result,
            )

    def on_idle(self) -> None:
        if self._last_deadlock_schedule_row is not None:
            mining_row = dict(self._last_deadlock_schedule_row)
            mining_row["event_type"] = "idle"
            self._append_deadlock_mining_row(mining_row)
            self._record_diagnostic_row(mining_row)
            self._last_deadlock_schedule_row = None
        if bool(getattr(self, "use_async_router_decisions", False)) and (
            self.pending_prefill_sample_detail is not None or self.pending_decode_sample_detail is not None
        ):
            self._finalize_async_router_sample("prefill", flush_event_type="idle-flush")
            self._finalize_async_router_sample("decode", flush_event_type="idle-flush")
            return
        if self.feedback_mode == "event":
            if self.pending_feedback_step_result is not None:
                self.pending_feedback_step_result.setdefault("feedback_event_type", "idle-flush")
                self._finalize_event_feedback(force=True)
            self.pending_sample_detail = None
            self.pending_prefill_observation = None
            self.pending_decode_observation = None
            self.pending_feedback_step_result = None
            self.pending_prefill_reward = 0.0
            self.pending_decode_reward = 0.0
            return
        if not self.greedy:
            self.agent.discard_last_sample()
        self.last_action_index = None
        self.last_policy_snapshot = None
        self.last_prefill_observation = None
        self.last_decode_observation = None

    def compute_reward_split(self, step_result: Dict[str, float]) -> Dict[str, float]:
        reward_split = compute_step_reward_split(step_result, self.config, self.reward_config)
        step_result["executed_reward"] = float(reward_split.get("executed_reward", reward_split.get("total_reward", 0.0)) or 0.0)
        step_result["actor_projection_debit"] = float(reward_split.get("actor_projection_debit", 0.0) or 0.0)
        step_result["actor_reward"] = float(reward_split.get("actor_reward", reward_split.get("total_reward", 0.0)) or 0.0)
        self.executed_reward_sum += float(step_result["executed_reward"])
        self.actor_projection_debit_sum += float(step_result["actor_projection_debit"])
        for name in REWARD_COMPONENT_NAMES:
            value = float(reward_split.get(name, 0.0) or 0.0)
            self.reward_component_sums[name] = self.reward_component_sums.get(name, 0.0) + value
            step_result[name] = value
        return reward_split

    def compute_reward(self, step_result: Dict[str, float]) -> float:
        return self.compute_reward_split(step_result)["total_reward"]

    def apply_terminal_objective(self, summary: Dict[str, object]) -> float:
        if bool(getattr(self, "use_async_router_decisions", False)) and (
            self.pending_prefill_sample_detail is not None or self.pending_decode_sample_detail is not None
        ):
            self._finalize_async_router_sample("prefill", flush_event_type="terminal-flush")
            self._finalize_async_router_sample("decode", flush_event_type="terminal-flush")
        elif self.feedback_mode == "event":
            self._finalize_event_feedback(force=True)
        terminal_reward = compute_terminal_objective_split(summary, self.reward_config)
        self.executed_reward_sum += float(terminal_reward.get("total", terminal_reward.get("total_reward", 0.0)) or 0.0)
        for name in REWARD_COMPONENT_NAMES:
            self.reward_component_sums[name] = self.reward_component_sums.get(name, 0.0) + float(
                terminal_reward.get(name, 0.0) or 0.0
            )
        self.cost_sums = add_costs(self.cost_sums, compute_terminal_costs(summary))
        shared_terminal_reward = terminal_reward["total"]
        self.prefill_agent.apply_terminal_reward(
            shared_terminal_reward,
            0.0,
            summary=summary,
        )
        if self.decode_agent is self.prefill_agent:
            self.decode_agent.apply_terminal_reward(0.0, shared_terminal_reward, summary=summary)
        else:
            self.decode_agent.apply_terminal_reward(0.0, shared_terminal_reward, summary=summary)
        return terminal_reward["total"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a neural RL teacher for the TTFT simulator.")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trace", type=Path)
    parser.add_argument(
        "--curriculum-trace",
        action="append",
        type=Path,
        help="Optional repeated trace path for cross-workload teacher curriculum. When set, training rotates across traces instead of replaying a single workload.",
    )
    parser.add_argument(
        "--curriculum-schedule",
        choices=("round-robin", "random"),
        default="round-robin",
        help="How to choose the active trace for each training episode when --curriculum-trace is provided.",
    )
    parser.add_argument(
        "--bucket-coverage-curriculum",
        action="store_true",
        help="Add the synthetic bucket-coverage workload as a teacher curriculum source.",
    )
    parser.add_argument(
        "--bucket-coverage-curriculum-requests",
        type=int,
        default=0,
        help="Request count for bucket-coverage curriculum episodes. Defaults to --requests when unset.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/rl_teacher.pt"))
    parser.add_argument(
        "--init-policy",
        type=Path,
        help="Optional teacher checkpoint to initialize actor, critic, and normalizers before training.",
    )
    parser.add_argument("--metrics-out", type=Path, default=Path("artifacts/rl_training_metrics.csv"))
    parser.add_argument("--trajectory-out", type=Path, help="Optional JSONL file with the best greedy teacher transitions.")
    parser.add_argument("--event-out", type=Path, help="Optional JSONL file with sparse best-eval events.")
    parser.add_argument(
        "--projection-correction-out",
        type=Path,
        help="Optional JSONL file of training states where the safety projection rewrote the requested action.",
    )
    parser.add_argument(
        "--deadlock-case-out",
        type=Path,
        help="Optional JSONL file of early-terminated training episode mining windows.",
    )
    parser.add_argument(
        "--deadlock-case-window",
        type=int,
        default=20,
        help="Number of final scheduler steps to retain for each early-termination/deadlock case.",
    )
    parser.add_argument(
        "--post-valid-drift-audit-out",
        type=Path,
        help="Optional JSONL file with post-first-valid actor hash, fixed-probe logits, normalizer, mask, and eval drift rows.",
    )
    parser.add_argument(
        "--post-valid-drift-probe-size",
        type=int,
        default=64,
        help="Maximum first-valid rollout observations to keep for fixed-logit drift probes.",
    )
    parser.add_argument(
        "--prefer-low-safety-projection-best",
        action="store_true",
        help="Among stable complete evaluations, choose the checkpoint with lower safety projection before tail latency.",
    )
    parser.add_argument(
        "--best-selection-objective",
        choices=("reward", "reward-margin", "tail"),
        default="reward",
        help=(
            "After hard stability/validation gates pass, choose best checkpoint by raw reward, "
            "SLO-margin-adjusted reward, or tail-latency rank."
        ),
    )
    parser.add_argument(
        "--max-safety-projection-best",
        type=float,
        help="Require best-policy candidates to have safety_projection_fraction at or below this value.",
    )
    parser.add_argument(
        "--validation-gate-trace",
        action="append",
        type=Path,
        help="Optional fixed validation trace for best-checkpoint selection. Candidates must pass this gate before replacing the best policy.",
    )
    parser.add_argument(
        "--validation-gate-seed",
        type=int,
        default=0,
        help="Seed used when loading validation gate traces.",
    )
    parser.add_argument(
        "--validation-gate-min-completed-fraction",
        type=float,
        default=1.0,
        help="Minimum completed request fraction required on validation gate traces.",
    )
    parser.add_argument(
        "--validation-gate-max-unfinished",
        type=int,
        default=0,
        help="Maximum unfinished requests allowed across validation gate traces.",
    )
    parser.add_argument(
        "--validation-gate-max-safety-projection",
        type=float,
        help="Optional maximum per-trace safety projection fraction allowed by the validation gate.",
    )
    parser.add_argument(
        "--validation-gate-max-ttft-p95",
        type=float,
        help="Optional maximum merged validation TTFT p95 allowed by the validation gate.",
    )
    parser.add_argument(
        "--validation-gate-max-ttft-p99",
        type=float,
        help="Optional maximum merged validation TTFT p99 allowed by the validation gate.",
    )
    parser.add_argument(
        "--validation-gate-max-e2e-p95",
        type=float,
        help="Optional maximum merged validation end-to-end p95 allowed by the validation gate.",
    )
    parser.add_argument(
        "--validation-margin-ttft-p95-target",
        type=float,
        default=87000.0,
        help="Soft TTFT p95 target used only by --best-selection-objective reward-margin.",
    )
    parser.add_argument(
        "--validation-margin-e2e-p95-target",
        type=float,
        default=87000.0,
        help="Soft E2E p95 target used only by --best-selection-objective reward-margin.",
    )
    parser.add_argument(
        "--validation-margin-ttft-p95-penalty",
        type=float,
        default=1.0,
        help="Reward penalty per millisecond above --validation-margin-ttft-p95-target.",
    )
    parser.add_argument(
        "--validation-margin-e2e-p95-penalty",
        type=float,
        default=1.0,
        help="Reward penalty per millisecond above --validation-margin-e2e-p95-target.",
    )
    parser.add_argument(
        "--validation-gate-allow-early-stop",
        action="store_true",
        help="Allow early-terminated validation gate traces when other gate thresholds pass.",
    )
    parser.add_argument("--write-metrics-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument(
        "--episode-step-cap",
        type=int,
        default=0,
        help="Abort a training episode simulator run after this many scheduling steps. 0 disables the cap.",
    )
    parser.add_argument(
        "--episode-wall-time-limit-s",
        type=float,
        default=0.0,
        help="Abort a training episode simulator run after this many wall-clock seconds. 0 disables the cap.",
    )
    parser.add_argument(
        "--episode-progress-every",
        type=int,
        default=0,
        help="Print training simulator progress every N scheduling steps. 0 disables progress output.",
    )
    parser.add_argument(
        "--eval-step-cap",
        type=int,
        default=0,
        help="Abort a greedy evaluation simulator run after this many scheduling steps. 0 disables the cap.",
    )
    parser.add_argument(
        "--eval-wall-time-limit-s",
        type=float,
        default=0.0,
        help="Abort a greedy evaluation simulator run after this many wall-clock seconds. 0 disables the cap.",
    )
    parser.add_argument(
        "--eval-progress-every",
        type=int,
        default=0,
        help="Print greedy evaluation simulator progress every N scheduling steps. 0 disables progress output.",
    )
    parser.add_argument("--token-budget", type=int, default=384)
    parser.add_argument("--prefill-chunk", type=int, default=128)
    parser.add_argument("--kv-capacity", type=int, default=16000)
    parser.add_argument("--prefill-ms-per-token", type=float, default=0.03)
    parser.add_argument("--decode-ms-per-token", type=float, default=0.40)
    parser.add_argument("--simulation-duration-ms", type=float, default=0.0)
    parser.add_argument(
        "--unfinished-age-grace-ms",
        type=float,
        default=0.0,
        help="When a fixed simulation duration ends, unfinished requests younger than this age are treated as horizon-censored and not terminally penalized. 0 treats all duration-censored unfinished requests as horizon-censored.",
    )
    parser.add_argument("--prefill-decision-interval-ms", type=float, default=0.0)
    parser.add_argument("--decode-decision-interval-ms", type=float, default=0.0)
    parser.add_argument("--shadow-kv-quantile", type=float, default=0.90)
    parser.add_argument("--prefill-padding-weight", type=float, default=1.0)
    parser.add_argument("--kv-allocator", choices=("token", "block", "paged"), default="token")
    parser.add_argument("--kv-block-size", type=int, default=16)
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--prefix-cache-max-resident-entries", type=int, default=0)
    parser.add_argument("--kv-pressure-threshold", type=float, default=0.85)
    parser.add_argument("--allow-shadow-oversubscription", type=float, default=1.0)
    parser.add_argument("--action-safety-kv-util-threshold", type=float, default=0.65)
    parser.add_argument("--action-safety-drain-kv-util-threshold", type=float, default=0.90)
    parser.add_argument(
        "--enable-first-token-drain-shield",
        action="store_true",
        help="Keep legacy shield behavior that enters drain mode for any sizable first-token pending load.",
    )
    parser.add_argument("--action-safety-max-new-admissions", type=int, default=2)
    parser.add_argument("--action-safety-pending-prompt-load-threshold", type=float, default=0.25)
    parser.add_argument(
        "--action-space-config",
        type=Path,
        help="Optional JSON file defining the scheduler action-space choices.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.985)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-clip", type=float, default=0.20)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--entropy-bonus", type=float, default=0.005)
    parser.add_argument("--value-coef", type=float, default=0.50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument(
        "--disable-observation-normalization",
        action="store_true",
        help="Use deterministic scheduler feature scaling directly and do not update or checkpoint observation normalizers.",
    )
    parser.add_argument("--disable-value-normalization", action="store_true")
    parser.add_argument("--shared-critic", action="store_true")
    parser.add_argument("--no-prefill-to-decode-condition", action="store_true")
    parser.add_argument("--no-prev-decode-to-prefill", action="store_true")
    parser.add_argument(
        "--router-boundary-mode",
        choices=("current", "actual_prefill_conditioned_decode", "strict_env_only"),
        default="strict_env_only",
        help=(
            "Router observation boundary. strict_env_only keeps shared environment state but removes "
            "cross-router policy/action conditioning; pass current for legacy joint-conditioned behavior."
        ),
    )
    parser.add_argument(
        "--router-update-role",
        choices=("prefill", "decode", "joint"),
        default="joint",
        help="Restrict PPO optimization terms to a single router when training split prefill/decode agents.",
    )
    parser.add_argument("--disable-serialized-joint-critic", action="store_true")
    parser.add_argument("--projection-imitation-coef", type=float, default=0.0)
    parser.add_argument(
        "--targeted-projection-bc-coef",
        type=float,
        default=0.0,
        help="Auxiliary CE loss weight for selected projection guard corrections.",
    )
    parser.add_argument(
        "--targeted-projection-bc-reasons",
        default=TrainingConfig.targeted_projection_bc_reasons,
        help="Comma-separated projection reasons used by targeted projection BC.",
    )
    parser.add_argument(
        "--targeted-projection-bc-reason-coefs",
        default="",
        help="Optional comma-separated reason=multiplier overrides for targeted projection BC, e.g. first_token_risky_cap=2.0.",
    )
    parser.add_argument(
        "--disable-targeted-projection-bc-group-by-reason",
        action="store_true",
        help="Use the legacy sample-frequency-weighted targeted projection BC reduction.",
    )
    parser.add_argument(
        "--targeted-projection-bc-min-weight",
        type=float,
        default=TrainingConfig.targeted_projection_bc_min_weight,
        help="Minimum loss weight for targeted projection BC rows attributed to the other router.",
    )
    parser.add_argument(
        "--late-update-start-fraction",
        type=float,
        default=1.10,
        help="Episode progress fraction at which late PPO update constraints start. Values >1 disable them.",
    )
    parser.add_argument(
        "--late-learning-rate-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to optimizer LR after --late-update-start-fraction.",
    )
    parser.add_argument(
        "--late-freeze-prefill-trunk",
        action="store_true",
        help="Freeze the actor prefill trunk after --late-update-start-fraction.",
    )
    parser.add_argument(
        "--late-freeze-admission-heads",
        action="store_true",
        help="Freeze actor admission, admission-limit, quantile, and risky-cap heads after --late-update-start-fraction.",
    )
    parser.add_argument("--teacher-stage-a-fraction", type=float, default=0.15)
    parser.add_argument("--teacher-stage-b-fraction", type=float, default=0.70)
    parser.add_argument("--stage-a-imitation-coef", type=float, default=1.0)
    parser.add_argument("--stage-c-learning-rate-scale", type=float, default=0.20)
    stage_c_freeze_group = parser.add_mutually_exclusive_group()
    stage_c_freeze_group.add_argument(
        "--stage-c-freeze-admission-heads",
        dest="stage_c_freeze_admission_heads",
        action="store_true",
        default=False,
        help="Freeze actor admission, admission-limit, quantile, and risky-cap heads during stage C.",
    )
    stage_c_freeze_group.add_argument(
        "--disable-stage-c-freeze-admission-heads",
        dest="stage_c_freeze_admission_heads",
        action="store_false",
        help="Keep stage-C admission-related action heads trainable. This is the default.",
    )
    parser.add_argument(
        "--sequential-router-update",
        action="store_true",
        help="Use two sequential PPO optimizer steps: prefill actor/value terms, then decode actor/value terms.",
    )
    parser.add_argument(
        "--first-valid-freeze-actor",
        action="store_true",
        help="After the first validation-gated 900/900 checkpoint, stop actor updates and continue critic updates.",
    )
    parser.add_argument(
        "--first-valid-anchor-audit",
        action="store_true",
        help="After the first validation-gated 900/900 checkpoint, anchor drift probes without freezing policy updates.",
    )
    parser.add_argument(
        "--first-valid-freeze-observation-normalizers",
        action="store_true",
        help="After the first validation-gated 900/900 checkpoint, stop updating actor and critic observation normalizers.",
    )
    parser.add_argument(
        "--anchor-kl-coef",
        type=float,
        default=0.0,
        help="After the first validation-gated 900/900 checkpoint, add this KL coefficient to keep policy near the anchor.",
    )
    parser.add_argument("--disable-strong-action-mask", action="store_true")
    parser.add_argument(
        "--enable-recovery-joint-profile",
        action="store_true",
        help=(
            "Enable the recovery joint profile head for greedy audit/evaluation and learning-rate-zero "
            "collection runs."
        ),
    )
    parser.add_argument(
        "--recovery-joint-profile-min-kv-util",
        type=float,
        default=0.65,
        help="KV/shadow utilization threshold for activating --enable-recovery-joint-profile.",
    )
    parser.add_argument(
        "--disable-action-safety-filter",
        action="store_true",
        help="Disable scheduler action safety projection/shield. Intended for short audit evaluations only.",
    )
    parser.add_argument(
        "--history-critic-window",
        type=int,
        default=0,
        help="Append the last N scheduler step summaries to the centralized critic observation. 0 disables history.",
    )
    parser.add_argument("--phased-curriculum", action="store_true")
    parser.add_argument("--curriculum-warmup-fraction", type=float, default=0.25)
    parser.add_argument("--curriculum-mid-fraction", type=float, default=0.35)
    parser.add_argument("--feedback-mode", choices=("step", "event"), default="event")
    parser.add_argument("--exploration-temperature", type=float, default=0.85)
    parser.add_argument("--exploration-top-k", type=int, default=3)
    parser.add_argument("--exploration-greedy-prob", type=float, default=0.15)
    parser.add_argument("--unsafe-state-temperature", type=float, default=0.35)
    parser.add_argument("--unsafe-state-top-k", type=int, default=1)
    parser.add_argument("--unsafe-state-greedy-prob", type=float, default=1.0)
    parser.add_argument("--disable-unsafe-state-heuristic-fallback", action="store_true")
    parser.add_argument(
        "--unsafe-state-heuristic-fallback-end-fraction",
        type=float,
        default=1.10,
        help="Disable training-time unsafe-state heuristic fallback after this episode progress fraction. Values >1 keep it enabled for the whole run.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--reward-profile",
        choices=("hybrid-stability", "ttft-focused", "stability-strict"),
        default="stability-strict",
    )
    parser.add_argument("--reward-first-token", type=float)
    parser.add_argument("--reward-completion", type=float)
    parser.add_argument("--reward-new-admission", type=float)
    parser.add_argument("--penalty-duration", type=float)
    parser.add_argument("--penalty-admission-wait", type=float)
    parser.add_argument("--penalty-ttft", type=float)
    parser.add_argument("--penalty-e2e", type=float)
    parser.add_argument("--penalty-pending-admission", type=float)
    parser.add_argument("--penalty-pending-admission-tail", type=float)
    parser.add_argument("--penalty-first-token-pending", type=float)
    parser.add_argument("--penalty-first-token-pending-tail", type=float)
    parser.add_argument("--penalty-kv", type=float)
    parser.add_argument("--kv-penalty-start-util", type=float)
    parser.add_argument("--penalty-prefill-padding", type=float)
    parser.add_argument("--penalty-eviction", type=float)
    parser.add_argument("--penalty-recompute-token", type=float)
    parser.add_argument("--penalty-wasted-kv", type=float)
    parser.add_argument("--bonus-terminal-all-complete", type=float)
    parser.add_argument("--penalty-terminal-early-stop", type=float)
    parser.add_argument("--penalty-terminal-unfinished-request", type=float)
    parser.add_argument("--penalty-terminal-deadlock-recovery", type=float)
    parser.add_argument("--penalty-terminal-evicted-request", type=float)
    parser.add_argument("--penalty-terminal-recompute-total", type=float)
    parser.add_argument("--penalty-terminal-wasted-kv-total", type=float)
    parser.add_argument("--penalty-terminal-external-fragmentation", type=float)
    parser.add_argument("--penalty-terminal-internal-fragmentation", type=float)
    parser.add_argument("--penalty-terminal-ttft-p95", type=float)
    parser.add_argument("--penalty-terminal-ttft-p99", type=float)
    parser.add_argument("--penalty-terminal-ttft-cvar95", type=float)
    parser.add_argument("--penalty-terminal-ttft-cvar99", type=float)
    parser.add_argument("--penalty-terminal-ttft-max", type=float)
    parser.add_argument("--penalty-terminal-e2e-p95", type=float)
    parser.add_argument("--penalty-terminal-admission-wait-cvar95", type=float)
    parser.add_argument("--terminal-peak-kv-target-min-util", type=float)
    parser.add_argument("--terminal-peak-kv-target-max-util", type=float)
    parser.add_argument("--penalty-terminal-peak-kv-underutil", type=float)
    parser.add_argument("--penalty-terminal-peak-kv-overutil", type=float)
    parser.add_argument("--reward-high-kv-fast-admission", type=float)
    parser.add_argument("--reward-high-kv-normal-admission", type=float)
    parser.add_argument("--penalty-high-kv-new-admission", type=float)
    parser.add_argument("--penalty-high-kv-risky-admission", type=float)
    parser.add_argument("--reward-high-kv-fast-first-token", type=float)
    parser.add_argument("--reward-high-kv-decode-token", type=float)
    parser.add_argument("--reward-high-kv-zero-admission", type=float)
    parser.add_argument("--reward-high-kv-decode-drain", type=float)
    parser.add_argument("--penalty-high-kv-prefill-priority", type=float)
    parser.add_argument("--penalty-high-kv-prefill-work", type=float)
    parser.add_argument("--penalty-projected-kv-admission", type=float)
    parser.add_argument("--penalty-projected-shadow-admission", type=float)
    parser.add_argument("--projected-kv-admission-penalty-start-util", type=float)
    parser.add_argument("--projected-shadow-admission-penalty-start-util", type=float)
    parser.add_argument("--penalty-high-kv-fast-pending-tail", type=float)
    parser.add_argument("--penalty-high-kv-normal-pending-tail", type=float)
    parser.add_argument("--penalty-high-kv-admission-stall", type=float)
    parser.add_argument("--penalty-short-admission-stall", type=float)
    parser.add_argument("--penalty-first-token-pending-age-quadratic", type=float)
    parser.add_argument("--penalty-risky-first-token-backlog", type=float)
    parser.add_argument("--penalty-action-safety-projection", type=float)
    parser.add_argument(
        "--actor-projection-debit-scale",
        type=float,
        default=0.0,
        help="Subtract a reason-specific debit from the actor reward when safety projection rewrites raw action.",
    )
    parser.add_argument(
        "--actor-projection-debit-reason-weights",
        default=DEFAULT_ACTOR_PROJECTION_DEBIT_REASON_WEIGHTS,
        help="Comma-separated reason=weight multipliers for --actor-projection-debit-scale.",
    )
    parser.add_argument("--safety-projection-penalty-start-fraction", type=float, default=0.60)
    parser.add_argument("--safety-projection-penalty-end-fraction", type=float, default=0.90)
    parser.add_argument(
        "--projection-lagrangian-target",
        type=float,
        help="When set, adapt the action-safety projection penalty toward this target fraction.",
    )
    parser.add_argument("--projection-lagrangian-lr", type=float, default=0.25)
    parser.add_argument("--projection-lagrangian-init", type=float)
    parser.add_argument(
        "--projection-lagrangian-min",
        type=float,
        default=0.0,
        help="Lower bound for the adaptive action-safety projection penalty dual.",
    )
    parser.add_argument("--projection-lagrangian-max", type=float, default=5.0)
    parser.add_argument("--enable-liveness-lagrangian", action="store_true")
    parser.add_argument(
        "--liveness-lagrangian-targets",
        default="deadlock_cost=0,unfinished_cost=0,prefill_liveness_violation=0,first_token_backlog_violation=0,unsafe_projection_cost=0.02",
    )
    parser.add_argument("--liveness-lagrangian-lr", type=float, default=0.25)
    parser.add_argument("--liveness-lagrangian-lrs", default="")
    parser.add_argument("--liveness-lagrangian-init", default="")
    parser.add_argument("--liveness-lagrangian-mins", default="")
    parser.add_argument("--liveness-lagrangian-max", type=float, default=10.0)
    parser.add_argument("--liveness-lagrangian-maxs", default="")
    add_dynamic_workload_args(parser)
    return parser.parse_args()


def clone_requests(requests: Sequence[Request]) -> List[Request]:
    return [
        Request(
            request_id=request.request_id,
            arrival_ms=request.arrival_ms,
            prompt_tokens=request.prompt_tokens,
            length_distribution=request.length_distribution,
            model_id=request.model_id,
            priority=request.priority,
            template_id=request.template_id,
            template_family=request.template_family,
            template_version=request.template_version,
            max_new_tokens=request.max_new_tokens,
            stop_strings=list(request.stop_strings),
            stop_at_eos=request.stop_at_eos,
            structured_output=request.structured_output,
            shared_prefix_id=request.shared_prefix_id,
            shared_prefix_tokens=request.shared_prefix_tokens,
            private_prompt_tokens=request.private_prompt_tokens,
            actual_output_tokens=request.actual_output_tokens,
        )
        for request in requests
    ]


def curriculum_trace_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    seen: set[Path] = set()
    for candidate in [args.trace, *((args.curriculum_trace or []))]:
        if candidate is None:
            continue
        path = Path(candidate)
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def bucket_coverage_curriculum_requests(args: argparse.Namespace) -> int:
    configured = int(getattr(args, "bucket_coverage_curriculum_requests", 0) or 0)
    return configured if configured > 0 else int(args.requests)


def curriculum_phase(args: argparse.Namespace, episode_index: int | None) -> str:
    if not getattr(args, "phased_curriculum", False):
        return "target"
    if episode_index is None:
        episode_index = 0
    total_span = max(int(getattr(args, "episodes", 1)) - 1, 1)
    progress = min(max(float(episode_index) / float(total_span), 0.0), 1.0)
    warmup_fraction = min(max(float(getattr(args, "curriculum_warmup_fraction", 0.0)), 0.0), 1.0)
    mid_fraction = min(max(float(getattr(args, "curriculum_mid_fraction", 0.0)), 0.0), 1.0)
    if progress < warmup_fraction:
        return "warmup"
    if progress < min(warmup_fraction + mid_fraction, 1.0):
        return "mid"
    return "target"


def phased_curriculum_sources(args: argparse.Namespace, phase: str) -> List[Tuple[str, Path | None]]:
    if phase == "warmup":
        return [("synthetic", None)]
    if phase == "mid":
        sources: List[Tuple[str, Path | None]] = [("synthetic", None)]
        sources.extend(("trace", path) for path in curriculum_trace_paths(args))
        if getattr(args, "bucket_coverage_curriculum", False):
            sources.append(("bucket-coverage", None))
        return sources
    sources = curriculum_sources(args)
    return sources or [("synthetic", None)]


def curriculum_request_count(args: argparse.Namespace, source_kind: str, phase: str) -> int:
    if source_kind == "bucket-coverage":
        target = bucket_coverage_curriculum_requests(args)
    else:
        target = int(args.requests)
    if phase == "warmup":
        return max(8, min(target, int(math.ceil(target * 0.50))))
    if phase == "mid":
        return max(8, min(target, int(math.ceil(target * 0.75))))
    return target


def training_progress_fraction(args: argparse.Namespace, episode_index: int | None) -> float:
    if episode_index is None:
        episode_index = 0
    total_span = max(int(getattr(args, "episodes", 1)) - 1, 1)
    return min(max(float(episode_index) / float(total_span), 0.0), 1.0)


def safety_projection_penalty_scale(args: argparse.Namespace, episode_index: int | None) -> float:
    start = min(max(float(getattr(args, "safety_projection_penalty_start_fraction", 0.60)), 0.0), 1.0)
    end = min(max(float(getattr(args, "safety_projection_penalty_end_fraction", 0.90)), 0.0), 1.0)
    if end < start:
        start, end = end, start
    progress = training_progress_fraction(args, episode_index)
    if progress <= start:
        return 0.0
    if progress >= end:
        return 1.0
    return (progress - start) / max(end - start, 1e-6)


def scheduled_reward_config(
    args: argparse.Namespace,
    reward_config: RewardConfig,
    episode_index: int | None,
) -> RewardConfig:
    scale = safety_projection_penalty_scale(args, episode_index)
    if scale == 1.0:
        return reward_config
    return replace(
        reward_config,
        action_safety_projection_penalty=reward_config.action_safety_projection_penalty * scale,
    )


def projection_lagrangian_enabled(args: argparse.Namespace) -> bool:
    return getattr(args, "projection_lagrangian_target", None) is not None


def parse_cost_spec(spec: str | None, default: float = 0.0) -> Dict[str, float]:
    values = {name: float(default) for name in LIVENESS_COST_NAMES}
    if not spec:
        return values
    for raw_item in str(spec).replace(";", ",").split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" in item:
            key, raw_value = item.split("=", 1)
        elif ":" in item:
            key, raw_value = item.split(":", 1)
        else:
            raise ValueError(f"expected cost=value item, got {item!r}")
        key = key.strip()
        if key not in values:
            raise ValueError(f"unknown liveness cost {key!r}; expected one of {', '.join(LIVENESS_COST_NAMES)}")
        values[key] = float(raw_value)
    return values


def multi_lagrangian_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "enable_liveness_lagrangian", False))


def initial_lagrangian_duals(args: argparse.Namespace, reward_config: RewardConfig) -> Dict[str, float]:
    unsafe_default = (
        reward_config.action_safety_projection_penalty
        if multi_lagrangian_enabled(args) or projection_lagrangian_enabled(args)
        else 0.0
    )
    defaults = {
        "deadlock_cost": reward_config.deadlock_cost_penalty,
        "unfinished_cost": reward_config.unfinished_cost_penalty,
        "prefill_liveness_violation": reward_config.prefill_liveness_violation_penalty,
        "first_token_backlog_violation": reward_config.first_token_backlog_violation_penalty,
        "unsafe_projection_cost": reward_config.unsafe_projection_cost_penalty or unsafe_default,
    }
    configured = parse_cost_spec(getattr(args, "liveness_lagrangian_init", ""), 0.0)
    return {
        name: float(configured[name] if getattr(args, "liveness_lagrangian_init", "") else defaults[name])
        for name in LIVENESS_COST_NAMES
    }


def reward_config_with_lagrangian_duals(reward_config: RewardConfig, duals: Dict[str, float]) -> RewardConfig:
    unsafe_dual = float(duals.get("unsafe_projection_cost", 0.0))
    return replace(
        reward_config,
        deadlock_cost_penalty=float(duals.get("deadlock_cost", 0.0)),
        unfinished_cost_penalty=float(duals.get("unfinished_cost", 0.0)),
        prefill_liveness_violation_penalty=float(duals.get("prefill_liveness_violation", 0.0)),
        first_token_backlog_violation_penalty=float(duals.get("first_token_backlog_violation", 0.0)),
        unsafe_projection_cost_penalty=unsafe_dual,
        action_safety_projection_penalty=0.0 if unsafe_dual > 0.0 else reward_config.action_safety_projection_penalty,
    )


def update_lagrangian_duals(
    current_duals: Dict[str, float],
    observed_costs: Dict[str, float],
    args: argparse.Namespace,
) -> Dict[str, float]:
    if not multi_lagrangian_enabled(args):
        return dict(current_duals)
    targets = parse_cost_spec(getattr(args, "liveness_lagrangian_targets", ""), 0.0)
    learning_rates = parse_cost_spec(
        getattr(args, "liveness_lagrangian_lrs", ""),
        float(getattr(args, "liveness_lagrangian_lr", 0.25)),
    )
    lower_bounds = parse_cost_spec(getattr(args, "liveness_lagrangian_mins", ""), 0.0)
    upper_bounds = parse_cost_spec(
        getattr(args, "liveness_lagrangian_maxs", ""),
        float(getattr(args, "liveness_lagrangian_max", 10.0)),
    )
    updated: Dict[str, float] = {}
    for name in LIVENESS_COST_NAMES:
        lower = max(float(lower_bounds[name]), 0.0)
        upper = max(float(upper_bounds[name]), lower)
        value = float(current_duals.get(name, 0.0)) + max(float(learning_rates[name]), 0.0) * (
            float(observed_costs.get(name, 0.0)) - float(targets[name])
        )
        updated[name] = min(max(value, lower), upper)
    return updated


def update_projection_lagrangian_dual(
    current_dual: float,
    observed_projection_fraction: float,
    args: argparse.Namespace,
) -> float:
    if not projection_lagrangian_enabled(args):
        return current_dual
    target = min(max(float(args.projection_lagrangian_target), 0.0), 1.0)
    learning_rate = max(float(args.projection_lagrangian_lr), 0.0)
    lower_bound = max(float(getattr(args, "projection_lagrangian_min", 0.0) or 0.0), 0.0)
    upper_bound = max(float(args.projection_lagrangian_max), 0.0)
    if upper_bound < lower_bound:
        upper_bound = lower_bound
    updated = current_dual + learning_rate * (observed_projection_fraction - target)
    return min(max(updated, lower_bound), upper_bound)


def allow_unsafe_state_heuristic_fallback(args: argparse.Namespace, episode_index: int | None) -> bool:
    if getattr(args, "disable_unsafe_state_heuristic_fallback", False):
        return False
    end_fraction = float(getattr(args, "unsafe_state_heuristic_fallback_end_fraction", 1.10))
    return training_progress_fraction(args, episode_index) <= end_fraction


def curriculum_sources(args: argparse.Namespace) -> List[Tuple[str, Path | None]]:
    sources: List[Tuple[str, Path | None]] = [("trace", path) for path in curriculum_trace_paths(args)]
    if getattr(args, "bucket_coverage_curriculum", False):
        sources.append(("bucket-coverage", None))
    return sources


def select_curriculum_source(
    args: argparse.Namespace,
    episode_index: int,
    episode_seed: int,
) -> Tuple[str, Path | None] | None:
    phase = curriculum_phase(args, episode_index)
    sources = phased_curriculum_sources(args, phase) if getattr(args, "phased_curriculum", False) else curriculum_sources(args)
    if not sources:
        return None
    if args.curriculum_schedule == "random":
        rng = np.random.default_rng(episode_seed)
        return sources[int(rng.integers(0, len(sources)))]
    return sources[int(episode_index) % len(sources)]


def select_curriculum_trace(args: argparse.Namespace, episode_index: int, episode_seed: int) -> Path | None:
    source = select_curriculum_source(args, episode_index, episode_seed)
    if source is None:
        return None
    source_kind, source_path = source
    return source_path if source_kind == "trace" else None


def curriculum_source_label(source: Tuple[str, Path | None] | None) -> str:
    if source is None:
        return "synthetic"
    source_kind, source_path = source
    return str(source_path) if source_path is not None else source_kind


def load_requests_from_trace(args: argparse.Namespace, trace_path: Path, episode_seed: int) -> List[Request]:
    return load_trace(trace_path, episode_seed, dynamic_config=dynamic_workload_config_from_args(args))


def build_validation_gate_request_sets(args: argparse.Namespace) -> List[Tuple[str, List[Request]]]:
    traces = list(getattr(args, "validation_gate_trace", None) or [])
    return [
        (
            str(trace_path),
            load_requests_from_trace(args, trace_path, int(args.validation_gate_seed) + trace_index),
        )
        for trace_index, trace_path in enumerate(traces)
    ]


def build_requests(args: argparse.Namespace, episode_seed: int, episode_index: int | None = None) -> List[Request]:
    resolved_episode_index = episode_index if episode_index is not None else max(0, episode_seed - int(args.seed))
    phase = curriculum_phase(args, resolved_episode_index)
    source = select_curriculum_source(
        args,
        resolved_episode_index,
        episode_seed,
    )
    if source is not None:
        source_kind, source_path = source
        if source_kind == "trace" and source_path is not None:
            return load_requests_from_trace(args, source_path, episode_seed)
        if source_kind == "synthetic":
            return generate_synthetic_requests(curriculum_request_count(args, source_kind, phase), episode_seed)
        if source_kind == "bucket-coverage":
            return generate_bucket_coverage_requests(curriculum_request_count(args, source_kind, phase), episode_seed)
    return generate_synthetic_requests(curriculum_request_count(args, "synthetic", phase), episode_seed)


def build_evaluation_request_sets(args: argparse.Namespace, episode_seed: int) -> List[Tuple[str, List[Request]]]:
    traces = curriculum_trace_paths(args)
    if traces or getattr(args, "bucket_coverage_curriculum", False):
        request_sets = [(str(trace_path), load_requests_from_trace(args, trace_path, episode_seed)) for trace_path in traces]
        if getattr(args, "bucket_coverage_curriculum", False):
            request_sets.append(
                (
                    "bucket-coverage",
                    generate_bucket_coverage_requests(bucket_coverage_curriculum_requests(args), episode_seed),
                )
            )
        return request_sets
    return [("synthetic", generate_synthetic_requests(args.requests, episode_seed))]


def build_config(args: argparse.Namespace) -> SimulatorConfig:
    action_space = load_action_space_config(args.action_space_config) if args.action_space_config else ActionSpaceConfig()
    return SimulatorConfig(
        token_budget_per_step=args.token_budget,
        prefill_chunk_size=args.prefill_chunk,
        kv_capacity_tokens=args.kv_capacity,
        kv_allocator_mode=args.kv_allocator,
        kv_block_size_tokens=args.kv_block_size,
        prefill_ms_per_token=args.prefill_ms_per_token,
        decode_ms_per_token=args.decode_ms_per_token,
        simulation_duration_ms=args.simulation_duration_ms,
        unfinished_age_grace_ms=args.unfinished_age_grace_ms,
        prefill_decision_interval_ms=args.prefill_decision_interval_ms,
        decode_decision_interval_ms=args.decode_decision_interval_ms,
        kv_reservation_quantile=args.shadow_kv_quantile,
        prefill_padding_weight=args.prefill_padding_weight,
        prefix_cache_enabled=not args.disable_prefix_cache,
        prefix_cache_max_resident_entries=args.prefix_cache_max_resident_entries,
        kv_pressure_threshold=args.kv_pressure_threshold,
        allow_shadow_oversubscription=args.allow_shadow_oversubscription,
        enable_action_safety_filter=not args.disable_action_safety_filter,
        action_safety_kv_util_threshold=args.action_safety_kv_util_threshold,
        action_safety_drain_kv_util_threshold=args.action_safety_drain_kv_util_threshold,
        action_safety_drain_on_first_token_pending=args.enable_first_token_drain_shield,
        action_safety_max_new_admissions=args.action_safety_max_new_admissions,
        action_safety_pending_prompt_load_threshold=args.action_safety_pending_prompt_load_threshold,
        action_space=action_space,
    )


def evaluate_policy(
    agent: TorchFactorizedAgent,
    config: SimulatorConfig,
    reward_config: RewardConfig,
    requests: Sequence[Request],
) -> Tuple[Dict[str, float], List[Dict[str, object]], List[Dict[str, object]]]:
    return evaluate_policy_with_capture(
        agent,
        config,
        reward_config,
        requests,
        capture_trajectory=True,
        capture_sparse_events=False,
    )


def evaluate_policy_with_capture(
    agent: TorchFactorizedAgent,
    config: SimulatorConfig,
    reward_config: RewardConfig,
    requests: Sequence[Request],
    capture_trajectory: bool,
    capture_sparse_events: bool,
    max_steps: int = 0,
    progress_every: int = 0,
    progress_label: str = "",
    wall_time_limit_s: float = 0.0,
    decode_agent: Optional[TorchFactorizedAgent] = None,
) -> Tuple[Dict[str, float], List[Dict[str, object]], List[Dict[str, object]]]:
    eval_agent = agent.clone_for_eval()
    eval_decode_agent = decode_agent.clone_for_eval() if decode_agent is not None else None
    scheduler = RLDistributionAwareScheduler(
        config,
        eval_agent,
        reward_config,
        decode_agent=eval_decode_agent,
        greedy=True,
        capture_trajectory=capture_trajectory,
        capture_sparse_events=capture_sparse_events,
        feedback_mode=eval_agent.config.feedback_mode,
    )
    simulator = LLMSingleNodeSimulator(clone_requests(requests), config, scheduler=scheduler)
    summary = simulator.run(
        max_steps=max_steps,
        progress_every=progress_every,
        progress_label=progress_label,
        wall_time_limit_s=wall_time_limit_s,
    )
    terminal_reward = scheduler.apply_terminal_objective(summary)
    total_reward = float(sum(eval_agent.rewards))
    eval_costs = scheduler.normalized_costs()
    diagnostic_metrics = scheduler.diagnostic_metrics()
    prefill_cf = sum(float(row.get("counterfactual_prefill_cost_delta", 0.0) or 0.0) for row in eval_agent.transition_rows)
    decode_rows = eval_decode_agent.transition_rows if eval_decode_agent is not None else eval_agent.transition_rows
    decode_cf = sum(float(row.get("counterfactual_decode_cost_delta", 0.0) or 0.0) for row in decode_rows)
    return (
        {
            "reward": total_reward,
            "executed_reward": float(scheduler.executed_reward_sum),
            "actor_projection_debit": float(scheduler.actor_projection_debit_sum),
            "actor_reward": total_reward,
            "terminal_reward": terminal_reward,
            "prefill_reward": float(sum(eval_agent.prefill_rewards)),
            "decode_reward": float(
                sum(eval_decode_agent.decode_rewards) if eval_decode_agent is not None else sum(eval_agent.decode_rewards)
            ),
            "completed": float(summary["completed"]),
            "ttft_mean": float(summary["ttft_ms"]["mean"] or 0.0),
            "ttft_p95": float(summary["ttft_ms"]["p95"] or 0.0),
            "ttft_p99": float(summary["ttft_ms"]["p99"] or 0.0),
            "ttft_cvar95": float(summary["ttft_ms"]["cvar95"] or 0.0),
            "ttft_cvar99": float(summary["ttft_ms"]["cvar99"] or 0.0),
            "ttft_max": float(summary["ttft_ms"]["max"] or 0.0),
            "e2e_p95": float(summary["e2e_ms"]["p95"] or 0.0),
            "e2e_p99": float(summary["e2e_ms"]["p99"] or 0.0),
            "e2e_cvar95": float(summary["e2e_ms"]["cvar95"] or 0.0),
            "e2e_cvar99": float(summary["e2e_ms"]["cvar99"] or 0.0),
            "admission_wait_p95": float(summary["admission_wait_ms"]["p95"] or 0.0),
            "admission_wait_cvar95": float(summary["admission_wait_ms"]["cvar95"] or 0.0),
            "peak_kv_tokens": float(summary["peak_kv_tokens"]),
            "peak_allocator_internal_fragmentation_tokens": float(
                summary.get("peak_allocator_internal_fragmentation_tokens", 0.0)
            ),
            "peak_allocator_external_fragmentation_tokens": float(
                summary.get("peak_allocator_external_fragmentation_tokens", 0.0)
            ),
            "safety_projection_fraction": float(scheduler.safety_projection_fraction()),
            **{name: float(value) for name, value in scheduler.decision_gate_fractions().items()},
            **{name: float(value) for name, value in eval_costs.items()},
            **diagnostic_metrics,
            **{name: float(scheduler.reward_component_sums.get(name, 0.0)) for name in REWARD_COMPONENT_NAMES},
            "counterfactual_prefill_cost_delta": float(prefill_cf),
            "counterfactual_decode_cost_delta": float(decode_cf),
            "terminated_early": float(bool(summary["terminated_early"])),
            "unfinished_requests": float(summary["unfinished_requests"]),
            "horizon_unfinished_requests": float(summary.get("horizon_unfinished_requests", 0.0)),
            "aged_unfinished_requests": float(summary.get("aged_unfinished_requests", 0.0)),
            "liveness_unfinished_requests": float(summary.get("liveness_unfinished_requests", 0.0)),
            "punishable_unfinished_requests": float(summary.get("punishable_unfinished_requests", 0.0)),
            "termination_reason": str(summary.get("termination_reason") or ""),
            "evictions_total": float(summary["evictions_total"]),
            "recompute_tokens_total": float(summary["recompute_tokens_total"]),
            "simulation_steps": float(summary.get("simulation_steps", 0.0)),
            "wall_time_seconds": float(summary.get("wall_time_seconds", 0.0)),
        },
        list(eval_agent.transition_rows),
        list(eval_agent.step_events),
    )


def merge_evaluation_suite(
    evaluations: Sequence[Tuple[str, Dict[str, float], List[Dict[str, object]], List[Dict[str, object]]]],
) -> Tuple[Dict[str, float | object], List[Dict[str, object]], List[Dict[str, object]]]:
    if not evaluations:
        raise ValueError("expected at least one evaluation result")

    numeric_keys = (
        "reward",
        "executed_reward",
        "actor_projection_debit",
        "actor_reward",
        "terminal_reward",
        "prefill_reward",
        "decode_reward",
        "completed",
        "ttft_mean",
        "ttft_p95",
        "ttft_p99",
        "ttft_cvar95",
        "ttft_cvar99",
        "ttft_max",
        "e2e_p95",
        "e2e_p99",
        "e2e_cvar95",
        "e2e_cvar99",
        "admission_wait_p95",
        "admission_wait_cvar95",
        "peak_kv_tokens",
        "peak_allocator_internal_fragmentation_tokens",
        "peak_allocator_external_fragmentation_tokens",
        "safety_projection_fraction",
        *REWARD_COMPONENT_NAMES,
        "diagnostic_step_rows",
        "projection_count",
        "projection_rate",
        "prefill_liveness_risky_cap_floor_count",
        "prefill_liveness_risky_cap_floor_rate",
        "fallback_count",
        "fallback_rate",
        "unfinished_mean",
        "unfinished_p95",
        "unfinished_max",
        "kv_util_mean",
        "kv_util_p95",
        "decode_starvation_steps",
        "prefill_starvation_steps",
        "policy_entropy_mean",
        "kl_to_anchor_teacher_mean",
        "prefill_decision_act_fraction",
        "decode_decision_act_fraction",
        "prefill_decision_checks",
        "decode_decision_checks",
        *LIVENESS_COST_NAMES,
        "counterfactual_prefill_cost_delta",
        "counterfactual_decode_cost_delta",
        "unfinished_requests",
        "horizon_unfinished_requests",
        "aged_unfinished_requests",
        "liveness_unfinished_requests",
        "punishable_unfinished_requests",
        "evictions_total",
        "recompute_tokens_total",
        "simulation_steps",
        "wall_time_seconds",
    )
    aggregate: Dict[str, float | object] = {
        key: float(np.mean([float(payload[key]) for _, payload, _, _ in evaluations]))
        for key in numeric_keys
    }
    terminated_early_count = sum(int(bool(payload.get("terminated_early", 0.0))) for _, payload, _, _ in evaluations)
    termination_reasons = sorted(
        {
            str(payload.get("termination_reason") or "")
            for _, payload, _, _ in evaluations
            if str(payload.get("termination_reason") or "")
        }
    )
    aggregate["terminated_early"] = float(terminated_early_count > 0)
    aggregate["terminated_early_trace_count"] = float(terminated_early_count)
    aggregate["curriculum_trace_count"] = float(len(evaluations))
    aggregate["termination_reason"] = "|".join(termination_reasons)
    aggregate["per_trace"] = [
        {
            "trace": trace_label,
            **payload,
        }
        for trace_label, payload, _, _ in evaluations
    ]

    merged_trajectory: List[Dict[str, object]] = []
    merged_events: List[Dict[str, object]] = []
    for trace_label, _payload, trajectory_rows, event_rows in evaluations:
        merged_trajectory.extend({"evaluation_trace": trace_label, **row} for row in trajectory_rows)
        merged_events.extend({"evaluation_trace": trace_label, **row} for row in event_rows)
    return aggregate, merged_trajectory, merged_events


def evaluate_policy_suite_with_capture(
    agent: TorchFactorizedAgent,
    config: SimulatorConfig,
    reward_config: RewardConfig,
    evaluation_requests: Sequence[Tuple[str, Sequence[Request]]],
    capture_trajectory: bool,
    capture_sparse_events: bool,
    max_steps: int = 0,
    progress_every: int = 0,
    progress_label: str = "",
    wall_time_limit_s: float = 0.0,
    decode_agent: Optional[TorchFactorizedAgent] = None,
) -> Tuple[Dict[str, float | object], List[Dict[str, object]], List[Dict[str, object]]]:
    evaluations: List[Tuple[str, Dict[str, float], List[Dict[str, object]], List[Dict[str, object]]]] = []
    for trace_label, requests in evaluation_requests:
        payload, trajectory_rows, event_rows = evaluate_policy_with_capture(
            agent,
            config,
            reward_config,
            requests,
            capture_trajectory=capture_trajectory,
            capture_sparse_events=capture_sparse_events,
            max_steps=max_steps,
            progress_every=progress_every,
            progress_label=f"{progress_label}:{trace_label}" if progress_label else trace_label,
            wall_time_limit_s=wall_time_limit_s,
            decode_agent=decode_agent,
        )
        evaluations.append((trace_label, payload, trajectory_rows, event_rows))
    return merge_evaluation_suite(evaluations)


def log_training_progress(payload: Dict[str, object]) -> None:
    print(
        "[train-progress] " + " ".join(f"{key}={value}" for key, value in payload.items()),
        file=sys.stderr,
        flush=True,
    )


def evaluation_rank_key(evaluation: Dict[str, float]) -> Tuple[float, ...]:
    return evaluation_rank_key_with_options(evaluation, prefer_low_safety_projection=False)


def _best_selection_objective(value: str) -> str:
    objective = str(value or "reward").strip().replace("_", "-").lower()
    if objective in {"reward", "reward-first"}:
        return "reward"
    if objective in {"reward-margin", "reward-with-margin", "slo-margin", "margin"}:
        return "reward-margin"
    if objective in {"tail", "tail-latency", "tail-first"}:
        return "tail"
    raise ValueError(f"unsupported best selection objective: {value!r}")


def evaluation_rank_key_with_options(
    evaluation: Dict[str, float],
    prefer_low_safety_projection: bool = False,
) -> Tuple[float, ...]:
    terminated_early = bool(
        evaluation.get(
            "terminated_early_trace_count",
            evaluation.get("terminated_early", 0.0),
        )
    )
    unfinished_requests = float(evaluation.get("unfinished_requests", 0.0) or 0.0)
    evictions = float(evaluation.get("evictions_total", 0.0) or 0.0)
    recompute_k = float(evaluation.get("recompute_tokens_total", 0.0) or 0.0) / 1024.0
    ttft_max = float(evaluation.get("ttft_max", 0.0) or 0.0)
    ttft_cvar99 = float(evaluation.get("ttft_cvar99", 0.0) or 0.0)
    ttft_p99 = float(evaluation.get("ttft_p99", 0.0) or 0.0)
    ttft_cvar95 = float(evaluation.get("ttft_cvar95", 0.0) or 0.0)
    ttft_p95 = float(evaluation.get("ttft_p95", 0.0) or 0.0)
    admission_wait_cvar95 = float(evaluation.get("admission_wait_cvar95", 0.0) or 0.0)
    admission_wait_p95 = float(evaluation.get("admission_wait_p95", 0.0) or 0.0)
    safety_projection_fraction = float(evaluation.get("safety_projection_fraction", 0.0) or 0.0)
    reward = float(evaluation.get("reward", -float("inf")))
    runtime = float(evaluation.get("simulation_steps", 0.0) or 0.0)
    # Stability is a hard gate. Among stable runs, use a lexicographic tail objective:
    # the worst request matters first, then upper-tail aggregates, then runtime/reward.
    tail_rank = (
        0.0 if terminated_early else 1.0,
        -unfinished_requests,
        -(evictions + recompute_k),
        -ttft_max,
        -ttft_cvar99,
        -ttft_p99,
        -ttft_cvar95,
        -ttft_p95,
        -admission_wait_cvar95,
        -admission_wait_p95,
        -runtime,
        reward,
    )
    if not prefer_low_safety_projection:
        return tail_rank
    return (
        tail_rank[0],
        tail_rank[1],
        tail_rank[2],
        -safety_projection_fraction,
        *tail_rank[3:],
    )


def reward_first_evaluation_rank_key_with_options(
    evaluation: Dict[str, float],
    prefer_low_safety_projection: bool = False,
) -> Tuple[float, ...]:
    reward = float(evaluation.get("reward", -float("inf")))
    safety_projection_fraction = float(evaluation.get("safety_projection_fraction", 0.0) or 0.0)
    tail_rank = evaluation_rank_key_with_options(evaluation, prefer_low_safety_projection=False)
    if prefer_low_safety_projection:
        return (reward, -safety_projection_fraction, *tail_rank)
    return (reward, *tail_rank)


def best_evaluation_rank_key_with_options(
    evaluation: Dict[str, float],
    prefer_low_safety_projection: bool = False,
    best_selection_objective: str = "reward",
) -> Tuple[float, ...]:
    if _best_selection_objective(best_selection_objective) in {"reward", "reward-margin"}:
        return reward_first_evaluation_rank_key_with_options(
            evaluation,
            prefer_low_safety_projection=prefer_low_safety_projection,
        )
    return evaluation_rank_key_with_options(
        evaluation,
        prefer_low_safety_projection=prefer_low_safety_projection,
    )


def evaluation_is_best_policy_eligible(
    evaluation: Dict[str, float] | Dict[str, object],
    max_safety_projection: Optional[float] = None,
) -> bool:
    if bool(
        evaluation.get(
            "terminated_early_trace_count",
            evaluation.get("terminated_early", 0.0),
        )
    ):
        return False
    if float(evaluation.get("unfinished_requests", 0.0) or 0.0) > 0.0:
        return False
    if max_safety_projection is not None:
        safety_projection = float(evaluation.get("safety_projection_fraction", 0.0) or 0.0)
        if safety_projection > float(max_safety_projection):
            return False
    return True


def should_replace_best_evaluation(
    candidate: Dict[str, float],
    incumbent: Optional[Dict[str, float]],
    prefer_low_safety_projection: bool = False,
    max_safety_projection: Optional[float] = None,
    best_selection_objective: str = "reward",
) -> bool:
    if not evaluation_is_best_policy_eligible(candidate, max_safety_projection=max_safety_projection):
        return False
    if incumbent is None:
        return True
    if not evaluation_is_best_policy_eligible(incumbent, max_safety_projection=max_safety_projection):
        return True
    return best_evaluation_rank_key_with_options(
        candidate,
        prefer_low_safety_projection=prefer_low_safety_projection,
        best_selection_objective=best_selection_objective,
    ) > best_evaluation_rank_key_with_options(
        incumbent,
        prefer_low_safety_projection=prefer_low_safety_projection,
        best_selection_objective=best_selection_objective,
    )


def validation_gate_rank_key(gate: Dict[str, object]) -> Tuple[float, ...]:
    completed_fraction = float(gate.get("validation_gate_completed_fraction", 0.0) or 0.0)
    terminated_early_count = float(gate.get("validation_gate_terminated_early_trace_count", 0.0) or 0.0)
    unfinished_requests = float(gate.get("validation_gate_unfinished_requests", 0.0) or 0.0)
    ttft_p95 = float(gate.get("validation_gate_ttft_p95", 0.0) or 0.0)
    ttft_p99 = float(gate.get("validation_gate_ttft_p99", 0.0) or 0.0)
    e2e_p95 = float(gate.get("validation_gate_e2e_p95", 0.0) or 0.0)
    safety_projection = float(gate.get("validation_gate_safety_projection_fraction", 0.0) or 0.0)
    return (
        completed_fraction,
        1.0 if terminated_early_count <= 0.0 else 0.0,
        -unfinished_requests,
        -ttft_p95,
        -ttft_p99,
        -e2e_p95,
        -safety_projection,
    )


def validation_gate_slo_margin_penalty(
    gate: Dict[str, object],
    *,
    ttft_p95_target: float = 0.0,
    e2e_p95_target: float = 0.0,
    ttft_p95_penalty: float = 0.0,
    e2e_p95_penalty: float = 0.0,
) -> float:
    ttft_p95 = float(gate.get("validation_gate_ttft_p95", 0.0) or 0.0)
    e2e_p95 = float(gate.get("validation_gate_e2e_p95", 0.0) or 0.0)
    penalty = 0.0
    if ttft_p95_target > 0.0 and ttft_p95_penalty > 0.0:
        penalty += float(ttft_p95_penalty) * max(0.0, ttft_p95 - float(ttft_p95_target))
    if e2e_p95_target > 0.0 and e2e_p95_penalty > 0.0:
        penalty += float(e2e_p95_penalty) * max(0.0, e2e_p95 - float(e2e_p95_target))
    return penalty


def validation_ranked_evaluation_key(
    evaluation: Dict[str, float],
    gate: Dict[str, object],
    prefer_low_safety_projection: bool = False,
    best_selection_objective: str = "reward",
    validation_margin_ttft_p95_target: float = 0.0,
    validation_margin_e2e_p95_target: float = 0.0,
    validation_margin_ttft_p95_penalty: float = 0.0,
    validation_margin_e2e_p95_penalty: float = 0.0,
) -> Tuple[float, ...]:
    objective = _best_selection_objective(best_selection_objective)
    if objective in {"reward", "reward-margin"}:
        reward = float(evaluation.get("reward", -float("inf")))
        safety_projection_fraction = float(evaluation.get("safety_projection_fraction", 0.0) or 0.0)
        tail_rank = evaluation_rank_key_with_options(evaluation, prefer_low_safety_projection=False)
        primary_reward = reward
        if objective == "reward-margin":
            primary_reward -= validation_gate_slo_margin_penalty(
                gate,
                ttft_p95_target=validation_margin_ttft_p95_target,
                e2e_p95_target=validation_margin_e2e_p95_target,
                ttft_p95_penalty=validation_margin_ttft_p95_penalty,
                e2e_p95_penalty=validation_margin_e2e_p95_penalty,
            )
        if prefer_low_safety_projection:
            return (
                primary_reward,
                reward,
                -safety_projection_fraction,
                *validation_gate_rank_key(gate),
                *tail_rank,
            )
        return (
            primary_reward,
            reward,
            *validation_gate_rank_key(gate),
            *tail_rank,
        )
    return (
        *validation_gate_rank_key(gate),
        *evaluation_rank_key_with_options(
            evaluation,
            prefer_low_safety_projection=prefer_low_safety_projection,
        ),
    )


def evaluation_satisfies_safety_projection_limit(
    evaluation: Dict[str, float] | Dict[str, object],
    max_safety_projection: Optional[float] = None,
) -> bool:
    if max_safety_projection is None:
        return True
    safety_projection = float(evaluation.get("safety_projection_fraction", 0.0) or 0.0)
    return safety_projection <= float(max_safety_projection)


def should_replace_best_validation_ranked_evaluation(
    candidate: Dict[str, float],
    candidate_gate: Dict[str, object],
    incumbent: Optional[Dict[str, float]],
    incumbent_gate: Optional[Dict[str, object]],
    prefer_low_safety_projection: bool = False,
    max_safety_projection: Optional[float] = None,
    best_selection_objective: str = "reward",
    validation_margin_ttft_p95_target: float = 0.0,
    validation_margin_e2e_p95_target: float = 0.0,
    validation_margin_ttft_p95_penalty: float = 0.0,
    validation_margin_e2e_p95_penalty: float = 0.0,
) -> bool:
    if not bool(candidate_gate.get("validation_gate_passed", False)):
        return False
    if not evaluation_is_best_policy_eligible(candidate, max_safety_projection=max_safety_projection):
        return False
    if incumbent is None or incumbent_gate is None or not bool(incumbent_gate.get("validation_gate_passed", False)):
        return True
    if not evaluation_is_best_policy_eligible(incumbent, max_safety_projection=max_safety_projection):
        return True
    return validation_ranked_evaluation_key(
        candidate,
        candidate_gate,
        prefer_low_safety_projection=prefer_low_safety_projection,
        best_selection_objective=best_selection_objective,
        validation_margin_ttft_p95_target=validation_margin_ttft_p95_target,
        validation_margin_e2e_p95_target=validation_margin_e2e_p95_target,
        validation_margin_ttft_p95_penalty=validation_margin_ttft_p95_penalty,
        validation_margin_e2e_p95_penalty=validation_margin_e2e_p95_penalty,
    ) > validation_ranked_evaluation_key(
        incumbent,
        incumbent_gate,
        prefer_low_safety_projection=prefer_low_safety_projection,
        best_selection_objective=best_selection_objective,
        validation_margin_ttft_p95_target=validation_margin_ttft_p95_target,
        validation_margin_e2e_p95_target=validation_margin_e2e_p95_target,
        validation_margin_ttft_p95_penalty=validation_margin_ttft_p95_penalty,
        validation_margin_e2e_p95_penalty=validation_margin_e2e_p95_penalty,
    )


def validation_gate_payload(
    evaluation: Dict[str, float | object],
    total_requests: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    per_trace = evaluation.get("per_trace", [])
    if isinstance(per_trace, list) and per_trace:
        completed = sum(float(payload.get("completed", 0.0) or 0.0) for payload in per_trace if isinstance(payload, dict))
        unfinished = sum(
            float(payload.get("unfinished_requests", 0.0) or 0.0)
            for payload in per_trace
            if isinstance(payload, dict)
        )
        terminated_early_count = sum(
            int(bool(payload.get("terminated_early", 0.0)))
            for payload in per_trace
            if isinstance(payload, dict)
        )
        safety_projection = max(
            (
                float(payload.get("safety_projection_fraction", 0.0) or 0.0)
                for payload in per_trace
                if isinstance(payload, dict)
            ),
            default=0.0,
        )
    else:
        completed = float(evaluation.get("completed", 0.0) or 0.0)
        unfinished = float(evaluation.get("unfinished_requests", 0.0) or 0.0)
        terminated_early_count = int(bool(evaluation.get("terminated_early", 0.0)))
        safety_projection = float(evaluation.get("safety_projection_fraction", 0.0) or 0.0)
    horizon_unfinished = float(evaluation.get("horizon_unfinished_requests", 0.0) or 0.0)
    aged_unfinished = float(evaluation.get("aged_unfinished_requests", 0.0) or 0.0)
    liveness_unfinished = float(evaluation.get("liveness_unfinished_requests", 0.0) or 0.0)
    punishable_unfinished = float(
        evaluation.get("punishable_unfinished_requests", aged_unfinished + liveness_unfinished) or 0.0
    )

    completed_fraction = completed / float(max(int(total_requests), 1))
    ttft_p95 = float(evaluation.get("ttft_p95", 0.0) or 0.0)
    ttft_p99 = float(evaluation.get("ttft_p99", 0.0) or 0.0)
    e2e_p95 = float(evaluation.get("e2e_p95", 0.0) or 0.0)
    ttft_margin_target = float(getattr(args, "validation_margin_ttft_p95_target", 0.0) or 0.0)
    e2e_margin_target = float(getattr(args, "validation_margin_e2e_p95_target", 0.0) or 0.0)
    ttft_margin_excess = max(0.0, ttft_p95 - ttft_margin_target) if ttft_margin_target > 0.0 else 0.0
    e2e_margin_excess = max(0.0, e2e_p95 - e2e_margin_target) if e2e_margin_target > 0.0 else 0.0
    slo_margin_penalty = (
        max(float(getattr(args, "validation_margin_ttft_p95_penalty", 0.0) or 0.0), 0.0) * ttft_margin_excess
        + max(float(getattr(args, "validation_margin_e2e_p95_penalty", 0.0) or 0.0), 0.0) * e2e_margin_excess
    )
    reject_reasons: List[str] = []
    min_completed_fraction = min(max(float(args.validation_gate_min_completed_fraction), 0.0), 1.0)
    max_unfinished = max(float(args.validation_gate_max_unfinished), 0.0)
    if completed_fraction + 1e-9 < min_completed_fraction:
        reject_reasons.append("completed_fraction")
    if unfinished > max_unfinished:
        reject_reasons.append("unfinished_requests")
    if terminated_early_count > 0 and not bool(args.validation_gate_allow_early_stop):
        reject_reasons.append("terminated_early")
    if args.validation_gate_max_safety_projection is not None:
        if safety_projection > float(args.validation_gate_max_safety_projection):
            reject_reasons.append("safety_projection")
    if args.validation_gate_max_ttft_p95 is not None:
        if ttft_p95 > float(args.validation_gate_max_ttft_p95):
            reject_reasons.append("ttft_p95")
    if args.validation_gate_max_ttft_p99 is not None:
        if ttft_p99 > float(args.validation_gate_max_ttft_p99):
            reject_reasons.append("ttft_p99")
    if args.validation_gate_max_e2e_p95 is not None:
        if e2e_p95 > float(args.validation_gate_max_e2e_p95):
            reject_reasons.append("e2e_p95")

    return {
        "validation_gate_passed": not reject_reasons,
        "validation_gate_completed": completed,
        "validation_gate_total_requests": int(total_requests),
        "validation_gate_completed_fraction": completed_fraction,
        "validation_gate_unfinished_requests": unfinished,
        "validation_gate_horizon_unfinished_requests": horizon_unfinished,
        "validation_gate_aged_unfinished_requests": aged_unfinished,
        "validation_gate_liveness_unfinished_requests": liveness_unfinished,
        "validation_gate_punishable_unfinished_requests": punishable_unfinished,
        "validation_gate_terminated_early_trace_count": float(terminated_early_count),
        "validation_gate_safety_projection_fraction": safety_projection,
        "validation_gate_ttft_p95": ttft_p95,
        "validation_gate_ttft_p99": ttft_p99,
        "validation_gate_e2e_p95": e2e_p95,
        "validation_gate_ttft_p95_margin_excess": ttft_margin_excess,
        "validation_gate_e2e_p95_margin_excess": e2e_margin_excess,
        "validation_gate_slo_margin_penalty": slo_margin_penalty,
        "validation_gate_termination_reason": str(evaluation.get("termination_reason") or ""),
        "validation_gate_reject_reason": ",".join(reject_reasons),
    }


def should_run_policy_evaluation(episode: int, total_episodes: int, eval_every: int) -> bool:
    interval = max(1, int(eval_every))
    return episode == 0 or (episode % interval == 0) or episode == total_episodes - 1


def write_metrics(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


PREFILL_ACTOR_STATE_PREFIXES = (
    "prefill_trunk.",
    "admission_head.",
    "admission_limit_head.",
    "quantile_head.",
    "risky_admission_cap_head.",
    "recovery_joint_profile_head.",
    "prefill_safety_filter_head.",
    "prefill_decision_gate_head.",
)
DECODE_ACTOR_STATE_PREFIXES = (
    "decode_trunk.",
    "phase_head.",
    "budget_profile_head.",
    "recovery_joint_profile_head.",
    "decode_safety_filter_head.",
    "decode_decision_gate_head.",
)
PREFILL_LOGIT_KEYS = (
    "admission_logits",
    "admission_limit_logits",
    "quantile_logits",
    "risky_admission_cap_logits",
    "recovery_joint_profile_logits",
    "prefill_safety_filter_logits",
    "prefill_decision_gate_logits",
)
DECODE_LOGIT_KEYS = (
    "phase_logits",
    "budget_profile_logits",
    "recovery_joint_profile_logits",
    "decode_safety_filter_logits",
    "decode_decision_gate_logits",
)


def hash_tensor_state(
    state: Dict[str, torch.Tensor],
    prefixes: Optional[Sequence[str]] = None,
) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        if prefixes is not None and not any(key.startswith(prefix) for prefix in prefixes):
            continue
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def hash_json_payload(payload: Dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def drift_audit_state_hashes(
    agent: TorchFactorizedAgent,
    decode_agent: TorchFactorizedAgent,
) -> Dict[str, str]:
    prefill_model_state = agent.model.state_dict()
    decode_model_state = decode_agent.model.state_dict()
    hashes = {
        "prefill_actor_hash": hash_tensor_state(prefill_model_state, PREFILL_ACTOR_STATE_PREFIXES),
        "decode_actor_hash": hash_tensor_state(decode_model_state, DECODE_ACTOR_STATE_PREFIXES),
        "prefill_model_hash": hash_tensor_state(prefill_model_state),
        "decode_model_hash": hash_tensor_state(decode_model_state),
        "prefill_critic_hash": hash_tensor_state(agent.central_critic.state_dict()),
        "decode_critic_hash": hash_tensor_state(decode_agent.central_critic.state_dict()),
        "prefill_return_norm_hash": hash_json_payload(agent.prefill_return_normalizer.state_dict()),
        "decode_return_norm_hash": hash_json_payload(decode_agent.decode_return_normalizer.state_dict()),
    }
    if agent.config.observation_normalization:
        hashes["prefill_obs_norm_hash"] = hash_json_payload(agent.prefill_normalizer.state_dict())
    else:
        hashes["prefill_obs_norm_hash"] = "disabled"
    if decode_agent.config.observation_normalization:
        hashes["decode_obs_norm_hash"] = hash_json_payload(decode_agent.decode_normalizer.state_dict())
    else:
        hashes["decode_obs_norm_hash"] = "disabled"
    return hashes


def capture_drift_probe(
    agent: TorchFactorizedAgent,
    decode_agent: TorchFactorizedAgent,
    limit: int,
) -> Dict[str, List[List[float]]]:
    limit = max(0, int(limit))
    prefill_rows = agent.transition_rows[:limit] if limit else []
    decode_rows = decode_agent.transition_rows[:limit] if limit else []
    prefill_normalized = [
        np.asarray(row, dtype=np.float64).reshape(-1).tolist()
        for row in agent.prefill_observations[:limit]
    ]
    decode_normalized = [
        np.asarray(row, dtype=np.float64).reshape(-1).tolist()
        for row in decode_agent.decode_observations[:limit]
    ]
    prefill_raw = [
        np.asarray(row, dtype=np.float64).reshape(-1).tolist()
        for row in agent.rollout.raw_prefill_observations[:limit]
        if np.asarray(row).size == agent.prefill_observation_dim
    ]
    decode_raw = [
        np.asarray(row, dtype=np.float64).reshape(-1).tolist()
        for row in decode_agent.rollout.raw_decode_observations[:limit]
        if np.asarray(row).size == decode_agent.decode_observation_dim
    ]
    return {
        "prefill_raw": [
            [float(value) for value in row.get("prefill_observation", [])]
            for row in prefill_rows
            if isinstance(row.get("prefill_observation"), list)
        ]
        or prefill_raw,
        "decode_raw": [
            [float(value) for value in row.get("decode_observation", [])]
            for row in decode_rows
            if isinstance(row.get("decode_observation"), list)
        ]
        or decode_raw,
        "prefill_normalized": [
            [float(value) for value in row.get("normalized_prefill_observation", [])]
            for row in prefill_rows
            if isinstance(row.get("normalized_prefill_observation"), list)
        ]
        or prefill_normalized,
        "decode_normalized": [
            [float(value) for value in row.get("normalized_decode_observation", [])]
            for row in decode_rows
            if isinstance(row.get("normalized_decode_observation"), list)
        ]
        or decode_normalized,
    }


def _flatten_logits(outputs: Dict[str, torch.Tensor], keys: Sequence[str]) -> np.ndarray:
    arrays = [
        outputs[key].detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
        for key in keys
        if key in outputs
    ]
    if not arrays:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(arrays)


def drift_probe_logits(
    agent: TorchFactorizedAgent,
    decode_agent: TorchFactorizedAgent,
    probe: Optional[Dict[str, List[List[float]]]],
    *,
    use_current_normalizer: bool,
) -> np.ndarray:
    if not probe:
        return np.asarray([], dtype=np.float64)
    if use_current_normalizer:
        prefill_rows = [
            agent.prefill_normalizer.normalize(row)
            if agent.config.observation_normalization
            else np.asarray(row, dtype=np.float64)
            for row in probe.get("prefill_raw", [])
            if len(row) == agent.prefill_observation_dim
        ]
        decode_rows = [
            decode_agent.decode_normalizer.normalize(row)
            if decode_agent.config.observation_normalization
            else np.asarray(row, dtype=np.float64)
            for row in probe.get("decode_raw", [])
            if len(row) == decode_agent.decode_observation_dim
        ]
    else:
        prefill_rows = [
            np.asarray(row, dtype=np.float64)
            for row in probe.get("prefill_normalized", [])
            if len(row) == agent.prefill_observation_dim
        ]
        decode_rows = [
            np.asarray(row, dtype=np.float64)
            for row in probe.get("decode_normalized", [])
            if len(row) == decode_agent.decode_observation_dim
        ]
    parts: List[np.ndarray] = []
    with torch.no_grad():
        if prefill_rows:
            prefill_tensor = torch.tensor(np.asarray(prefill_rows), dtype=torch.float32, device=agent.device)
            parts.append(_flatten_logits(agent.model.prefill_outputs(prefill_tensor), PREFILL_LOGIT_KEYS))
        if decode_rows:
            decode_tensor = torch.tensor(np.asarray(decode_rows), dtype=torch.float32, device=decode_agent.device)
            parts.append(_flatten_logits(decode_agent.model.decode_outputs(decode_tensor), DECODE_LOGIT_KEYS))
    if not parts:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(parts)


def max_abs_diff(current: np.ndarray, anchor: Optional[np.ndarray]) -> float:
    if anchor is None or current.shape != anchor.shape or current.size == 0:
        return float("nan")
    return float(np.max(np.abs(current - anchor)))


def rollout_mask_stats(agent: TorchFactorizedAgent) -> Dict[str, object]:
    masks = list(agent.rollout.action_masks)
    nonempty = [mask for mask in masks if mask]
    per_head_counts: Dict[str, List[int]] = {}
    for mask in nonempty:
        for key, values in mask.items():
            per_head_counts.setdefault(str(key), []).append(len(values))
    return {
        "mask_samples": len(masks),
        "mask_nonempty_samples": len(nonempty),
        "mask_nonempty_fraction": round(len(nonempty) / max(len(masks), 1), 6),
        "mask_allowed_mean": {
            key: round(float(np.mean(values)), 6)
            for key, values in sorted(per_head_counts.items())
            if values
        },
    }


def main() -> None:
    args = parse_args()
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ppo_clip=args.ppo_clip,
        ppo_epochs=args.ppo_epochs,
        entropy_bonus=args.entropy_bonus,
        value_coef=args.value_coef,
        grad_clip=args.grad_clip,
        hidden_dim=args.hidden_dim,
        hidden_depth=args.hidden_depth,
        observation_normalization=not args.disable_observation_normalization,
        feedback_mode=args.feedback_mode,
        exploration_temperature=args.exploration_temperature,
        exploration_top_k=args.exploration_top_k,
        exploration_greedy_prob=args.exploration_greedy_prob,
        unsafe_state_temperature=args.unsafe_state_temperature,
        unsafe_state_top_k=args.unsafe_state_top_k,
        unsafe_state_greedy_prob=args.unsafe_state_greedy_prob,
        unsafe_state_heuristic_fallback=not args.disable_unsafe_state_heuristic_fallback,
        value_normalization=not args.disable_value_normalization,
        shared_critic=args.shared_critic,
        no_prefill_to_decode_condition=args.no_prefill_to_decode_condition,
        no_prev_decode_to_prefill=args.no_prev_decode_to_prefill,
        router_boundary_mode=args.router_boundary_mode,
        router_update_role=args.router_update_role,
        serialized_joint_critic=not args.disable_serialized_joint_critic,
        projection_imitation_coef=args.projection_imitation_coef,
        targeted_projection_bc_coef=args.targeted_projection_bc_coef,
        targeted_projection_bc_reasons=args.targeted_projection_bc_reasons,
        targeted_projection_bc_reason_coefs=args.targeted_projection_bc_reason_coefs,
        targeted_projection_bc_group_by_reason=not args.disable_targeted_projection_bc_group_by_reason,
        targeted_projection_bc_min_weight=args.targeted_projection_bc_min_weight,
        late_update_start_fraction=args.late_update_start_fraction,
        late_learning_rate_scale=args.late_learning_rate_scale,
        late_freeze_prefill_trunk=args.late_freeze_prefill_trunk,
        late_freeze_admission_heads=args.late_freeze_admission_heads,
        teacher_stage_a_fraction=args.teacher_stage_a_fraction,
        teacher_stage_b_fraction=args.teacher_stage_b_fraction,
        stage_a_imitation_coef=args.stage_a_imitation_coef,
        stage_c_learning_rate_scale=args.stage_c_learning_rate_scale,
        stage_c_freeze_admission_heads=args.stage_c_freeze_admission_heads,
        sequential_router_update=args.sequential_router_update,
        first_valid_freeze_actor=args.first_valid_freeze_actor,
        anchor_kl_coef=args.anchor_kl_coef,
        strong_action_mask=not args.disable_strong_action_mask,
        recovery_joint_profile_enabled=args.enable_recovery_joint_profile,
        recovery_joint_profile_min_kv_util=args.recovery_joint_profile_min_kv_util,
        history_critic_window=args.history_critic_window,
        phased_curriculum=args.phased_curriculum,
        curriculum_warmup_fraction=args.curriculum_warmup_fraction,
        curriculum_mid_fraction=args.curriculum_mid_fraction,
    )
    reward_config = reward_profile_defaults(args.reward_profile)
    if args.reward_first_token is not None:
        reward_config.first_token_reward = args.reward_first_token
    if args.reward_completion is not None:
        reward_config.completion_reward = args.reward_completion
    if args.reward_new_admission is not None:
        reward_config.new_admission_reward = args.reward_new_admission
    if args.penalty_duration is not None:
        reward_config.duration_penalty = args.penalty_duration
    if args.penalty_admission_wait is not None:
        reward_config.admission_wait_penalty = args.penalty_admission_wait
    if args.penalty_ttft is not None:
        reward_config.ttft_penalty = args.penalty_ttft
    if args.penalty_e2e is not None:
        reward_config.e2e_penalty = args.penalty_e2e
    if args.penalty_pending_admission is not None:
        reward_config.pending_admission_penalty = args.penalty_pending_admission
    if args.penalty_pending_admission_tail is not None:
        reward_config.pending_admission_tail_penalty = args.penalty_pending_admission_tail
    if args.penalty_first_token_pending is not None:
        reward_config.first_token_pending_penalty = args.penalty_first_token_pending
    if args.penalty_first_token_pending_tail is not None:
        reward_config.first_token_pending_tail_penalty = args.penalty_first_token_pending_tail
    if args.penalty_kv is not None:
        reward_config.kv_penalty = args.penalty_kv
    if args.kv_penalty_start_util is not None:
        reward_config.kv_penalty_start_util = args.kv_penalty_start_util
    if args.penalty_prefill_padding is not None:
        reward_config.prefill_padding_penalty = args.penalty_prefill_padding
    if args.penalty_eviction is not None:
        reward_config.eviction_penalty = args.penalty_eviction
    if args.penalty_recompute_token is not None:
        reward_config.recompute_token_penalty = args.penalty_recompute_token
    if args.penalty_wasted_kv is not None:
        reward_config.wasted_kv_penalty = args.penalty_wasted_kv
    if args.bonus_terminal_all_complete is not None:
        reward_config.terminal_all_complete_bonus = args.bonus_terminal_all_complete
    if args.penalty_terminal_early_stop is not None:
        reward_config.terminal_early_stop_penalty = args.penalty_terminal_early_stop
    if args.penalty_terminal_unfinished_request is not None:
        reward_config.terminal_unfinished_request_penalty = args.penalty_terminal_unfinished_request
    if args.penalty_terminal_deadlock_recovery is not None:
        reward_config.terminal_deadlock_recovery_penalty = args.penalty_terminal_deadlock_recovery
    if args.penalty_terminal_evicted_request is not None:
        reward_config.terminal_evicted_request_penalty = args.penalty_terminal_evicted_request
    if args.penalty_terminal_recompute_total is not None:
        reward_config.terminal_recompute_total_penalty = args.penalty_terminal_recompute_total
    if args.penalty_terminal_wasted_kv_total is not None:
        reward_config.terminal_wasted_kv_total_penalty = args.penalty_terminal_wasted_kv_total
    if args.penalty_terminal_external_fragmentation is not None:
        reward_config.terminal_external_fragmentation_penalty = args.penalty_terminal_external_fragmentation
    if args.penalty_terminal_internal_fragmentation is not None:
        reward_config.terminal_internal_fragmentation_penalty = args.penalty_terminal_internal_fragmentation
    if args.penalty_terminal_ttft_p95 is not None:
        reward_config.terminal_ttft_p95_penalty = args.penalty_terminal_ttft_p95
    if args.penalty_terminal_ttft_p99 is not None:
        reward_config.terminal_ttft_p99_penalty = args.penalty_terminal_ttft_p99
    if args.penalty_terminal_ttft_cvar95 is not None:
        reward_config.terminal_ttft_cvar95_penalty = args.penalty_terminal_ttft_cvar95
    if args.penalty_terminal_ttft_cvar99 is not None:
        reward_config.terminal_ttft_cvar99_penalty = args.penalty_terminal_ttft_cvar99
    if args.penalty_terminal_ttft_max is not None:
        reward_config.terminal_ttft_max_penalty = args.penalty_terminal_ttft_max
    if args.penalty_terminal_e2e_p95 is not None:
        reward_config.terminal_e2e_p95_penalty = args.penalty_terminal_e2e_p95
    if args.penalty_terminal_admission_wait_cvar95 is not None:
        reward_config.terminal_admission_wait_cvar95_penalty = args.penalty_terminal_admission_wait_cvar95
    if args.terminal_peak_kv_target_min_util is not None:
        reward_config.terminal_peak_kv_target_min_util = args.terminal_peak_kv_target_min_util
    if args.terminal_peak_kv_target_max_util is not None:
        reward_config.terminal_peak_kv_target_max_util = args.terminal_peak_kv_target_max_util
    if args.penalty_terminal_peak_kv_underutil is not None:
        reward_config.terminal_peak_kv_underutil_penalty = args.penalty_terminal_peak_kv_underutil
    if args.penalty_terminal_peak_kv_overutil is not None:
        reward_config.terminal_peak_kv_overutil_penalty = args.penalty_terminal_peak_kv_overutil
    if args.reward_high_kv_fast_admission is not None:
        reward_config.high_kv_fast_admission_reward = args.reward_high_kv_fast_admission
    if args.reward_high_kv_normal_admission is not None:
        reward_config.high_kv_normal_admission_reward = args.reward_high_kv_normal_admission
    if args.penalty_high_kv_new_admission is not None:
        reward_config.high_kv_new_admission_penalty = args.penalty_high_kv_new_admission
    if args.penalty_high_kv_risky_admission is not None:
        reward_config.high_kv_risky_admission_penalty = args.penalty_high_kv_risky_admission
    if args.reward_high_kv_fast_first_token is not None:
        reward_config.high_kv_fast_first_token_reward = args.reward_high_kv_fast_first_token
    if args.reward_high_kv_decode_token is not None:
        reward_config.high_kv_decode_token_reward = args.reward_high_kv_decode_token
    if args.reward_high_kv_zero_admission is not None:
        reward_config.high_kv_zero_admission_reward = args.reward_high_kv_zero_admission
    if args.reward_high_kv_decode_drain is not None:
        reward_config.high_kv_decode_drain_reward = args.reward_high_kv_decode_drain
    if args.penalty_high_kv_prefill_priority is not None:
        reward_config.high_kv_prefill_priority_penalty = args.penalty_high_kv_prefill_priority
    if args.penalty_high_kv_prefill_work is not None:
        reward_config.high_kv_prefill_work_penalty = args.penalty_high_kv_prefill_work
    if args.penalty_projected_kv_admission is not None:
        reward_config.projected_kv_admission_penalty = args.penalty_projected_kv_admission
    if args.penalty_projected_shadow_admission is not None:
        reward_config.projected_shadow_admission_penalty = args.penalty_projected_shadow_admission
    if args.projected_kv_admission_penalty_start_util is not None:
        reward_config.projected_kv_admission_penalty_start_util = args.projected_kv_admission_penalty_start_util
    if args.projected_shadow_admission_penalty_start_util is not None:
        reward_config.projected_shadow_admission_penalty_start_util = (
            args.projected_shadow_admission_penalty_start_util
        )
    if args.penalty_high_kv_fast_pending_tail is not None:
        reward_config.high_kv_fast_pending_tail_penalty = args.penalty_high_kv_fast_pending_tail
    if args.penalty_high_kv_normal_pending_tail is not None:
        reward_config.high_kv_normal_pending_tail_penalty = args.penalty_high_kv_normal_pending_tail
    if args.penalty_high_kv_admission_stall is not None:
        reward_config.high_kv_admission_stall_penalty = args.penalty_high_kv_admission_stall
    if args.penalty_short_admission_stall is not None:
        reward_config.short_admission_stall_penalty = args.penalty_short_admission_stall
    if args.penalty_first_token_pending_age_quadratic is not None:
        reward_config.first_token_pending_age_quadratic_penalty = args.penalty_first_token_pending_age_quadratic
    if args.penalty_risky_first_token_backlog is not None:
        reward_config.risky_first_token_backlog_penalty = args.penalty_risky_first_token_backlog
    if args.penalty_action_safety_projection is not None:
        reward_config.action_safety_projection_penalty = args.penalty_action_safety_projection
    reward_config.actor_projection_debit_scale = max(float(args.actor_projection_debit_scale), 0.0)
    reward_config.actor_projection_debit_reason_weights = args.actor_projection_debit_reason_weights
    config = build_config(args)
    config.router_boundary_mode = args.router_boundary_mode
    set_current_action_space(config.action_space)
    empty_engine = LLMSingleNodeSimulator([], config)
    observation_builder = DistributionAwareScheduler(config)
    prefill_observation_dim = len(observation_builder.build_prefill_router_observation(empty_engine))
    decode_observation_dim = len(observation_builder.build_decode_router_observation(empty_engine))
    central_observation_dim = (
        prefill_observation_dim
        + decode_observation_dim
        + max(0, int(training_config.history_critic_window)) * HISTORY_CRITIC_FEATURES_PER_STEP
    )
    prefill_training_config, decode_training_config = split_router_training_configs(training_config)
    agent = TorchFactorizedAgent(
        prefill_observation_dim=prefill_observation_dim,
        decode_observation_dim=decode_observation_dim,
        rng_seed=args.seed,
        config=prefill_training_config,
        device=args.device,
        central_observation_dim=central_observation_dim,
    )
    decode_agent = TorchFactorizedAgent(
        prefill_observation_dim=prefill_observation_dim,
        decode_observation_dim=decode_observation_dim,
        rng_seed=args.seed + 9973,
        config=decode_training_config,
        device=args.device,
        central_observation_dim=central_observation_dim,
    )
    if args.init_policy is not None:
        checkpoint = torch.load(args.init_policy, map_location=args.device, weights_only=False)
        policy_payload = checkpoint.get("policy", checkpoint)
        agent.load_checkpoint_payload(policy_payload, strict=False)
        decode_agent.load_checkpoint_payload(
            checkpoint.get("decode_policy", policy_payload) if isinstance(checkpoint, dict) else policy_payload,
            strict=False,
        )
        log_training_progress(
            {
                "event": "init_policy_loaded",
                "path": str(args.init_policy),
                "best_policy_found": bool(checkpoint.get("best_policy_found", False))
                if isinstance(checkpoint, dict)
                else False,
            }
        )

    metrics_rows: List[Dict[str, object]] = []
    best_reward = -float("inf")
    best_trajectory: List[Dict[str, object]] = []
    best_events: List[Dict[str, object]] = []
    projection_corrections: List[Dict[str, object]] = []
    deadlock_cases: List[Dict[str, object]] = []
    best_evaluation: Optional[Dict[str, float]] = None
    best_validation_gate: Optional[Dict[str, object]] = None
    best_policy_checkpoint: Optional[Dict[str, object]] = None
    best_decode_policy_checkpoint: Optional[Dict[str, object]] = None
    init_policy_evaluation: Optional[Dict[str, float | object]] = None
    init_policy_validation_gate: Optional[Dict[str, object]] = None
    first_valid_anchor_triggered = False
    validation_gate_requests = build_validation_gate_request_sets(args)
    validation_gate_total_requests = sum(len(requests) for _, requests in validation_gate_requests)
    drift_audit_rows: List[Dict[str, object]] = []
    drift_probe: Optional[Dict[str, List[List[float]]]] = None
    drift_anchor_hashes: Optional[Dict[str, str]] = None
    drift_anchor_fixed_norm_logits: Optional[np.ndarray] = None
    drift_anchor_raw_current_norm_logits: Optional[np.ndarray] = None
    first_valid_anchor_audit_enabled = bool(
        args.first_valid_anchor_audit or args.post_valid_drift_audit_out is not None
    )

    def finite_round(value: float, digits: int = 12) -> object:
        if not math.isfinite(float(value)):
            return ""
        return round(float(value), digits)

    def append_drift_audit_row(
        *,
        checkpoint_label: str,
        episode_label: object,
        episode_seed: object,
        evaluation: Optional[Dict[str, object]],
        gate: Optional[Dict[str, object]],
        heuristic_fallback_enabled: object,
        training_stage: object,
        eval_kind: str,
        train_safety_projection_fraction: object = "",
        train_gate_stats: Optional[Dict[str, float]] = None,
    ) -> None:
        if args.post_valid_drift_audit_out is None:
            return
        if not first_valid_anchor_triggered:
            return
        current_hashes = drift_audit_state_hashes(agent, decode_agent)
        fixed_norm_logits = drift_probe_logits(
            agent,
            decode_agent,
            drift_probe,
            use_current_normalizer=False,
        )
        raw_current_norm_logits = drift_probe_logits(
            agent,
            decode_agent,
            drift_probe,
            use_current_normalizer=True,
        )
        anchor_hashes = drift_anchor_hashes or {}
        gate_stats = train_gate_stats or {}
        row: Dict[str, object] = {
            "checkpoint": checkpoint_label,
            "eval_kind": eval_kind,
            "episode": episode_label,
            "seed": episode_seed,
            "training_stage": training_stage,
            "first_valid_anchor_active": int(bool(first_valid_anchor_triggered)),
            "first_valid_anchor_audit": int(bool(first_valid_anchor_audit_enabled)),
            "actor_updates_frozen": int(bool(agent.actor_updates_frozen() or decode_agent.actor_updates_frozen())),
            "observation_normalization_enabled": int(
                bool(agent.config.observation_normalization or decode_agent.config.observation_normalization)
            ),
            "observation_normalizers_frozen": int(
                bool(agent.observation_normalizers_frozen() or decode_agent.observation_normalizers_frozen())
            ),
            "heuristic_fallback_enabled": heuristic_fallback_enabled,
            "strong_action_mask": int(not args.disable_strong_action_mask),
            "action_safety_filter": int(not args.disable_action_safety_filter),
            "train_safety_projection_fraction": train_safety_projection_fraction,
            "train_prefill_decision_act_fraction": round(
                float(gate_stats.get("prefill_decision_act_fraction", 0.0)),
                6,
            )
            if gate_stats
            else "",
            "train_decode_decision_act_fraction": round(
                float(gate_stats.get("decode_decision_act_fraction", 0.0)),
                6,
            )
            if gate_stats
            else "",
            "eval_completed": round(float(evaluation.get("completed", 0.0)), 6) if evaluation else "",
            "eval_unfinished_requests": round(float(evaluation.get("unfinished_requests", 0.0)), 6)
            if evaluation
            else "",
            "eval_terminated_early": int(bool(evaluation.get("terminated_early", 0.0))) if evaluation else "",
            "eval_safety_projection_fraction": round(
                float(evaluation.get("safety_projection_fraction", 0.0)),
                6,
            )
            if evaluation
            else "",
            "eval_prefill_decision_act_fraction": round(
                float(evaluation.get("prefill_decision_act_fraction", 0.0)),
                6,
            )
            if evaluation
            else "",
            "eval_decode_decision_act_fraction": round(
                float(evaluation.get("decode_decision_act_fraction", 0.0)),
                6,
            )
            if evaluation
            else "",
            "eval_termination_reason": str(evaluation.get("termination_reason") or "") if evaluation else "",
            "gate_passed": int(bool(gate.get("validation_gate_passed", False))) if gate else "",
            "gate_completed": round(float(gate.get("validation_gate_completed", 0.0)), 6) if gate else "",
            "gate_unfinished_requests": round(
                float(gate.get("validation_gate_unfinished_requests", 0.0)),
                6,
            )
            if gate
            else "",
            "gate_safety_projection_fraction": round(
                float(gate.get("validation_gate_safety_projection_fraction", 0.0)),
                6,
            )
            if gate
            else "",
            "gate_reject_reason": str(gate.get("validation_gate_reject_reason") or "") if gate else "",
            "probe_rows_prefill": len((drift_probe or {}).get("prefill_normalized", [])),
            "probe_rows_decode": len((drift_probe or {}).get("decode_normalized", [])),
            "probe_rows_prefill_raw": len((drift_probe or {}).get("prefill_raw", [])),
            "probe_rows_decode_raw": len((drift_probe or {}).get("decode_raw", [])),
            "probe_fixed_norm_logits_diff": finite_round(
                max_abs_diff(fixed_norm_logits, drift_anchor_fixed_norm_logits)
            ),
            "probe_raw_current_norm_logits_diff": finite_round(
                max_abs_diff(raw_current_norm_logits, drift_anchor_raw_current_norm_logits)
            ),
            "best_policy_found_so_far": int(best_policy_checkpoint is not None),
            "export_policy_source_if_finished": "best_validation_gate"
            if best_policy_checkpoint is not None
            else "final_current",
        }
        for key, value in current_hashes.items():
            row[key] = value
            row[f"{key}_changed"] = int(bool(anchor_hashes) and value != anchor_hashes.get(key))
        row.update(rollout_mask_stats(agent))
        drift_audit_rows.append(row)

    def maybe_activate_first_valid_anchor(
        evaluation: Dict[str, object],
        gate: Optional[Dict[str, object]],
        *,
        episode_label: object,
    ) -> None:
        nonlocal drift_anchor_fixed_norm_logits
        nonlocal drift_anchor_hashes
        nonlocal drift_anchor_raw_current_norm_logits
        nonlocal drift_probe
        nonlocal first_valid_anchor_triggered
        if first_valid_anchor_triggered:
            return
        if not (
            bool(first_valid_anchor_audit_enabled)
            or bool(args.first_valid_freeze_actor)
            or bool(args.first_valid_freeze_observation_normalizers)
            or float(args.anchor_kl_coef) > 0.0
        ):
            return
        if gate is None or not bool(gate.get("validation_gate_passed", False)):
            return
        if float(evaluation.get("completed", 0.0) or 0.0) < float(args.requests):
            return
        agent.set_anchor_policy_from_current()
        decode_agent.set_anchor_policy_from_current()
        if bool(args.first_valid_freeze_actor):
            agent.freeze_actor_updates()
            decode_agent.freeze_actor_updates()
        if bool(args.first_valid_freeze_observation_normalizers):
            agent.freeze_observation_normalizers()
            decode_agent.freeze_observation_normalizers()
        first_valid_anchor_triggered = True
        if args.post_valid_drift_audit_out is not None:
            drift_probe = capture_drift_probe(agent, decode_agent, args.post_valid_drift_probe_size)
            drift_anchor_hashes = drift_audit_state_hashes(agent, decode_agent)
            drift_anchor_fixed_norm_logits = drift_probe_logits(
                agent,
                decode_agent,
                drift_probe,
                use_current_normalizer=False,
            )
            drift_anchor_raw_current_norm_logits = drift_probe_logits(
                agent,
                decode_agent,
                drift_probe,
                use_current_normalizer=True,
            )
            append_drift_audit_row(
                checkpoint_label="first-valid",
                episode_label=episode_label,
                episode_seed="",
                evaluation=evaluation,
                gate=gate,
                heuristic_fallback_enabled="",
                training_stage=agent.training_stage,
                eval_kind="validation_gate",
            )
        log_training_progress(
            {
                "event": "first_valid_anchor_activated",
                "episode": episode_label,
                "anchor_audit": bool(first_valid_anchor_audit_enabled),
                "freeze_actor": bool(args.first_valid_freeze_actor),
                "freeze_observation_normalizers": bool(args.first_valid_freeze_observation_normalizers),
                "anchor_kl_coef": float(args.anchor_kl_coef),
                "completed": round(float(evaluation.get("completed", 0.0) or 0.0), 3),
                "validation_gate_completed": round(float(gate.get("validation_gate_completed", 0.0) or 0.0), 3),
            }
        )
    projection_dual = float(
        args.projection_lagrangian_init
        if args.projection_lagrangian_init is not None
        else reward_config.action_safety_projection_penalty
    )
    liveness_duals = initial_lagrangian_duals(args, reward_config)
    if projection_lagrangian_enabled(args):
        projection_dual = update_projection_lagrangian_dual(
            projection_dual,
            float(args.projection_lagrangian_target),
            args,
        )
        liveness_duals["unsafe_projection_cost"] = projection_dual
    reward_config = reward_config_with_lagrangian_duals(reward_config, liveness_duals)

    if args.init_policy is not None:
        init_evaluation_seed = args.seed + 10000
        init_evaluation_reward_config = scheduled_reward_config(
            args,
            replace(
                reward_config,
                action_safety_projection_penalty=projection_dual,
            ),
            args.episodes - 1,
        )
        log_training_progress(
            {
                "event": "init_policy_eval_start",
                "seed": init_evaluation_seed,
                "step_cap": int(args.eval_step_cap or 0),
                "wall_time_limit_s": float(args.eval_wall_time_limit_s or 0.0),
            }
        )
        init_evaluation_requests = build_evaluation_request_sets(args, init_evaluation_seed)
        init_eval_started_at = time.monotonic()
        init_evaluation, _, _ = evaluate_policy_suite_with_capture(
            agent,
            config,
            init_evaluation_reward_config,
            init_evaluation_requests,
            capture_trajectory=False,
            capture_sparse_events=False,
            max_steps=args.eval_step_cap,
            progress_every=args.eval_progress_every,
            progress_label=f"eval:init:seed{init_evaluation_seed}",
            wall_time_limit_s=args.eval_wall_time_limit_s,
            decode_agent=decode_agent,
        )
        init_policy_evaluation = init_evaluation
        log_training_progress(
            {
                "event": "init_policy_eval_done",
                "termination": init_evaluation.get("termination_reason"),
                "completed": round(float(init_evaluation.get("completed", 0.0)), 3),
                "unfinished": round(float(init_evaluation.get("unfinished_requests", 0.0)), 3),
                "steps": round(float(init_evaluation.get("simulation_steps", 0.0)), 3),
                "sim_wall_s": round(float(init_evaluation.get("wall_time_seconds", 0.0)), 3),
                "outer_wall_s": round(time.monotonic() - init_eval_started_at, 3),
                "reward": round(float(init_evaluation.get("reward", 0.0)), 6),
            }
        )
        init_gate: Optional[Dict[str, object]] = None
        init_can_replace = False
        if validation_gate_requests:
            log_training_progress(
                {
                    "event": "init_policy_validation_gate_start",
                    "traces": len(validation_gate_requests),
                    "requests": validation_gate_total_requests,
                }
            )
            init_gate_started_at = time.monotonic()
            init_gate_evaluation, _, _ = evaluate_policy_suite_with_capture(
                agent,
                config,
                init_evaluation_reward_config,
                validation_gate_requests,
                capture_trajectory=False,
                capture_sparse_events=False,
                max_steps=args.eval_step_cap,
                progress_every=args.eval_progress_every,
                progress_label=f"validation-gate:init:seed{args.validation_gate_seed}",
                wall_time_limit_s=args.eval_wall_time_limit_s,
                decode_agent=decode_agent,
            )
            init_gate = validation_gate_payload(init_gate_evaluation, validation_gate_total_requests, args)
            init_policy_validation_gate = init_gate
            log_training_progress(
                {
                    "event": "init_policy_validation_gate_done",
                    "passed": bool(init_gate["validation_gate_passed"]),
                    "completed": round(float(init_gate["validation_gate_completed"]), 3),
                    "unfinished": round(float(init_gate["validation_gate_unfinished_requests"]), 3),
                    "completed_fraction": round(float(init_gate["validation_gate_completed_fraction"]), 6),
                    "terminated_early_traces": round(
                        float(init_gate["validation_gate_terminated_early_trace_count"]),
                        3,
                    ),
                    "reject_reason": init_gate["validation_gate_reject_reason"],
                    "outer_wall_s": round(time.monotonic() - init_gate_started_at, 3),
                }
            )
            maybe_activate_first_valid_anchor(init_evaluation, init_gate, episode_label="init")
            init_can_replace = should_replace_best_validation_ranked_evaluation(
                init_evaluation,
                init_gate,
                best_evaluation,
                best_validation_gate,
                prefer_low_safety_projection=args.prefer_low_safety_projection_best,
                max_safety_projection=args.max_safety_projection_best,
                best_selection_objective=args.best_selection_objective,
                validation_margin_ttft_p95_target=args.validation_margin_ttft_p95_target,
                validation_margin_e2e_p95_target=args.validation_margin_e2e_p95_target,
                validation_margin_ttft_p95_penalty=args.validation_margin_ttft_p95_penalty,
                validation_margin_e2e_p95_penalty=args.validation_margin_e2e_p95_penalty,
            )
        else:
            init_can_replace = should_replace_best_evaluation(
                init_evaluation,
                best_evaluation,
                prefer_low_safety_projection=args.prefer_low_safety_projection_best,
                max_safety_projection=args.max_safety_projection_best,
                best_selection_objective=args.best_selection_objective,
            )
        if init_can_replace:
            if args.trajectory_out or args.event_out:
                best_evaluation, best_trajectory, best_events = evaluate_policy_suite_with_capture(
                    agent,
                    config,
                    init_evaluation_reward_config,
                    init_evaluation_requests,
                    capture_trajectory=bool(args.trajectory_out),
                    capture_sparse_events=bool(args.event_out),
                    max_steps=args.eval_step_cap,
                    progress_every=args.eval_progress_every,
                    progress_label=f"best-eval:init:seed{init_evaluation_seed}",
                    wall_time_limit_s=args.eval_wall_time_limit_s,
                    decode_agent=decode_agent,
                )
            else:
                best_evaluation = init_evaluation
                best_trajectory = []
                best_events = []
            best_policy_checkpoint = agent.checkpoint()
            best_decode_policy_checkpoint = decode_agent.checkpoint()
            best_validation_gate = init_gate
            best_reward = float(best_evaluation["reward"])
            log_training_progress(
                {
                    "event": "best_policy_updated",
                    "source": "init_policy",
                    "reward": round(best_reward, 6),
                    "ttft_p95": round(float(best_evaluation.get("ttft_p95", 0.0)), 6),
                    "safety_projection_fraction": round(
                        float(best_evaluation.get("safety_projection_fraction", 0.0)),
                        6,
                    ),
                }
            )

    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        training_phase = curriculum_phase(args, episode)
        projection_dual_before = projection_dual
        liveness_duals_before = dict(liveness_duals)
        if projection_lagrangian_enabled(args):
            liveness_duals_before["unsafe_projection_cost"] = projection_dual_before
        episode_base_reward_config = reward_config_with_lagrangian_duals(reward_config, liveness_duals_before)
        episode_reward_config = scheduled_reward_config(args, episode_base_reward_config, episode)
        episode_projection_penalty = float(episode_reward_config.action_safety_projection_penalty)
        heuristic_fallback_enabled = allow_unsafe_state_heuristic_fallback(args, episode)
        update_control = agent.configure_update(episode / max(args.episodes, 1))
        decode_update_control = decode_agent.configure_update(episode / max(args.episodes, 1))
        training_source = select_curriculum_source(args, episode, episode_seed)
        requests = build_requests(args, episode_seed, episode_index=episode)
        train_label = f"train:e{episode}:seed{episode_seed}:{training_phase}:{curriculum_source_label(training_source)}"
        log_training_progress(
            {
                "event": "episode_start",
                "episode": episode,
                "seed": episode_seed,
                "curriculum_phase": training_phase,
                "trace": curriculum_source_label(training_source),
                "requests": len(requests),
                "action_safety_projection_penalty": episode_projection_penalty,
                "projection_lagrangian_dual": projection_dual_before,
                "liveness_lagrangian_duals": json.dumps(liveness_duals_before, ensure_ascii=True),
                "training_stage": update_control["training_stage"],
                "decode_training_stage": decode_update_control["training_stage"],
                "heuristic_fallback_enabled": heuristic_fallback_enabled,
                "step_cap": int(args.episode_step_cap or 0),
                "wall_time_limit_s": float(args.episode_wall_time_limit_s or 0.0),
            }
        )
        scheduler = RLDistributionAwareScheduler(
            config,
            agent,
            episode_reward_config,
            decode_agent=decode_agent,
            greedy=False,
            feedback_mode=agent.config.feedback_mode,
            allow_heuristic_fallback=heuristic_fallback_enabled,
            deadlock_case_window=args.deadlock_case_window,
        )
        simulator = LLMSingleNodeSimulator(requests, config, scheduler=scheduler)
        episode_started_at = time.monotonic()
        summary = simulator.run(
            max_steps=args.episode_step_cap,
            progress_every=args.episode_progress_every,
            progress_label=train_label,
            wall_time_limit_s=args.episode_wall_time_limit_s,
        )
        for correction in scheduler.projection_corrections:
            projection_corrections.append(
                {
                    "episode": episode,
                    "seed": episode_seed,
                    "curriculum_phase": training_phase,
                    "train_trace": curriculum_source_label(training_source),
                    **correction,
                }
            )
        episode_deadlock_case = scheduler.build_deadlock_case(
            summary,
            episode=episode,
            seed=episode_seed,
            train_trace=curriculum_source_label(training_source),
            curriculum_phase=training_phase,
        )
        if episode_deadlock_case is not None:
            deadlock_cases.append(episode_deadlock_case)
        episode_deadlock_classes = (
            "|".join(str(code) for code in episode_deadlock_case.get("classification", []))
            if episode_deadlock_case is not None
            else ""
        )
        episode_deadlock_labels = (
            "|".join(str(label) for label in episode_deadlock_case.get("classification_labels", []))
            if episode_deadlock_case is not None
            else ""
        )
        log_training_progress(
            {
                "event": "episode_sim_done",
                "episode": episode,
                "termination": summary.get("termination_reason"),
                "completed": summary.get("completed"),
                "unfinished": summary.get("unfinished_requests"),
                "steps": summary.get("simulation_steps"),
                "sim_wall_s": summary.get("wall_time_seconds"),
                "outer_wall_s": round(time.monotonic() - episode_started_at, 3),
                "runtime_ms": summary.get("total_runtime_ms"),
                "peak_kv": summary.get("peak_kv_tokens"),
            }
        )
        terminal_reward = scheduler.apply_terminal_objective(summary)
        total_reward = float(sum(agent.rewards))
        update_control = agent.configure_update((episode + 1) / max(args.episodes, 1))
        decode_agent.configure_update((episode + 1) / max(args.episodes, 1))
        update_stats = agent.finish_episode()
        decode_update_stats = decode_agent.finish_episode()
        update_stats = {
            **update_stats,
            "policy_loss": float(update_stats["prefill_policy_loss"] + decode_update_stats["decode_policy_loss"]),
            "value_loss": float(update_stats["prefill_value_loss"] + decode_update_stats["decode_value_loss"]),
            "mean_return": float(
                0.5 * (update_stats["prefill_mean_return"] + decode_update_stats["decode_mean_return"])
            ),
            "decode_policy_loss": float(decode_update_stats["decode_policy_loss"]),
            "decode_value_loss": float(decode_update_stats["decode_value_loss"]),
            "decode_mean_return": float(decode_update_stats["decode_mean_return"]),
            "anchor_kl_loss": float(update_stats["anchor_kl_loss"] + decode_update_stats["anchor_kl_loss"]),
        }
        train_safety_projection_fraction = scheduler.safety_projection_fraction()
        train_gate_stats = scheduler.decision_gate_fractions()
        train_diagnostic_metrics = scheduler.diagnostic_metrics()
        evaluation_base_reward_config = reward_config_with_lagrangian_duals(reward_config, liveness_duals_before)
        evaluation_reward_config = scheduled_reward_config(args, evaluation_base_reward_config, args.episodes - 1)

        eval_reward = ""
        eval_ttft_mean = ""
        eval_ttft_p95 = ""
        eval_ttft_p99 = ""
        eval_ttft_cvar95 = ""
        eval_e2e_p95 = ""
        eval_e2e_p99 = ""
        eval_admission_wait_p95 = ""
        eval_admission_wait_cvar95 = ""
        eval_peak_kv_tokens = ""
        eval_peak_kv_util = ""
        eval_peak_allocator_internal_fragmentation_tokens = ""
        eval_peak_allocator_external_fragmentation_tokens = ""
        eval_safety_projection_fraction = ""
        eval_prefill_decision_act_fraction = ""
        eval_decode_decision_act_fraction = ""
        eval_prefill_decision_checks = ""
        eval_decode_decision_checks = ""
        eval_completed = ""
        eval_terminated_early = ""
        eval_unfinished_requests = ""
        eval_horizon_unfinished_requests = ""
        eval_aged_unfinished_requests = ""
        eval_liveness_unfinished_requests = ""
        eval_punishable_unfinished_requests = ""
        eval_termination_reason = ""
        eval_terminated_early_trace_count = ""
        validation_gate_passed = ""
        validation_gate_completed = ""
        validation_gate_total = ""
        validation_gate_completed_fraction = ""
        validation_gate_unfinished_requests = ""
        validation_gate_horizon_unfinished_requests = ""
        validation_gate_aged_unfinished_requests = ""
        validation_gate_liveness_unfinished_requests = ""
        validation_gate_punishable_unfinished_requests = ""
        validation_gate_terminated_early_trace_count = ""
        validation_gate_safety_projection_fraction = ""
        validation_gate_ttft_p95 = ""
        validation_gate_ttft_p99 = ""
        validation_gate_e2e_p95 = ""
        validation_gate_termination_reason = ""
        validation_gate_reject_reason = ""
        if should_run_policy_evaluation(episode, args.episodes, args.eval_every):
            log_training_progress(
                {
                    "event": "eval_start",
                    "episode": episode,
                    "seed": episode_seed + 10000,
                    "step_cap": int(args.eval_step_cap or 0),
                    "wall_time_limit_s": float(args.eval_wall_time_limit_s or 0.0),
                }
            )
            evaluation_requests = build_evaluation_request_sets(args, episode_seed + 10000)
            eval_started_at = time.monotonic()
            evaluation, _, _ = evaluate_policy_suite_with_capture(
                agent,
                config,
                evaluation_reward_config,
                evaluation_requests,
                capture_trajectory=False,
                capture_sparse_events=False,
                max_steps=args.eval_step_cap,
                progress_every=args.eval_progress_every,
                progress_label=f"eval:e{episode}:seed{episode_seed + 10000}",
                wall_time_limit_s=args.eval_wall_time_limit_s,
                decode_agent=decode_agent,
            )
            log_training_progress(
                {
                    "event": "eval_done",
                    "episode": episode,
                    "termination": evaluation.get("termination_reason"),
                    "completed": round(float(evaluation.get("completed", 0.0)), 3),
                    "unfinished": round(float(evaluation.get("unfinished_requests", 0.0)), 3),
                    "steps": round(float(evaluation.get("simulation_steps", 0.0)), 3),
                    "sim_wall_s": round(float(evaluation.get("wall_time_seconds", 0.0)), 3),
                    "outer_wall_s": round(time.monotonic() - eval_started_at, 3),
                    "reward": round(float(evaluation.get("reward", 0.0)), 6),
                    "prefill_act": round(float(evaluation.get("prefill_decision_act_fraction", 0.0)), 6),
                    "decode_act": round(float(evaluation.get("decode_decision_act_fraction", 0.0)), 6),
                }
            )
            eval_reward = round(evaluation["reward"], 6)
            eval_ttft_mean = round(evaluation["ttft_mean"], 6)
            eval_ttft_p95 = round(evaluation["ttft_p95"], 6)
            eval_ttft_p99 = round(evaluation["ttft_p99"], 6)
            eval_ttft_cvar95 = round(evaluation["ttft_cvar95"], 6)
            eval_e2e_p95 = round(evaluation["e2e_p95"], 6)
            eval_e2e_p99 = round(evaluation["e2e_p99"], 6)
            eval_admission_wait_p95 = round(evaluation["admission_wait_p95"], 6)
            eval_admission_wait_cvar95 = round(evaluation["admission_wait_cvar95"], 6)
            eval_peak_kv_tokens = round(evaluation["peak_kv_tokens"], 6)
            eval_peak_kv_util = round(evaluation["peak_kv_tokens"] / max(config.kv_capacity_tokens, 1), 6)
            eval_peak_allocator_internal_fragmentation_tokens = round(
                evaluation["peak_allocator_internal_fragmentation_tokens"],
                6,
            )
            eval_peak_allocator_external_fragmentation_tokens = round(
                evaluation["peak_allocator_external_fragmentation_tokens"],
                6,
            )
            eval_safety_projection_fraction = round(evaluation["safety_projection_fraction"], 6)
            eval_prefill_decision_act_fraction = round(evaluation["prefill_decision_act_fraction"], 6)
            eval_decode_decision_act_fraction = round(evaluation["decode_decision_act_fraction"], 6)
            eval_prefill_decision_checks = round(evaluation["prefill_decision_checks"], 6)
            eval_decode_decision_checks = round(evaluation["decode_decision_checks"], 6)
            eval_completed = round(evaluation["completed"], 6)
            eval_terminated_early = int(bool(evaluation["terminated_early"]))
            eval_terminated_early_trace_count = round(float(evaluation["terminated_early_trace_count"]), 6)
            eval_unfinished_requests = round(evaluation["unfinished_requests"], 6)
            eval_horizon_unfinished_requests = round(evaluation["horizon_unfinished_requests"], 6)
            eval_aged_unfinished_requests = round(evaluation["aged_unfinished_requests"], 6)
            eval_liveness_unfinished_requests = round(evaluation["liveness_unfinished_requests"], 6)
            eval_punishable_unfinished_requests = round(evaluation["punishable_unfinished_requests"], 6)
            eval_termination_reason = evaluation["termination_reason"]
            candidate_gate: Optional[Dict[str, object]] = None
            if validation_gate_requests:
                log_training_progress(
                    {
                        "event": "validation_gate_start",
                        "episode": episode,
                        "traces": len(validation_gate_requests),
                        "requests": validation_gate_total_requests,
                    }
                )
                gate_started_at = time.monotonic()
                gate_evaluation, _, _ = evaluate_policy_suite_with_capture(
                    agent,
                    config,
                    evaluation_reward_config,
                    validation_gate_requests,
                    capture_trajectory=False,
                    capture_sparse_events=False,
                    max_steps=args.eval_step_cap,
                    progress_every=args.eval_progress_every,
                    progress_label=f"validation-gate:e{episode}:seed{args.validation_gate_seed}",
                    wall_time_limit_s=args.eval_wall_time_limit_s,
                    decode_agent=decode_agent,
                )
                candidate_gate = validation_gate_payload(gate_evaluation, validation_gate_total_requests, args)
                log_training_progress(
                    {
                        "event": "validation_gate_done",
                        "episode": episode,
                        "passed": bool(candidate_gate["validation_gate_passed"]),
                        "completed": round(float(candidate_gate["validation_gate_completed"]), 3),
                        "unfinished": round(float(candidate_gate["validation_gate_unfinished_requests"]), 3),
                        "completed_fraction": round(float(candidate_gate["validation_gate_completed_fraction"]), 6),
                        "terminated_early_traces": round(
                            float(candidate_gate["validation_gate_terminated_early_trace_count"]),
                            3,
                        ),
                        "reject_reason": candidate_gate["validation_gate_reject_reason"],
                        "outer_wall_s": round(time.monotonic() - gate_started_at, 3),
                    }
                )
                validation_gate_passed = int(bool(candidate_gate["validation_gate_passed"]))
                validation_gate_completed = round(float(candidate_gate["validation_gate_completed"]), 6)
                validation_gate_total = int(candidate_gate["validation_gate_total_requests"])
                validation_gate_completed_fraction = round(
                    float(candidate_gate["validation_gate_completed_fraction"]),
                    6,
                )
                validation_gate_unfinished_requests = round(
                    float(candidate_gate["validation_gate_unfinished_requests"]),
                    6,
                )
                validation_gate_horizon_unfinished_requests = round(
                    float(candidate_gate["validation_gate_horizon_unfinished_requests"]),
                    6,
                )
                validation_gate_aged_unfinished_requests = round(
                    float(candidate_gate["validation_gate_aged_unfinished_requests"]),
                    6,
                )
                validation_gate_liveness_unfinished_requests = round(
                    float(candidate_gate["validation_gate_liveness_unfinished_requests"]),
                    6,
                )
                validation_gate_punishable_unfinished_requests = round(
                    float(candidate_gate["validation_gate_punishable_unfinished_requests"]),
                    6,
                )
                validation_gate_terminated_early_trace_count = round(
                    float(candidate_gate["validation_gate_terminated_early_trace_count"]),
                    6,
                )
                validation_gate_safety_projection_fraction = round(
                    float(candidate_gate["validation_gate_safety_projection_fraction"]),
                    6,
                )
                validation_gate_ttft_p95 = round(float(candidate_gate["validation_gate_ttft_p95"]), 6)
                validation_gate_ttft_p99 = round(float(candidate_gate["validation_gate_ttft_p99"]), 6)
                validation_gate_e2e_p95 = round(float(candidate_gate["validation_gate_e2e_p95"]), 6)
                validation_gate_termination_reason = candidate_gate["validation_gate_termination_reason"]
                validation_gate_reject_reason = candidate_gate["validation_gate_reject_reason"]
                drift_audit_was_active = first_valid_anchor_triggered
                maybe_activate_first_valid_anchor(evaluation, candidate_gate, episode_label=episode)
                if first_valid_anchor_triggered and (drift_audit_was_active or episode == args.episodes - 1):
                    append_drift_audit_row(
                        checkpoint_label="final" if episode == args.episodes - 1 else "post-valid eval",
                        episode_label=episode,
                        episode_seed=episode_seed,
                        evaluation=evaluation,
                        gate=candidate_gate,
                        heuristic_fallback_enabled=heuristic_fallback_enabled,
                        training_stage=update_control["training_stage"],
                        eval_kind="validation_gate",
                        train_safety_projection_fraction=round(train_safety_projection_fraction, 6),
                        train_gate_stats=train_gate_stats,
                    )
                candidate_can_replace = should_replace_best_validation_ranked_evaluation(
                    evaluation,
                    candidate_gate,
                    best_evaluation,
                    best_validation_gate,
                    prefer_low_safety_projection=args.prefer_low_safety_projection_best,
                    max_safety_projection=args.max_safety_projection_best,
                    best_selection_objective=args.best_selection_objective,
                    validation_margin_ttft_p95_target=args.validation_margin_ttft_p95_target,
                    validation_margin_e2e_p95_target=args.validation_margin_e2e_p95_target,
                    validation_margin_ttft_p95_penalty=args.validation_margin_ttft_p95_penalty,
                    validation_margin_e2e_p95_penalty=args.validation_margin_e2e_p95_penalty,
                )
            else:
                candidate_can_replace = should_replace_best_evaluation(
                    evaluation,
                    best_evaluation,
                    prefer_low_safety_projection=args.prefer_low_safety_projection_best,
                    max_safety_projection=args.max_safety_projection_best,
                    best_selection_objective=args.best_selection_objective,
                )
                validation_gate_passed = 1
            if candidate_can_replace:
                if args.trajectory_out or args.event_out:
                    best_evaluation, best_trajectory, best_events = evaluate_policy_suite_with_capture(
                        agent,
                        config,
                        evaluation_reward_config,
                        evaluation_requests,
                        capture_trajectory=bool(args.trajectory_out),
                        capture_sparse_events=bool(args.event_out),
                        max_steps=args.eval_step_cap,
                        progress_every=args.eval_progress_every,
                        progress_label=f"best-eval:e{episode}:seed{episode_seed + 10000}",
                        wall_time_limit_s=args.eval_wall_time_limit_s,
                        decode_agent=decode_agent,
                    )
                else:
                    best_evaluation = evaluation
                    best_trajectory = []
                    best_events = []
                best_policy_checkpoint = agent.checkpoint()
                best_decode_policy_checkpoint = decode_agent.checkpoint()
                best_validation_gate = candidate_gate
                best_reward = float(best_evaluation["reward"])
        projection_dual = update_projection_lagrangian_dual(
            projection_dual_before,
            train_safety_projection_fraction,
            args,
        )
        observed_liveness_costs = scheduler.normalized_costs()
        if projection_lagrangian_enabled(args):
            observed_liveness_costs["unsafe_projection_cost"] = train_safety_projection_fraction
        liveness_duals = update_lagrangian_duals(
            liveness_duals_before,
            observed_liveness_costs,
            args,
        )
        if projection_lagrangian_enabled(args):
            liveness_duals["unsafe_projection_cost"] = projection_dual

        metrics_rows.append(
            {
                "episode": episode,
                "seed": episode_seed,
                "curriculum_phase": training_phase,
                "action_safety_projection_penalty": round(episode_projection_penalty, 6),
                "projection_lagrangian_dual": round(projection_dual_before, 6),
                "projection_lagrangian_dual_next": round(projection_dual, 6),
                "liveness_lagrangian_duals": json.dumps(liveness_duals_before, ensure_ascii=True),
                "liveness_lagrangian_duals_next": json.dumps(liveness_duals, ensure_ascii=True),
                **{name: round(float(observed_liveness_costs.get(name, 0.0)), 6) for name in LIVENESS_COST_NAMES},
                "heuristic_fallback_enabled": heuristic_fallback_enabled,
                "strong_action_mask": int(not args.disable_strong_action_mask),
                "action_safety_filter": int(not args.disable_action_safety_filter),
                "training_stage": update_control["training_stage"],
                "late_update_active": int(bool(update_control["late_update_active"])),
                "optimizer_lr": round(float(update_control["optimizer_lr"]), 10),
                "late_freeze_prefill_trunk": int(bool(update_control["late_freeze_prefill_trunk"])),
                "late_freeze_admission_heads": int(bool(update_control["late_freeze_admission_heads"])),
                "first_valid_anchor_active": int(bool(first_valid_anchor_triggered)),
                "actor_updates_frozen": int(bool(agent.actor_updates_frozen() or decode_agent.actor_updates_frozen())),
                "train_trace": curriculum_source_label(training_source),
                "reward": round(total_reward, 6),
                "executed_reward": round(float(scheduler.executed_reward_sum), 6),
                "actor_projection_debit": round(float(scheduler.actor_projection_debit_sum), 6),
                "actor_reward": round(total_reward, 6),
                "prefill_reward": round(float(sum(agent.prefill_rewards)), 6),
                "decode_reward": round(float(sum(decode_agent.decode_rewards)), 6),
                "terminal_reward": round(terminal_reward, 6),
                **{
                    name: round(float(scheduler.reward_component_sums.get(name, 0.0)), 6)
                    for name in REWARD_COMPONENT_NAMES
                },
                "policy_loss": round(update_stats["policy_loss"], 6),
                "prefill_policy_loss": round(update_stats["prefill_policy_loss"], 6),
                "decode_policy_loss": round(update_stats["decode_policy_loss"], 6),
                "value_loss": round(update_stats["value_loss"], 6),
                "prefill_value_loss": round(update_stats["prefill_value_loss"], 6),
                "decode_value_loss": round(update_stats["decode_value_loss"], 6),
                "projection_imitation_loss": round(update_stats["projection_imitation_loss"], 6),
                "targeted_projection_bc_loss": round(update_stats["targeted_projection_bc_loss"], 6),
                "anchor_kl_loss": round(update_stats["anchor_kl_loss"], 6),
                "targeted_projection_bc_rows": round(update_stats["targeted_projection_bc_rows"], 6),
                "targeted_projection_bc_heads": round(update_stats["targeted_projection_bc_heads"], 6),
                "targeted_projection_bc_groups": round(update_stats["targeted_projection_bc_groups"], 6),
                "mean_return": round(update_stats["mean_return"], 6),
                "prefill_mean_return": round(update_stats["prefill_mean_return"], 6),
                "decode_mean_return": round(update_stats["decode_mean_return"], 6),
                "ttft_p95": round(summary["ttft_ms"]["p95"] or 0.0, 6),
                "ttft_p99": round(summary["ttft_ms"]["p99"] or 0.0, 6),
                "ttft_cvar95": round(summary["ttft_ms"]["cvar95"] or 0.0, 6),
                "e2e_p95": round(summary["e2e_ms"]["p95"] or 0.0, 6),
                "e2e_p99": round(summary["e2e_ms"]["p99"] or 0.0, 6),
                "admission_wait_p95": round(summary["admission_wait_ms"]["p95"] or 0.0, 6),
                "admission_wait_cvar95": round(summary["admission_wait_ms"]["cvar95"] or 0.0, 6),
                "safety_projection_fraction": round(train_safety_projection_fraction, 6),
                "prefill_decision_act_fraction": round(
                    float(train_gate_stats["prefill_decision_act_fraction"]),
                    6,
                ),
                "decode_decision_act_fraction": round(
                    float(train_gate_stats["decode_decision_act_fraction"]),
                    6,
                ),
                "prefill_decision_checks": round(float(train_gate_stats["prefill_decision_checks"]), 6),
                "decode_decision_checks": round(float(train_gate_stats["decode_decision_checks"]), 6),
                "terminated_early": summary["terminated_early"],
                "completed_all_requests_rate": 1.0
                if str(summary.get("termination_reason") or "") == "completed_all_requests"
                else 0.0,
                "deadlock_case_rows": 1 if episode_deadlock_case is not None else 0,
                "deadlock_case_classification": episode_deadlock_classes,
                "deadlock_case_labels": episode_deadlock_labels,
                **{
                    f"deadlock_case_classification_{code}": 1
                    if code in set(str(part) for part in episode_deadlock_classes.split("|") if part)
                    else 0
                    for code in ("A", "B", "C", "D", "E", "F", "G")
                },
                **train_diagnostic_metrics,
                "unfinished_requests": summary["unfinished_requests"],
                "unfinished_request_ids": json.dumps(summary.get("unfinished_request_ids", []), ensure_ascii=True),
                "horizon_unfinished_requests": summary.get("horizon_unfinished_requests", 0),
                "aged_unfinished_requests": summary.get("aged_unfinished_requests", 0),
                "liveness_unfinished_requests": summary.get("liveness_unfinished_requests", 0),
                "punishable_unfinished_requests": summary.get("punishable_unfinished_requests", 0),
                "termination_reason": summary["termination_reason"],
                "termination_details": json.dumps(summary.get("termination_details", {}), ensure_ascii=True),
                "evictions_total": summary["evictions_total"],
                "recompute_tokens_total": summary["recompute_tokens_total"],
                "deadlock_recoveries_total": summary["deadlock_recoveries_total"],
                "simulation_steps": summary.get("simulation_steps", 0),
                "wall_time_seconds": summary.get("wall_time_seconds", 0.0),
                "peak_allocator_external_fragmentation_tokens": round(
                    summary["peak_allocator_external_fragmentation_tokens"],
                    6,
                ),
                "peak_allocator_internal_fragmentation_tokens": round(
                    summary["peak_allocator_internal_fragmentation_tokens"],
                    6,
                ),
                "peak_kv_tokens": summary["peak_kv_tokens"],
                "peak_kv_util": round(summary["peak_kv_tokens"] / max(config.kv_capacity_tokens, 1), 6),
                "eval_reward": eval_reward,
                "eval_executed_reward": round(evaluation["executed_reward"], 6) if eval_reward != "" else "",
                "eval_actor_projection_debit": round(evaluation["actor_projection_debit"], 6) if eval_reward != "" else "",
                "eval_actor_reward": round(evaluation["actor_reward"], 6) if eval_reward != "" else "",
                "eval_terminal_reward": round(evaluation["terminal_reward"], 6) if eval_reward != "" else "",
                "eval_prefill_reward": round(evaluation["prefill_reward"], 6) if eval_reward != "" else "",
                "eval_decode_reward": round(evaluation["decode_reward"], 6) if eval_reward != "" else "",
                **{
                    f"eval_{name}": round(float(evaluation.get(name, 0.0)), 6)
                    if eval_reward != ""
                    else ""
                    for name in REWARD_COMPONENT_NAMES
                },
                "eval_completed": eval_completed,
                "eval_terminated_early": eval_terminated_early,
                "eval_terminated_early_trace_count": eval_terminated_early_trace_count,
                "eval_unfinished_requests": eval_unfinished_requests,
                "eval_horizon_unfinished_requests": eval_horizon_unfinished_requests,
                "eval_aged_unfinished_requests": eval_aged_unfinished_requests,
                "eval_liveness_unfinished_requests": eval_liveness_unfinished_requests,
                "eval_punishable_unfinished_requests": eval_punishable_unfinished_requests,
                "eval_termination_reason": eval_termination_reason,
                "eval_simulation_steps": round(evaluation["simulation_steps"], 6) if eval_reward != "" else "",
                "eval_wall_time_seconds": round(evaluation["wall_time_seconds"], 6) if eval_reward != "" else "",
                "eval_ttft_mean": eval_ttft_mean,
                "eval_ttft_p95": eval_ttft_p95,
                "eval_ttft_p99": eval_ttft_p99,
                "eval_ttft_cvar95": eval_ttft_cvar95,
                "eval_e2e_p95": eval_e2e_p95,
                "eval_e2e_p99": eval_e2e_p99,
                "eval_admission_wait_p95": eval_admission_wait_p95,
                "eval_admission_wait_cvar95": eval_admission_wait_cvar95,
                "eval_peak_kv_tokens": eval_peak_kv_tokens,
                "eval_peak_kv_util": eval_peak_kv_util,
                "eval_peak_allocator_internal_fragmentation_tokens": (
                    eval_peak_allocator_internal_fragmentation_tokens
                ),
                "eval_peak_allocator_external_fragmentation_tokens": (
                    eval_peak_allocator_external_fragmentation_tokens
                ),
                "eval_safety_projection_fraction": eval_safety_projection_fraction,
                "eval_prefill_decision_act_fraction": eval_prefill_decision_act_fraction,
                "eval_decode_decision_act_fraction": eval_decode_decision_act_fraction,
                "eval_prefill_decision_checks": eval_prefill_decision_checks,
                "eval_decode_decision_checks": eval_decode_decision_checks,
                "validation_gate_passed": validation_gate_passed,
                "validation_gate_completed": validation_gate_completed,
                "validation_gate_total_requests": validation_gate_total,
                "validation_gate_completed_fraction": validation_gate_completed_fraction,
                "validation_gate_unfinished_requests": validation_gate_unfinished_requests,
                "validation_gate_horizon_unfinished_requests": validation_gate_horizon_unfinished_requests,
                "validation_gate_aged_unfinished_requests": validation_gate_aged_unfinished_requests,
                "validation_gate_liveness_unfinished_requests": validation_gate_liveness_unfinished_requests,
                "validation_gate_punishable_unfinished_requests": validation_gate_punishable_unfinished_requests,
                "validation_gate_terminated_early_trace_count": validation_gate_terminated_early_trace_count,
                "validation_gate_safety_projection_fraction": validation_gate_safety_projection_fraction,
                "validation_gate_ttft_p95": validation_gate_ttft_p95,
                "validation_gate_ttft_p99": validation_gate_ttft_p99,
                "validation_gate_e2e_p95": validation_gate_e2e_p95,
                "validation_gate_termination_reason": validation_gate_termination_reason,
                "validation_gate_reject_reason": validation_gate_reject_reason,
            }
        )
        if args.write_metrics_every > 0 and ((episode + 1) % args.write_metrics_every == 0 or episode == args.episodes - 1):
            write_metrics(args.metrics_out, metrics_rows)
        agent.reset_episode()
        decode_agent.reset_episode()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_prefill_policy = best_policy_checkpoint or agent.checkpoint()
    final_decode_policy = best_decode_policy_checkpoint or decode_agent.checkpoint()
    final_combined_policy = combined_router_policy_checkpoint(final_prefill_policy, final_decode_policy)
    checkpoint = {
        "teacher_type": "torch-factorized-actor-critic-split-router-reward",
        "policy": final_combined_policy,
        "prefill_policy": final_prefill_policy,
        "decode_policy": final_decode_policy,
        "best_evaluation": best_evaluation,
        "best_validation_gate": best_validation_gate,
        "best_policy_found": best_policy_checkpoint is not None,
        "init_policy": str(args.init_policy) if args.init_policy is not None else None,
        "curriculum_traces": [str(path) for path in curriculum_trace_paths(args)],
        "curriculum_sources": [curriculum_source_label(source) for source in curriculum_sources(args)],
        "curriculum_schedule": args.curriculum_schedule,
        "prefer_low_safety_projection_best": bool(args.prefer_low_safety_projection_best),
        "best_selection_objective": args.best_selection_objective,
        "max_safety_projection_best": args.max_safety_projection_best,
        "validation_gate_traces": [str(path) for path, _ in validation_gate_requests],
        "validation_gate_seed": int(args.validation_gate_seed),
        "validation_gate_min_completed_fraction": float(args.validation_gate_min_completed_fraction),
        "validation_gate_max_unfinished": int(args.validation_gate_max_unfinished),
        "validation_gate_max_safety_projection": args.validation_gate_max_safety_projection,
        "validation_gate_max_ttft_p95": args.validation_gate_max_ttft_p95,
        "validation_gate_max_ttft_p99": args.validation_gate_max_ttft_p99,
        "validation_gate_max_e2e_p95": args.validation_gate_max_e2e_p95,
        "validation_margin_ttft_p95_target": args.validation_margin_ttft_p95_target,
        "validation_margin_e2e_p95_target": args.validation_margin_e2e_p95_target,
        "validation_margin_ttft_p95_penalty": args.validation_margin_ttft_p95_penalty,
        "validation_margin_e2e_p95_penalty": args.validation_margin_e2e_p95_penalty,
        "validation_gate_allow_early_stop": bool(args.validation_gate_allow_early_stop),
        "projection_lagrangian_target": args.projection_lagrangian_target,
        "projection_lagrangian_lr": args.projection_lagrangian_lr,
        "projection_lagrangian_final_dual": projection_dual,
        "actor_projection_debit_scale": args.actor_projection_debit_scale,
        "actor_projection_debit_reason_weights": args.actor_projection_debit_reason_weights,
        "enable_liveness_lagrangian": bool(args.enable_liveness_lagrangian),
        "liveness_lagrangian_targets": args.liveness_lagrangian_targets,
        "liveness_lagrangian_final_duals": liveness_duals,
        "targeted_projection_bc_coef": args.targeted_projection_bc_coef,
        "targeted_projection_bc_reasons": args.targeted_projection_bc_reasons,
        "targeted_projection_bc_reason_coefs": args.targeted_projection_bc_reason_coefs,
        "targeted_projection_bc_group_by_reason": not args.disable_targeted_projection_bc_group_by_reason,
        "targeted_projection_bc_min_weight": args.targeted_projection_bc_min_weight,
        "deadlock_case_window": int(args.deadlock_case_window),
        "late_update_start_fraction": args.late_update_start_fraction,
        "late_learning_rate_scale": args.late_learning_rate_scale,
        "late_freeze_prefill_trunk": bool(args.late_freeze_prefill_trunk),
        "late_freeze_admission_heads": bool(args.late_freeze_admission_heads),
        "teacher_stage_a_fraction": args.teacher_stage_a_fraction,
        "teacher_stage_b_fraction": args.teacher_stage_b_fraction,
        "stage_c_learning_rate_scale": args.stage_c_learning_rate_scale,
        "stage_c_freeze_admission_heads": bool(args.stage_c_freeze_admission_heads),
        "first_valid_anchor_audit": bool(first_valid_anchor_audit_enabled),
        "first_valid_freeze_actor": bool(args.first_valid_freeze_actor),
        "anchor_kl_coef": float(args.anchor_kl_coef),
        "first_valid_anchor_triggered": bool(first_valid_anchor_triggered),
        "strong_action_mask": not args.disable_strong_action_mask,
        "action_safety_filter": not args.disable_action_safety_filter,
        "phased_curriculum": bool(args.phased_curriculum),
        "curriculum_warmup_fraction": float(args.curriculum_warmup_fraction),
        "curriculum_mid_fraction": float(args.curriculum_mid_fraction),
        "bucket_coverage_curriculum": bool(args.bucket_coverage_curriculum),
        "bucket_coverage_curriculum_requests": bucket_coverage_curriculum_requests(args),
        "reward_config": asdict(reward_config),
        "reward_split": {
            "prefill_router_reward": True,
            "decode_router_reward": True,
        },
        "simulator_config": {
            "token_budget_per_step": config.token_budget_per_step,
            "prefill_chunk_size": config.prefill_chunk_size,
            "kv_capacity_tokens": config.kv_capacity_tokens,
            "prefill_ms_per_token": config.prefill_ms_per_token,
            "decode_ms_per_token": config.decode_ms_per_token,
            "kv_reservation_quantile": config.kv_reservation_quantile,
            "prefill_padding_weight": config.prefill_padding_weight,
            "enable_action_safety_filter": config.enable_action_safety_filter,
        },
        "evaluation": best_evaluation,
        "reward_profile": reward_config.profile,
    }
    torch.save(checkpoint, args.output)
    write_metrics(args.metrics_out, metrics_rows)
    if args.trajectory_out:
        write_jsonl(args.trajectory_out, best_trajectory)
    if args.event_out:
        write_jsonl(args.event_out, best_events)
    if args.projection_correction_out:
        write_jsonl(args.projection_correction_out, projection_corrections)
    if args.deadlock_case_out:
        write_jsonl(args.deadlock_case_out, deadlock_cases)
    if args.post_valid_drift_audit_out:
        write_jsonl(args.post_valid_drift_audit_out, drift_audit_rows)

    print(
        json.dumps(
            {
                "episodes": args.episodes,
                "output": str(args.output),
                "metrics_out": str(args.metrics_out),
                "best_reward": best_reward,
                "best_evaluation": best_evaluation,
                "best_validation_gate": best_validation_gate,
                "init_evaluation": init_policy_evaluation,
                "init_validation_gate": init_policy_validation_gate,
                "best_policy_found": best_policy_checkpoint is not None,
                "trajectory_rows": len(best_trajectory),
                "event_rows": len(best_events),
                "projection_correction_rows": len(projection_corrections),
                "deadlock_case_rows": len(deadlock_cases),
                "post_valid_drift_audit_rows": len(drift_audit_rows),
                "teacher_type": "torch-factorized-actor-critic-split-router-reward",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
