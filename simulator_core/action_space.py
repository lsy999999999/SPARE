from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from .types import ActionSpaceConfig, PhaseMode, SchedulerAction


def budget_control_action_space() -> ActionSpaceConfig:
    """A smaller quota-oriented action space for serving control."""
    return ActionSpaceConfig(
        control_mode="budget-control",
        phase_choices=["heuristic", PhaseMode.MIXED.value, PhaseMode.DECODE_PRIORITY.value],
        admission_choices=["balanced", "ttft-greedy", "prefill-drain"],
        new_admission_limit_choices=[0, 1, 2, 4],
        quantile_choices=[0.80, 0.90, 0.95],
        budget_profile_choices=["balanced", "decode-heavy", "decode-drain", "prefill-heavy"],
        risky_admission_cap_choices=[0, 1, 2],
    )


def phase_label_from_preference(phase_preference: PhaseMode | None) -> str:
    return phase_preference.value if phase_preference is not None else "heuristic"


def load_action_space_config(path: str | Path | None) -> ActionSpaceConfig:
    if path is None:
        return ActionSpaceConfig()
    payload = json.loads(Path(path).read_text())
    if payload.get("control_mode") == "budget-control" and not any(
        key in payload
        for key in (
            "phase_choices",
            "admission_choices",
            "new_admission_limit_choices",
            "quantile_choices",
            "budget_profile_choices",
            "risky_admission_cap_choices",
        )
    ):
        return budget_control_action_space()
    return ActionSpaceConfig.from_payload(payload)


def build_action_table(action_space: ActionSpaceConfig) -> list[SchedulerAction]:
    table: list[SchedulerAction] = []
    for phase_label in action_space.phase_choices:
        phase_preference = None if phase_label == "heuristic" else PhaseMode(phase_label)
        for admission_profile in action_space.admission_choices:
            for new_admission_limit in action_space.new_admission_limit_choices:
                for shadow_quantile in action_space.quantile_choices:
                    for budget_profile in action_space.budget_profile_choices:
                        for risky_admission_cap in action_space.risky_admission_cap_choices:
                            table.append(
                                SchedulerAction(
                                    phase_preference=phase_preference,
                                    admission_profile=admission_profile,
                                    new_admission_limit=new_admission_limit,
                                    shadow_quantile=shadow_quantile,
                                    budget_profile=budget_profile,
                                    risky_admission_cap=risky_admission_cap,
                                )
                            )
    return table


def build_action_index(
    action_space: ActionSpaceConfig,
    action_table: Sequence[SchedulerAction] | None = None,
) -> dict[tuple[str, str, int, float, str, int], int]:
    table = action_table if action_table is not None else build_action_table(action_space)
    return {
        action_key(action, action_space): index
        for index, action in enumerate(table)
    }


def action_key(
    action: SchedulerAction,
    action_space: ActionSpaceConfig,
) -> tuple[str, str, int, float, str, int]:
    new_admission_limit = (
        action.new_admission_limit
        if action.new_admission_limit is not None
        else action_space.new_admission_limit_choices[0]
    )
    risky_admission_cap = (
        action.risky_admission_cap
        if action.risky_admission_cap is not None
        else action_space.risky_admission_cap_choices[0]
    )
    return (
        phase_label_from_preference(action.phase_preference),
        str(action.admission_profile),
        int(new_admission_limit),
        round(float(action.shadow_quantile or action_space.quantile_choices[0]), 2),
        str(action.budget_profile),
        int(risky_admission_cap),
    )


def allowed_admission_profiles(phase_label: str, choices: Sequence[str]) -> list[str]:
    blocked: set[str] = set()
    if phase_label == PhaseMode.PREFILL_PRIORITY.value:
        blocked.add("tail-relief")
    allowed = [choice for choice in choices if choice not in blocked]
    return allowed or list(choices)


def allowed_budget_profiles(
    phase_label: str,
    admission_profile: str,
    choices: Sequence[str],
) -> list[str]:
    blocked: set[str] = set()
    if phase_label == PhaseMode.PREFILL_PRIORITY.value:
        blocked.update({"decode-heavy", "decode-drain", "decode-strict"})
    if admission_profile == "prefill-drain":
        blocked.update({"decode-drain", "decode-strict"})
    allowed = [choice for choice in choices if choice not in blocked]
    return allowed or list(choices)


def allowed_new_admission_limits(
    phase_label: str,
    admission_profile: str,
    budget_profile: str,
    choices: Sequence[int],
) -> list[int]:
    max_limit = max(int(choice) for choice in choices)
    if phase_label == PhaseMode.PREFILL_PRIORITY.value:
        max_limit = min(max_limit, 4)
    if admission_profile in {"tail-relief", "prefill-drain"}:
        max_limit = min(max_limit, 4)
    allowed = [int(choice) for choice in choices if int(choice) <= max_limit]
    return allowed or [min(int(choice) for choice in choices)]


def allowed_shadow_quantiles(
    phase_label: str,
    admission_profile: str,
    budget_profile: str,
    choices: Sequence[float],
) -> list[float]:
    max_quantile = max(float(choice) for choice in choices)
    if phase_label == PhaseMode.PREFILL_PRIORITY.value or budget_profile == "decode-strict":
        max_quantile = min(max_quantile, 0.95)
    if admission_profile == "tail-relief":
        max_quantile = min(max_quantile, 0.95)
    allowed = [float(choice) for choice in choices if float(choice) <= max_quantile + 1e-9]
    return allowed or [min(float(choice) for choice in choices)]


def allowed_risky_admission_caps(
    phase_label: str,
    admission_profile: str,
    budget_profile: str,
    choices: Sequence[int],
) -> list[int]:
    max_cap = max(int(choice) for choice in choices)
    if phase_label == PhaseMode.PREFILL_PRIORITY.value or budget_profile == "decode-strict":
        max_cap = min(max_cap, 1)
    if admission_profile == "tail-relief":
        max_cap = min(max_cap, 2)
    allowed = [int(choice) for choice in choices if int(choice) <= max_cap]
    return allowed or [min(int(choice) for choice in choices)]


def _nearest_numeric_choice(value: float, allowed: Iterable[float]) -> float:
    allowed_list = list(allowed)
    return min(allowed_list, key=lambda candidate: (abs(float(candidate) - float(value)), float(candidate)))


def _pick_string_choice(value: str, allowed: Sequence[str]) -> str:
    if value in allowed:
        return value
    return allowed[0]


def project_action_to_stable_space(
    action: SchedulerAction,
    *,
    admission_choices: Sequence[str],
    new_admission_limit_choices: Sequence[int],
    quantile_choices: Sequence[float],
    budget_profile_choices: Sequence[str],
    risky_admission_cap_choices: Sequence[int],
) -> SchedulerAction:
    phase_label = phase_label_from_preference(action.phase_preference)

    allowed_admission = allowed_admission_profiles(phase_label, admission_choices)
    admission_profile = _pick_string_choice(str(action.admission_profile), allowed_admission)

    allowed_budget = allowed_budget_profiles(phase_label, admission_profile, budget_profile_choices)
    budget_profile = _pick_string_choice(str(action.budget_profile), allowed_budget)

    raw_limit = int(action.new_admission_limit) if action.new_admission_limit is not None else min(new_admission_limit_choices)
    allowed_limits = allowed_new_admission_limits(
        phase_label,
        admission_profile,
        budget_profile,
        new_admission_limit_choices,
    )
    new_admission_limit = int(_nearest_numeric_choice(raw_limit, allowed_limits))

    raw_quantile = float(action.shadow_quantile or min(quantile_choices))
    allowed_quantiles = allowed_shadow_quantiles(
        phase_label,
        admission_profile,
        budget_profile,
        quantile_choices,
    )
    shadow_quantile = float(_nearest_numeric_choice(raw_quantile, allowed_quantiles))

    raw_risky_cap = int(action.risky_admission_cap) if action.risky_admission_cap is not None else min(risky_admission_cap_choices)
    allowed_risky_caps = allowed_risky_admission_caps(
        phase_label,
        admission_profile,
        budget_profile,
        risky_admission_cap_choices,
    )
    risky_admission_cap = int(_nearest_numeric_choice(raw_risky_cap, allowed_risky_caps))

    return SchedulerAction(
        phase_preference=action.phase_preference,
        admission_profile=admission_profile,
        new_admission_limit=new_admission_limit,
        shadow_quantile=shadow_quantile,
        budget_profile=budget_profile,
        risky_admission_cap=risky_admission_cap,
        prefill_safety_filter_enabled=action.prefill_safety_filter_enabled,
        decode_safety_filter_enabled=action.decode_safety_filter_enabled,
    )
