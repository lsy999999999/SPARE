from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .engine import LLMSingleNodeSimulator
from .types import Request


@dataclass(slots=True)
class ValidationReport:
    name: str
    passed: bool
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "issues": list(self.issues),
            "metrics": dict(self.metrics),
        }


def _append_issue(issues: List[str], condition: bool, message: str) -> None:
    if not condition:
        issues.append(message)


def validate_request_lifecycle(
    requests: Sequence[Request],
    tolerance_ms: float = 1e-6,
) -> ValidationReport:
    issues: List[str] = []
    for request in requests:
        _append_issue(
            issues,
            request.actual_output_tokens >= 1,
            f"request {request.request_id}: actual_output_tokens must be >= 1",
        )
        _append_issue(
            issues,
            request.prefill_done_tokens <= request.active_prefill_target_tokens,
            f"request {request.request_id}: prefill_done_tokens exceeds active_prefill_target_tokens",
        )
        if request.batch_admitted_at_ms is not None:
            _append_issue(
                issues,
                request.batch_admitted_at_ms + tolerance_ms >= request.arrival_ms,
                f"request {request.request_id}: batch_admitted_at_ms precedes arrival_ms",
            )
        if request.prefill_started_at_ms is not None:
            _append_issue(
                issues,
                request.batch_admitted_at_ms is not None,
                f"request {request.request_id}: prefill_started_at_ms set without batch admission",
            )
            _append_issue(
                issues,
                request.prefill_started_at_ms + tolerance_ms >= request.arrival_ms,
                f"request {request.request_id}: prefill_started_at_ms precedes arrival_ms",
            )
        if request.prefill_completed_at_ms is not None:
            _append_issue(
                issues,
                request.prefill_started_at_ms is not None,
                f"request {request.request_id}: prefill_completed_at_ms set without prefill start",
            )
            _append_issue(
                issues,
                request.prefill_completed_at_ms + tolerance_ms >= (request.prefill_started_at_ms or 0.0),
                f"request {request.request_id}: prefill_completed_at_ms precedes prefill_started_at_ms",
            )
        if request.first_token_at_ms is not None:
            _append_issue(
                issues,
                request.prefill_completed_at_ms is not None,
                f"request {request.request_id}: first_token_at_ms set without prefill completion",
            )
            _append_issue(
                issues,
                request.first_decode_started_at_ms is not None,
                f"request {request.request_id}: first_token_at_ms set without first decode start",
            )
            _append_issue(
                issues,
                request.first_token_at_ms + tolerance_ms >= (request.prefill_completed_at_ms or 0.0),
                f"request {request.request_id}: first_token_at_ms precedes prefill_completed_at_ms",
            )
            _append_issue(
                issues,
                request.first_token_at_ms + tolerance_ms >= (request.first_decode_started_at_ms or 0.0),
                f"request {request.request_id}: first_token_at_ms precedes first_decode_started_at_ms",
            )
        if request.first_decode_started_at_ms is not None:
            _append_issue(
                issues,
                request.prefill_completed_at_ms is not None,
                f"request {request.request_id}: first_decode_started_at_ms set without prefill completion",
            )
            _append_issue(
                issues,
                request.first_decode_started_at_ms + tolerance_ms >= (request.prefill_completed_at_ms or 0.0),
                f"request {request.request_id}: first_decode_started_at_ms precedes prefill_completed_at_ms",
            )
        if request.finished_at_ms is not None:
            _append_issue(
                issues,
                request.first_token_at_ms is not None,
                f"request {request.request_id}: finished_at_ms set without first token",
            )
            _append_issue(
                issues,
                request.finished_at_ms + tolerance_ms >= (request.first_token_at_ms or 0.0),
                f"request {request.request_id}: finished_at_ms precedes first_token_at_ms",
            )
            _append_issue(
                issues,
                request.generated_tokens == request.actual_output_tokens,
                f"request {request.request_id}: completed request generated_tokens mismatch",
            )
            _append_issue(
                issues,
                request.kv_tokens == 0,
                f"request {request.request_id}: completed request must release all KV",
            )
        if request.ttft_ms is not None:
            parts = [
                value
                for value in (
                    request.admission_wait_ms,
                    request.prefill_queue_wait_ms,
                    request.prefill_service_ms,
                    request.ready_to_decode_wait_ms,
                    request.first_decode_service_ms,
                )
                if value is not None
            ]
            if len(parts) == 5:
                lhs = request.ttft_ms
                rhs = sum(parts)
                _append_issue(
                    issues,
                    abs(lhs - rhs) <= max(1e-3, tolerance_ms),
                    f"request {request.request_id}: ttft decomposition mismatch ({lhs} vs {rhs})",
                )
        if request.recompute_count == 0:
            _append_issue(
                issues,
                request.recompute_tokens == 0,
                f"request {request.request_id}: recompute_tokens set but recompute_count is zero",
            )
        _append_issue(
            issues,
            request.eviction_count <= request.recompute_count,
            f"request {request.request_id}: eviction_count exceeds recompute_count",
        )

    return ValidationReport(
        name="request_lifecycle",
        passed=not issues,
        issues=issues,
        metrics={
            "requests": len(requests),
            "completed": sum(1 for request in requests if request.is_complete),
            "with_evictions": sum(1 for request in requests if request.eviction_count > 0),
        },
    )


def validate_event_log(events: Sequence[Dict[str, object]], requests: Sequence[Request]) -> ValidationReport:
    issues: List[str] = []
    last_time = None
    counts: Dict[str, int] = {}
    arrival_by_request: Dict[int, int] = {}
    first_token_by_request: Dict[int, int] = {}
    complete_by_request: Dict[int, int] = {}
    for index, event in enumerate(events):
        event_name = str(event.get("event"))
        counts[event_name] = counts.get(event_name, 0) + 1
        time_ms = event.get("time_ms")
        if isinstance(time_ms, (int, float)):
            if last_time is not None and time_ms < last_time:
                issues.append(f"event {index}: time_ms is not monotonic")
            last_time = float(time_ms)
        request_id = event.get("request_id")
        if isinstance(request_id, int):
            if event_name == "arrival":
                arrival_by_request[request_id] = arrival_by_request.get(request_id, 0) + 1
            if event_name == "first_token":
                first_token_by_request[request_id] = first_token_by_request.get(request_id, 0) + 1
            if event_name == "complete":
                complete_by_request[request_id] = complete_by_request.get(request_id, 0) + 1

    for request in requests:
        _append_issue(
            issues,
            arrival_by_request.get(request.request_id, 0) == 1,
            f"request {request.request_id}: arrival event count must be exactly 1",
        )
        _append_issue(
            issues,
            first_token_by_request.get(request.request_id, 0) == (1 if request.first_token_at_ms is not None else 0),
            f"request {request.request_id}: first_token event count mismatch",
        )
        _append_issue(
            issues,
            complete_by_request.get(request.request_id, 0) == (1 if request.finished_at_ms is not None else 0),
            f"request {request.request_id}: complete event count mismatch",
        )

    return ValidationReport(
        name="event_log",
        passed=not issues,
        issues=issues,
        metrics={
            "events": len(events),
            "event_counts": counts,
        },
    )


def validate_summary_consistency(
    simulator: LLMSingleNodeSimulator,
    summary: Dict[str, object],
) -> ValidationReport:
    issues: List[str] = []
    requests = simulator.requests
    completed = sum(1 for request in requests if request.is_complete)
    queue_mix = {
        "fast": sum(1 for request in requests if request.queue_class.value == "fast"),
        "normal": sum(1 for request in requests if request.queue_class.value == "normal"),
        "risky": sum(1 for request in requests if request.queue_class.value == "risky"),
    }
    _append_issue(issues, int(summary["requests"]) == len(requests), "summary requests count mismatch")
    _append_issue(issues, int(summary["completed"]) == completed, "summary completed count mismatch")
    _append_issue(issues, summary["queue_mix"] == queue_mix, "summary queue_mix mismatch")
    _append_issue(
        issues,
        int(summary["evictions_total"]) == sum(request.eviction_count for request in requests),
        "summary evictions_total mismatch",
    )
    _append_issue(
        issues,
        int(summary["recompute_tokens_total"]) == sum(request.recompute_tokens for request in requests),
        "summary recompute_tokens_total mismatch",
    )
    _append_issue(
        issues,
        int(summary["wasted_kv_tokens_total"]) == sum(request.wasted_kv_tokens for request in requests),
        "summary wasted_kv_tokens_total mismatch",
    )
    _append_issue(
        issues,
        not simulator.active_decode,
        "simulator active_decode must be empty after run",
    )
    _append_issue(
        issues,
        all(len(queue) == 0 for queue in simulator.prefill_queues.values()),
        "simulator prefill queues must be empty after run",
    )

    return ValidationReport(
        name="summary_consistency",
        passed=not issues,
        issues=issues,
        metrics={
            "requests": len(requests),
            "completed": completed,
            "peak_kv_tokens": summary.get("peak_kv_tokens"),
            "total_runtime_ms": summary.get("total_runtime_ms"),
        },
    )


def validate_simulation_run(
    simulator: LLMSingleNodeSimulator,
    summary: Dict[str, object],
) -> List[ValidationReport]:
    return [
        validate_request_lifecycle(simulator.requests),
        validate_event_log(simulator.events, simulator.requests),
        validate_summary_consistency(simulator, summary),
    ]
