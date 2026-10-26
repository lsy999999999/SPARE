from __future__ import annotations

from typing import Dict, List, Sequence

from .types import Request


class OutputMetricsCollector:
    name = "output-metrics"

    def record_completion(
        self,
        request: Request,
        released_logical_kv_tokens: int,
        released_physical_kv_tokens: int,
    ) -> Dict[str, object]:
        return {
            "request_id": request.request_id,
            "output_tokens": request.actual_output_tokens,
            "output_length_source": request.output_length_source,
            "ttft_ms": round(request.ttft_ms or 0.0, 3),
            "e2e_ms": round(request.e2e_ms or 0.0, 3),
            "arrival_ms": round(request.arrival_ms, 3),
            "admitted_ms": round(request.batch_admitted_at_ms or 0.0, 3),
            "prefill_start_ms": round(request.prefill_started_at_ms or 0.0, 3),
            "prefill_done_ms": round(request.prefill_completed_at_ms or 0.0, 3),
            "first_decode_start_ms": round(request.first_decode_started_at_ms or 0.0, 3),
            "first_token_ms": round(request.first_token_at_ms or 0.0, 3),
            "completed_ms": round(request.finished_at_ms or 0.0, 3),
            "admission_wait_ms": round(request.admission_wait_ms or 0.0, 3),
            "prefill_queue_wait_ms": round(request.prefill_queue_wait_ms or 0.0, 3),
            "prefill_service_ms": round(request.prefill_service_ms or 0.0, 3),
            "prefill_compute_ms": round(request.prefill_compute_ms or 0.0, 3),
            "ready_to_decode_wait_ms": round(request.ready_to_decode_wait_ms or 0.0, 3),
            "first_decode_service_ms": round(request.first_decode_service_ms or 0.0, 3),
            "decode_to_first_token_ms": round(request.decode_to_first_token_ms or 0.0, 3),
            "eviction_count": request.eviction_count,
            "recompute_tokens": request.recompute_tokens,
            "wasted_kv_tokens": request.wasted_kv_tokens,
            "released_logical_kv_tokens": int(released_logical_kv_tokens),
            "released_physical_kv_tokens": int(released_physical_kv_tokens),
        }

    def build_summary(
        self,
        base_summary: Dict[str, object],
        completion_rows: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        total_output_tokens = sum(int(row["output_tokens"]) for row in completion_rows)
        total_physical_kv_released = sum(int(row["released_physical_kv_tokens"]) for row in completion_rows)
        total_logical_kv_released = sum(int(row["released_logical_kv_tokens"]) for row in completion_rows)
        traced_outputs = sum(1 for row in completion_rows if row["output_length_source"] == "trace")
        sampled_outputs = len(completion_rows) - traced_outputs
        return {
            "completed_requests": len(completion_rows),
            "total_output_tokens": total_output_tokens,
            "total_released_logical_kv_tokens": total_logical_kv_released,
            "total_released_physical_kv_tokens": total_physical_kv_released,
            "output_length_sources": {
                "trace": traced_outputs,
                "sampled": sampled_outputs,
            },
            "rl_objective_inputs": {
                "ttft_p95_ms": base_summary["ttft_ms"]["p95"],
                "ttft_cvar95_ms": base_summary["ttft_ms"]["cvar95"],
                "e2e_p95_ms": base_summary["e2e_ms"]["p95"],
                "admission_wait_cvar95_ms": base_summary["admission_wait_ms"]["cvar95"],
                "evictions_total": base_summary["evictions_total"],
                "recompute_tokens_total": base_summary["recompute_tokens_total"],
                "wasted_kv_tokens_total": base_summary["wasted_kv_tokens_total"],
                "peak_kv_tokens": base_summary["peak_kv_tokens"],
                "peak_allocator_internal_fragmentation_tokens": base_summary[
                    "peak_allocator_internal_fragmentation_tokens"
                ],
                "peak_allocator_external_fragmentation_tokens": base_summary[
                    "peak_allocator_external_fragmentation_tokens"
                ],
                "terminated_early": base_summary["terminated_early"],
                "unfinished_requests": base_summary["unfinished_requests"],
                "horizon_unfinished_requests": base_summary.get("horizon_unfinished_requests", 0),
                "aged_unfinished_requests": base_summary.get("aged_unfinished_requests", 0),
                "liveness_unfinished_requests": base_summary.get("liveness_unfinished_requests", 0),
                "punishable_unfinished_requests": base_summary.get("punishable_unfinished_requests", 0),
            },
        }
