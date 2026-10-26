from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Sequence

from .types import Request


def write_events(path: Path, events: Sequence[Dict[str, object]]) -> None:
    if not events:
        path.write_text("")
        return
    keys = sorted({key for event in events for key in event.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(events)


def write_request_summary(path: Path, requests: Sequence[Request]) -> None:
    keys = [
        "request_id",
        "model_id",
        "arrival_ms",
        "template_id",
        "template_family",
        "template_version",
        "queue_class",
        "model_route_name",
        "prefill_route_name",
        "decode_route_name",
        "distribution_source",
        "output_length_source",
        "prompt_tokens",
        "shared_prefix_id",
        "shared_prefix_tokens",
        "private_prompt_tokens",
        "prefix_cache_status",
        "max_new_tokens",
        "structured_output",
        "stop_at_eos",
        "stop_strings",
        "output_mean",
        "output_std",
        "output_p50",
        "output_p90",
        "output_p95",
        "output_p99",
        "actual_output_tokens",
        "eviction_count",
        "recompute_count",
        "recompute_tokens",
        "wasted_kv_tokens",
        "admitted_ms",
        "prefill_start_ms",
        "prefill_done_ms",
        "first_decode_start_ms",
        "first_token_ms",
        "completed_ms",
        "admission_wait_ms",
        "prefill_queue_wait_ms",
        "prefill_service_ms",
        "prefill_compute_ms",
        "ready_to_decode_wait_ms",
        "first_decode_service_ms",
        "decode_to_first_token_ms",
        "ttft_ms",
        "e2e_ms",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for request in requests:
            writer.writerow(
                {
                    "request_id": request.request_id,
                    "model_id": request.model_id,
                    "arrival_ms": round(request.arrival_ms, 3),
                    "template_id": request.template_id,
                    "template_family": request.template_family,
                    "template_version": request.template_version,
                    "queue_class": request.queue_class.value,
                    "model_route_name": request.model_route_name,
                    "prefill_route_name": request.prefill_route_name,
                    "decode_route_name": request.decode_route_name,
                    "distribution_source": request.distribution_source,
                    "output_length_source": request.output_length_source,
                    "prompt_tokens": request.prompt_tokens,
                    "shared_prefix_id": request.shared_prefix_id,
                    "shared_prefix_tokens": request.shared_prefix_tokens,
                    "private_prompt_tokens": request.private_prompt_tokens,
                    "prefix_cache_status": request.prefix_cache_status,
                    "max_new_tokens": request.max_new_tokens,
                    "structured_output": int(request.structured_output),
                    "stop_at_eos": int(request.stop_at_eos),
                    "stop_strings": json.dumps(request.stop_strings),
                    "output_mean": round(request.effective_length_distribution.mean, 3),
                    "output_std": round(request.effective_length_distribution.std, 3),
                    "output_p50": round(
                        request.effective_length_distribution.p50 or request.effective_length_distribution.quantile(0.50),
                        3,
                    ),
                    "output_p90": round(request.effective_length_distribution.p90, 3),
                    "output_p95": round(
                        request.effective_length_distribution.p95 or request.effective_length_distribution.quantile(0.95),
                        3,
                    ),
                    "output_p99": round(
                        request.effective_length_distribution.p99 or request.effective_length_distribution.quantile(0.99),
                        3,
                    ),
                    "actual_output_tokens": request.actual_output_tokens,
                    "eviction_count": request.eviction_count,
                    "recompute_count": request.recompute_count,
                    "recompute_tokens": request.recompute_tokens,
                    "wasted_kv_tokens": request.wasted_kv_tokens,
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
                    "ttft_ms": round(request.ttft_ms or 0.0, 3),
                    "e2e_ms": round(request.e2e_ms or 0.0, 3),
                }
            )
