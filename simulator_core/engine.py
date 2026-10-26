from __future__ import annotations

import math
import copy
import hashlib
import random
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .kv_allocator import build_kv_allocator
from .metrics import OutputMetricsCollector
from .timing import build_timing_model, TimingBatch
from .types import PhaseMode, QueueClass, Request, ScheduleDecision, SimulatorConfig, percentile


def _request_id_list(requests: Iterable[Request]) -> str:
    return ",".join(str(int(request.request_id)) for request in requests)


def _contains_identity(requests: Sequence[Request], target: Request) -> bool:
    return any(request is target for request in requests)


def _remove_identity(requests: List[Request], target: Request) -> None:
    for index, request in enumerate(requests):
        if request is target:
            requests.pop(index)
            return
    raise ValueError("request object is not present in the queue")


@dataclass(slots=True)
class PrefixCacheEntry:
    prefix_id: str
    model_id: str
    template_id: str
    template_family: str
    logical_tokens: int
    cacheable_tokens: int
    block_size_tokens: int
    total_blocks: int
    cached_blocks: int = 0
    state: str = "allocating"
    owner_request_id: Optional[int] = None
    active_references: int = 0
    hit_count: int = 0
    reuse_tokens: int = 0
    last_used_ms: float = 0.0

    @property
    def physical_tokens(self) -> int:
        return self.total_blocks * self.block_size_tokens

    @property
    def cached_tokens(self) -> int:
        return self.cached_blocks * self.block_size_tokens

    @property
    def warm(self) -> bool:
        return self.state == "ready"


class LLMSingleNodeSimulator:
    RECENT_QUERY_WINDOW_SIZES = (8, 32, 128)
    EVENT_ORDER = {
        "arrival": 0,
        "prefix_cache_reserve": 0,
        "prefix_cache_evict": 0,
        "batch_admit": 1,
        "prefill_reset": 1,
        "prefill_step": 2,
        "prefix_cache_warm": 2,
        "prefill_complete": 3,
        "recompute_complete": 3,
        "decode_step": 4,
        "first_token": 5,
        "complete": 6,
        "evict_recompute": 7,
        "termination": 8,
    }

    def __init__(
        self,
        requests: Iterable[Request],
        config: SimulatorConfig,
        scheduler=None,
        metrics_collector: Optional[OutputMetricsCollector] = None,
    ):
        from .scheduler import DistributionAwareScheduler

        self.config = config
        self.timing_model = build_timing_model(config.timing_profile) if config.timing_profile is not None else None
        self.scheduler = scheduler or DistributionAwareScheduler(config)
        self.requests = sorted(list(requests), key=lambda req: req.arrival_ms)
        if len({request.request_id for request in self.requests}) != len(self.requests):
            raise ValueError("request IDs must be unique within an episode")
        self._true_length_distributions = {
            request.request_id: copy.deepcopy(request.true_length_distribution or request.length_distribution)
            for request in self.requests
        }
        self._batch_protected_ids: set[int] = set()
        for request in self.requests:
            request.output_length_source = "trace" if request.actual_output_tokens > 0 else "unsampled"
        self.pending_index = 0
        self.now_ms = 0.0
        self.rng = random.Random(config.runtime_sampling_seed)
        self.prefill_queues: Dict[QueueClass, List[Request]] = {
            QueueClass.FAST: [],
            QueueClass.NORMAL: [],
            QueueClass.RISKY: [],
        }
        self.active_decode: List[Request] = []
        self.events: List[Dict[str, object]] = []
        self.phase_counts: Dict[str, int] = {
            PhaseMode.PREFILL_PRIORITY.value: 0,
            PhaseMode.DECODE_PRIORITY.value: 0,
            PhaseMode.MIXED.value: 0,
        }
        self.kv_allocator = build_kv_allocator(
            config.kv_allocator_mode,
            config.kv_capacity_tokens,
            config.kv_block_size_tokens,
            scope=config.kv_cache_scope,
        )
        self.metrics_collector = metrics_collector or OutputMetricsCollector()
        self.completed_request_metrics: List[Dict[str, object]] = []
        self.prefix_cache: Dict[str, PrefixCacheEntry] = {}
        self.prefix_cache_hit_requests = 0
        self.prefix_cache_miss_requests = 0
        self.prefix_cache_bypass_requests = 0
        self.prefix_cache_reuse_tokens = 0
        self.prefix_cache_evictions_total = 0
        self.prefix_cache_peak_active_references = 0
        self.peak_kv_tokens = 0
        self.peak_logical_kv_tokens = 0
        self.peak_allocator_internal_fragmentation_tokens = 0.0
        self.peak_allocator_external_fragmentation_tokens = 0.0
        self.deadlock_recoveries_total = 0
        self.termination_reason = "not_started"
        self.termination_details: Dict[str, object] = {}
        self.simulation_steps = 0
        self.wall_time_seconds = 0.0
        self.recent_query_history = deque(maxlen=max(self.RECENT_QUERY_WINDOW_SIZES))

    @property
    def current_kv_tokens(self) -> int:
        return self.kv_allocator.current_physical_tokens() + self.prefix_cache_physical_tokens

    @property
    def prefix_cache_physical_tokens(self) -> int:
        if not self.config.prefix_cache_enabled:
            return 0
        return sum(entry.physical_tokens for entry in self.prefix_cache.values())

    @property
    def reclaimable_prefix_tokens(self) -> int:
        if not self.config.prefix_cache_enable_lru_eviction:
            return 0
        return sum(entry.physical_tokens for entry in self.prefix_cache.values()
                   if entry.warm and entry.active_references == 0)

    @property
    def current_logical_kv_tokens(self) -> int:
        return sum(request.logical_kv_tokens for request in self.requests if not request.is_complete)

    def quantize_kv_tokens(self, logical_tokens: int) -> int:
        return self.kv_allocator.quantize_tokens(logical_tokens)

    def current_shadow_kv_commitment(self, quantile: Optional[float] = None) -> int:
        resolved_quantile = quantile if quantile is not None else self.config.kv_reservation_quantile
        committed = 0
        for request in self.requests:
            if request.is_complete or not request.has_started:
                continue
            committed += self.quantize_kv_tokens(request.shadow_kv_target(resolved_quantile))
        return committed

    def estimate_kv_growth(self, request: Request, delta_tokens: int) -> int:
        delta_tokens = max(0, delta_tokens - self._reserved_prefill_growth(request))
        if delta_tokens <= 0:
            return 0
        attempt = self.kv_allocator.estimate_growth(request.request_id, request.kv_tokens, delta_tokens)
        if attempt.success:
            return attempt.physical_delta_tokens
        target = self.quantize_kv_tokens(request.kv_tokens + delta_tokens)
        current = self.quantize_kv_tokens(request.kv_tokens)
        return max(0, target - current)

    def _reserved_prefill_growth(self, request: Request) -> int:
        """Prefill work backed by an already allocated shared prefix."""
        entry = self._prefix_cache_owner_entry(request)
        if entry is None:
            return 0
        return max(0, entry.cacheable_tokens - request.shared_kv_tokens)

    @property
    def prefix_cache_block_size_tokens(self) -> int:
        return max(1, int(self.config.prefix_cache_block_size_tokens or self.config.kv_block_size_tokens))

    def _prefix_cache_key(self, request: Request) -> str:
        if not self.config.prefix_cache_enabled:
            return ""
        cacheable_tokens = request.cacheable_shared_prefix_tokens(self.prefix_cache_block_size_tokens)
        if cacheable_tokens <= 0 or not request.shared_prefix_id:
            return ""
        return f"{request.model_id}::{request.shared_prefix_id}::{cacheable_tokens}"

    def _prefix_cache_physical_tokens_for(self, logical_tokens: int) -> int:
        block_size = self.prefix_cache_block_size_tokens
        blocks = max(0, int(logical_tokens)) // block_size
        return blocks * block_size

    def _evict_prefix_cache_entries(
        self,
        required_physical_tokens: int,
        *,
        exclude_keys: Optional[set[str]] = None,
    ) -> int:
        if required_physical_tokens <= 0:
            return 0
        protected = exclude_keys or set()
        evicted = 0
        for cache_key, entry in sorted(
            self.prefix_cache.items(),
            key=lambda item: (item[1].last_used_ms, item[1].hit_count, item[0]),
        ):
            if cache_key in protected:
                continue
            if entry.active_references > 0 or not entry.warm:
                continue
            del self.prefix_cache[cache_key]
            evicted += 1
            self.prefix_cache_evictions_total += 1
            self._record_event(
                {
                    "time_ms": round(self.now_ms, 3),
                    "event": "prefix_cache_evict",
                    "shared_prefix_id": cache_key,
                    "template_id": entry.template_id,
                    "cacheable_tokens": entry.cacheable_tokens,
                    "cached_blocks": entry.cached_blocks,
                    "total_blocks": entry.total_blocks,
                    "prefix_cache_tokens": entry.physical_tokens,
                },
                importance="dense",
            )
            if self.current_kv_tokens + required_physical_tokens <= self.config.kv_capacity_tokens:
                break
            if (
                self.config.prefix_cache_max_resident_entries > 0
                and len(self.prefix_cache) <= self.config.prefix_cache_max_resident_entries
                and self.current_kv_tokens + required_physical_tokens <= self.config.kv_capacity_tokens
            ):
                break
        return evicted

    def _ensure_prefix_cache_capacity(self, logical_tokens: int, *, exclude_keys: Optional[set[str]] = None) -> bool:
        physical_tokens = self._prefix_cache_physical_tokens_for(logical_tokens)
        if physical_tokens <= 0:
            return True
        needs_eviction = (
            self.current_kv_tokens + physical_tokens > self.config.kv_capacity_tokens
            or (
                self.config.prefix_cache_max_resident_entries > 0
                and len(self.prefix_cache) >= self.config.prefix_cache_max_resident_entries
            )
        )
        if needs_eviction and self.config.prefix_cache_enable_lru_eviction:
            self._evict_prefix_cache_entries(physical_tokens, exclude_keys=exclude_keys)
        if self.current_kv_tokens + physical_tokens > self.config.kv_capacity_tokens:
            return False
        if (
            self.config.prefix_cache_max_resident_entries > 0
            and len(self.prefix_cache) >= self.config.prefix_cache_max_resident_entries
        ):
            return False
        return True

    def _prefix_cache_entry(self, request: Request) -> Optional[PrefixCacheEntry]:
        cache_key = self._prefix_cache_key(request)
        if not cache_key:
            return None
        return self.prefix_cache.get(cache_key)

    def _prefix_cache_owner_entry(self, request: Request) -> Optional[PrefixCacheEntry]:
        entry = self._prefix_cache_entry(request)
        if entry is None or entry.owner_request_id != request.request_id:
            return None
        return entry

    def _release_prefix_cache_reference(self, request: Request) -> None:
        entry = self._prefix_cache_entry(request)
        if entry is None:
            return
        if entry.owner_request_id == request.request_id or request.prefix_cache_status == "hit":
            entry.active_references = max(0, entry.active_references - 1)
            entry.last_used_ms = self.now_ms

    def _attach_prefix_cache(self, request: Request) -> None:
        request.shared_kv_tokens = 0
        if not self.config.prefix_cache_enabled:
            request.prefix_cache_status = "disabled"
            return
        cacheable_tokens = request.cacheable_shared_prefix_tokens(self.prefix_cache_block_size_tokens)
        if cacheable_tokens <= 0 or request.shared_prefix_tokens <= 0 or not request.shared_prefix_id:
            request.prefix_cache_status = "uncacheable"
            return
        cache_key = self._prefix_cache_key(request)
        entry = self.prefix_cache.get(cache_key)
        if entry is not None and entry.warm:
            # A cache stores KV, not the logits needed to sample the first token.
            # Keep the final prompt block in the compute plan on a full hit.
            reused_tokens = cacheable_tokens
            if reused_tokens == request.prompt_tokens:
                reused_tokens = max(0, reused_tokens - self.prefix_cache_block_size_tokens)
            request.prefix_cache_status = "hit"
            request.shared_kv_tokens = reused_tokens
            request.prefill_done_tokens = max(request.prefill_done_tokens, reused_tokens)
            entry.active_references += 1
            entry.hit_count += 1
            entry.reuse_tokens += reused_tokens
            entry.last_used_ms = self.now_ms
            self.prefix_cache_hit_requests += 1
            self.prefix_cache_reuse_tokens += reused_tokens
            self.prefix_cache_peak_active_references = max(
                self.prefix_cache_peak_active_references,
                max(item.active_references for item in self.prefix_cache.values()),
            )
            return
        if entry is None and self._ensure_prefix_cache_capacity(cacheable_tokens, exclude_keys={cache_key}):
            physical_tokens = self._prefix_cache_physical_tokens_for(cacheable_tokens)
            self.prefix_cache[cache_key] = PrefixCacheEntry(
                prefix_id=cache_key,
                model_id=request.model_id,
                template_id=request.template_id,
                template_family=request.template_family,
                logical_tokens=request.shared_prefix_tokens,
                cacheable_tokens=cacheable_tokens,
                block_size_tokens=self.prefix_cache_block_size_tokens,
                total_blocks=max(0, physical_tokens // self.prefix_cache_block_size_tokens),
                cached_blocks=0,
                state="allocating",
                owner_request_id=request.request_id,
                active_references=1,
                last_used_ms=self.now_ms,
            )
            request.prefix_cache_status = "warming"
            self.prefix_cache_miss_requests += 1
            self.prefix_cache_peak_active_references = max(
                self.prefix_cache_peak_active_references,
                max(item.active_references for item in self.prefix_cache.values()),
            )
            self._record_event(
                {
                    "time_ms": round(self.now_ms, 3),
                    "event": "prefix_cache_reserve",
                    "request_id": request.request_id,
                    "template_id": request.template_id,
                    "template_family": request.template_family,
                    "shared_prefix_id": request.shared_prefix_id,
                    "shared_prefix_tokens": request.shared_prefix_tokens,
                    "cacheable_shared_prefix_tokens": cacheable_tokens,
                    "uncacheable_shared_prefix_tokens": request.shared_prefix_tokens - cacheable_tokens,
                    "cached_blocks": 0,
                    "total_blocks": max(0, physical_tokens // self.prefix_cache_block_size_tokens),
                    "prefix_cache_state": "allocating",
                    "prefix_cache_tokens": physical_tokens,
                },
                importance="dense",
            )
            return
        if entry is not None:
            request.prefix_cache_status = "warming"
            self.prefix_cache_miss_requests += 1
            return
        request.prefix_cache_status = "bypass"
        self.prefix_cache_bypass_requests += 1

    def _mark_prefix_cache_warm(self, request: Request) -> None:
        entry = self._prefix_cache_owner_entry(request)
        if entry is None:
            return
        if entry.warm:
            return
        entry.cached_blocks = entry.total_blocks
        entry.state = "ready"
        entry.last_used_ms = self.now_ms
        self._record_event(
            {
                "time_ms": round(self.now_ms, 3),
                "event": "prefix_cache_warm",
                "request_id": request.request_id,
                "template_id": entry.template_id,
                "shared_prefix_id": request.shared_prefix_id,
                "shared_prefix_tokens": request.shared_prefix_tokens,
                "cacheable_shared_prefix_tokens": entry.cacheable_tokens,
                "cached_blocks": entry.cached_blocks,
                "total_blocks": entry.total_blocks,
                "prefix_cache_state": entry.state,
                "prefix_cache_tokens": entry.physical_tokens,
            },
            importance="dense",
        )

    def _duration_value(self, prefill_tokens: int, prefill_padding_tokens: int, decode_tokens: int) -> float:
        if self.config.time_unit == "token":
            return float(
                self.config.step_base_ms
                + prefill_tokens
                + self.config.prefill_padding_weight * prefill_padding_tokens
                + decode_tokens
            )
        return float(
            self.config.step_base_ms
            + (prefill_tokens + self.config.prefill_padding_weight * prefill_padding_tokens)
            * self.config.prefill_ms_per_token
            + decode_tokens * self.config.decode_ms_per_token
        )

    def _decode_duration_value(self, decode_tokens: int) -> float:
        if decode_tokens <= 0:
            return 0.0
        if self.config.time_unit == "token":
            return float(decode_tokens)
        return float(decode_tokens * self.config.decode_ms_per_token)

    def _record_event(self, event: Dict[str, object], importance: str = "important") -> None:
        if self.config.event_record_mode == "important" and importance == "dense":
            return
        self.events.append(event)

    def _recent_query_mix_window_summary(self, rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
        if not rows:
            return {
                "prompt_mean": 0.0,
                "output_mean": 0.0,
                "tail_ratio_mean": 0.0,
                "output_cv_mean": 0.0,
                "risky_fraction": 0.0,
                "interarrival_mean": 0.0,
            }
        prompt_mean = statistics.fmean(float(row["prompt_tokens"]) for row in rows)
        output_mean = statistics.fmean(float(row["output_mean"]) for row in rows)
        tail_ratio_mean = statistics.fmean(float(row["tail_ratio"]) for row in rows)
        output_cv_mean = statistics.fmean(float(row["output_cv"]) for row in rows)
        risky_fraction = statistics.fmean(float(row["is_risky"]) for row in rows)
        interarrival_gaps = [
            max(0.0, float(rows[index]["arrival_ms"]) - float(rows[index - 1]["arrival_ms"]))
            for index in range(1, len(rows))
        ]
        interarrival_mean = statistics.fmean(interarrival_gaps) if interarrival_gaps else 0.0
        return {
            "prompt_mean": min(prompt_mean / 1024.0, 4.0),
            "output_mean": min(output_mean / 1024.0, 4.0),
            "tail_ratio_mean": min(tail_ratio_mean / 4.0, 4.0),
            "output_cv_mean": min(output_cv_mean / 2.0, 4.0),
            "risky_fraction": min(max(risky_fraction, 0.0), 1.0),
            "interarrival_mean": min(interarrival_mean / 100.0, 4.0),
        }

    def recent_query_mix_summary(self) -> Dict[str, float]:
        rows = list(self.recent_query_history)
        summary: Dict[str, float] = {}
        for window_size in self.RECENT_QUERY_WINDOW_SIZES:
            window_rows = rows[-window_size:]
            window_summary = self._recent_query_mix_window_summary(window_rows)
            prefix = f"recent_q{window_size}_"
            for key, value in window_summary.items():
                summary[f"{prefix}{key}"] = float(value)
        return summary

    def run(
        self,
        *,
        max_steps: int = 0,
        progress_every: int = 0,
        progress_label: str = "",
        wall_time_limit_s: float = 0.0,
    ) -> Dict[str, object]:
        self.termination_reason = "running"
        self.termination_details = {}
        self.simulation_steps = 0
        start_time = time.monotonic()
        max_steps = max(0, int(max_steps or 0))
        progress_every = max(0, int(progress_every or 0))
        wall_time_limit_s = max(0.0, float(wall_time_limit_s or 0.0))
        simulation_duration_ms = max(0.0, float(self.config.simulation_duration_ms or 0.0))
        while True:
            elapsed_s = time.monotonic() - start_time
            if simulation_duration_ms > 0.0 and self.now_ms >= simulation_duration_ms:
                self.termination_reason = "simulation_duration_elapsed"
                self.termination_details = self._build_termination_details("simulation-duration")
                self.termination_details["simulation_duration_ms"] = round(simulation_duration_ms, 3)
                self._record_event(
                    {
                        "time_ms": round(self.now_ms, 3),
                        "event": "termination",
                        "reason": self.termination_reason,
                        "simulation_steps": self.simulation_steps,
                        "simulation_duration_ms": round(simulation_duration_ms, 3),
                    },
                )
                break
            if wall_time_limit_s > 0.0 and elapsed_s >= wall_time_limit_s:
                self.termination_reason = "wall_time_limit_exceeded"
                self.termination_details = self._build_termination_details("wall-time-limit")
                self.termination_details["wall_time_limit_s"] = wall_time_limit_s
                self.termination_details["wall_time_elapsed_s"] = round(elapsed_s, 3)
                self._record_event(
                    {
                        "time_ms": round(self.now_ms, 3),
                        "event": "termination",
                        "reason": self.termination_reason,
                        "simulation_steps": self.simulation_steps,
                        "wall_time_elapsed_s": round(elapsed_s, 3),
                    },
                )
                break
            if max_steps > 0 and self.simulation_steps >= max_steps:
                self.termination_reason = "step_cap_exceeded"
                self.termination_details = self._build_termination_details("step-cap")
                self.termination_details["step_cap"] = max_steps
                self.termination_details["simulation_steps"] = self.simulation_steps
                self._record_event(
                    {
                        "time_ms": round(self.now_ms, 3),
                        "event": "termination",
                        "reason": self.termination_reason,
                        "simulation_steps": self.simulation_steps,
                    },
                )
                break
            self._ingest_arrivals()
            if self._is_finished():
                self.termination_reason = "completed_all_requests"
                break
            decision = self.scheduler.schedule(self)
            self.simulation_steps += 1
            if progress_every > 0 and self.simulation_steps % progress_every == 0:
                self._emit_progress(progress_label, start_time)
            self.phase_counts[decision.phase_mode.value] += 1
            if not decision.decode_requests and not decision.prefill_requests:
                self.scheduler.on_idle()
                if self.config.enable_deadlock_recovery and self._recover_unschedulable_state():
                    continue
                next_arrival = self._next_arrival_time()
                if next_arrival is None:
                    self.termination_reason = "deadlock_unschedulable"
                    self.termination_details = self._build_termination_details(decision.phase_mode.value)
                    self._record_event(
                        {
                            "time_ms": round(self.now_ms, 3),
                            "event": "termination",
                            "reason": self.termination_reason,
                            "unfinished_requests": self.termination_details["unfinished_requests"],
                            "active_decode": self.termination_details["active_decode"],
                            "started_prefill_waiting": self.termination_details["started_prefill_waiting"],
                            "pending_unadmitted": self.termination_details["pending_unadmitted"],
                            "current_kv_tokens": self.termination_details["current_kv_tokens"],
                        },
                    )
                    break
                bounded_next_arrival = (
                    min(next_arrival, simulation_duration_ms) if simulation_duration_ms > 0.0 else next_arrival
                )
                self.now_ms = max(self.now_ms, bounded_next_arrival)
                continue
            step_result = self._apply_decision(decision)
            self.scheduler.on_step_result(step_result)
            made_progress = (
                step_result["actual_prefill_progress_tokens"] > 0
                or step_result["actual_decode_progress_tokens"] > 0
                or step_result["completed_requests"] > 0
                or step_result["first_tokens_emitted"] > 0
            )
            if not made_progress:
                if self.config.enable_deadlock_recovery and self._recover_unschedulable_state():
                    continue
                if self._next_arrival_time() is None:
                    self.termination_reason = "deadlock_no_progress"
                    self.termination_details = self._build_termination_details(decision.phase_mode.value)
                    self._record_event(
                        {
                            "time_ms": round(self.now_ms, 3),
                            "event": "termination",
                            "reason": self.termination_reason,
                            "unfinished_requests": self.termination_details["unfinished_requests"],
                            "active_decode": self.termination_details["active_decode"],
                            "started_prefill_waiting": self.termination_details["started_prefill_waiting"],
                            "pending_unadmitted": self.termination_details["pending_unadmitted"],
                            "current_kv_tokens": self.termination_details["current_kv_tokens"],
                        },
                    )
                    break
        self.events.sort(key=self._event_sort_key)
        self.wall_time_seconds = time.monotonic() - start_time
        return self._build_summary()

    def _emit_progress(self, label: str, start_time: float) -> None:
        prefix = f"[sim-progress {label}]" if label else "[sim-progress]"
        snapshot = self._progress_snapshot()
        snapshot_text = " ".join(f"{key}={value}" for key, value in snapshot.items())
        print(
            f"{prefix} wall_s={time.monotonic() - start_time:.1f} steps={self.simulation_steps} {snapshot_text}",
            file=sys.stderr,
            flush=True,
        )

    def _progress_snapshot(self) -> Dict[str, object]:
        queued_prefill = sum(len(queue) for queue in self.prefill_queues.values())
        active_decode = [request for request in self.active_decode if not request.is_complete]
        unfinished = [request for request in self.requests if not request.is_complete]
        longest_prefill_request = max(
            unfinished,
            key=lambda request: request.remaining_prefill_tokens,
            default=None,
        )
        longest_decode_request = max(
            active_decode,
            key=lambda request: request.remaining_output_tokens,
            default=None,
        )
        return {
            "now_ms": round(self.now_ms, 3),
            "completed": sum(1 for request in self.requests if request.is_complete),
            "requests": len(self.requests),
            "pending_index": self.pending_index,
            "queued_prefill": queued_prefill,
            "active_decode": len(active_decode),
            "kv": self.current_kv_tokens,
            "logical_kv": self.current_logical_kv_tokens,
            "max_prefill_remaining": max((request.remaining_prefill_tokens for request in unfinished), default=0),
            "max_decode_remaining": max((request.remaining_output_tokens for request in active_decode), default=0),
            "prefill_req": getattr(longest_prefill_request, "request_id", ""),
            "decode_req": getattr(longest_decode_request, "request_id", ""),
            "deadlock_recoveries": self.deadlock_recoveries_total,
        }

    def _ingest_arrivals(self) -> None:
        while self.pending_index < len(self.requests):
            request = self.requests[self.pending_index]
            if request.arrival_ms > self.now_ms:
                break
            self._attach_prefix_cache(request)
            self.scheduler.route_prefill_request(request, self)
            self.prefill_queues[request.queue_class].append(request)
            self.recent_query_history.append(
                {
                    "arrival_ms": float(request.arrival_ms),
                    "template_id": str(request.template_id),
                    "template_family": str(request.template_family),
                    "template_version": str(request.template_version),
                    "shared_prefix_id": str(request.shared_prefix_id),
                    "prompt_tokens": float(request.prompt_tokens),
                    "shared_prefix_tokens": float(request.shared_prefix_tokens),
                    "private_prompt_tokens": float(request.private_prompt_tokens),
                    "prefix_cache_hit": 1.0 if request.prefix_cache_status == "hit" else 0.0,
                    "output_mean": float(request.query_length_distribution.mean),
                    "tail_ratio": float(request.query_length_distribution.tail_ratio),
                    "output_cv": float(request.query_length_distribution.cv),
                    "is_risky": 1.0 if request.queue_class == QueueClass.RISKY else 0.0,
                }
            )
            self.pending_index += 1
            self._record_event(
                {
                    "time_ms": round(request.arrival_ms, 3),
                    "event": "arrival",
                    "request_id": request.request_id,
                    "template_id": request.template_id,
                    "template_family": request.template_family,
                    "template_version": request.template_version,
                    "queue": request.queue_class.value,
                    "query_distribution_source": request.query_distribution_source,
                    "query_distribution_adapter": request.query_distribution_adapter_name,
                    "prefill_router": request.prefill_route_name,
                    "distribution_source": request.distribution_source,
                    "prompt_tokens": request.prompt_tokens,
                    "shared_prefix_id": request.shared_prefix_id,
                    "shared_prefix_tokens": request.shared_prefix_tokens,
                    "cacheable_shared_prefix_tokens": request.cacheable_shared_prefix_tokens(
                        self.prefix_cache_block_size_tokens
                    ),
                    "uncacheable_shared_prefix_tokens": request.uncacheable_shared_prefix_tokens(
                        self.prefix_cache_block_size_tokens
                    ),
                    "private_prompt_tokens": request.private_prompt_tokens,
                    "prefix_cache_status": request.prefix_cache_status,
                    "max_new_tokens": request.max_new_tokens,
                    "structured_output": int(request.structured_output),
                    "stop_at_eos": int(request.stop_at_eos),
                    "stop_strings": list(request.stop_strings),
                    "base_output_mean": round(request.length_distribution.mean, 3),
                    "base_output_p90": round(request.length_distribution.p90, 3),
                    "query_output_mean": round(request.query_length_distribution.mean, 3),
                    "query_output_p90": round(request.query_length_distribution.p90, 3),
                    "output_mean": round(request.effective_length_distribution.mean, 3),
                    "output_p90": round(request.effective_length_distribution.p90, 3),
                },
            )

    def _apply_decision(self, decision: ScheduleDecision) -> Dict[str, float]:
        self._batch_protected_ids = set()
        if self.config.execution_mode == "scheduled-tokens":
            selected = decision.prefill_requests + decision.decode_requests
            ids = [request.request_id for request, _ in selected]
            if len(ids) != len(set(ids)):
                raise ValueError("a request may appear only once per scheduling iteration")
            if any(not isinstance(tokens, int) or tokens <= 0 for _, tokens in selected):
                raise ValueError("scheduled token counts must be positive integers")
            if sum(tokens for _, tokens in selected) > self.config.token_budget_per_step:
                raise ValueError("schedule exceeds token_budget_per_step")
        if self.timing_model is not None and any(tokens != 1 for _, tokens in decision.decode_requests):
            raise ValueError("profiled timing requires one decode token per request per iteration")
        timing_contexts = {
            request.request_id: request.logical_kv_tokens
            for request, _ in decision.prefill_requests + decision.decode_requests
        } if self.timing_model is not None else {}
        current_kv = self.current_kv_tokens
        current_logical_kv = self.current_logical_kv_tokens
        logical_kv_high_water = current_logical_kv
        action_quantile = (
            decision.scheduler_action.shadow_quantile
            if decision.scheduler_action is not None and decision.scheduler_action.shadow_quantile is not None
            else self.config.kv_reservation_quantile
        )
        newly_admitted_candidates = [
            request
            for request, _ in decision.prefill_requests
            if request.batch_admitted_at_ms is None
        ]
        projected_new_admission_prompt_tokens = sum(
            self.quantize_kv_tokens(request.prompt_tokens)
            for request in newly_admitted_candidates
        )
        projected_new_admission_shadow_tokens = sum(
            self.quantize_kv_tokens(request.shadow_kv_target(action_quantile))
            for request in newly_admitted_candidates
        )
        projected_new_admission_decode_shadow_tokens = sum(
            max(
                0,
                self.quantize_kv_tokens(request.shadow_kv_target(action_quantile))
                - self.quantize_kv_tokens(request.prompt_tokens),
            )
            for request in newly_admitted_candidates
        )
        current_shadow_kv = self.current_shadow_kv_commitment(action_quantile)
        capacity = max(1, int(self.config.kv_capacity_tokens))
        projected_real_kv_util_after_new_admissions = (
            float(current_kv + projected_new_admission_prompt_tokens) / float(capacity)
        )
        projected_shadow_kv_util_after_new_admissions = (
            float(current_shadow_kv + projected_new_admission_shadow_tokens) / float(capacity)
        )
        first_tokens_emitted = 0
        first_tokens_by_queue = {queue: 0 for queue in QueueClass}
        completed_requests = 0
        prefill_completed_requests = 0
        new_admissions = 0
        new_admissions_by_queue = {queue: 0 for queue in QueueClass}
        resumed_prefills = 0
        evictions = 0
        reclaimed_kv_tokens = 0
        recompute_tokens_added = 0
        allocation_failures = 0
        actual_prefill_progress_tokens = 0
        actual_decode_progress_tokens = 0
        admission_wait_sum_ms = 0.0
        ttft_sum_ms = 0.0
        e2e_sum_ms = 0.0
        completed_output_tokens = 0.0
        deferred_finish_events: List[Tuple[Dict[str, object], str]] = []
        completed_prefills: List[Request] = []
        newly_admitted_requests: List[Request] = []
        first_token_records: List[Tuple[Request, int]] = []
        completed_decode_records: List[Tuple[Request, int, int]] = []
        prefill_progressed_requests: List[Request] = []
        decode_progressed_requests: List[Request] = []
        prefill_progress_chunks: List[int] = []
        decode_progress_chunks: List[int] = []

        for request, _ in decision.prefill_requests:
            _remove_identity(self.prefill_queues[request.queue_class], request)

        # Canonical resource admission is independent of list order. No completion
        # in this batch may release storage for another member of the same batch.
        for request, scheduled_tokens in sorted(decision.prefill_requests, key=lambda row: row[0].request_id):
            if request.batch_admitted_at_ms is None:
                request.batch_admitted_at_ms = self.now_ms
                newly_admitted_requests.append(request)
                new_admissions += 1
                new_admissions_by_queue[request.queue_class] += 1
                admission_wait_sum_ms += request.admission_wait_ms or 0.0
                self._record_event(
                    {
                        "time_ms": round(self.now_ms, 3),
                        "event": "batch_admit",
                        "request_id": request.request_id,
                        "queue": request.queue_class.value,
                    },
                )
            else:
                resumed_prefills += 1
            if request.prefill_started_at_ms is None:
                request.prefill_started_at_ms = self.now_ms
            upper_bound = request.remaining_prefill_tokens
            if self.config.execution_mode == "scheduled-tokens":
                upper_bound = min(upper_bound, scheduled_tokens)
            chunk = self._resolve_boundary_growth(request, upper_bound)
            if chunk <= 0:
                self.prefill_queues[request.queue_class].append(request)
                continue

            physical_before = self.kv_allocator.request_physical_tokens(request.request_id)
            cacheable_shared_tokens = request.cacheable_shared_prefix_tokens(self.prefix_cache_block_size_tokens)
            shared_chunk = 0
            shared_tail_chunk = 0
            owner_entry = self._prefix_cache_owner_entry(request)
            if owner_entry is not None and request.shared_kv_tokens < cacheable_shared_tokens:
                shared_chunk = min(chunk, cacheable_shared_tokens - request.shared_kv_tokens)
                if shared_chunk > 0:
                    request.prefill_done_tokens += shared_chunk
                    request.shared_kv_tokens += shared_chunk
                    actual_prefill_progress_tokens += shared_chunk
                    owner_entry.cached_blocks = min(
                        owner_entry.total_blocks,
                        request.shared_kv_tokens // self.prefix_cache_block_size_tokens,
                    )
                    owner_entry.state = "warming" if owner_entry.cached_blocks < owner_entry.total_blocks else "ready"
                    owner_entry.last_used_ms = self.now_ms
                    if request.shared_kv_tokens >= cacheable_shared_tokens:
                        self._mark_prefix_cache_warm(request)

            private_chunk_target = max(0, chunk - shared_chunk)
            private_chunk = private_chunk_target
            growth_evictions = 0
            growth_reclaimed = 0
            growth_recompute = 0
            growth_failed = False
            if private_chunk_target > 0:
                shared_tail_done = max(
                    0,
                    min(request.prefill_done_tokens, request.shared_prefix_tokens) - min(request.shared_kv_tokens, request.shared_prefix_tokens),
                )
                shared_tail_remaining = max(0, request.uncacheable_shared_prefix_tokens(self.prefix_cache_block_size_tokens) - shared_tail_done)
                growth_evictions, growth_reclaimed, growth_recompute, growth_failed = self._ensure_kv_growth(
                    request=request,
                    delta_tokens=private_chunk_target,
                    event_time=None,
                    exclude_request_id=None,
                    deferred_events=deferred_finish_events,
                )
                evictions += growth_evictions
                reclaimed_kv_tokens += growth_reclaimed
                recompute_tokens_added += growth_recompute
                allocation_failures += int(growth_failed)
                if growth_failed:
                    private_chunk = 0
                else:
                    shared_tail_chunk = min(private_chunk, shared_tail_remaining)
                    request.prefill_done_tokens += private_chunk
                    request.kv_tokens += private_chunk
                    actual_prefill_progress_tokens += private_chunk

            total_prefill_progress = shared_chunk + private_chunk
            if total_prefill_progress <= 0:
                self.prefill_queues[request.queue_class].append(request)
                continue

            prefill_progressed_requests.append(request)
            self._batch_protected_ids.add(request.request_id)
            prefill_progress_chunks.append(total_prefill_progress)
            physical_after = self.kv_allocator.request_physical_tokens(request.request_id)
            self.peak_kv_tokens = max(self.peak_kv_tokens, self.current_kv_tokens)
            current_logical_kv += total_prefill_progress
            if growth_evictions:
                current_logical_kv = self.current_logical_kv_tokens
            logical_kv_high_water = max(logical_kv_high_water, current_logical_kv)
            deferred_finish_events.append(
                (
                    {
                        "event": "prefill_step",
                        "request_id": request.request_id,
                        "queue": request.queue_class.value,
                        "chunk_tokens": total_prefill_progress,
                        "shared_chunk_tokens": shared_chunk,
                        "shared_tail_chunk_tokens": shared_tail_chunk,
                        "private_chunk_tokens": private_chunk,
                        "kv_logical_delta_tokens": total_prefill_progress,
                        "kv_physical_delta_tokens": max(0, physical_after - physical_before),
                        "prefill_done_tokens": request.prefill_done_tokens,
                        "kv_tokens": request.logical_kv_tokens,
                        "owned_kv_tokens": request.kv_tokens,
                        "shared_kv_tokens": request.shared_kv_tokens,
                        "request_physical_kv_tokens": physical_after,
                        "physical_kv_tokens": self.current_kv_tokens,
                    },
                    "dense",
                )
            )
            if request.remaining_prefill_tokens == 0:
                prefill_completed_requests += 1
                completed_prefills.append(request)
            else:
                self.prefill_queues[request.queue_class].append(request)

        for request, scheduled_tokens in sorted(decision.decode_requests, key=lambda row: row[0].request_id):
            if not _contains_identity(self.active_decode, request) or request.is_complete:
                continue
            upper_bound = request.remaining_output_tokens
            if self.config.execution_mode == "scheduled-tokens":
                upper_bound = min(upper_bound, scheduled_tokens)
            chunk = self._resolve_boundary_growth(request, upper_bound)
            if chunk <= 0:
                continue

            physical_before = self.kv_allocator.request_physical_tokens(request.request_id)
            growth_evictions, growth_reclaimed, growth_recompute, growth_failed = self._ensure_kv_growth(
                request=request,
                delta_tokens=chunk,
                event_time=None,
                exclude_request_id=request.request_id,
                deferred_events=deferred_finish_events,
            )
            evictions += growth_evictions
            reclaimed_kv_tokens += growth_reclaimed
            recompute_tokens_added += growth_recompute
            allocation_failures += int(growth_failed)
            if growth_failed or not _contains_identity(self.active_decode, request) or request.is_complete:
                continue

            prior_generated_tokens = request.generated_tokens
            request.generated_tokens += chunk
            request.kv_tokens += chunk
            actual_decode_progress_tokens += chunk
            decode_progressed_requests.append(request)
            self._batch_protected_ids.add(request.request_id)
            decode_progress_chunks.append(chunk)
            physical_after = self.kv_allocator.request_physical_tokens(request.request_id)
            self.peak_kv_tokens = max(self.peak_kv_tokens, self.current_kv_tokens)
            current_logical_kv += chunk
            if growth_evictions:
                current_logical_kv = self.current_logical_kv_tokens
            logical_kv_high_water = max(logical_kv_high_water, current_logical_kv)
            deferred_finish_events.append(
                (
                    {
                        "event": "decode_step",
                        "request_id": request.request_id,
                        "chunk_tokens": chunk,
                        "generated_tokens": request.generated_tokens,
                        "remaining_output_tokens": request.remaining_output_tokens,
                        "kv_logical_delta_tokens": chunk,
                        "kv_physical_delta_tokens": max(0, physical_after - physical_before),
                        "kv_tokens": request.logical_kv_tokens,
                        "owned_kv_tokens": request.kv_tokens,
                        "shared_kv_tokens": request.shared_kv_tokens,
                        "request_physical_kv_tokens": physical_after,
                        "physical_kv_tokens": self.current_kv_tokens,
                    },
                    "dense",
                )
            )
            if request.first_token_at_ms is None and chunk > 0 and prior_generated_tokens == 0:
                first_token_records.append((request, chunk))
                first_tokens_emitted += 1
                first_tokens_by_queue[request.queue_class] += 1
            if request.remaining_output_tokens == 0:
                completed_requests += 1
                completed_output_tokens += float(request.actual_output_tokens)
                released_logical_kv = request.logical_kv_tokens
                released_physical_kv = self.kv_allocator.request_physical_tokens(request.request_id)
                _remove_identity(self.active_decode, request)
                completed_decode_records.append((request, released_logical_kv, released_physical_kv))

        if self.config.overflow_policy == "evict-recompute" and self.current_kv_tokens > self.config.kv_capacity_tokens:
            overflow_evictions, overflow_reclaimed, overflow_recompute = self._resolve_overflow(
                event_time=None,
                deferred_events=deferred_finish_events,
            )
            evictions += overflow_evictions
            reclaimed_kv_tokens += overflow_reclaimed
            recompute_tokens_added += overflow_recompute

        prefill_tokens = sum(prefill_progress_chunks)
        prefill_batch_size = len(prefill_progress_chunks)
        max_prefill_chunk = max(prefill_progress_chunks, default=0)
        padded_prefill_tokens = max_prefill_chunk * prefill_batch_size
        prefill_padding_tokens = max(0, padded_prefill_tokens - prefill_tokens)
        decode_tokens = sum(decode_progress_chunks)
        duration_ms = self._duration_value(prefill_tokens, prefill_padding_tokens, decode_tokens)
        timing_warnings = ()
        if self.timing_model is not None:
            duration_ms, timing_warnings = self.timing_model.estimate(TimingBatch(
                tuple(prefill_progress_chunks),
                tuple(timing_contexts[r.request_id] for r in prefill_progressed_requests),
                tuple(timing_contexts[r.request_id] for r in decode_progressed_requests),
            ))
        duration_ms += self.config.decision_overhead_ms
        finish_time = self.now_ms + duration_ms

        for request in prefill_progressed_requests:
            request.last_service_at_ms = finish_time
        for request in decode_progressed_requests:
            request.last_service_at_ms = finish_time

        for request in completed_prefills:
            if request.active_prefill_target_tokens == request.prompt_tokens:
                request.prefill_completed_at_ms = finish_time
            self._resolve_output_length(request, finish_time)
            self.scheduler.route_decode_request(request, self)
            if not _contains_identity(self.active_decode, request):
                self.active_decode.append(request)
            if self.config.first_token_mode == "prefill" and request.generated_tokens == 0:
                # The last prompt forward pass samples token 1; its KV is only
                # computed if a later decode consumes that sampled token.
                request.generated_tokens = 1
                request.first_decode_started_at_ms = finish_time
                first_token_records.append((request, 1))
                first_tokens_emitted += 1
                first_tokens_by_queue[request.queue_class] += 1
                if request.remaining_output_tokens == 0:
                    completed_requests += 1
                    completed_output_tokens += float(request.actual_output_tokens)
                    completed_decode_records.append((request, request.logical_kv_tokens,
                        self.kv_allocator.request_physical_tokens(request.request_id)))
                    _remove_identity(self.active_decode, request)
            deferred_finish_events.append(
                (
                    {
                        "event": (
                            "prefill_complete"
                            if request.active_prefill_target_tokens == request.prompt_tokens
                            else "recompute_complete"
                        ),
                        "request_id": request.request_id,
                        "queue": request.queue_class.value,
                        "decode_router": request.decode_route_name,
                        "decode_batch_tag": request.decode_route_metadata.get("batch_tag", "default"),
                        "actual_output_tokens": request.actual_output_tokens,
                        "output_length_source": request.output_length_source,
                        "kv_tokens": request.logical_kv_tokens,
                        "owned_kv_tokens": request.kv_tokens,
                        "shared_kv_tokens": request.shared_kv_tokens,
                        "physical_kv_tokens": self.current_kv_tokens,
                    },
                    "important",
                )
            )

        for request, chunk in first_token_records:
            request.first_token_at_ms = finish_time - self._decode_duration_value(max(0, chunk - 1))
            if request.first_decode_started_at_ms is None:
                request.first_decode_started_at_ms = max(
                    request.prefill_completed_at_ms or request.arrival_ms,
                    self.now_ms if self.timing_model is not None
                    else request.first_token_at_ms - self._decode_duration_value(1),
                )
            ttft_sum_ms += request.ttft_ms or 0.0
            self._record_event(
                {
                    "time_ms": round(request.first_token_at_ms, 3),
                    "event": "first_token",
                    "request_id": request.request_id,
                    "ttft_ms": round(request.ttft_ms or 0.0, 3),
                },
            )

        for request, released_logical_kv, released_physical_kv in completed_decode_records:
            self._release_prefix_cache_reference(request)
            self.kv_allocator.release(request.request_id)
            request.kv_tokens = 0
            request.shared_kv_tokens = 0
            request.finished_at_ms = finish_time
            e2e_sum_ms += request.e2e_ms or 0.0
            self.completed_request_metrics.append(
                self.metrics_collector.record_completion(
                    request,
                    released_logical_kv_tokens=released_logical_kv,
                    released_physical_kv_tokens=released_physical_kv,
                )
            )
            deferred_finish_events.append(
                (
                    {
                        "event": "complete",
                        "request_id": request.request_id,
                        "e2e_ms": round(request.e2e_ms or 0.0, 3),
                        "released_kv_tokens": released_logical_kv,
                        "released_physical_kv_tokens": released_physical_kv,
                    },
                    "important",
                )
            )

        for event, importance in deferred_finish_events:
            event["time_ms"] = round(finish_time, 3)
            self._record_event(event, importance=importance)

        self.peak_kv_tokens = max(self.peak_kv_tokens, current_kv, self.current_kv_tokens)
        self.peak_logical_kv_tokens = max(
            self.peak_logical_kv_tokens,
            logical_kv_high_water,
            current_logical_kv,
            self.current_logical_kv_tokens,
        )
        self._update_allocator_peak_metrics()
        self.now_ms = finish_time
        self._batch_protected_ids = set()
        pending_unadmitted_after = [
            request
            for queue in QueueClass
            for request in self.prefill_queues[queue]
            if request.batch_admitted_at_ms is None
        ]
        pending_unadmitted_after_by_queue = {
            queue: [
                request
                for request in self.prefill_queues[queue]
                if request.batch_admitted_at_ms is None
            ]
            for queue in QueueClass
        }
        pending_unadmitted_ages = [self.now_ms - request.arrival_ms for request in pending_unadmitted_after]
        pending_unadmitted_ages_by_queue = {
            queue: [self.now_ms - request.arrival_ms for request in requests]
            for queue, requests in pending_unadmitted_after_by_queue.items()
        }
        first_token_pending_after = [
            request for request in self.active_decode if not request.is_complete and request.generated_tokens == 0
        ]
        first_token_pending_after_ages = [self.now_ms - request.arrival_ms for request in first_token_pending_after]
        evict_recompute_request_ids = [
            int(event.get("request_id"))
            for event, _importance in deferred_finish_events
            if event.get("event") == "evict_recompute" and event.get("request_id") is not None
        ]

        return {
            "phase_mode": decision.phase_mode.value,
            "decision_time_ms": float(self.now_ms - duration_ms),
            "current_time_ms": float(self.now_ms),
            "duration_ms": duration_ms,
            "timing_profile_used": self.timing_model is not None,
            "timing_domain_warnings": ",".join(timing_warnings),
            "prefill_tokens": float(prefill_tokens),
            "prefill_padding_tokens": float(prefill_padding_tokens),
            "padded_prefill_tokens": float(padded_prefill_tokens),
            "decode_tokens": float(decode_tokens),
            "first_tokens_emitted": float(first_tokens_emitted),
            "new_admission_request_ids": _request_id_list(newly_admitted_requests),
            "prefill_progress_request_ids": _request_id_list(prefill_progressed_requests),
            "prefill_complete_request_ids": _request_id_list(completed_prefills),
            "decode_progress_request_ids": _request_id_list(decode_progressed_requests),
            "first_token_request_ids": _request_id_list(request for request, _chunk in first_token_records),
            "complete_request_ids": _request_id_list(
                request for request, _released_logical_kv, _released_physical_kv in completed_decode_records
            ),
            "evicted_request_ids": ",".join(str(request_id) for request_id in evict_recompute_request_ids),
            "recomputed_request_ids": ",".join(str(request_id) for request_id in evict_recompute_request_ids),
            "first_tokens_fast": float(first_tokens_by_queue[QueueClass.FAST]),
            "first_tokens_normal": float(first_tokens_by_queue[QueueClass.NORMAL]),
            "first_tokens_risky": float(first_tokens_by_queue[QueueClass.RISKY]),
            "completed_requests": float(completed_requests),
            "prefill_completed_requests": float(prefill_completed_requests),
            "completed_output_tokens": float(completed_output_tokens),
            "new_admissions": float(new_admissions),
            "new_admissions_fast": float(new_admissions_by_queue[QueueClass.FAST]),
            "new_admissions_normal": float(new_admissions_by_queue[QueueClass.NORMAL]),
            "new_admissions_risky": float(new_admissions_by_queue[QueueClass.RISKY]),
            "resumed_prefills": float(resumed_prefills),
            "evictions": float(evictions),
            "reclaimed_kv_tokens": float(reclaimed_kv_tokens),
            "recompute_tokens_added": float(recompute_tokens_added),
            "allocation_failures": float(allocation_failures),
            "actual_prefill_progress_tokens": float(actual_prefill_progress_tokens),
            "actual_decode_progress_tokens": float(actual_decode_progress_tokens),
            "admission_wait_sum_ms": admission_wait_sum_ms,
            "ttft_sum_ms": ttft_sum_ms,
            "e2e_sum_ms": e2e_sum_ms,
            "kv_before": float(current_kv),
            "kv_after": float(self.current_kv_tokens),
            "kv_peak": float(self.peak_kv_tokens),
            "logical_kv_after": float(self.current_logical_kv_tokens),
            "projected_new_admission_prompt_tokens": float(projected_new_admission_prompt_tokens),
            "projected_new_admission_decode_shadow_tokens": float(projected_new_admission_decode_shadow_tokens),
            "projected_new_admission_shadow_tokens": float(projected_new_admission_shadow_tokens),
            "projected_real_kv_util_after_new_admissions": float(projected_real_kv_util_after_new_admissions),
            "projected_shadow_kv_util_after_new_admissions": float(projected_shadow_kv_util_after_new_admissions),
            "active_decode_after": float(len(self.active_decode)),
            "prefill_waiting_after": float(sum(len(queue) for queue in self.prefill_queues.values())),
            "pending_unadmitted_after": float(len(pending_unadmitted_after)),
            "pending_unadmitted_fast_after": float(len(pending_unadmitted_after_by_queue[QueueClass.FAST])),
            "pending_unadmitted_normal_after": float(len(pending_unadmitted_after_by_queue[QueueClass.NORMAL])),
            "pending_unadmitted_risky_after": float(len(pending_unadmitted_after_by_queue[QueueClass.RISKY])),
            "oldest_pending_admission_after_ms": float(max(pending_unadmitted_ages, default=0.0)),
            "oldest_pending_admission_fast_after_ms": float(
                max(pending_unadmitted_ages_by_queue[QueueClass.FAST], default=0.0)
            ),
            "oldest_pending_admission_normal_after_ms": float(
                max(pending_unadmitted_ages_by_queue[QueueClass.NORMAL], default=0.0)
            ),
            "oldest_pending_admission_risky_after_ms": float(
                max(pending_unadmitted_ages_by_queue[QueueClass.RISKY], default=0.0)
            ),
            "mean_pending_admission_after_ms": float(
                statistics.fmean(pending_unadmitted_ages) if pending_unadmitted_ages else 0.0
            ),
            "first_token_pending_after": float(len(first_token_pending_after)),
            "oldest_first_token_pending_after_ms": float(max(first_token_pending_after_ages, default=0.0)),
            "scheduler_admission_profile": (
                decision.scheduler_action.admission_profile
                if decision.scheduler_action is not None
                else "balanced"
            ),
            "requested_scheduler_admission_profile": (
                decision.requested_scheduler_action.admission_profile
                if decision.requested_scheduler_action is not None
                else "balanced"
            ),
            "scheduler_new_admission_limit": (
                decision.scheduler_action.new_admission_limit
                if decision.scheduler_action is not None and decision.scheduler_action.new_admission_limit is not None
                else self.config.max_prefill_batch
            ),
            "requested_scheduler_new_admission_limit": (
                decision.requested_scheduler_action.new_admission_limit
                if decision.requested_scheduler_action is not None
                and decision.requested_scheduler_action.new_admission_limit is not None
                else self.config.max_prefill_batch
            ),
            "scheduler_shadow_quantile": (
                decision.scheduler_action.shadow_quantile
                if decision.scheduler_action is not None and decision.scheduler_action.shadow_quantile is not None
                else self.config.kv_reservation_quantile
            ),
            "requested_scheduler_shadow_quantile": (
                decision.requested_scheduler_action.shadow_quantile
                if decision.requested_scheduler_action is not None
                and decision.requested_scheduler_action.shadow_quantile is not None
                else self.config.kv_reservation_quantile
            ),
            "scheduler_budget_profile": (
                decision.scheduler_action.budget_profile
                if decision.scheduler_action is not None
                else "balanced"
            ),
            "requested_scheduler_budget_profile": (
                decision.requested_scheduler_action.budget_profile
                if decision.requested_scheduler_action is not None
                else "balanced"
            ),
            "scheduler_risky_admission_cap": (
                decision.scheduler_action.risky_admission_cap
                if decision.scheduler_action is not None and decision.scheduler_action.risky_admission_cap is not None
                else int(self.config.prefill_parallelism or self.config.max_prefill_batch)
            ),
            "scheduler_prefill_safety_filter_enabled": (
                float(bool(decision.scheduler_action.prefill_safety_filter_enabled))
                if decision.scheduler_action is not None
                and decision.scheduler_action.prefill_safety_filter_enabled is not None
                else 1.0
            ),
            "scheduler_decode_safety_filter_enabled": (
                float(bool(decision.scheduler_action.decode_safety_filter_enabled))
                if decision.scheduler_action is not None
                and decision.scheduler_action.decode_safety_filter_enabled is not None
                else 1.0
            ),
            "requested_scheduler_risky_admission_cap": (
                decision.requested_scheduler_action.risky_admission_cap
                if decision.requested_scheduler_action is not None
                and decision.requested_scheduler_action.risky_admission_cap is not None
                else int(self.config.prefill_parallelism or self.config.max_prefill_batch)
            ),
            "requested_scheduler_prefill_safety_filter_enabled": (
                float(bool(decision.requested_scheduler_action.prefill_safety_filter_enabled))
                if decision.requested_scheduler_action is not None
                and decision.requested_scheduler_action.prefill_safety_filter_enabled is not None
                else 1.0
            ),
            "requested_scheduler_decode_safety_filter_enabled": (
                float(bool(decision.requested_scheduler_action.decode_safety_filter_enabled))
                if decision.requested_scheduler_action is not None
                and decision.requested_scheduler_action.decode_safety_filter_enabled is not None
                else 1.0
            ),
            "requested_phase_mode": (
                decision.requested_scheduler_action.phase_preference.value
                if decision.requested_scheduler_action is not None
                and decision.requested_scheduler_action.phase_preference is not None
                else "heuristic"
            ),
            "action_safety_triggered": float(decision.action_safety_triggered),
            "action_safety_reasons": ",".join(decision.action_safety_reasons),
        }

    def _resolve_boundary_growth(self, request: Request, upper_bound: int) -> int:
        if upper_bound <= 0:
            return 0
        if self.config.overflow_policy != "block":
            return upper_bound
        return self._fit_growth_to_capacity(request, upper_bound)

    def _fit_growth_to_capacity(self, request: Request, upper_bound: int) -> int:
        if upper_bound <= 0:
            return 0
        low = 1
        high = upper_bound
        best = 0
        while low <= high:
            mid = (low + high) // 2
            private_growth = max(0, mid - self._reserved_prefill_growth(request))
            attempt = self.kv_allocator.estimate_growth(request.request_id, request.kv_tokens, private_growth)
            if (
                attempt.success
                and attempt.physical_tokens_after + self.prefix_cache_physical_tokens
                    - self.reclaimable_prefix_tokens <= self.config.kv_capacity_tokens
            ):
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best

    def _ensure_kv_growth(
        self,
        request: Request,
        delta_tokens: int,
        event_time: Optional[float],
        exclude_request_id: Optional[int],
        deferred_events: Optional[List[Tuple[Dict[str, object], str]]] = None,
    ) -> Tuple[int, int, int, bool]:
        if delta_tokens <= 0:
            return (0, 0, 0, False)
        attempt = self._apply_kv_growth(request, delta_tokens)
        if attempt.success and self.current_kv_tokens <= self.config.kv_capacity_tokens:
            return (0, 0, 0, False)
        if attempt.success and self.config.overflow_policy != "evict-recompute":
            self.kv_allocator.release(request.request_id)
            if request.kv_tokens > 0:
                self.kv_allocator.apply_growth(request.request_id, 0, request.kv_tokens)
            return (0, 0, 0, True)
        if self.config.overflow_policy != "evict-recompute":
            return (0, 0, 0, True)

        evictions = 0
        reclaimed = 0
        recompute_added = 0
        while not attempt.success:
            victim = self._choose_eviction_victim(exclude_request_id=exclude_request_id)
            if victim is None:
                return (evictions, reclaimed, recompute_added, True)
            rebuild_tokens = victim.kv_tokens
            evicted_kv = victim.kv_tokens
            victim.eviction_count += 1
            victim.recompute_count += 1
            victim.recompute_tokens += rebuild_tokens
            victim.wasted_kv_tokens += evicted_kv
            victim.prefill_target_tokens = victim.shared_kv_tokens + rebuild_tokens
            victim.prefill_done_tokens = victim.shared_kv_tokens
            self.kv_allocator.release(victim.request_id)
            victim.kv_tokens = 0
            self.active_decode.remove(victim)
            self.prefill_queues[victim.queue_class].append(victim)
            evictions += 1
            reclaimed += evicted_kv
            recompute_added += rebuild_tokens
            event_payload = {
                "event": "evict_recompute",
                "request_id": victim.request_id,
                "queue": victim.queue_class.value,
                "generated_tokens": victim.generated_tokens,
                "evicted_kv_tokens": evicted_kv,
                "recompute_tokens": rebuild_tokens,
                "eviction_count": victim.eviction_count,
                "allocator_mode": self.config.kv_allocator_mode,
            }
            if deferred_events is not None:
                deferred_events.append((event_payload, "important"))
            elif event_time is not None:
                self._record_event({"time_ms": round(event_time, 3), **event_payload})
            attempt = self._apply_kv_growth(request, delta_tokens)
            if attempt.success and self.current_kv_tokens <= self.config.kv_capacity_tokens:
                break
        return (evictions, reclaimed, recompute_added, False)

    def _apply_kv_growth(self, request: Request, delta_tokens: int):
        preview = self.kv_allocator.estimate_growth(request.request_id, request.kv_tokens, delta_tokens)
        if (preview.physical_tokens_after + self.prefix_cache_physical_tokens > self.config.kv_capacity_tokens
                and self.config.prefix_cache_enable_lru_eviction):
            self._evict_prefix_cache_entries(max(1, preview.physical_delta_tokens),
                                            exclude_keys={self._prefix_cache_key(request)})
            preview = self.kv_allocator.estimate_growth(request.request_id, request.kv_tokens, delta_tokens)
        if self.config.execution_mode == "scheduled-tokens":
            if preview.physical_tokens_after + self.prefix_cache_physical_tokens > self.config.kv_capacity_tokens:
                preview.success = False
                preview.reason = "physical_capacity_exceeded"
                return preview
        return self.kv_allocator.apply_growth(request.request_id, request.kv_tokens, delta_tokens)

    def _choose_eviction_victim(self, exclude_request_id: Optional[int]) -> Optional[Request]:
        candidates = [
            request
            for request in self.active_decode
            if exclude_request_id is None or request.request_id != exclude_request_id
            if request.request_id not in self._batch_protected_ids
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda request: (
                request.kv_tokens,
                request.estimated_remaining_output_tokens(),
                request.generated_tokens,
            ),
        )

    def _resolve_overflow(
        self,
        event_time: Optional[float],
        deferred_events: Optional[List[Tuple[Dict[str, object], str]]] = None,
    ) -> Tuple[int, int, int]:
        evictions = 0
        reclaimed = 0
        recompute_added = 0
        while self.current_kv_tokens > self.config.kv_capacity_tokens and self.active_decode:
            victim = self._choose_eviction_victim(exclude_request_id=None)
            if victim is None:
                break
            rebuild_tokens = victim.kv_tokens
            evicted_kv = victim.kv_tokens
            victim.eviction_count += 1
            victim.recompute_count += 1
            victim.recompute_tokens += rebuild_tokens
            victim.wasted_kv_tokens += evicted_kv
            victim.prefill_target_tokens = victim.shared_kv_tokens + rebuild_tokens
            victim.prefill_done_tokens = victim.shared_kv_tokens
            self.kv_allocator.release(victim.request_id)
            victim.kv_tokens = 0
            self.active_decode.remove(victim)
            self.prefill_queues[victim.queue_class].append(victim)
            evictions += 1
            reclaimed += evicted_kv
            recompute_added += rebuild_tokens
            event_payload = {
                "event": "evict_recompute",
                "request_id": victim.request_id,
                "queue": victim.queue_class.value,
                "generated_tokens": victim.generated_tokens,
                "evicted_kv_tokens": evicted_kv,
                "recompute_tokens": rebuild_tokens,
                "eviction_count": victim.eviction_count,
                "allocator_mode": self.config.kv_allocator_mode,
            }
            if deferred_events is not None:
                deferred_events.append((event_payload, "important"))
            elif event_time is not None:
                self._record_event({"time_ms": round(event_time, 3), **event_payload})
        return evictions, reclaimed, recompute_added

    def _next_arrival_time(self) -> Optional[float]:
        if self.pending_index >= len(self.requests):
            return None
        return self.requests[self.pending_index].arrival_ms

    def _recover_unschedulable_state(self) -> bool:
        started_prefills = [
            request
            for queue in QueueClass
            for request in self.prefill_queues[queue]
            if request.batch_admitted_at_ms is not None and not request.is_complete and request.logical_kv_tokens > 0
        ]
        if not started_prefills:
            return False
        victim = max(
            started_prefills,
            key=lambda request: (request.kv_tokens, request.remaining_prefill_tokens, -request.prefill_done_tokens),
        )
        reclaimed_kv = victim.kv_tokens
        recompute_tokens = victim.kv_tokens
        if reclaimed_kv <= 0 and victim.shared_kv_tokens <= 0:
            return False
        victim.recompute_count += 1
        victim.recompute_tokens += recompute_tokens
        victim.wasted_kv_tokens += reclaimed_kv
        victim.prefill_done_tokens = victim.shared_kv_tokens
        victim.prefill_target_tokens = victim.active_prefill_target_tokens
        victim.last_service_at_ms = self.now_ms
        self.kv_allocator.release(victim.request_id)
        victim.kv_tokens = 0
        self.deadlock_recoveries_total += 1
        self._record_event(
            {
                "time_ms": round(self.now_ms, 3),
                "event": "prefill_reset",
                "request_id": victim.request_id,
                "queue": victim.queue_class.value,
                "reclaimed_kv_tokens": reclaimed_kv,
                "recompute_tokens": recompute_tokens,
                "reason": "unschedulable_prefill_deadlock",
                "allocator_mode": self.config.kv_allocator_mode,
            },
        )
        return True

    def _is_finished(self) -> bool:
        if self.pending_index < len(self.requests):
            return False
        if any(self.prefill_queues.values()):
            return False
        return not self.active_decode

    def _build_summary(self) -> Dict[str, object]:
        ttfts = [request.ttft_ms for request in self.requests if request.ttft_ms is not None]
        e2es = [request.e2e_ms for request in self.requests if request.e2e_ms is not None]
        waits = [request.admission_wait_ms for request in self.requests if request.admission_wait_ms is not None]
        prefill_compute = [request.prefill_compute_ms for request in self.requests if request.prefill_compute_ms is not None]
        decode_to_first = [
            request.decode_to_first_token_ms for request in self.requests if request.decode_to_first_token_ms is not None
        ]
        queue_mix = {queue.value: sum(1 for request in self.requests if request.queue_class == queue) for queue in QueueClass}
        allocator_metrics = self.kv_allocator.metrics(
            {
                request.request_id: request.kv_tokens
                for request in self.requests
                if not request.is_complete and request.kv_tokens > 0
            }
        )
        unfinished = [request for request in self.requests if not request.is_complete]
        unfinished_requests = [request.request_id for request in unfinished]
        unfinished_age_grace_ms = max(0.0, float(self.config.unfinished_age_grace_ms or 0.0))
        termination_reason = str(self.termination_reason or "")
        deadlock_like = termination_reason.startswith("deadlock")
        horizon_like = termination_reason == "simulation_duration_elapsed"
        horizon_unfinished_request_ids: List[int] = []
        aged_unfinished_request_ids: List[int] = []
        liveness_unfinished_request_ids: List[int] = []
        unfinished_request_ages_ms: Dict[int, float] = {}
        for request in unfinished:
            age_ms = max(0.0, float(self.now_ms) - float(request.arrival_ms))
            unfinished_request_ages_ms[request.request_id] = round(age_ms, 3)
            if deadlock_like:
                liveness_unfinished_request_ids.append(request.request_id)
            elif horizon_like:
                if unfinished_age_grace_ms > 0.0 and age_ms >= unfinished_age_grace_ms:
                    aged_unfinished_request_ids.append(request.request_id)
                else:
                    horizon_unfinished_request_ids.append(request.request_id)
            else:
                aged_unfinished_request_ids.append(request.request_id)
        punishable_unfinished_request_ids = sorted(
            set(aged_unfinished_request_ids) | set(liveness_unfinished_request_ids)
        )
        if self.termination_reason == "running":
            self.termination_reason = "completed_all_requests" if not unfinished_requests else "unknown_termination"
        summary = {
            "requests": len(self.requests),
            "completed": sum(1 for request in self.requests if request.is_complete),
            "unfinished_requests": len(unfinished_requests),
            "unfinished_request_ids": unfinished_requests,
            "horizon_unfinished_requests": len(horizon_unfinished_request_ids),
            "horizon_unfinished_request_ids": horizon_unfinished_request_ids,
            "aged_unfinished_requests": len(aged_unfinished_request_ids),
            "aged_unfinished_request_ids": aged_unfinished_request_ids,
            "liveness_unfinished_requests": len(liveness_unfinished_request_ids),
            "liveness_unfinished_request_ids": liveness_unfinished_request_ids,
            "punishable_unfinished_requests": len(punishable_unfinished_request_ids),
            "punishable_unfinished_request_ids": punishable_unfinished_request_ids,
            "unfinished_request_ages_ms": unfinished_request_ages_ms,
            "unfinished_age_grace_ms": round(unfinished_age_grace_ms, 3),
            "terminated_early": bool(unfinished_requests),
            "termination_reason": self.termination_reason,
            "termination_details": self.termination_details,
            "deadlock_recoveries_total": self.deadlock_recoveries_total,
            "simulation_steps": self.simulation_steps,
            "wall_time_seconds": round(self.wall_time_seconds, 3),
            "total_runtime_ms": round(self.now_ms, 3),
            "peak_kv_tokens": self.peak_kv_tokens,
            "kv_capacity_tokens": self.config.kv_capacity_tokens,
            "peak_logical_kv_tokens": self.peak_logical_kv_tokens,
            "prefix_cache_physical_tokens": self.prefix_cache_physical_tokens,
            "prefix_cache_enabled": self.config.prefix_cache_enabled,
            "prefix_cache_cached_tokens": sum(entry.cached_tokens for entry in self.prefix_cache.values()),
            "prefix_cache_entries": len(self.prefix_cache),
            "prefix_cache_warm_entries": sum(1 for entry in self.prefix_cache.values() if entry.warm),
            "prefix_cache_warming_entries": sum(1 for entry in self.prefix_cache.values() if entry.state == "warming"),
            "prefix_cache_allocating_entries": sum(1 for entry in self.prefix_cache.values() if entry.state == "allocating"),
            "prefix_cache_cached_blocks": sum(entry.cached_blocks for entry in self.prefix_cache.values()),
            "prefix_cache_total_blocks": sum(entry.total_blocks for entry in self.prefix_cache.values()),
            "prefix_cache_hit_requests": self.prefix_cache_hit_requests,
            "prefix_cache_miss_requests": self.prefix_cache_miss_requests,
            "prefix_cache_bypass_requests": self.prefix_cache_bypass_requests,
            "prefix_cache_reuse_tokens": self.prefix_cache_reuse_tokens,
            "prefix_cache_evictions_total": self.prefix_cache_evictions_total,
            "prefix_cache_peak_active_references": self.prefix_cache_peak_active_references,
            "peak_allocator_internal_fragmentation_tokens": round(
                self.peak_allocator_internal_fragmentation_tokens,
                3,
            ),
            "peak_allocator_external_fragmentation_tokens": round(
                self.peak_allocator_external_fragmentation_tokens,
                3,
            ),
            "kv_cache_scope": self.config.kv_cache_scope,
            "time_unit": self.config.time_unit,
            "execution_mode": self.config.execution_mode,
            "first_token_mode": self.config.first_token_mode,
            "decision_overhead_ms": self.config.decision_overhead_ms,
            "simulation_duration_ms": round(float(self.config.simulation_duration_ms or 0.0), 3),
            "event_record_mode": self.config.event_record_mode,
            "kv_allocator_mode": self.config.kv_allocator_mode,
            "kv_block_size_tokens": self.config.kv_block_size_tokens,
            "prefix_cache_block_size_tokens": self.prefix_cache_block_size_tokens,
            "decode_chunk_size_tokens": self.config.decode_chunk_size_tokens,
            "prefill_parallelism": int(self.config.prefill_parallelism or self.config.max_prefill_batch),
            "decode_parallelism": int(self.config.decode_parallelism or self.config.max_decode_batch),
            "query_distribution_mode": self.config.query_distribution_mode,
            "query_distribution_adapter": getattr(
                self.scheduler.query_distribution_adapter,
                "name",
                type(self.scheduler.query_distribution_adapter).__name__,
            ),
            "prefill_router": getattr(self.scheduler.prefill_router, "name", type(self.scheduler.prefill_router).__name__),
            "decode_router": getattr(self.scheduler.decode_router, "name", type(self.scheduler.decode_router).__name__),
            "metrics_collector": getattr(self.metrics_collector, "name", type(self.metrics_collector).__name__),
            "queue_mix": queue_mix,
            "phase_counts": self.phase_counts,
            "evictions_total": sum(request.eviction_count for request in self.requests),
            "evicted_requests": sum(1 for request in self.requests if request.eviction_count > 0),
            "recompute_tokens_total": sum(request.recompute_tokens for request in self.requests),
            "wasted_kv_tokens_total": sum(request.wasted_kv_tokens for request in self.requests),
            **allocator_metrics,
            "ttft_ms": self._metric_block(ttfts),
            "e2e_ms": self._metric_block(e2es),
            "admission_wait_ms": self._metric_block(waits),
            "prefill_compute_ms": self._metric_block(prefill_compute),
            "decode_to_first_token_ms": self._metric_block(decode_to_first),
        }
        summary["output_metrics"] = self.metrics_collector.build_summary(summary, self.completed_request_metrics)
        return summary

    def _resolve_output_length(self, request: Request, event_time: float) -> None:
        if request.output_length_source == "sampled" or request.output_length_sampled_at_ms is not None:
            return
        if request.actual_output_tokens > 0 and not self.config.force_output_length_sampling:
            if request.max_new_tokens > 0:
                request.actual_output_tokens = min(request.actual_output_tokens, request.max_new_tokens)
            request.output_length_sampled_at_ms = event_time
            return
        seed = f"{self.config.runtime_sampling_seed}|{request.model_id}|{request.request_id}|output"
        rng = random.Random(int.from_bytes(hashlib.sha256(seed.encode()).digest()[:16], "big"))
        distribution = self._true_length_distributions[request.request_id]
        request.actual_output_tokens = max(1, distribution.sample(rng))
        if request.max_new_tokens > 0:
            request.actual_output_tokens = min(request.actual_output_tokens, request.max_new_tokens)
        request.output_length_source = "sampled"
        request.output_length_sampled_at_ms = event_time

    def _build_termination_details(self, phase_mode: str) -> Dict[str, object]:
        pending_unadmitted = [
            request
            for queue in QueueClass
            for request in self.prefill_queues[queue]
            if request.batch_admitted_at_ms is None
        ]
        started_prefill_waiting = [
            request
            for queue in QueueClass
            for request in self.prefill_queues[queue]
            if request.batch_admitted_at_ms is not None
        ]
        return {
            "phase_mode": phase_mode,
            "unfinished_requests": sum(1 for request in self.requests if not request.is_complete),
            "pending_unadmitted": len(pending_unadmitted),
            "started_prefill_waiting": len(started_prefill_waiting),
            "active_decode": len([request for request in self.active_decode if not request.is_complete]),
            "current_kv_tokens": self.current_kv_tokens,
            "current_logical_kv_tokens": self.current_logical_kv_tokens,
            "prefix_cache_physical_tokens": self.prefix_cache_physical_tokens,
            "prefix_cache_entries": len(self.prefix_cache),
            "pending_index": self.pending_index,
            "deadlock_recoveries_total": self.deadlock_recoveries_total,
        }

    def _update_allocator_peak_metrics(self) -> None:
        allocator_metrics = self.kv_allocator.metrics(
            {
                request.request_id: request.kv_tokens
                for request in self.requests
                if not request.is_complete and request.kv_tokens > 0
            }
        )
        self.peak_allocator_internal_fragmentation_tokens = max(
            self.peak_allocator_internal_fragmentation_tokens,
            float(allocator_metrics.get("allocator_internal_fragmentation_tokens", 0.0)),
        )
        self.peak_allocator_external_fragmentation_tokens = max(
            self.peak_allocator_external_fragmentation_tokens,
            float(allocator_metrics.get("allocator_external_fragmentation_tokens", 0.0)),
        )

    @classmethod
    def _event_sort_key(cls, event: Dict[str, object]) -> Tuple[float, int, int]:
        time_ms = float(event.get("time_ms", 0.0))
        event_name = str(event.get("event", ""))
        event_rank = cls.EVENT_ORDER.get(event_name, 999)
        request_id = int(event.get("request_id", -1))
        return (time_ms, event_rank, request_id)

    @staticmethod
    def _metric_block(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
        clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        if not clean:
            return {
                "mean": None,
                "p50": None,
                "p95": None,
                "p99": None,
                "cvar95": None,
                "cvar99": None,
                "max": None,
            }
        p95_value = percentile(clean, 95)
        p99_value = percentile(clean, 99)
        tail95 = [value for value in clean if value >= p95_value]
        tail99 = [value for value in clean if value >= p99_value]
        if not tail95:
            tail95 = [max(clean)]
        if not tail99:
            tail99 = [max(clean)]
        return {
            "mean": round(statistics.fmean(clean), 3),
            "p50": round(percentile(clean, 50), 3),
            "p95": round(p95_value, 3),
            "p99": round(p99_value, 3),
            "cvar95": round(statistics.fmean(tail95), 3),
            "cvar99": round(statistics.fmean(tail99), 3),
            "max": round(max(clean), 3),
        }
