"""Paper scheduling rules, independent of teacher/student action selection.

Accepted-paper sources include Orca (OSDI 2022), S3 (NeurIPS 2023),
vLLM (SOSP 2023), Sarathi-Serve (OSDI 2024), and Learning to Rank
(NeurIPS 2024). Engine costs and predictor outputs remain separate inputs.
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .scheduler import DistributionAwareScheduler
from .types import PhaseMode, QueueClass, Request, ScheduleDecision, SimulatorConfig


def future_peak_tokens(states: Sequence[tuple[int, int, int]], block_size: int = 1) -> int:
    """(prompt, generated, predicted total output), peak BEFORE completion frees KV."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if any(prompt < 1 or generated < 0 or predicted <= generated for prompt, generated, predicted in states):
        raise ValueError("predicted output must exceed generated tokens")
    ordered = sorted(((p + g, predicted - g) for p, g, predicted in states), key=lambda x: -x[1])
    if block_size == 1:
        prefix = 0
        peak = 0
        for count, (base, remaining) in enumerate(ordered, 1):
            prefix += base
            peak = max(peak, prefix + remaining * count)
        return peak
    # Physical allocation extension: evaluate every completion breakpoint.
    return max((sum(math.ceil((base + t) / block_size) * block_size
                    for base, remaining in ordered if remaining >= t)
                for t in {remaining for _, remaining in ordered}), default=0)


@dataclass(frozen=True)
class TIEPrediction:
    mean: float
    cvar90: float
    ready_after_ms: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.mean, self.cvar90, self.ready_after_ms)):
            raise ValueError("TIE predictions must be finite")
        if self.mean <= 0 or self.cvar90 < self.mean or self.ready_after_ms < 0:
            raise ValueError("invalid censored moments or prediction delay")

    @classmethod
    def from_log_t(cls, mu: float, sigma: float, max_tokens: int, *, seed: int = 0,
                   samples: int = 10_000, ready_after_ms: float = 0.0):
        if not math.isfinite(mu) or not math.isfinite(sigma) or sigma <= 0 or max_tokens <= 0 or samples < 10:
            raise ValueError("invalid log-t parameters")
        draws = np.random.default_rng(seed).standard_t(3.5, samples)
        values = np.minimum(np.exp(np.minimum(mu + sigma * draws, math.log(max_tokens))), float(max_tokens))
        # Quantile-integral CVaR retains the top 10% mass even at the censoring atom.
        tail_count = samples / 10.0
        whole = int(tail_count)
        ordered = np.sort(values)[::-1]
        tail = float(ordered[:whole].sum())
        if tail_count > whole:
            tail += float(ordered[whole]) * (tail_count - whole)
        mean = float(values.mean())
        return cls(mean, max(mean, tail / tail_count), ready_after_ms)

    def score(self, waiting: int, batch_size: int, age_ms: float) -> float:
        beta = min(0.5, max(0.1, 0.1 * waiting / batch_size))
        return (self.mean + beta * self.cvar90) * 0.9 ** (max(0.0, age_ms) / 30_000.0)


class PaperScheduler(DistributionAwareScheduler):
    POLICIES = (
        "fcfs", "sarathi", "orca-max", "orca-pow2", "orca-oracle", "s3",
        "ltr", "ltr-oracle", "mlfq", "conservative", "past-future", "tie",
    )
    ORACLE_POLICIES = frozenset({"orca-pow2", "orca-oracle", "ltr-oracle"})

    def __init__(self, config: SimulatorConfig, policy: str, *, max_batch_size: int = 32,
                 max_output_tokens: int = 4096, history: Sequence[int] = (), seed: int = 0,
                 predictions: Mapping[int, TIEPrediction] | None = None,
                 s3_bucket_count: int = 10, ltr_starvation_threshold: int = 100,
                 ltr_priority_quantum: int = 10, mlfq_base_quantum_ms: float = 16_000.0,
                 mlfq_quantum_growth: float = 2.0):
        super().__init__(config)
        if policy not in self.POLICIES:
            raise ValueError(f"unknown paper policy: {policy}")
        if max_batch_size < 1 or max_output_tokens < 1 or config.token_budget_per_step < max_batch_size:
            raise ValueError("positive batch/output limits required; token budget must fit all decodes")
        if config.prefix_cache_enabled:
            raise ValueError("paper adapters currently require prefix_cache_enabled=False")
        if config.query_distribution_mode != "static":
            raise ValueError("paper adapters require static inputs")
        if config.kv_allocator_mode not in {"token", "paged"}:
            raise ValueError("paper adapters support token or paged KV")
        if s3_bucket_count < 1:
            raise ValueError("s3_bucket_count must be positive")
        if ltr_starvation_threshold < 1 or ltr_priority_quantum < 1:
            raise ValueError("LTR starvation parameters must be positive")
        if not math.isfinite(mlfq_base_quantum_ms) or mlfq_base_quantum_ms <= 0:
            raise ValueError("mlfq_base_quantum_ms must be positive and finite")
        if not math.isfinite(mlfq_quantum_growth) or mlfq_quantum_growth < 1:
            raise ValueError("mlfq_quantum_growth must be finite and at least one")
        self.policy = policy
        self.max_batch_size = max_batch_size
        self.max_output_tokens = max_output_tokens
        if any(int(length) <= 0 for length in history):
            raise ValueError("history must contain positive completed output lengths")
        self.history = deque((int(length) for length in history), maxlen=1000)
        self.observed: set[int] = set()
        self.rng = random.Random(seed)
        self.predictions = dict(predictions or {})
        self.s3_bucket_count = int(s3_bucket_count)
        self.ltr_starvation_threshold = int(ltr_starvation_threshold)
        self.ltr_priority_quantum = int(ltr_priority_quantum)
        self.mlfq_base_quantum_ms = float(mlfq_base_quantum_ms)
        self.mlfq_quantum_growth = float(mlfq_quantum_growth)
        self.ltr_starvation: dict[int, int] = {}
        self.ltr_priority_remaining: dict[int, int] = {}
        self.ltr_scores: dict[int, int] = {}
        self.mlfq_level: dict[int, int] = {}
        self.mlfq_level_service_ms: dict[int, float] = {}
        self.s3_reservations: dict[int, int] = {}
        self.s3_preemptions = 0
        self.steps: list[dict[str, object]] = []
        self.last_predicted_peak = 0
        self.admission_rejections = 0

    def route_prefill_request(self, request: Request, engine) -> None:
        request.queue_class = QueueClass.NORMAL
        request.prefill_route_name = f"paper-{self.policy}"
        request.online_length_distribution = request.length_distribution
        request.routed_length_distribution = request.length_distribution

    def route_decode_request(self, request: Request, engine) -> None:
        request.decode_route_name = f"paper-{self.policy}"

    def on_step_result(self, step_result):
        super().on_step_result(step_result)
        self.steps.append(dict(step_result))
        if self.policy == "mlfq":
            progressed = {
                int(request_id)
                for key in ("prefill_progress_request_ids", "decode_progress_request_ids")
                for request_id in str(step_result.get(key, "")).split(",")
                if request_id
            }
            duration_ms = float(step_result.get("duration_ms", 0.0))
            for request_id in progressed:
                level = self.mlfq_level.setdefault(request_id, 0)
                service = self.mlfq_level_service_ms.get(request_id, 0.0) + duration_ms
                quantum = self.mlfq_base_quantum_ms * self.mlfq_quantum_growth ** level
                while service >= quantum:
                    service -= quantum
                    level += 1
                    quantum = self.mlfq_base_quantum_ms * self.mlfq_quantum_growth ** level
                self.mlfq_level[request_id] = level
                self.mlfq_level_service_ms[request_id] = service

    def _cap(self, request: Request) -> int:
        return request.max_new_tokens or self.max_output_tokens

    def _sample(self, request: Request) -> int:
        eligible = [length for length in self.history if length > request.generated_tokens]
        value = self.rng.choice(eligible) if eligible else self._cap(request)
        return max(request.generated_tokens + 1, min(value, self._cap(request)))

    def _predicted_output(self, request: Request) -> int:
        distribution = request.effective_length_distribution
        value = distribution.p50 if distribution.p50 is not None else distribution.mean
        return max(request.generated_tokens + 1, min(int(math.ceil(value)), self._cap(request)))

    def _s3_prediction(self, request: Request) -> int:
        width = max(1, int(math.ceil(self.max_output_tokens / self.s3_bucket_count)))
        predicted = int(math.ceil(self._predicted_output(request) / width) * width)
        return max(request.generated_tokens + 1, min(predicted, self._cap(request)))

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        return 1 << max(0, int(value - 1).bit_length())

    def _reservation_output(self, request: Request) -> int:
        if self.policy in {"orca-max", "conservative"}:
            return self._cap(request)
        if self.policy == "orca-pow2":
            return min(self._next_power_of_two(request.actual_output_tokens), self._cap(request))
        if self.policy == "orca-oracle":
            return min(request.actual_output_tokens, self._cap(request))
        if self.policy == "s3":
            return self.s3_reservations.setdefault(request.request_id, self._s3_prediction(request))
        raise ValueError(f"{self.policy} does not define reservation output")

    def _reservation_tokens(self, request: Request, engine) -> int:
        return engine.quantize_kv_tokens(request.prompt_tokens + self._reservation_output(request))

    @staticmethod
    def _unique_requests(requests: Sequence[Request]) -> list[Request]:
        seen = set()
        unique = []
        for request in requests:
            if request.request_id not in seen:
                unique.append(request)
                seen.add(request.request_id)
        return unique

    def _s3_preempt_overruns(self, engine) -> None:
        for request in list(engine.active_decode):
            reserved = self.s3_reservations.setdefault(request.request_id, self._s3_prediction(request))
            if request.generated_tokens < reserved:
                continue
            new_reservation = min(self._cap(request), max(request.generated_tokens + 1, reserved * 2))
            self.s3_reservations[request.request_id] = new_reservation
            rebuild_tokens = request.kv_tokens
            request.eviction_count += 1
            request.recompute_count += 1
            request.recompute_tokens += rebuild_tokens
            request.wasted_kv_tokens += request.kv_tokens
            request.prefill_target_tokens = request.shared_kv_tokens + rebuild_tokens
            request.prefill_done_tokens = request.shared_kv_tokens
            engine.kv_allocator.release(request.request_id)
            request.kv_tokens = 0
            engine.active_decode.remove(request)
            engine.prefill_queues[request.queue_class].append(request)
            self.s3_preemptions += 1

    def _rank_key(self, request: Request):
        if self.policy in {"ltr", "ltr-oracle"}:
            priority = self.ltr_priority_remaining.get(request.request_id, 0) > 0
            score = self.ltr_scores[request.request_id]
            return (0 if priority else 1, score, request.arrival_ms, request.request_id)
        if self.policy == "mlfq":
            return (self.mlfq_level.get(request.request_id, 0), request.arrival_ms, request.request_id)
        raise ValueError(f"{self.policy} is not a preemptive rank policy")

    def _update_ltr_fairness(self, candidates: Sequence[Request], selected_ids: set[int]) -> None:
        for request in candidates:
            request_id = request.request_id
            if request_id in selected_ids:
                self.ltr_starvation[request_id] = 0
                if self.ltr_priority_remaining.get(request_id, 0) > 0:
                    self.ltr_priority_remaining[request_id] -= 1
            else:
                self.ltr_starvation[request_id] = self.ltr_starvation.get(request_id, 0) + 1
        for request in candidates:
            request_id = request.request_id
            if self.ltr_starvation.get(request_id, 0) >= self.ltr_starvation_threshold:
                self.ltr_starvation[request_id] = 0
                self.ltr_priority_remaining[request_id] = self.ltr_priority_quantum
            elif self.ltr_priority_remaining.get(request_id, 0) <= 0:
                self.ltr_priority_remaining.pop(request_id, None)

    def _schedule_ranked(self, engine) -> ScheduleDecision:
        queued = [request for queue in engine.prefill_queues.values() for request in queue]
        candidates = self._unique_requests(list(engine.active_decode) + queued)
        for request in candidates:
            if self.policy == "mlfq":
                self.mlfq_level.setdefault(request.request_id, 0)
                self.mlfq_level_service_ms.setdefault(request.request_id, 0.0)
            else:
                self.ltr_starvation.setdefault(request.request_id, 0)
                self.ltr_scores.setdefault(
                    request.request_id,
                    request.actual_output_tokens if self.policy == "ltr-oracle"
                    else self._predicted_output(request),
                )
        candidates.sort(key=self._rank_key)
        budget = self.config.token_budget_per_step
        free = self.config.kv_capacity_tokens - engine.current_kv_tokens
        prefill_plan = []
        decode_plan = []
        selected_ids: set[int] = set()
        for request in candidates:
            if len(selected_ids) >= self.max_batch_size:
                break
            if request in engine.active_decode:
                tokens = 1
                growth = engine.estimate_kv_growth(request, tokens)
                can_recover = self.config.overflow_policy == "evict-recompute" and len(engine.active_decode) > 1
                if tokens <= budget and (growth <= free or can_recover):
                    decode_plan.append((request, tokens))
                else:
                    continue
            else:
                tokens = request.remaining_prefill_tokens
                growth = engine.estimate_kv_growth(request, tokens)
                if tokens <= 0 or tokens > budget or growth > free:
                    continue
                prefill_plan.append((request, tokens))
            selected_ids.add(request.request_id)
            budget -= tokens
            free -= growth
        if self.policy in {"ltr", "ltr-oracle"}:
            self._update_ltr_fairness(candidates, selected_ids)
        return ScheduleDecision(
            phase_mode=(PhaseMode.MIXED if prefill_plan and decode_plan else
                        PhaseMode.PREFILL_PRIORITY if prefill_plan else PhaseMode.DECODE_PRIORITY),
            prefill_requests=prefill_plan,
            decode_requests=decode_plan,
        )

    def _queue_key(self, request, engine, waiting):
        if self.policy != "tie":
            return request.arrival_ms, request.request_id
        if request.request_id not in self.predictions:
            raise ValueError(f"missing external TIE prediction for request {request.request_id}")
        prediction = self.predictions[request.request_id]
        age = engine.now_ms - request.arrival_ms
        score = (prediction.score(waiting, self.max_batch_size, age)
                 if age >= prediction.ready_after_ms else float(self._cap(request)))
        return score, request.arrival_ms, request.request_id

    def schedule(self, engine):
        if self.policy in {"ltr", "ltr-oracle", "mlfq"}:
            return self._schedule_ranked(engine)
        if self.policy == "s3":
            self._s3_preempt_overruns(engine)
        if self.policy == "past-future":
            for request in engine.requests:
                if request.is_complete and request.request_id not in self.observed:
                    self.history.append(request.actual_output_tokens)
                    self.observed.add(request.request_id)
        queued = [request for queue in engine.prefill_queues.values() for request in queue]
        started = sorted((r for r in queued if r.has_started), key=lambda r: (r.arrival_ms, r.request_id))
        waiting = [r for r in queued if not r.has_started]
        waiting_count = len(waiting)
        if self.policy == "s3":
            waiting.sort(key=lambda r: (-self._reservation_tokens(r, engine), r.arrival_ms, r.request_id))
        else:
            waiting.sort(key=lambda r: self._queue_key(r, engine, waiting_count))
        decodes = sorted(engine.active_decode, key=lambda r: (r.arrival_ms, r.request_id))
        running = decodes + started
        if len(running) > self.max_batch_size:
            raise ValueError("running batch exceeds configured maximum")
        self.last_predicted_peak = 0
        predictions = {r.request_id: self._sample(r) for r in running} if self.policy == "past-future" else {}
        budget = self.config.token_budget_per_step
        free = self.config.kv_capacity_tokens - engine.current_kv_tokens
        decode_plan = [(r, 1) for r in decodes]
        decode_growth = sum(engine.estimate_kv_growth(r, 1) for r in decodes)
        mixed = self.policy in {"sarathi", "tie"}
        if mixed:
            budget -= len(decode_plan)
            free -= decode_growth
        admission_free = free - sum(engine.estimate_kv_growth(r, r.remaining_prefill_tokens) for r in started)
        prefill_plan = []
        admitted = list(running)
        reservation_policies = {"orca-max", "orca-pow2", "orca-oracle", "s3", "conservative"}
        reserved = (sum(self._reservation_tokens(r, engine) for r in admitted)
                    if self.policy in reservation_policies else 0)
        block = self.config.kv_block_size_tokens if self.config.kv_allocator_mode == "paged" else 1
        for request in started + waiting:
            is_new = not request.has_started
            if is_new and len(admitted) >= self.max_batch_size:
                break
            remaining = request.remaining_prefill_tokens
            chunk = min(remaining, budget) if mixed else remaining
            if chunk <= 0:
                break
            if chunk > budget:
                if not prefill_plan and not running:
                    raise ValueError("non-chunked prefill exceeds token budget; increase token_budget_per_step")
                break
            growth = engine.estimate_kv_growth(request, chunk)
            if growth > free:
                break
            full_prompt_growth = engine.estimate_kv_growth(request, remaining)
            if is_new and full_prompt_growth > admission_free:
                break
            if is_new and self.policy in reservation_policies:
                peak = reserved + self._reservation_tokens(request, engine)
                self.last_predicted_peak = max(self.last_predicted_peak, peak)
                if peak > self.config.kv_capacity_tokens:
                    self.admission_rejections += 1
                    if self.policy == "s3":
                        continue
                    break
            elif is_new and self.policy == "past-future":
                candidates = admitted + [request]
                predictions[request.request_id] = self._sample(request)
                states = [(r.prompt_tokens, r.generated_tokens, predictions[r.request_id]) for r in candidates]
                peak = future_peak_tokens(states, block)
                self.last_predicted_peak = max(self.last_predicted_peak, peak)
                if peak > self.config.kv_capacity_tokens:
                    self.admission_rejections += 1
                    break
            prefill_plan.append((request, chunk))
            if is_new:
                admitted.append(request)
                admission_free -= full_prompt_growth
                if self.policy in reservation_policies:
                    reserved += self._reservation_tokens(request, engine)
            free -= growth
            budget -= chunk
        if prefill_plan and not mixed:
            decode_plan = []
        return ScheduleDecision(
            phase_mode=(PhaseMode.MIXED if prefill_plan and decode_plan else
                        PhaseMode.PREFILL_PRIORITY if prefill_plan else PhaseMode.DECODE_PRIORITY),
            prefill_requests=prefill_plan, decode_requests=decode_plan,
        )
