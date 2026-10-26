"""Online-only chunk/budget allocation environment for complete mean-TTFT episodes.

No reservation, risk gate, aging bonus or NOW/WAIT state machine is inherited.
The engine owns hidden lengths; policies receive numeric candidate observations.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass

import numpy as np

from .engine import LLMSingleNodeSimulator
from .scheduler import DistributionAwareScheduler
from .timing import TimingBatch
from .types import LengthDistribution, PhaseMode, QueueClass, Request, ScheduleDecision, SimulatorConfig, percentile


SCHEMA = "prefill-budget-rl-v1"


class PlainScheduler(DistributionAwareScheduler):
    """Only the engine routing/event hooks; decisions come from the environment."""
    def route_prefill_request(self, request, engine):
        request.queue_class = QueueClass.NORMAL
        request.online_length_distribution = request.length_distribution
        request.routed_length_distribution = request.length_distribution

    def route_decode_request(self, request, engine):
        request.decode_route_name = "rl-environment"

    def schedule(self, engine):
        raise RuntimeError("use SchedulingEnvironment.step(), not engine.run()")


@dataclass(frozen=True)
class AllocationAction:
    candidate_id: int | None
    prefill_budget: int
    chunk_limit: int
    decode_limit: int


@dataclass
class Candidate:
    action: AllocationAction
    decision: ScheduleDecision
    features: np.ndarray
    is_baseline: bool = False


FEATURE_NAMES = (
    "kv_used_fraction", "running_fraction", "waiting_fraction", "prefill_work_fraction",
    "oldest_wait_scaled", "decode_remaining_mean", "decode_release_soon_fraction",
    "budget_capacity_fraction", "block_capacity_fraction", "batch_capacity_fraction",
    "reference_prefill_time", "reference_decode_time", "reference_mixed_time",
    "head_prefill_fraction", "head_predicted_remaining", "head_wait_scaled",
    "prefill_budget_fraction", "chunk_budget_fraction", "decode_limit_fraction",
    "planned_prefill_fraction", "planned_decode_fraction", "planned_duration_scaled",
    "planned_kv_growth_fraction", "finishing_prefills_fraction", "baseline_action",
    "active_remaining_min", "active_remaining_max", "next_release_kv_fraction",
    "last_round_duration_scaled", "decision_overhead_scaled",
)


def waiting_cost(requests, start: float, end: float) -> float:
    """Exact integral of arrived requests that have not produced token 1."""
    return sum(max(0.0, min(end, r.first_token_at_ms if r.first_token_at_ms is not None else end)
                       - max(start, r.arrival_ms)) for r in requests)


class SchedulingEnvironment:
    def __init__(self, requests: list[Request], config: SimulatorConfig, *, max_running: int = 16,
                 baseline_chunk: int = 128, extended_actions: bool = True, max_steps: int = 20000,
                 action_space: str = "joint"):
        if config.execution_mode != "scheduled-tokens" or config.first_token_mode != "prefill":
            raise ValueError("RL environment requires chunk execution and prefill first token")
        if config.prefix_cache_enabled:
            raise ValueError("this experiment excludes prefix caching")
        if config.simulation_duration_ms:
            raise ValueError("training uses complete request sets, not censored horizons")
        if max_running < 1 or max_steps < 1 or baseline_chunk < 1:
            raise ValueError("invalid environment limits")
        self.config = copy.deepcopy(config)
        self.engine = LLMSingleNodeSimulator(copy.deepcopy(requests), self.config, PlainScheduler(self.config))
        self.max_running = int(max_running)
        self.baseline_chunk = min(baseline_chunk, config.prefill_chunk_size, config.token_budget_per_step)
        self.extended_actions = extended_actions
        if action_space not in {"request", "joint"}:
            raise ValueError("action_space must be request or joint")
        self.action_space = action_space
        self.max_steps = max_steps
        self.steps = 0
        self.total_cost = 0.0
        self.last_duration = 0.0
        self.preemptions = 0
        self.done = False
        self._candidates: list[Candidate] | None = None
        self._advance_idle()

    def _groups(self):
        queued = [r for q in self.engine.prefill_queues.values() for r in q]
        decodes = list(self.engine.active_decode)
        return queued, decodes

    def _advance_idle(self):
        e = self.engine
        e._ingest_arrivals()
        queued, decodes = self._groups()
        if not queued and not decodes and e._next_arrival_time() is not None:
            e.now_ms = max(e.now_ms, e._next_arrival_time())
            e._ingest_arrivals()
        self.done = e._is_finished()

    def _plan(self, action: AllocationAction) -> ScheduleDecision:
        e = self.engine
        queued, decodes = self._groups()
        allocator = e.kv_allocator.clone()
        budget = self.config.token_budget_per_step
        dplan, pplan = [], []

        def fit(r, wanted):
            lo, hi, best = 1, int(wanted), 0
            while lo <= hi:
                n = (lo + hi) // 2
                attempt = allocator.estimate_growth(r.request_id, r.kv_tokens, n)
                if attempt.success and attempt.physical_tokens_after <= self.config.kv_capacity_tokens:
                    best, lo = n, n + 1
                else:
                    hi = n - 1
            if best:
                allocator.apply_growth(r.request_id, r.kv_tokens, best)
            return best

        # Age here is a deterministic service tie-break, not an extra reward or gate.
        for r in sorted(decodes, key=lambda r: (r.last_service_at_ms, r.request_id))[:action.decode_limit]:
            if budget > 0 and fit(r, 1):
                dplan.append((r, 1))
                budget -= 1
        running = len(decodes) + sum(r.has_started for r in queued)
        remaining_p = min(budget, action.prefill_budget)
        ordered = sorted(queued, key=lambda r: (
            r.request_id != action.candidate_id, r.remaining_prefill_tokens, r.arrival_ms, r.request_id))
        for r in ordered:
            if remaining_p <= 0 or len(pplan) >= self.config.max_prefill_batch:
                break
            if not r.has_started and running >= self.max_running:
                continue
            chunk = fit(r, min(r.remaining_prefill_tokens, action.chunk_limit, remaining_p))
            if chunk:
                pplan.append((r, chunk))
                remaining_p -= chunk
                if not r.has_started:
                    running += 1
        return ScheduleDecision(PhaseMode.MIXED if pplan and dplan else
            PhaseMode.PREFILL_PRIORITY if pplan else PhaseMode.DECODE_PRIORITY,
            decode_requests=dplan, prefill_requests=pplan)

    def _estimate(self, decision):
        batch = TimingBatch(tuple(n for r, n in decision.prefill_requests),
                            tuple(r.logical_kv_tokens for r, n in decision.prefill_requests),
                            tuple(r.logical_kv_tokens for r, n in decision.decode_requests))
        if self.engine.timing_model:
            return self.engine.timing_model.estimate(batch)[0] + self.config.decision_overhead_ms
        p = [n for r, n in decision.prefill_requests]
        padding = max(p, default=0) * len(p) - sum(p)
        return self.engine._duration_value(sum(p), padding, len(decision.decode_requests)) + self.config.decision_overhead_ms

    def _features(self, action, plan, baseline):
        e, c = self.engine, self.config
        queued, decodes = self._groups()
        capacity, budget = max(c.kv_capacity_tokens, 1), c.token_budget_per_step
        head = next((r for r in queued if r.request_id == action.candidate_id), None)
        active = sorted(((r.effective_length_distribution.conditional_remaining_quantile(
            r.generated_tokens, cap=r.max_new_tokens), r.logical_kv_tokens) for r in decodes))
        remain = [x[0] for x in active]
        if e.timing_model:
            ref_p = e.timing_model.estimate(TimingBatch((128,), (0,), ()))[0]
            ref_d = e.timing_model.estimate(TimingBatch((), (), (512,)))[0]
            ref_m = e.timing_model.estimate(TimingBatch((128,), (512,), (512,) * 4))[0]
        else:
            ref_p, ref_d, ref_m = (e._duration_value(128, 0, 0), e._duration_value(0, 0, 1),
                                    e._duration_value(128, 0, 4))
        growth = sum(e.estimate_kv_growth(r, n) for r, n in plan.prefill_requests + plan.decode_requests)
        values = (
            e.current_kv_tokens / capacity, (len(decodes) + sum(r.has_started for r in queued)) / self.max_running,
            len(queued) / 32, sum(r.remaining_prefill_tokens for r in queued) / capacity,
            max((e.now_ms - r.arrival_ms for r in queued), default=0) / 1000,
            (sum(remain) / len(remain) if remain else 0) / 512,
            sum(kv for rem, kv in active if rem <= 16) / capacity,
            budget / capacity, c.kv_block_size_tokens / capacity, self.max_running / 32,
            ref_p / 10, ref_d / 10, ref_m / 10,
            head.remaining_prefill_tokens / budget if head else 0,
            head.effective_length_distribution.conditional_remaining_quantile(0, cap=head.max_new_tokens) / 512 if head else 0,
            (e.now_ms - head.arrival_ms) / 1000 if head else 0,
            action.prefill_budget / budget, action.chunk_limit / budget,
            action.decode_limit / self.max_running,
            sum(n for r, n in plan.prefill_requests) / budget, len(plan.decode_requests) / self.max_running,
            self._estimate(plan) / 10, growth / capacity,
            sum(n == r.remaining_prefill_tokens for r, n in plan.prefill_requests) / self.max_running,
            float(baseline), min(remain, default=0) / 512, max(remain, default=0) / 512,
            active[0][1] / capacity if active else 0, self.last_duration / 10, c.decision_overhead_ms / 10,
        )
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _key(plan):
        return (tuple(sorted((r.request_id, n) for r, n in plan.prefill_requests)),
                tuple(sorted((r.request_id, n) for r, n in plan.decode_requests)))

    def candidates(self) -> list[Candidate]:
        if self.done:
            return []
        if self._candidates is not None:
            return self._candidates
        queued, decodes = self._groups()
        e, c = self.engine, self.config
        heads = sorted(queued, key=lambda r: (r.remaining_prefill_tokens, r.arrival_ms, r.request_id))
        baseline = AllocationAction(heads[0].request_id if heads else None,
                                    c.token_budget_per_step, self.baseline_chunk,
                                    min(len(decodes), c.max_decode_batch))
        if self._key(self._plan(baseline)) == ((), ()):
            # Common physical fallback, independent of whether extended actions
            # are enabled. It cannot consult output truth or future arrivals.
            fallback = AllocationAction(baseline.candidate_id, c.token_budget_per_step,
                                        self.baseline_chunk, 0)
            if self._key(self._plan(fallback)) == ((), ()):
                self._recover_capacity()
                return self.candidates()
            baseline = fallback
        actions = [baseline]
        if self.extended_actions and heads:
            chunks = sorted({min(64, c.prefill_chunk_size, c.token_budget_per_step), self.baseline_chunk,
                             min(c.prefill_chunk_size, c.token_budget_per_step)})
            pbudgets = sorted({max(1, c.token_budget_per_step // 4),
                               max(1, c.token_budget_per_step // 2), c.token_budget_per_step})
            dlimits = sorted({0, min(c.max_decode_batch, max(1, len(decodes) // 2)),
                             min(c.max_decode_batch, len(decodes))})
            if self.action_space == "request":
                chunks, pbudgets, dlimits = [baseline.chunk_limit], [baseline.prefill_budget], [baseline.decode_limit]
            for head in heads:
                for p in pbudgets:
                    for chunk in chunks:
                        for d in dlimits:
                            actions.append(AllocationAction(head.request_id, p, chunk, d))
            if decodes and self.action_space == "joint":
                actions.append(AllocationAction(None, 0, self.baseline_chunk, baseline.decode_limit))
        result, seen = [], set()
        for index, action in enumerate(actions):
            plan = self._plan(action)
            key = self._key(plan)
            if key == ((), ()) or key in seen:
                continue
            seen.add(key)
            result.append(Candidate(action, plan, self._features(action, plan, index == 0), index == 0))
        if not result:
            self._recover_capacity()
            return self.candidates()
        # Empty baseline is a physically infeasible plan. Recovery or the first
        # legal allocation is shared by baseline evaluation and learned policies.
        if not result[0].is_baseline:
            result[0].is_baseline = True
            result[0].features[FEATURE_NAMES.index("baseline_action")] = 1
        self._candidates = result
        return result

    def _recover_capacity(self):
        e = self.engine
        queued, decodes = self._groups()
        running = [r for r in decodes + queued if r.kv_tokens > 0]
        if not running or self.preemptions >= self.max_steps:
            raise RuntimeError("workload cannot make physical progress")
        victim = max(running, key=lambda r: (r.kv_tokens, r.request_id))
        rebuild = victim.kv_tokens
        victim.eviction_count += 1
        victim.recompute_count += 1
        victim.recompute_tokens += rebuild
        victim.wasted_kv_tokens += rebuild
        victim.prefill_target_tokens = max(victim.active_prefill_target_tokens,
                                          victim.shared_kv_tokens + rebuild)
        victim.prefill_done_tokens = victim.shared_kv_tokens
        victim.kv_tokens = 0
        e.kv_allocator.release(victim.request_id)
        if victim in e.active_decode:
            e.active_decode.remove(victim)
            e.prefill_queues[victim.queue_class].append(victim)
        e._record_event({"time_ms": e.now_ms, "event": "evict_recompute",
                         "request_id": victim.request_id, "recompute_tokens": rebuild})
        self.preemptions += 1

    def step(self, action_index: int):
        candidates = self.candidates()
        if not 0 <= action_index < len(candidates):
            raise ValueError("action is outside the legal candidate set")
        start = self.engine.now_ms
        result = self.engine._apply_decision(candidates[action_index].decision)
        self.engine.scheduler.on_step_result(result)
        if result["actual_prefill_progress_tokens"] + result["actual_decode_progress_tokens"] <= 0:
            raise RuntimeError("legal allocation made no progress")
        self.last_duration = result["duration_ms"]
        self.steps += 1
        self.engine.simulation_steps = self.steps
        self._candidates = None
        self._advance_idle()
        cost = waiting_cost(self.engine.requests, start, self.engine.now_ms)
        self.total_cost += cost
        if self.steps >= self.max_steps and not self.done:
            raise RuntimeError("episode step cap reached; incomplete episodes must not be scored as successes")
        reward = -cost / max(1, len(self.engine.requests))
        return reward, self.done, result

    def summary(self):
        if not self.done:
            raise RuntimeError("mean TTFT evaluation requires a complete episode")
        expected = sum(r.ttft_ms for r in self.engine.requests)
        if not math.isclose(expected, self.total_cost, rel_tol=1e-9, abs_tol=1e-6):
            raise AssertionError("waiting integral does not reconcile with total TTFT")
        self.engine.termination_reason = "completed_all_requests"
        return {"schema": SCHEMA, "requests": len(self.engine.requests), "steps": self.steps,
                "mean_ttft_ms": expected / max(1, len(self.engine.requests)),
                "p95_ttft_ms": percentile([r.ttft_ms for r in self.engine.requests], 95),
                "mean_e2e_ms": sum(r.e2e_ms for r in self.engine.requests) / max(1, len(self.engine.requests)),
                "total_cost_ms": self.total_cost, "reconciliation_error_ms": self.total_cost - expected,
                "preemptions": self.preemptions, "peak_kv_tokens": self.engine.peak_kv_tokens,
                "makespan_ms": self.engine.now_ms}


def make_episode(seed: int, *, regime: str = "train", requests: int = 24,
                 smoke: bool = False, overhead_ms: float = 0.0, family: dict | None = None):
    """Synthetic latent workload and independent noisy prediction, seeded per episode.

    Validation/test use disjoint random streams; OOD also holds out cost/capacity
    ranges. No fitting or parameter selection reads the test stream.
    """
    offsets = {"train": 0, "validation": 1_000_000, "test": 2_000_000, "ood": 3_000_000}
    if regime not in offsets or not 0 <= seed < 1_000_000 or requests < 1:
        raise ValueError("invalid episode specification")
    episode_seed = seed + offsets[regime]
    rng = random.Random(episode_seed)
    family = family or {}
    if set(family) - {"in_distribution", "ood"}:
        raise ValueError("family must contain only in_distribution and/or ood")
    options = family.get("ood" if regime == "ood" else "in_distribution", {})
    allowed = {"scale_range", "budget_choices", "capacity_choices", "max_running_choices",
        "block_size", "timing_ranges", "arrival_intervals", "prompt_range", "output_base",
        "output_prompt_slope", "output_cap", "predictor_bias_range", "predictor_noise_std",
        "length_cv_range", "prediction_cv_range"}
    if set(options) - allowed:
        raise ValueError(f"unknown environment family keys: {set(options) - allowed}")

    def span(name, default):
        bounds = options.get(name, default)
        if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]:
            raise ValueError(f"invalid {name}")
        return rng.uniform(*bounds)

    def choice(name, default):
        values = options.get(name, default)
        if not values or any(not isinstance(v, int) or v < 1 for v in values):
            raise ValueError(f"invalid {name}")
        return rng.choice(values)

    scale = span("scale_range", [0.5, 2.0] if regime != "ood" else [2.5, 4.0])
    if scale <= 0:
        raise ValueError("environment scale must be positive")
    budget = choice("budget_choices", [128, 256, 512] if not smoke else [32, 64])
    capacity = choice("capacity_choices", [2048, 4096, 8192] if regime != "ood" else [1536, 12288])
    max_running = choice("max_running_choices", [4, 8, 16])
    timing_ranges = options.get("timing_ranges", {})
    default_ranges = {"launch_ms": [0.1, 0.1], "weight_read_ms": [0.4, 2.0],
        "prefill_ms_per_token": [0.01, 0.05], "prefill_context_ms_per_token2": [0.000001, 0.00002],
        "decode_ms_per_request": [0.01, 0.06], "decode_context_ms_per_token": [0.00001, 0.0002]}
    if set(timing_ranges) - set(default_ranges):
        raise ValueError("unknown timing parameter")
    parameters = {}
    for key, defaults in default_ranges.items():
        bounds = timing_ranges.get(key, defaults)
        if len(bounds) != 2 or not 0 <= bounds[0] <= bounds[1] or not all(math.isfinite(v) for v in bounds):
            raise ValueError(f"invalid timing range: {key}")
        parameters[key] = rng.uniform(*bounds) * scale
    profile = {"kind": "resource-surrogate-v1", "measurement_scope": "synthetic-uncalibrated",
        "source": "parametric environment family; not GPU measurements", "parameters": parameters}
    block_size = options.get("block_size", 16)
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    c = SimulatorConfig(execution_mode="scheduled-tokens", first_token_mode="prefill",
        timing_profile=profile, token_budget_per_step=budget, prefill_chunk_size=min(256, budget),
        kv_capacity_tokens=capacity, kv_allocator_mode="paged", kv_block_size_tokens=block_size,
        max_decode_batch=max_running, max_prefill_batch=max_running,
        query_distribution_mode="static", prefix_cache_enabled=False,
        enable_action_safety_filter=False, enable_deadlock_recovery=False,
        prefill_padding_weight=0, runtime_sampling_seed=episode_seed, decision_overhead_ms=overhead_ms)
    rows, arrival = [], 0.0
    workload_kind = rng.choice(["burst", "steady", "light"])
    intervals = {"burst": 0.2, "steady": 4.0, "light": 30.0}
    intervals.update(options.get("arrival_intervals", {}))
    if any(not math.isfinite(v) or v <= 0 for v in intervals.values()):
        raise ValueError("arrival intervals must be positive and finite")
    interval = intervals[workload_kind] * scale
    bias = span("predictor_bias_range", [-0.35, 0.35])
    noise_std = float(options.get("predictor_noise_std", 0.3))
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("predictor_noise_std must be finite and nonnegative")
    for i in range(requests):
        if i:
            arrival += rng.expovariate(1 / interval)
        prompt = rng.randint(8, 48) if smoke else rng.choice([rng.randint(16, 128), rng.randint(128, 768)])
        if "prompt_range" in options:
            bounds = options["prompt_range"]
            if len(bounds) != 2 or any(not isinstance(v, int) for v in bounds) or not 1 <= bounds[0] <= bounds[1]:
                raise ValueError("invalid prompt_range")
            prompt = rng.randint(*bounds)
        category = rng.choice([0.5, 1.0, 2.0])
        conditional_mean = (options.get("output_base", 8 if smoke else 24)
            + options.get("output_prompt_slope", 0 if smoke else 0.12) * prompt) * category
        true_distribution = LengthDistribution.from_moments(conditional_mean, conditional_mean * span("length_cv_range", [0.3, 1.2]))
        # Prediction is generated without consulting the realized length draw.
        predicted_mean = conditional_mean * math.exp(bias + rng.gauss(0, noise_std))
        prediction = LengthDistribution.from_moments(predicted_mean, predicted_mean * span("prediction_cv_range", [0.4, 1.0]))
        cap = options.get("output_cap", 32 if smoke else 384)
        if not isinstance(cap, int) or cap < 1 or math.ceil((prompt + cap - 1) / block_size) * block_size > capacity:
            raise ValueError("each request's capped context must fit the physical capacity")
        rows.append(Request(i, arrival, prompt, prediction, max_new_tokens=cap,
                            true_length_distribution=true_distribution))
    metadata = {"schema": SCHEMA, "seed": seed, "episode_seed": episode_seed, "regime": regime,
                "workload_kind": workload_kind, "requests": requests,
                "config": asdict(c), "max_running": max_running,
                "family_options": copy.deepcopy(options),
                "baseline_chunk": min(128, budget), "synthetic_uncalibrated": True}
    return rows, c, metadata
