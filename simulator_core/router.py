from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict

from .types import LengthDistribution, QueueClass, Request

if TYPE_CHECKING:
    from .engine import LLMSingleNodeSimulator
    from .scheduler import DistributionAwareScheduler
    from .types import ScheduleContext


def mean_only_distribution(distribution: LengthDistribution) -> LengthDistribution:
    mean_value = max(1.0, float(distribution.mean))
    return LengthDistribution(
        mean=mean_value,
        std=1.0,
        p50=mean_value,
        p90=mean_value,
        p95=mean_value,
        p99=mean_value,
        family="deterministic",
    )


@dataclass(slots=True)
class PrefillRoute:
    queue_class: QueueClass
    routing_distribution: LengthDistribution
    route_name: str = "prefill-default"
    priority_delta: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class DecodeRoute:
    route_name: str = "decode-default"
    priority_delta: float = 0.0
    batch_tag: str = "default"
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class QueryDistributionDecision:
    distribution: LengthDistribution
    source: str = "static"
    adapter_name: str = "query-distribution-static"
    metadata: Dict[str, object] = field(default_factory=dict)


class QueryDistributionAdapter:
    name = "query-distribution-adapter"

    def adapt_distribution(
        self,
        request: Request,
        scheduler: "DistributionAwareScheduler",
        engine: "LLMSingleNodeSimulator | None",
    ) -> QueryDistributionDecision:
        raise NotImplementedError


class StaticQueryDistributionAdapter(QueryDistributionAdapter):
    name = "query-distribution-static"

    def adapt_distribution(
        self,
        request: Request,
        scheduler: "DistributionAwareScheduler",
        engine: "LLMSingleNodeSimulator | None",
    ) -> QueryDistributionDecision:
        return QueryDistributionDecision(
            distribution=request.length_distribution,
            source="static",
            adapter_name=self.name,
            metadata={
                "base_mean": round(float(request.length_distribution.mean), 6),
                "base_p90": round(float(request.length_distribution.p90), 6),
            },
        )


class OnlineLengthPredictionAdapter(QueryDistributionAdapter):
    name = "query-distribution-online-predictor"

    def _rng_for_request(
        self,
        request: Request,
        scheduler: "DistributionAwareScheduler",
    ) -> random.Random:
        seed_payload = (
            f"{scheduler.config.runtime_sampling_seed}|"
            f"{request.request_id}|"
            f"{request.arrival_ms:.6f}|"
            f"{request.model_id}|"
            f"{request.template_id}|"
            f"{request.template_version}"
        )
        seed_bytes = hashlib.sha256(seed_payload.encode("utf-8")).digest()[:8]
        return random.Random(int.from_bytes(seed_bytes, byteorder="big", signed=False))

    @staticmethod
    def _bounded_lognormal_scale(
        rng: random.Random,
        sigma: float,
        low: float,
        high: float,
    ) -> float:
        if sigma <= 0.0:
            return 1.0
        # Mean-normalized lognormal noise keeps predictions unbiased over many requests.
        scale = rng.lognormvariate(-0.5 * sigma * sigma, sigma)
        return max(low, min(high, scale))

    @staticmethod
    def _recent_template_mean(
        request: Request,
        engine: "LLMSingleNodeSimulator | None",
    ) -> float | None:
        if engine is None:
            return None
        rows = list(engine.recent_query_history)[-32:]
        if not rows:
            return None
        family = str(request.template_family or "")
        template_id = str(request.template_id or "")
        matching = [
            float(row["output_mean"])
            for row in rows
            if (
                family
                and str(row.get("template_family") or "") == family
            )
            or (
                template_id
                and str(row.get("template_id") or "") == template_id
            )
        ]
        if not matching:
            return None
        return sum(matching) / len(matching)

    @staticmethod
    def _cap_distribution(distribution: LengthDistribution, max_new_tokens: int) -> LengthDistribution:
        if max_new_tokens <= 0:
            return distribution
        cap = float(max_new_tokens)
        mean = min(max(1.0, distribution.mean), cap)
        p50 = min(max(1.0, distribution.p50 if distribution.p50 is not None else mean), cap)
        p90 = min(max(mean, distribution.p90), cap)
        p95 = (
            min(max(p90, distribution.p95), cap)
            if distribution.p95 is not None
            else None
        )
        p99 = (
            min(max(p95 or p90, distribution.p99), cap)
            if distribution.p99 is not None
            else None
        )
        std = max(1.0, min(distribution.std, max(cap - mean, 1.0)))
        return LengthDistribution(
            mean=mean,
            std=std,
            p50=p50,
            p90=p90,
            p95=p95,
            p99=p99,
            family=distribution.family,
        )

    def adapt_distribution(
        self,
        request: Request,
        scheduler: "DistributionAwareScheduler",
        engine: "LLMSingleNodeSimulator | None",
    ) -> QueryDistributionDecision:
        base_distribution = request.length_distribution
        config = scheduler.config
        rng = self._rng_for_request(request, scheduler)
        jitter = max(0.0, float(config.query_distribution_jitter or 0.0))
        has_template_signal = bool(request.template_id or request.template_family or request.shared_prefix_id)
        confidence = 0.70
        if has_template_signal:
            confidence += 0.12
        if request.structured_output or request.stop_strings or not request.stop_at_eos:
            confidence += 0.08
        if request.max_new_tokens > 0:
            confidence += 0.05
        confidence = min(confidence, 0.95)

        effective_jitter = jitter * (1.05 - confidence)
        # Keep a small floor so online-predictor is visibly different from static unless jitter is zero.
        if jitter > 0.0:
            effective_jitter = max(effective_jitter, jitter * 0.20)

        mean_scale = self._bounded_lognormal_scale(
            rng,
            sigma=effective_jitter,
            low=float(config.query_distribution_mean_floor),
            high=float(config.query_distribution_mean_ceiling),
        )
        tail_scale = self._bounded_lognormal_scale(
            rng,
            sigma=effective_jitter * 1.35,
            low=max(0.50, float(config.query_distribution_mean_floor)),
            high=float(config.query_distribution_tail_ceiling),
        )

        recent_mean = self._recent_template_mean(request, engine)
        recent_blend_weight = 0.0
        predicted_mean_seed = max(1.0, base_distribution.mean * mean_scale)
        if recent_mean is not None:
            recent_blend_weight = min(0.25, max(0.0, jitter))
            predicted_mean_seed = (
                (1.0 - recent_blend_weight) * predicted_mean_seed
                + recent_blend_weight * max(1.0, recent_mean)
            )
            mean_scale = predicted_mean_seed / max(base_distribution.mean, 1.0)

        predicted = scale_length_distribution(
            base_distribution,
            mean_scale=mean_scale,
            tail_scale=tail_scale,
        )
        predicted = self._cap_distribution(predicted, request.max_new_tokens)
        return QueryDistributionDecision(
            distribution=predicted,
            source="online-predicted",
            adapter_name=self.name,
            metadata={
                "base_mean": round(float(base_distribution.mean), 6),
                "base_p90": round(float(base_distribution.p90), 6),
                "predicted_mean": round(float(predicted.mean), 6),
                "predicted_p90": round(float(predicted.p90), 6),
                "mean_scale": round(float(mean_scale), 6),
                "tail_scale": round(float(tail_scale), 6),
                "prediction_jitter": round(float(jitter), 6),
                "effective_jitter": round(float(effective_jitter), 6),
                "prediction_confidence": round(float(confidence), 6),
                "recent_template_mean": round(float(recent_mean), 6) if recent_mean is not None else None,
                "recent_blend_weight": round(float(recent_blend_weight), 6),
                "predictor_variant": "calibrated-online-length",
            },
        )


def scale_length_distribution(
    distribution: LengthDistribution,
    mean_scale: float,
    tail_scale: float,
) -> LengthDistribution:
    mean_value = max(1.0, float(distribution.mean) * mean_scale)
    p50_value = (
        max(1.0, float(distribution.p50) * mean_scale)
        if distribution.p50 is not None
        else mean_value
    )
    p90_value = max(mean_value, float(distribution.p90) * tail_scale)
    p95_value = (
        max(p90_value, float(distribution.p95) * tail_scale)
        if distribution.p95 is not None
        else None
    )
    p99_value = (
        max(p95_value or p90_value, float(distribution.p99) * tail_scale)
        if distribution.p99 is not None
        else None
    )
    std_floor = 1.0 if distribution.family == "deterministic" else max(1.0, float(distribution.std) * mean_scale)
    std_value = max(std_floor, float(distribution.std) * max(mean_scale, tail_scale))
    family = distribution.family if distribution.family != "deterministic" else "deterministic"
    if family == "deterministic":
        mean_value = max(1.0, mean_value)
        p50_value = mean_value
        p90_value = mean_value
        p95_value = mean_value
        p99_value = mean_value
        std_value = 1.0
    return LengthDistribution(
        mean=mean_value,
        std=std_value,
        p50=p50_value,
        p90=p90_value,
        p95=p95_value,
        p99=p99_value,
        family=family,
    )


class LoadAwareQueryDistributionAdapter(QueryDistributionAdapter):
    name = "query-distribution-load-aware"

    def adapt_distribution(
        self,
        request: Request,
        scheduler: "DistributionAwareScheduler",
        engine: "LLMSingleNodeSimulator | None",
    ) -> QueryDistributionDecision:
        if engine is None:
            return StaticQueryDistributionAdapter().adapt_distribution(request, scheduler, engine)

        pending_prefill = sum(len(queue) for queue in engine.prefill_queues.values())
        active_decode = len([live for live in engine.active_decode if not live.is_complete])
        queue_pressure = pending_prefill / max(scheduler._resolved_prefill_parallelism(), 1)
        decode_pressure = active_decode / max(scheduler._resolved_decode_parallelism(), 1)
        kv_pressure = engine.current_kv_tokens / max(scheduler.config.kv_capacity_tokens, 1)
        load_signal = min(2.5, 0.45 * queue_pressure + 0.35 * decode_pressure + 0.80 * kv_pressure)

        base_distribution = request.length_distribution
        return QueryDistributionDecision(
            distribution=base_distribution,
            source="static",
            adapter_name=self.name,
            metadata={
                "base_mean": round(float(base_distribution.mean), 6),
                "base_p90": round(float(base_distribution.p90), 6),
                "load_signal": round(float(load_signal), 6),
                "queue_pressure": round(float(queue_pressure), 6),
                "decode_pressure": round(float(decode_pressure), 6),
                "kv_pressure": round(float(kv_pressure), 6),
                "adapter_variant": "load-aware-observer",
            },
        )


class PrefillRouter:
    name = "prefill-router"

    def route_request(
        self,
        request: Request,
        scheduler: "DistributionAwareScheduler",
        engine: "LLMSingleNodeSimulator | None",
    ) -> PrefillRoute:
        raise NotImplementedError

    def score_prefill(
        self,
        now: float,
        request: Request,
        base_score: float,
        context: "ScheduleContext",
        scheduler: "DistributionAwareScheduler",
        engine: "LLMSingleNodeSimulator | None",
    ) -> float:
        return base_score + request.priority


class HeuristicPrefillRouter(PrefillRouter):
    name = "heuristic-prefill-router"

    def __init__(self, use_distribution_features: bool = True):
        self.use_distribution_features = use_distribution_features

    def _should_use_full_distribution(
        self,
        distribution: LengthDistribution,
        scheduler: "DistributionAwareScheduler",
        engine: "LLMSingleNodeSimulator | None",
    ) -> bool:
        if not self.use_distribution_features:
            return False
        if engine is None:
            return distribution.tail_ratio >= 1.25 or distribution.cv >= 0.25
        kv_pressure = engine.current_kv_tokens / max(scheduler.config.kv_capacity_tokens, 1)
        queue_pressure = (
            sum(len(queue) for queue in engine.prefill_queues.values())
            / max(scheduler._resolved_prefill_parallelism(), 1)
        )
        uncertainty_signal = distribution.tail_ratio + 0.75 * distribution.cv
        pressure_signal = kv_pressure + 0.15 * math.log1p(max(queue_pressure, 0.0))
        return uncertainty_signal + pressure_signal >= 1.65

    def route_request(
        self,
        request: Request,
        scheduler: "DistributionAwareScheduler",
        engine: "LLMSingleNodeSimulator | None",
    ) -> PrefillRoute:
        base_distribution = request.query_length_distribution
        use_full_distribution = self._should_use_full_distribution(base_distribution, scheduler, engine)
        distribution = base_distribution if use_full_distribution else mean_only_distribution(base_distribution)
        if use_full_distribution:
            risk = (
                0.55 * distribution.p90
                + 0.30 * distribution.mean
                + 0.15 * request.prompt_tokens
            )
            tail_ratio = distribution.tail_ratio
        else:
            risk = 0.85 * distribution.mean + 0.15 * request.prompt_tokens
            tail_ratio = 1.0
        predicted_decode_mean = request._bounded_output_estimate(base_distribution.mean)
        predicted_decode_p90 = request._bounded_output_estimate(base_distribution.quantile(0.90))
        predicted_decode_shadow = request._bounded_output_estimate(
            base_distribution.quantile(scheduler.config.kv_reservation_quantile)
        )
        current_kv_tokens = float(engine.current_kv_tokens) if engine is not None else 0.0
        current_shadow_kv_tokens = (
            float(scheduler._shadow_commitment(engine, scheduler.config.kv_reservation_quantile))
            if engine is not None
            else 0.0
        )
        kv_capacity_tokens = float(scheduler.config.kv_capacity_tokens)
        prefix_cache_tokens = float(engine.prefix_cache_physical_tokens) if engine is not None else 0.0
        active_decode_count = (
            sum(1 for live in engine.active_decode if not live.is_complete)
            if engine is not None
            else 0
        )
        first_token_pending_count = (
            sum(1 for live in engine.active_decode if not live.is_complete and live.generated_tokens == 0)
            if engine is not None
            else 0
        )
        projected_shadow_kv_tokens = scheduler._physicalize_tokens(
            request.prompt_tokens + predicted_decode_shadow
        )
        projected_prefill_kv_tokens = scheduler._physicalize_tokens(request.prompt_tokens)
        projected_total_kv_tokens = current_kv_tokens + projected_prefill_kv_tokens
        if risk <= 180 and tail_ratio <= 1.8 and request.prompt_tokens <= 256:
            queue_class = QueueClass.FAST
        elif risk <= 520 and tail_ratio <= 2.8:
            queue_class = QueueClass.NORMAL
        else:
            queue_class = QueueClass.RISKY
        return PrefillRoute(
            queue_class=queue_class,
            routing_distribution=distribution,
            route_name=self.name,
            metadata={
                "distribution_variant": "full" if use_full_distribution else "mean-only",
                "tail_ratio": round(float(tail_ratio), 6),
                "risk_signal": round(float(risk), 6),
                "query_distribution_source": request.query_distribution_source,
                "query_distribution_adapter": request.query_distribution_adapter_name,
                "predicted_decode_kv_tokens_mean": int(predicted_decode_mean),
                "predicted_decode_kv_tokens_p90": int(predicted_decode_p90),
                "predicted_decode_kv_tokens_shadow": int(predicted_decode_shadow),
                "projected_prefill_kv_tokens": int(projected_prefill_kv_tokens),
                "projected_shadow_kv_tokens": int(projected_shadow_kv_tokens),
                "current_kv_tokens": round(current_kv_tokens, 6),
                "current_shadow_kv_tokens": round(current_shadow_kv_tokens, 6),
                "kv_capacity_tokens": int(kv_capacity_tokens),
                "kv_util": round(current_kv_tokens / max(kv_capacity_tokens, 1.0), 6),
                "shadow_kv_util": round(current_shadow_kv_tokens / max(kv_capacity_tokens, 1.0), 6),
                "projected_kv_util_after_prefill": round(
                    projected_total_kv_tokens / max(kv_capacity_tokens, 1.0),
                    6,
                ),
                "projected_shadow_kv_util_after_prefill": round(
                    (current_shadow_kv_tokens + projected_shadow_kv_tokens) / max(kv_capacity_tokens, 1.0),
                    6,
                ),
                "prefix_cache_enabled": bool(scheduler.config.prefix_cache_enabled),
                "prefix_cache_physical_tokens": round(prefix_cache_tokens, 6),
                "active_decode_count": int(active_decode_count),
                "first_token_pending_count": int(first_token_pending_count),
            },
        )

    def score_prefill(
        self,
        now: float,
        request: Request,
        base_score: float,
        context: "ScheduleContext",
        scheduler: "DistributionAwareScheduler",
        engine: "LLMSingleNodeSimulator | None",
    ) -> float:
        route_bonus = float(request.prefill_route_metadata.get("route_priority_delta", 0.0))
        return base_score + route_bonus + request.priority


class DecodeRouter:
    name = "decode-router"

    def route_request(
        self,
        request: Request,
        scheduler: "DistributionAwareScheduler",
        engine: "LLMSingleNodeSimulator | None",
    ) -> DecodeRoute:
        raise NotImplementedError

    def score_decode(
        self,
        now: float,
        request: Request,
        base_score: float,
        scheduler: "DistributionAwareScheduler",
        engine: "LLMSingleNodeSimulator | None",
    ) -> float:
        return base_score + request.decode_priority + request.priority


class HeuristicDecodeRouter(DecodeRouter):
    name = "heuristic-decode-router"

    def route_request(
        self,
        request: Request,
        scheduler: "DistributionAwareScheduler",
        engine: "LLMSingleNodeSimulator | None",
    ) -> DecodeRoute:
        remaining = request.estimated_remaining_output_tokens()
        if remaining <= 64:
            batch_tag = "short"
            priority_delta = 40.0
        elif remaining <= 256:
            batch_tag = "medium"
            priority_delta = 15.0
        else:
            batch_tag = "long"
            priority_delta = -10.0
        return DecodeRoute(
            route_name=self.name,
            priority_delta=priority_delta,
            batch_tag=batch_tag,
            metadata={
                "predicted_remaining_output_tokens": int(remaining),
            },
        )

    def score_decode(
        self,
        now: float,
        request: Request,
        base_score: float,
        scheduler: "DistributionAwareScheduler",
        engine: "LLMSingleNodeSimulator | None",
    ) -> float:
        batch_tag = str(request.decode_route_metadata.get("batch_tag", "default"))
        group_bonus = {"short": 12.0, "medium": 4.0, "long": -4.0}.get(batch_tag, 0.0)
        return base_score + request.decode_priority + request.priority + group_bonus
