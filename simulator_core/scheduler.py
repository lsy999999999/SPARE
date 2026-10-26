from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from .router import (
    DecodeRouter,
    HeuristicDecodeRouter,
    HeuristicPrefillRouter,
    LoadAwareQueryDistributionAdapter,
    OnlineLengthPredictionAdapter,
    PrefillRouter,
    QueryDistributionAdapter,
    StaticQueryDistributionAdapter,
)
from .types import (
    DecodeRouterPolicy,
    PhaseMode,
    PrefillRouterPolicy,
    QueueClass,
    Request,
    ScheduleContext,
    ScheduleDecision,
    SchedulerAction,
    SimulatorConfig,
)

if TYPE_CHECKING:
    from .engine import LLMSingleNodeSimulator


class DistributionAwareScheduler:
    def __init__(
        self,
        config: SimulatorConfig,
        use_distribution_features: bool = True,
        shadow_mode: str = "quantile",
        query_distribution_adapter: Optional[QueryDistributionAdapter] = None,
        prefill_router: Optional[PrefillRouter] = None,
        decode_router: Optional[DecodeRouter] = None,
    ):
        self.config = config
        self.use_distribution_features = use_distribution_features
        self.shadow_mode = shadow_mode
        if query_distribution_adapter is None:
            query_distribution_adapter = self._build_query_distribution_adapter(config.query_distribution_mode)
        self.query_distribution_adapter = query_distribution_adapter
        self.prefill_router = prefill_router or HeuristicPrefillRouter(
            use_distribution_features=use_distribution_features
        )
        self.decode_router = decode_router or HeuristicDecodeRouter()
        self.last_prefill_policy: Optional[PrefillRouterPolicy] = None
        self.last_decode_policy: Optional[DecodeRouterPolicy] = None
        self.last_requested_prefill_policy: Optional[PrefillRouterPolicy] = None
        self.last_requested_decode_policy: Optional[DecodeRouterPolicy] = None
        self.last_prefill_decision_ms: Optional[float] = None
        self.last_decode_decision_ms: Optional[float] = None
        self.last_step_result: Dict[str, float] = {}

    @staticmethod
    def _build_query_distribution_adapter(mode: str) -> QueryDistributionAdapter:
        normalized = str(mode or "static").strip().lower().replace("_", "-")
        if normalized == "static":
            return StaticQueryDistributionAdapter()
        if normalized in {"dynamic", "online", "online-predictor", "calibrated-predictor"}:
            return OnlineLengthPredictionAdapter()
        if normalized in {"load-aware", "observer", "load-aware-observer"}:
            return LoadAwareQueryDistributionAdapter()
        raise ValueError(f"unsupported query_distribution_mode: {mode}")

    def _resolved_decode_parallelism(self) -> int:
        return max(1, int(self.config.decode_parallelism or self.config.max_decode_batch))

    def _resolved_prefill_parallelism(self) -> int:
        return max(1, int(self.config.prefill_parallelism or self.config.max_prefill_batch))

    def classify_request(self, request: Request) -> QueueClass:
        query_decision = self.query_distribution_adapter.adapt_distribution(request, self, None)
        request.online_length_distribution = query_decision.distribution
        request.query_distribution_source = query_decision.source
        request.query_distribution_adapter_name = query_decision.adapter_name
        request.query_distribution_metadata = dict(query_decision.metadata)
        route = self.prefill_router.route_request(request, self, None)
        return route.queue_class

    def route_prefill_request(self, request: Request, engine: Optional["LLMSingleNodeSimulator"]) -> None:
        query_decision = self.query_distribution_adapter.adapt_distribution(request, self, engine)
        request.online_length_distribution = query_decision.distribution
        request.query_distribution_source = query_decision.source
        request.query_distribution_adapter_name = query_decision.adapter_name
        request.query_distribution_metadata = dict(query_decision.metadata)
        route = self.prefill_router.route_request(request, self, engine)
        request.queue_class = route.queue_class
        request.routed_length_distribution = route.routing_distribution
        request.distribution_source = str(route.metadata.get("distribution_variant", "full"))
        request.prefill_route_name = route.route_name
        request.prefill_route_metadata = dict(route.metadata)
        request.prefill_route_metadata["route_priority_delta"] = float(route.priority_delta)

    def route_decode_request(self, request: Request, engine: Optional["LLMSingleNodeSimulator"]) -> None:
        route = self.decode_router.route_request(request, self, engine)
        request.decode_route_name = route.route_name
        request.decode_priority = float(route.priority_delta)
        request.decode_route_metadata = dict(route.metadata)
        request.decode_route_metadata["batch_tag"] = route.batch_tag

    def choose_action(self, engine: "LLMSingleNodeSimulator") -> Optional[SchedulerAction]:
        return None

    def choose_prefill_policy(self, engine: "LLMSingleNodeSimulator") -> Optional[PrefillRouterPolicy]:
        return None

    def choose_decode_policy(
        self,
        engine: "LLMSingleNodeSimulator",
        prefill_policy: Optional[PrefillRouterPolicy] = None,
    ) -> Optional[DecodeRouterPolicy]:
        return None

    def on_step_result(self, step_result: Dict[str, float]) -> None:
        self.last_step_result = dict(step_result)
        return None

    def on_idle(self) -> None:
        return None

    @staticmethod
    def _choice_index(value: object, choices: Sequence[object]) -> float:
        try:
            return float(list(choices).index(value) + 1) / float(max(len(choices), 1))
        except ValueError:
            return 0.0

    def _last_prefill_admission_profile_index(self) -> float:
        if self.last_prefill_policy is None:
            return 0.0
        return self._choice_index(
            self.last_prefill_policy.admission_profile,
            self.config.action_space.admission_choices,
        )

    def _last_prefill_new_admission_limit_value(self) -> float:
        if self.last_prefill_policy is None or self.last_prefill_policy.new_admission_limit is None:
            return 0.0
        return self._prefill_new_admission_limit_value(self.last_prefill_policy)

    def _prefill_new_admission_limit_value(self, policy: PrefillRouterPolicy) -> float:
        if policy.new_admission_limit is None:
            return 0.0
        limit = max(self.config.action_space.new_admission_limit_choices or [1])
        return min(float(policy.new_admission_limit) / max(float(limit), 1.0), 1.0)

    def _last_prefill_shadow_quantile_value(self) -> float:
        if self.last_prefill_policy is None or self.last_prefill_policy.shadow_quantile is None:
            return 0.0
        return min(max(float(self.last_prefill_policy.shadow_quantile), 0.0), 1.0)

    def _last_prefill_risky_admission_cap_value(self) -> float:
        if self.last_prefill_policy is None or self.last_prefill_policy.risky_admission_cap is None:
            return 0.0
        return self._prefill_risky_admission_cap_value(self.last_prefill_policy)

    def _prefill_risky_admission_cap_value(self, policy: PrefillRouterPolicy) -> float:
        if policy.risky_admission_cap is None:
            return 0.0
        cap = max(self.config.action_space.risky_admission_cap_choices or [1])
        return min(float(policy.risky_admission_cap) / max(float(cap), 1.0), 1.0)

    def _last_decode_phase_index(self) -> float:
        if self.last_decode_policy is None:
            return 0.0
        phase_label = (
            self.last_decode_policy.phase_preference.value
            if self.last_decode_policy.phase_preference is not None
            else "heuristic"
        )
        return self._choice_index(phase_label, self.config.action_space.phase_choices)

    def _last_decode_budget_profile_index(self) -> float:
        if self.last_decode_policy is None:
            return 0.0
        return self._choice_index(
            self.last_decode_policy.budget_profile,
            self.config.action_space.budget_profile_choices,
        )

    def prefill_policy_condition_vector(self, policy: Optional[PrefillRouterPolicy]) -> List[float]:
        if policy is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        return [
            self._choice_index(policy.admission_profile, self.config.action_space.admission_choices),
            self._prefill_new_admission_limit_value(policy),
            min(max(float(policy.shadow_quantile or 0.0), 0.0), 1.0),
            self._prefill_risky_admission_cap_value(policy),
            1.0 if policy.safety_filter_enabled is not False else 0.0,
        ]

    def router_boundary_mode(self) -> str:
        return str(getattr(self.config, "router_boundary_mode", "current") or "current")

    def _observation_state(self, engine: "LLMSingleNodeSimulator") -> Dict[str, object]:
        live_decodes = [req for req in engine.active_decode if not req.is_complete]
        first_token_pending = [req for req in live_decodes if req.generated_tokens == 0]
        started_prefill = [
            req
            for queue in QueueClass
            for req in engine.prefill_queues[queue]
            if req.batch_admitted_at_ms is not None
        ]
        pending_unadmitted = {
            queue: [req for req in engine.prefill_queues[queue] if req.batch_admitted_at_ms is None]
            for queue in QueueClass
        }
        pending_unadmitted_flat = [req for queue in QueueClass for req in pending_unadmitted[queue]]
        oldest_pending_ages = {
            queue: max(((engine.now_ms - req.arrival_ms) for req in pending_unadmitted[queue]), default=0.0)
            for queue in QueueClass
        }
        pending_ages = [engine.now_ms - req.arrival_ms for req in pending_unadmitted_flat]
        first_token_pending_ages = [engine.now_ms - req.arrival_ms for req in first_token_pending]
        mean_decode_remaining = (
            statistics.fmean(req.estimated_remaining_output_tokens() for req in live_decodes)
            if live_decodes
            else 0.0
        )
        mean_decode_kv = statistics.fmean(req.logical_kv_tokens for req in live_decodes) if live_decodes else 0.0
        started_prefill_remaining = sum(req.remaining_prefill_tokens for req in started_prefill)
        pending_shadow_commitment = sum(
            self._shadow_reservation_tokens(req, self.config.kv_reservation_quantile) for req in pending_unadmitted_flat
        )
        pending_decode_mean = sum(req.estimated_remaining_output_tokens() for req in pending_unadmitted_flat)
        pending_decode_shadow = sum(
            req.estimated_remaining_output_tokens(self.config.kv_reservation_quantile)
            for req in pending_unadmitted_flat
        )
        started_prefill_decode_shadow = sum(
            req.estimated_remaining_output_tokens(self.config.kv_reservation_quantile)
            for req in started_prefill
        )
        shadow_util = self._shadow_commitment(engine, self.config.kv_reservation_quantile) / max(
            self.config.kv_capacity_tokens * self.config.allow_shadow_oversubscription,
            1.0,
        )
        recent_query_mix = engine.recent_query_mix_summary()
        prefix_template_state = self._prefix_template_observation_state(
            engine,
            pending_unadmitted_flat=pending_unadmitted_flat,
            started_prefill=started_prefill,
            live_decodes=live_decodes,
        )
        capacity = max(float(self.config.kv_capacity_tokens), 1.0)
        decode_parallelism = max(float(self._resolved_decode_parallelism()), 1.0)
        prefill_parallelism = max(float(self._resolved_prefill_parallelism()), 1.0)
        prefill_load_scale = max(prefill_parallelism * 4.0, 1.0)
        return {
            "now": self._clip(math.log1p(max(float(engine.now_ms), 0.0)) / 10.0, 0.0, 1.0),
            "kv_util": self._ratio(float(engine.current_kv_tokens), capacity, 2.0),
            "shadow_util": self._clip(shadow_util, 0.0, 2.0),
            "decode_load": self._ratio(len(live_decodes), decode_parallelism, 4.0),
            "first_token_pending_load": self._ratio(len(first_token_pending), decode_parallelism, 4.0),
            "started_prefill_load": self._ratio(len(started_prefill), prefill_load_scale, 4.0),
            "pending_unadmitted_load": self._ratio(len(pending_unadmitted_flat), prefill_load_scale, 8.0),
            "pending_fast_load": self._ratio(len(pending_unadmitted[QueueClass.FAST]), prefill_load_scale, 8.0),
            "pending_normal_load": self._ratio(len(pending_unadmitted[QueueClass.NORMAL]), prefill_load_scale, 8.0),
            "pending_risky_load": self._ratio(len(pending_unadmitted[QueueClass.RISKY]), prefill_load_scale, 8.0),
            "oldest_pending_fast": self._ratio(oldest_pending_ages[QueueClass.FAST], 500.0, 4.0),
            "oldest_pending_normal": self._ratio(oldest_pending_ages[QueueClass.NORMAL], 500.0, 4.0),
            "oldest_pending_risky": self._ratio(oldest_pending_ages[QueueClass.RISKY], 500.0, 4.0),
            "mean_pending_age": self._ratio(statistics.fmean(pending_ages), 500.0, 4.0) if pending_ages else 0.0,
            "max_pending_age": self._ratio(max(pending_ages, default=0.0), 500.0, 4.0),
            "mean_first_token_pending_age": (
                self._ratio(statistics.fmean(first_token_pending_ages), 500.0, 4.0)
                if first_token_pending_ages
                else 0.0
            ),
            "max_first_token_pending_age": self._ratio(max(first_token_pending_ages, default=0.0), 500.0, 4.0),
            "mean_decode_remaining": self._ratio(mean_decode_remaining, 512.0, 4.0),
            "mean_decode_kv": self._ratio(mean_decode_kv, capacity, 1.0),
            "started_prefill_remaining": self._ratio(started_prefill_remaining, capacity, 4.0),
            "pending_shadow_commitment": self._ratio(pending_shadow_commitment, capacity, 4.0),
            "pending_decode_kv_mean": self._ratio(pending_decode_mean, capacity, 4.0),
            "pending_decode_kv_shadow": self._ratio(pending_decode_shadow, capacity, 4.0),
            "started_prefill_decode_kv_shadow": self._ratio(started_prefill_decode_shadow, capacity, 4.0),
            "prefix_cache_enabled": 1.0 if self.config.prefix_cache_enabled else 0.0,
            "prefix_cache_util": self._ratio(float(engine.prefix_cache_physical_tokens), capacity, 2.0),
            "last_prefill_admission_profile": self._last_prefill_admission_profile_index(),
            "last_prefill_new_admission_limit": self._last_prefill_new_admission_limit_value(),
            "last_prefill_shadow_quantile": self._last_prefill_shadow_quantile_value(),
            "last_prefill_risky_admission_cap": self._last_prefill_risky_admission_cap_value(),
            "last_decode_phase_preference": self._last_decode_phase_index(),
            "last_decode_budget_profile": self._last_decode_budget_profile_index(),
            "last_decode_tokens": min(float(self.last_step_result.get("decode_tokens", 0.0)) / 256.0, 4.0),
            "last_first_tokens_emitted": min(
                float(self.last_step_result.get("first_tokens_emitted", 0.0))
                / max(self._resolved_decode_parallelism(), 1),
                1.0,
            ),
            "recent_q8_prompt_mean": float(recent_query_mix["recent_q8_prompt_mean"]),
            "recent_q8_output_mean": float(recent_query_mix["recent_q8_output_mean"]),
            "recent_q8_tail_ratio_mean": float(recent_query_mix["recent_q8_tail_ratio_mean"]),
            "recent_q8_output_cv_mean": float(recent_query_mix["recent_q8_output_cv_mean"]),
            "recent_q8_risky_fraction": float(recent_query_mix["recent_q8_risky_fraction"]),
            "recent_q8_interarrival_mean": float(recent_query_mix["recent_q8_interarrival_mean"]),
            "recent_q32_prompt_mean": float(recent_query_mix["recent_q32_prompt_mean"]),
            "recent_q32_output_mean": float(recent_query_mix["recent_q32_output_mean"]),
            "recent_q32_tail_ratio_mean": float(recent_query_mix["recent_q32_tail_ratio_mean"]),
            "recent_q32_output_cv_mean": float(recent_query_mix["recent_q32_output_cv_mean"]),
            "recent_q32_risky_fraction": float(recent_query_mix["recent_q32_risky_fraction"]),
            "recent_q32_interarrival_mean": float(recent_query_mix["recent_q32_interarrival_mean"]),
            "recent_q128_prompt_mean": float(recent_query_mix["recent_q128_prompt_mean"]),
            "recent_q128_output_mean": float(recent_query_mix["recent_q128_output_mean"]),
            "recent_q128_tail_ratio_mean": float(recent_query_mix["recent_q128_tail_ratio_mean"]),
            "recent_q128_output_cv_mean": float(recent_query_mix["recent_q128_output_cv_mean"]),
            "recent_q128_risky_fraction": float(recent_query_mix["recent_q128_risky_fraction"]),
            "recent_q128_interarrival_mean": float(recent_query_mix["recent_q128_interarrival_mean"]),
            **prefix_template_state,
        }

    @staticmethod
    def _safe_fraction(numerator: float, denominator: float) -> float:
        if denominator <= 0.0:
            return 0.0
        return float(numerator) / float(denominator)

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return min(max(float(value), float(lower)), float(upper))

    @classmethod
    def _ratio(cls, numerator: float, denominator: float, cap: float) -> float:
        return cls._clip(float(numerator) / max(float(denominator), 1.0), 0.0, float(cap))

    def _warm_prefix_hit_potential(
        self,
        engine: "LLMSingleNodeSimulator",
        requests: Sequence[Request],
    ) -> float:
        if not self.config.prefix_cache_enabled:
            return 0.0
        if not requests:
            return 0.0
        warm_hits = 0
        for request in requests:
            cacheable_tokens = request.cacheable_shared_prefix_tokens(engine.prefix_cache_block_size_tokens)
            if cacheable_tokens <= 0 or not request.shared_prefix_id:
                continue
            cache_key = f"{request.model_id}::{request.shared_prefix_id}::{cacheable_tokens}"
            entry = engine.prefix_cache.get(cache_key)
            if entry is not None and entry.warm:
                warm_hits += 1
        return min(self._safe_fraction(warm_hits, len(requests)), 1.0)

    def _prefix_template_observation_state(
        self,
        engine: "LLMSingleNodeSimulator",
        *,
        pending_unadmitted_flat: Sequence[Request],
        started_prefill: Sequence[Request],
        live_decodes: Sequence[Request],
    ) -> Dict[str, float]:
        rows = list(engine.recent_query_history)
        recent32 = rows[-32:]
        recent128 = rows[-128:]

        def top_family_share(window_rows: Sequence[Dict[str, object]]) -> float:
            families = [str(row.get("template_family") or "") for row in window_rows if str(row.get("template_family") or "")]
            if not families:
                return 0.0
            counts = Counter(families)
            return min(self._safe_fraction(max(counts.values()), len(families)), 1.0)

        def rollout_pressure(window_rows: Sequence[Dict[str, object]]) -> float:
            family_versions: dict[str, set[str]] = defaultdict(set)
            total = 0
            rollover = 0
            for row in window_rows:
                family = str(row.get("template_family") or "")
                version = str(row.get("template_version") or "")
                if not family:
                    continue
                if version:
                    family_versions[family].add(version)
                total += 1
            if total <= 0:
                return 0.0
            for row in window_rows:
                family = str(row.get("template_family") or "")
                if family and len(family_versions.get(family, set())) > 1:
                    rollover += 1
            return min(self._safe_fraction(rollover, total), 1.0)

        inflight_requests = list(pending_unadmitted_flat) + list(started_prefill) + list(live_decodes)
        pending_requests = list(pending_unadmitted_flat) + list(started_prefill)
        hottest_family = ""
        if recent32:
            hot_counts = Counter(
                str(row.get("template_family") or "")
                for row in recent32
                if str(row.get("template_family") or "")
            )
            if hot_counts:
                hottest_family = hot_counts.most_common(1)[0][0]

        hottest_family_pending_share = 0.0
        if hottest_family and pending_requests:
            hottest_family_pending_share = min(
                self._safe_fraction(
                    sum(1 for request in pending_requests if request.template_family == hottest_family),
                    len(pending_requests),
                ),
                1.0,
            )

        return {
            "template_hotness_recent32": top_family_share(recent32),
            "template_hotness_recent128": top_family_share(recent128),
            "template_hotness_pending_alignment": hottest_family_pending_share,
            "version_rollout_pressure_recent32": rollout_pressure(recent32),
            "version_rollout_pressure_recent128": rollout_pressure(recent128),
            "version_rollout_pressure_inflight": min(
                self._safe_fraction(
                    sum(1 for request in inflight_requests if str(request.template_version or "") != "v1"),
                    len(inflight_requests),
                ),
                1.0,
            ) if inflight_requests else 0.0,
            "prefix_cache_warm_hit_potential_pending": self._warm_prefix_hit_potential(engine, pending_requests),
            "prefix_cache_warm_hit_potential_unadmitted": self._warm_prefix_hit_potential(engine, pending_unadmitted_flat),
            "prefix_cache_warm_hit_potential_decode": self._warm_prefix_hit_potential(engine, live_decodes),
        }

    def _copy_action(self, action: SchedulerAction) -> SchedulerAction:
        return SchedulerAction(
            phase_preference=action.phase_preference,
            admission_profile=action.admission_profile,
            new_admission_limit=action.new_admission_limit,
            shadow_quantile=action.shadow_quantile,
            budget_profile=action.budget_profile,
            risky_admission_cap=action.risky_admission_cap,
            prefill_safety_filter_enabled=action.prefill_safety_filter_enabled,
            decode_safety_filter_enabled=action.decode_safety_filter_enabled,
        )

    def _prefill_safety_filter_enabled(self, action: SchedulerAction) -> bool:
        if action.prefill_safety_filter_enabled is None:
            return True
        return bool(action.prefill_safety_filter_enabled)

    def _decode_safety_filter_enabled(self, action: SchedulerAction) -> bool:
        if action.decode_safety_filter_enabled is None:
            return True
        return bool(action.decode_safety_filter_enabled)

    def _stabilize_action(
        self,
        engine: "LLMSingleNodeSimulator",
        action: Optional[SchedulerAction],
    ) -> Tuple[Optional[SchedulerAction], bool, List[str]]:
        if action is None or not self.config.enable_action_safety_filter:
            return action, False, []
        prefill_safety_enabled = self._prefill_safety_filter_enabled(action)
        decode_safety_enabled = self._decode_safety_filter_enabled(action)

        state = self._observation_state(engine)
        kv_util = float(state["kv_util"])
        first_token_pending_load = float(state["first_token_pending_load"])
        pending_unadmitted_load = float(state["pending_unadmitted_load"])
        live_decode_count = len([req for req in engine.active_decode if not req.is_complete])
        first_token_pending_count = len(
            [req for req in engine.active_decode if not req.is_complete and req.generated_tokens == 0]
        )
        decode_load = live_decode_count / max(self._resolved_decode_parallelism(), 1)
        pending_unadmitted_count = sum(
            1
            for queue in QueueClass
            for req in engine.prefill_queues[queue]
            if req.batch_admitted_at_ms is None
        )
        pending_short_count = sum(
            1
            for queue in (QueueClass.FAST, QueueClass.NORMAL)
            for req in engine.prefill_queues[queue]
            if req.batch_admitted_at_ms is None
        )
        pending_risky_count = sum(
            1
            for req in engine.prefill_queues[QueueClass.RISKY]
            if req.batch_admitted_at_ms is None
        )
        started_prefill_count = sum(
            1
            for queue in QueueClass
            for req in engine.prefill_queues[queue]
            if req.batch_admitted_at_ms is not None and not req.is_complete
        )
        severe_kv_threshold = min(max(float(self.config.action_safety_kv_util_threshold), 0.0), 1.0)
        drain_kv_threshold = min(max(float(self.config.action_safety_drain_kv_util_threshold), 0.0), 1.0)
        severe_kv_pressure = kv_util >= severe_kv_threshold
        first_token_backlog_threshold = max(
            2,
            int(math.ceil(self._resolved_decode_parallelism() * self.config.action_safety_first_token_load_threshold)),
        )
        severe_decode_pressure = (
            first_token_pending_count >= first_token_backlog_threshold
            and first_token_pending_load >= self.config.action_safety_first_token_load_threshold
        )
        drain_required = live_decode_count > 0 and (
            kv_util >= drain_kv_threshold
            or (
                first_token_pending_load >= max(0.25, self.config.action_safety_first_token_load_threshold)
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
        needs_risky_admission_floor = (
            needs_prefill_liveness
            and pending_short_count <= 0
            and pending_risky_count > 0
        )
        liveness_prefill_safety_enabled = prefill_safety_enabled or needs_prefill_liveness
        liveness_decode_safety_enabled = decode_safety_enabled or needs_prefill_liveness
        needs_short_admission_floor = (
            pending_short_count > 0
            and not drain_required
            and (
                action.new_admission_limit is None
                or action.new_admission_limit < 1
                or action.budget_profile == "decode-drain"
                or action.phase_preference == PhaseMode.DECODE_PRIORITY
            )
        )
        risky_cap_base = (
            self._resolved_prefill_parallelism()
            if action.risky_admission_cap is None
            else max(0, int(action.risky_admission_cap))
        )
        guarded_risky_cap, risky_cap_reason = self._guarded_risky_admission_cap(
            first_token_pending_count=first_token_pending_count,
            decode_load=decode_load,
            base_cap=risky_cap_base,
        )
        needs_risky_cap_guard = risky_cap_reason is not None and guarded_risky_cap < risky_cap_base
        if not prefill_safety_enabled and not decode_safety_enabled and not needs_prefill_liveness:
            return action, False, []
        if not (
            severe_kv_pressure
            or severe_decode_pressure
            or severe_prefill_pressure
            or drain_required
            or needs_prefill_liveness
            or needs_risky_admission_floor
            or needs_short_admission_floor
            or needs_risky_cap_guard
        ):
            return action, False, []

        stabilized = self._copy_action(action)
        reasons: List[str] = []

        if (
            decode_safety_enabled
            and
            stabilized.phase_preference == PhaseMode.PREFILL_PRIORITY
            and live_decode_count > 0
            and (kv_util >= 0.55 or pending_unadmitted_count > 0)
        ):
            stabilized.phase_preference = PhaseMode.DECODE_PRIORITY if severe_decode_pressure else PhaseMode.MIXED
            reasons.append("live_decode_prefill_priority_guard")

        if severe_decode_pressure or severe_kv_pressure:
            safe_phase = PhaseMode.DECODE_PRIORITY if severe_decode_pressure else PhaseMode.MIXED
            if decode_safety_enabled and stabilized.phase_preference == PhaseMode.PREFILL_PRIORITY:
                stabilized.phase_preference = safe_phase
                reasons.append("phase_decode_backlog_guard")
            if prefill_safety_enabled and drain_required and stabilized.admission_profile in {"tail-relief", "prefill-drain"}:
                stabilized.admission_profile = "balanced"
                reasons.append("admission_profile_guard")
            max_new_admissions = max(1, int(self.config.action_safety_max_new_admissions))
            if (
                prefill_safety_enabled
                and
                stabilized.new_admission_limit is not None
                and stabilized.new_admission_limit > max_new_admissions
            ):
                stabilized.new_admission_limit = max_new_admissions
                reasons.append("new_admission_limit_guard")
            if (
                prefill_safety_enabled
                and severe_kv_pressure
                and stabilized.shadow_quantile is not None
                and stabilized.shadow_quantile > 0.90
            ):
                stabilized.shadow_quantile = 0.90
                reasons.append("shadow_quantile_guard")

            if drain_required:
                if decode_safety_enabled and stabilized.phase_preference != PhaseMode.DECODE_PRIORITY:
                    stabilized.phase_preference = PhaseMode.DECODE_PRIORITY
                    reasons.append("kv_recovery_phase_guard")
                if (
                    prefill_safety_enabled
                    and stabilized.new_admission_limit is not None
                    and stabilized.new_admission_limit > 0
                ):
                    stabilized.new_admission_limit = 0
                    reasons.append("kv_recovery_admission_guard")
                if (
                    prefill_safety_enabled
                    and stabilized.risky_admission_cap is not None
                    and stabilized.risky_admission_cap > 0
                ):
                    stabilized.risky_admission_cap = 0
                    reasons.append("kv_recovery_risky_guard")
                if decode_safety_enabled and stabilized.budget_profile != "decode-drain":
                    stabilized.budget_profile = "decode-drain"
                    reasons.append("kv_recovery_budget_guard")

        if drain_required and not (severe_decode_pressure or severe_kv_pressure):
            if decode_safety_enabled and stabilized.phase_preference != PhaseMode.DECODE_PRIORITY:
                stabilized.phase_preference = PhaseMode.DECODE_PRIORITY
                reasons.append("kv_drain_phase_guard")
            if (
                prefill_safety_enabled
                and stabilized.new_admission_limit is not None
                and stabilized.new_admission_limit > 0
            ):
                stabilized.new_admission_limit = 0
                reasons.append("kv_drain_admission_guard")
            if (
                prefill_safety_enabled
                and stabilized.risky_admission_cap is not None
                and stabilized.risky_admission_cap > 0
            ):
                stabilized.risky_admission_cap = 0
                reasons.append("kv_drain_risky_guard")
            if decode_safety_enabled and stabilized.budget_profile != "decode-drain":
                stabilized.budget_profile = "decode-drain"
                reasons.append("kv_drain_budget_guard")

        if decode_safety_enabled and severe_decode_pressure:
            if stabilized.budget_profile in {"prefill-heavy", "decode-strict"}:
                stabilized.budget_profile = "decode-heavy"
                reasons.append("budget_decode_guard")
        elif decode_safety_enabled and severe_kv_pressure and stabilized.budget_profile == "prefill-heavy":
            stabilized.budget_profile = "balanced"
            reasons.append("budget_kv_guard")

        if (
            prefill_safety_enabled
            and severe_prefill_pressure
            and drain_required
            and stabilized.admission_profile == "tail-relief"
        ):
            stabilized.admission_profile = "balanced"
            reasons.append("prefill_queue_guard")
        if prefill_safety_enabled and severe_prefill_pressure and stabilized.new_admission_limit is not None:
            max_new_admissions = max(1, int(self.config.action_safety_max_new_admissions))
            if stabilized.new_admission_limit > max_new_admissions:
                stabilized.new_admission_limit = max_new_admissions
                reasons.append("prefill_queue_admission_guard")
        if (severe_prefill_pressure or needs_prefill_liveness) and live_decode_count == 0:
            if liveness_decode_safety_enabled and stabilized.phase_preference == PhaseMode.DECODE_PRIORITY:
                stabilized.phase_preference = PhaseMode.MIXED
                reasons.append("prefill_liveness_phase_guard")
            if liveness_decode_safety_enabled and stabilized.budget_profile == "decode-drain":
                stabilized.budget_profile = "balanced"
                reasons.append("prefill_liveness_budget_guard")
            if (
                liveness_prefill_safety_enabled
                and stabilized.new_admission_limit is not None
                and stabilized.new_admission_limit < 1
            ):
                stabilized.new_admission_limit = 1
                reasons.append("prefill_liveness_admission_guard")
            if (
                liveness_prefill_safety_enabled
                and needs_risky_admission_floor
                and stabilized.risky_admission_cap is not None
                and stabilized.risky_admission_cap < 1
            ):
                stabilized.risky_admission_cap = 1
                reasons.append("prefill_liveness_risky_cap_floor")

        if needs_short_admission_floor:
            if decode_safety_enabled and stabilized.budget_profile == "decode-drain":
                stabilized.budget_profile = "decode-heavy" if live_decode_count > 0 else "balanced"
                reasons.append("short_admission_budget_floor")
            if decode_safety_enabled and stabilized.phase_preference == PhaseMode.DECODE_PRIORITY:
                stabilized.phase_preference = PhaseMode.MIXED
                reasons.append("short_admission_phase_floor")
            short_admission_floor = min(
                max(1, pending_short_count),
                max(1, int(self.config.action_safety_max_new_admissions)),
                self._resolved_prefill_parallelism(),
            )
            if (
                prefill_safety_enabled
                and stabilized.new_admission_limit is not None
                and stabilized.new_admission_limit < short_admission_floor
            ):
                stabilized.new_admission_limit = short_admission_floor
                reasons.append("short_admission_limit_floor")

        if prefill_safety_enabled and needs_risky_cap_guard:
            stabilized.risky_admission_cap = guarded_risky_cap
            reasons.append(risky_cap_reason)

        return stabilized, bool(reasons), reasons

    def _guarded_risky_admission_cap(
        self,
        *,
        first_token_pending_count: int,
        decode_load: float,
        base_cap: int,
    ) -> Tuple[int, Optional[str]]:
        cap = max(0, int(base_cap))
        if first_token_pending_count > 0:
            guarded = min(cap, max(0, int(self.config.action_safety_first_token_max_risky_admissions)))
            return guarded, "first_token_risky_cap" if guarded < cap else None
        if decode_load >= float(self.config.action_safety_decode_guard_load_threshold):
            guarded = min(cap, max(0, int(self.config.action_safety_decode_guard_max_risky_admissions)))
            return guarded, "decode_guard_risky_cap" if guarded < cap else None
        return cap, None

    def build_prefill_router_observation(self, engine: "LLMSingleNodeSimulator") -> List[float]:
        state = self._observation_state(engine)
        values = [
            float(state["now"]),
            float(state["kv_util"]),
            float(state["shadow_util"]),
            float(state["started_prefill_load"]),
            float(state["pending_unadmitted_load"]),
            float(state["pending_fast_load"]),
            float(state["pending_normal_load"]),
            float(state["pending_risky_load"]),
            float(state["oldest_pending_fast"]),
            float(state["oldest_pending_normal"]),
            float(state["oldest_pending_risky"]),
            float(state["mean_pending_age"]),
            float(state["max_pending_age"]),
            float(state["started_prefill_remaining"]),
            float(state["pending_shadow_commitment"]),
            float(state["pending_decode_kv_mean"]),
            float(state["pending_decode_kv_shadow"]),
            float(state["started_prefill_decode_kv_shadow"]),
            float(state["prefix_cache_enabled"]),
            float(state["prefix_cache_util"]),
            float(state["decode_load"]),
            float(state["first_token_pending_load"]),
            float(state["last_decode_phase_preference"]),
            float(state["last_decode_budget_profile"]),
            float(state["last_decode_tokens"]),
            float(state["last_first_tokens_emitted"]),
            float(state["recent_q8_prompt_mean"]),
            float(state["recent_q8_output_mean"]),
            float(state["recent_q8_tail_ratio_mean"]),
            float(state["recent_q8_output_cv_mean"]),
            float(state["recent_q8_risky_fraction"]),
            float(state["recent_q8_interarrival_mean"]),
            float(state["recent_q32_prompt_mean"]),
            float(state["recent_q32_output_mean"]),
            float(state["recent_q32_tail_ratio_mean"]),
            float(state["recent_q32_output_cv_mean"]),
            float(state["recent_q32_risky_fraction"]),
            float(state["recent_q32_interarrival_mean"]),
            float(state["recent_q128_prompt_mean"]),
            float(state["recent_q128_output_mean"]),
            float(state["recent_q128_tail_ratio_mean"]),
            float(state["recent_q128_output_cv_mean"]),
            float(state["recent_q128_risky_fraction"]),
            float(state["recent_q128_interarrival_mean"]),
            float(state["template_hotness_recent32"]),
            float(state["template_hotness_recent128"]),
            float(state["template_hotness_pending_alignment"]),
            float(state["version_rollout_pressure_recent32"]),
            float(state["version_rollout_pressure_recent128"]),
            float(state["version_rollout_pressure_inflight"]),
            float(state["prefix_cache_warm_hit_potential_pending"]),
            float(state["prefix_cache_warm_hit_potential_unadmitted"]),
            float(state["prefix_cache_warm_hit_potential_decode"]),
        ]
        if self.router_boundary_mode() == "strict_env_only" and len(values) >= 26:
            values[22:26] = [0.0, 0.0, 0.0, 0.0]
        return values

    def build_decode_router_observation(
        self,
        engine: "LLMSingleNodeSimulator",
        prefill_policy: Optional[PrefillRouterPolicy] = None,
    ) -> List[float]:
        state = self._observation_state(engine)
        mode = self.router_boundary_mode()
        if mode == "strict_env_only":
            prefill_condition = [0.0, 0.0, 0.0, 0.0, 0.0]
            last_prefill_values = [0.0, 0.0, 0.0, 0.0]
        elif mode == "actual_prefill_conditioned_decode":
            prefill_condition = self.prefill_policy_condition_vector(prefill_policy)
            last_prefill_values = [
                float(state["last_prefill_admission_profile"]),
                float(state["last_prefill_new_admission_limit"]),
                float(state["last_prefill_shadow_quantile"]),
                float(state["last_prefill_risky_admission_cap"]),
            ]
        else:
            prefill_condition = self.prefill_policy_condition_vector(prefill_policy)
            last_prefill_values = [
                float(state["last_prefill_admission_profile"]),
                float(state["last_prefill_new_admission_limit"]),
                float(state["last_prefill_shadow_quantile"]),
                float(state["last_prefill_risky_admission_cap"]),
            ]
        return [
            float(state["now"]),
            float(state["kv_util"]),
            float(state["shadow_util"]),
            float(state["decode_load"]),
            float(state["first_token_pending_load"]),
            float(state["mean_first_token_pending_age"]),
            float(state["max_first_token_pending_age"]),
            float(state["mean_decode_remaining"]),
            float(state["mean_decode_kv"]),
            float(state["started_prefill_load"]),
            float(state["started_prefill_remaining"]),
            float(state["started_prefill_decode_kv_shadow"]),
            float(state["pending_unadmitted_load"]),
            float(state["pending_decode_kv_mean"]),
            float(state["pending_decode_kv_shadow"]),
            float(state["mean_pending_age"]),
            float(state["prefix_cache_enabled"]),
            float(state["prefix_cache_util"]),
            *last_prefill_values,
            float(state["recent_q8_prompt_mean"]),
            float(state["recent_q8_output_mean"]),
            float(state["recent_q8_tail_ratio_mean"]),
            float(state["recent_q8_output_cv_mean"]),
            float(state["recent_q8_risky_fraction"]),
            float(state["recent_q8_interarrival_mean"]),
            float(state["recent_q32_prompt_mean"]),
            float(state["recent_q32_output_mean"]),
            float(state["recent_q32_tail_ratio_mean"]),
            float(state["recent_q32_output_cv_mean"]),
            float(state["recent_q32_risky_fraction"]),
            float(state["recent_q32_interarrival_mean"]),
            float(state["recent_q128_prompt_mean"]),
            float(state["recent_q128_output_mean"]),
            float(state["recent_q128_tail_ratio_mean"]),
            float(state["recent_q128_output_cv_mean"]),
            float(state["recent_q128_risky_fraction"]),
            float(state["recent_q128_interarrival_mean"]),
            float(state["template_hotness_recent32"]),
            float(state["template_hotness_recent128"]),
            float(state["template_hotness_pending_alignment"]),
            float(state["version_rollout_pressure_recent32"]),
            float(state["version_rollout_pressure_recent128"]),
            float(state["version_rollout_pressure_inflight"]),
            float(state["prefix_cache_warm_hit_potential_pending"]),
            float(state["prefix_cache_warm_hit_potential_unadmitted"]),
            float(state["prefix_cache_warm_hit_potential_decode"]),
            *prefill_condition,
        ]

    def build_observation(self, engine: "LLMSingleNodeSimulator") -> List[float]:
        return self.build_prefill_router_observation(engine) + self.build_decode_router_observation(engine)

    def _context_from_action(
        self,
        engine: "LLMSingleNodeSimulator",
        action: Optional[SchedulerAction],
    ) -> ScheduleContext:
        admission_bias, started_prefill_boost, new_admission_bonus, shadow_risk_weight = self._admission_profile(
            action.admission_profile if action else "balanced"
        )
        shadow_quantile = self.config.kv_reservation_quantile
        if action and action.shadow_quantile is not None:
            shadow_quantile = action.shadow_quantile
        heuristic_phase = self._heuristic_phase_mode(engine)
        phase_mode = action.phase_preference if action and action.phase_preference is not None else heuristic_phase
        new_admission_limit = self._resolved_prefill_parallelism()
        if action and action.new_admission_limit is not None:
            new_admission_limit = max(0, min(self._resolved_prefill_parallelism(), action.new_admission_limit))
        decode_quota_scale, prefill_chunk_cap = self._budget_profile(action.budget_profile if action else "balanced")
        risky_admission_cap = self._resolved_prefill_parallelism()
        if action and action.risky_admission_cap is not None:
            risky_admission_cap = max(0, min(self._resolved_prefill_parallelism(), action.risky_admission_cap))
        return ScheduleContext(
            phase_mode=phase_mode,
            shadow_quantile=shadow_quantile,
            admission_bias=admission_bias,
            started_prefill_boost=started_prefill_boost,
            new_admission_bonus=new_admission_bonus,
            shadow_risk_weight=shadow_risk_weight,
            new_admission_limit=new_admission_limit,
            decode_quota_scale=decode_quota_scale,
            prefill_chunk_cap=prefill_chunk_cap,
            risky_admission_cap=risky_admission_cap,
        )

    def _heuristic_phase_mode(self, engine: "LLMSingleNodeSimulator") -> PhaseMode:
        live_decodes = [req for req in engine.active_decode if not req.is_complete]
        first_token_pending = [req for req in live_decodes if req.generated_tokens == 0]
        kv_util = engine.current_kv_tokens / max(self.config.kv_capacity_tokens, 1)
        if first_token_pending or (kv_util >= self.config.kv_pressure_threshold and live_decodes):
            return PhaseMode.DECODE_PRIORITY
        if engine.prefill_queues[QueueClass.FAST] and len(live_decodes) <= 2:
            return PhaseMode.PREFILL_PRIORITY
        return PhaseMode.MIXED

    def _request_risk_signal(self, request: Request) -> float:
        distribution = request.effective_length_distribution
        if self.use_distribution_features:
            return (
                0.55 * distribution.p90
                + 0.30 * distribution.mean
                + 0.15 * request.prompt_tokens
            )
        return 0.85 * distribution.mean + 0.15 * request.prompt_tokens

    def _physicalize_tokens(self, logical_tokens: int) -> int:
        if self.config.kv_allocator_mode == "token":
            return int(logical_tokens)
        block_size = max(1, int(self.config.kv_block_size_tokens))
        if logical_tokens <= 0:
            return 0
        return int(math.ceil(logical_tokens / block_size) * block_size)

    def _shadow_reservation_tokens(self, request: Request, quantile: float) -> int:
        logical_tokens = request.mean_kv_target() if self.shadow_mode == "mean" else request.shadow_kv_target(quantile)
        return self._physicalize_tokens(logical_tokens)

    def _shadow_commitment(self, engine: "LLMSingleNodeSimulator", quantile: float) -> int:
        committed = 0
        for request in engine.requests:
            if request.is_complete or not request.has_started:
                continue
            committed += self._shadow_reservation_tokens(request, quantile)
        return committed

    def _admission_shadow_fraction_limit(
        self,
        *,
        kv_util: float,
        shadow_util: float,
        live_decode_count: int,
    ) -> float:
        pressure = max(kv_util, shadow_util)
        if pressure < 0.55:
            return 1.0
        if live_decode_count > 0 and pressure >= 0.55:
            return 0.18
        if pressure >= 0.90:
            return 0.08
        if pressure >= 0.80:
            return 0.12
        if pressure >= 0.70:
            return 0.20
        return 0.30

    def _should_defer_new_admission_for_kv_pressure(
        self,
        request: Request,
        *,
        extra_shadow_tokens: int,
        kv_budget: int,
        local_shadow_budget: int,
        kv_util: float,
        shadow_util: float,
        live_decode_count: int,
        admitted_any_new: bool,
    ) -> bool:
        capacity = max(1, int(self.config.kv_capacity_tokens))
        if self.config.overflow_policy != "block":
            return False
        pressure = max(kv_util, shadow_util)
        if live_decode_count <= 0 and pressure < 0.55 and not admitted_any_new:
            return False
        projected_shadow_util = (
            float(self._physicalize_tokens(request.logical_kv_tokens) + extra_shadow_tokens)
            / float(capacity)
        )
        projected_total_shadow_util = shadow_util + float(extra_shadow_tokens) / float(capacity)
        if projected_total_shadow_util >= 0.90 and projected_shadow_util >= 0.12:
            return True
        if projected_total_shadow_util >= 0.80 and projected_shadow_util >= 0.20:
            return True
        if kv_util >= 0.60 and projected_shadow_util >= 0.30:
            return True
        if pressure < 0.55:
            return False
        if live_decode_count <= 0 and not admitted_any_new:
            if projected_shadow_util >= 0.25 or projected_total_shadow_util >= 0.95:
                return True
            return False

        shadow_fraction = float(extra_shadow_tokens) / float(capacity)
        limit = self._admission_shadow_fraction_limit(
            kv_util=kv_util,
            shadow_util=shadow_util,
            live_decode_count=live_decode_count,
        )
        if shadow_fraction > limit:
            return True
        if live_decode_count > 0 and request.queue_class == QueueClass.RISKY and pressure >= 0.55:
            return True
        if pressure >= 0.70 and extra_shadow_tokens > max(kv_budget, 1) * 2:
            return True
        if pressure >= 0.80 and extra_shadow_tokens > max(local_shadow_budget, 1) * 2 // 3:
            return True
        return False

    def _admission_profile(
        self,
        profile: str,
    ) -> Tuple[Dict[QueueClass, float], float, float, float]:
        base = {
            QueueClass.FAST: self.config.queue_bias_fast,
            QueueClass.NORMAL: self.config.queue_bias_normal,
            QueueClass.RISKY: self.config.queue_bias_risky,
        }
        if profile == "ttft-greedy":
            return (
                {
                    QueueClass.FAST: base[QueueClass.FAST] + 60.0,
                    QueueClass.NORMAL: base[QueueClass.NORMAL] + 10.0,
                    QueueClass.RISKY: base[QueueClass.RISKY] - 40.0,
                },
                120.0,
                150.0,
                0.020,
            )
        if profile == "balanced":
            return (base, 150.0, 60.0, 0.010)
        if profile == "prefill-drain":
            return (
                {
                    QueueClass.FAST: base[QueueClass.FAST] + 20.0,
                    QueueClass.NORMAL: base[QueueClass.NORMAL] + 10.0,
                    QueueClass.RISKY: base[QueueClass.RISKY] - 10.0,
                },
                240.0,
                15.0,
                0.015,
            )
        if profile == "tail-relief":
            return (
                {
                    QueueClass.FAST: base[QueueClass.FAST] - 20.0,
                    QueueClass.NORMAL: base[QueueClass.NORMAL] + 10.0,
                    QueueClass.RISKY: base[QueueClass.RISKY] + 90.0,
                },
                180.0,
                35.0,
                0.005,
            )
        return (base, 150.0, 60.0, 0.010)

    def _budget_profile(self, profile: str) -> Tuple[float, int]:
        if profile == "decode-heavy":
            return (1.45, 2)
        if profile == "decode-drain":
            return (2.20, 0)
        if profile == "prefill-heavy":
            return (0.55, 8)
        if profile == "decode-strict":
            return (1.80, 1)
        return (1.0, self.config.max_prefill_batch)

    def _legacy_joint_action_override(self) -> bool:
        return (
            type(self).choose_action is not DistributionAwareScheduler.choose_action
            and not bool(getattr(self, "use_async_router_decisions", False))
        )

    def _default_prefill_decision_interval_ms(self) -> float:
        configured = float(self.config.prefill_decision_interval_ms or 0.0)
        if configured > 0.0:
            return configured
        if self.config.time_unit == "token":
            return float(max(1, int(self.config.prefill_chunk_size)))
        return max(float(self.config.prefill_chunk_size) * float(self.config.prefill_ms_per_token), 1e-9)

    def _default_decode_decision_interval_ms(self) -> float:
        configured = float(self.config.decode_decision_interval_ms or 0.0)
        if configured > 0.0:
            return configured
        if self.config.time_unit == "token":
            return float(max(1, int(self.config.decode_chunk_size_tokens)))
        return max(float(self.config.decode_chunk_size_tokens) * float(self.config.decode_ms_per_token), 1e-9)

    @staticmethod
    def _router_policy_due(last_decision_ms: Optional[float], now: float, interval_ms: float) -> bool:
        if last_decision_ms is None:
            return True
        return now - last_decision_ms >= interval_ms

    def _has_prefill_work(self, engine: "LLMSingleNodeSimulator") -> bool:
        return any(engine.prefill_queues[queue] for queue in QueueClass)

    def _has_decode_work(self, engine: "LLMSingleNodeSimulator") -> bool:
        return any(not request.is_complete for request in engine.active_decode)

    def _requested_action_from_async_router_policies(
        self,
        engine: "LLMSingleNodeSimulator",
    ) -> Optional[SchedulerAction]:
        now = engine.now_ms
        has_prefill_work = self._has_prefill_work(engine)
        has_decode_work = self._has_decode_work(engine)
        if not has_prefill_work:
            self.last_prefill_policy = None
            self.last_requested_prefill_policy = None
        if not has_decode_work:
            self.last_decode_policy = None
            self.last_requested_decode_policy = None

        if has_prefill_work:
            requested_prefill_policy = self.choose_prefill_policy(engine)
            if requested_prefill_policy is not None:
                self.last_requested_prefill_policy = requested_prefill_policy
                self.last_prefill_policy = requested_prefill_policy
                self.last_prefill_decision_ms = now

        if has_decode_work:
            requested_decode_policy = self.choose_decode_policy(engine, self.last_prefill_policy)
            if requested_decode_policy is not None:
                self.last_requested_decode_policy = requested_decode_policy
                self.last_decode_policy = requested_decode_policy
                self.last_decode_decision_ms = now

        if self.last_prefill_policy is None and self.last_decode_policy is None:
            return None
        prefill_policy = self.last_prefill_policy or PrefillRouterPolicy()
        decode_policy = self.last_decode_policy or DecodeRouterPolicy()
        return SchedulerAction.from_router_policies(prefill_policy, decode_policy)

    def schedule(self, engine: "LLMSingleNodeSimulator") -> ScheduleDecision:
        now = engine.now_ms
        requested_action = (
            self.choose_action(engine)
            if self._legacy_joint_action_override()
            else self._requested_action_from_async_router_policies(engine)
        )
        action, action_safety_triggered, action_safety_reasons = self._stabilize_action(engine, requested_action)
        if action is not None:
            self.last_prefill_policy = action.prefill_policy()
            self.last_decode_policy = action.decode_policy()
        context = self._context_from_action(engine, action)
        live_decodes = [req for req in engine.active_decode if not req.is_complete]
        first_token_pending = [req for req in live_decodes if req.generated_tokens == 0]

        token_budget = self.config.token_budget_per_step
        kv_budget = max(0, self.config.kv_capacity_tokens - engine.current_kv_tokens
                        + engine.reclaimable_prefix_tokens)
        shadow_budget = max(
            0,
            int(self.config.kv_capacity_tokens * self.config.allow_shadow_oversubscription)
            - self._shadow_commitment(engine, context.shadow_quantile),
        )

        decode_quota = self._decode_quota(context.phase_mode, len(live_decodes), len(first_token_pending), token_budget)
        decode_quota = max(0, min(len(live_decodes), int(math.ceil(decode_quota * context.decode_quota_scale))))
        decode_requests = self._select_decode_requests(now, live_decodes, decode_quota, kv_budget, engine)
        decode_tokens = sum(chunk for _, chunk in decode_requests)
        token_budget -= decode_tokens
        if self.config.enforce_decode_kv_budget:
            for request, chunk in decode_requests:
                kv_budget -= engine.estimate_kv_growth(request, chunk)

        prefill_token_budget = min(token_budget, self.config.prefill_chunk_size * max(0, context.prefill_chunk_cap))
        prefill_requests = self._select_prefill_requests(
            now=now,
            engine=engine,
            token_budget=prefill_token_budget,
            kv_budget=kv_budget,
            shadow_budget=shadow_budget,
            context=context,
        )
        prefill_tokens = sum(chunk for _, chunk in prefill_requests)
        token_budget -= prefill_tokens
        for request, chunk in prefill_requests:
            kv_budget -= engine.estimate_kv_growth(request, chunk)

        if token_budget > 0 and kv_budget >= 0:
            selected_decode_ids = {request.request_id for request, _ in decode_requests}
            extra_decode_candidates = [req for req in live_decodes if req.request_id not in selected_decode_ids]
            decode_requests.extend(
                self._select_decode_requests(now, extra_decode_candidates, token_budget, kv_budget, engine,
                                             max_requests=self._resolved_decode_parallelism() - len(decode_requests))
            )

        return ScheduleDecision(
            phase_mode=context.phase_mode,
            decode_requests=decode_requests,
            prefill_requests=prefill_requests,
            scheduler_action=action,
            requested_scheduler_action=requested_action,
            action_safety_triggered=action_safety_triggered,
            action_safety_reasons=action_safety_reasons,
        )

    def _decode_quota(
        self,
        phase_mode: PhaseMode,
        total_decode: int,
        first_token_pending: int,
        token_budget: int,
    ) -> int:
        if total_decode == 0 or token_budget == 0:
            return 0
        if phase_mode == PhaseMode.DECODE_PRIORITY:
            return min(total_decode, self._resolved_decode_parallelism(), token_budget)
        if phase_mode == PhaseMode.PREFILL_PRIORITY:
            return min(max(1, first_token_pending), total_decode, token_budget // 4 or token_budget)
        quota = max(first_token_pending, token_budget // 6)
        return min(total_decode, self._resolved_decode_parallelism(), max(1, quota))

    def _select_decode_requests(
        self,
        now: float,
        requests: Sequence[Request],
        token_budget: int,
        kv_budget: int,
        engine: "LLMSingleNodeSimulator",
        max_requests: Optional[int] = None,
    ) -> List[Tuple[Request, int]]:
        strict_kv_budget = self.config.enforce_decode_kv_budget or self.config.overflow_policy == "block"
        if token_budget <= 0 or (strict_kv_budget and kv_budget < 0):
            return []
        scored = sorted(requests, key=lambda req: self._decode_score(now, req, engine), reverse=True)
        selected: List[Tuple[Request, int]] = []
        remaining_budget = kv_budget
        remaining_token_budget = token_budget
        limit = min(self._resolved_decode_parallelism(), len(scored))
        if max_requests is not None:
            limit = min(limit, max(0, max_requests))
        for request in scored:
            if len(selected) >= limit:
                break
            max_chunk = min(
                max(1, int(self.config.decode_chunk_size_tokens)),
                remaining_token_budget,
            )
            if max_chunk <= 0:
                continue
            chunk = max_chunk
            growth = engine.estimate_kv_growth(request, chunk)
            if strict_kv_budget and growth > remaining_budget:
                chunk = self._fit_decode_chunk_to_budget(engine, request, max_chunk, remaining_budget)
                if chunk <= 0:
                    continue
                growth = engine.estimate_kv_growth(request, chunk)
            if chunk <= 0:
                continue
            selected.append((request, chunk))
            remaining_token_budget -= chunk
            if strict_kv_budget:
                remaining_budget -= growth
            if remaining_token_budget <= 0:
                break
        return selected

    def _fit_decode_chunk_to_budget(
        self,
        engine: "LLMSingleNodeSimulator",
        request: Request,
        max_chunk: int,
        kv_budget: int,
    ) -> int:
        if kv_budget < 0 or max_chunk <= 0:
            return 0
        low = 1
        high = max_chunk
        best = 0
        while low <= high:
            mid = (low + high) // 2
            growth = engine.estimate_kv_growth(request, mid)
            if growth <= kv_budget:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best

    def _decode_score(
        self,
        now: float,
        request: Request,
        engine: Optional["LLMSingleNodeSimulator"],
    ) -> float:
        wait_since_service = now - request.last_service_at_ms
        release_bonus = self.config.release_boost / max(request.estimated_remaining_output_tokens(), 1)
        first_token_bonus = self.config.first_token_boost if request.generated_tokens == 0 else 0.0
        base_score = first_token_bonus + release_bonus + self.config.aging_weight * wait_since_service
        return self.decode_router.score_decode(
            now=now,
            request=request,
            base_score=base_score,
            scheduler=self,
            engine=engine,
        )

    def _select_prefill_requests(
        self,
        now: float,
        engine: "LLMSingleNodeSimulator",
        token_budget: int,
        kv_budget: int,
        shadow_budget: int,
        context: ScheduleContext,
    ) -> List[Tuple[Request, int]]:
        strict_kv_budget = self.config.overflow_policy == "block"
        if token_budget <= 0 or (strict_kv_budget and kv_budget < 0):
            return []
        candidates: List[Request] = []
        for queue_items in engine.prefill_queues.values():
            candidates.extend(queue_items)
        if not candidates:
            return []

        scored = sorted(
            candidates,
            key=lambda req: self._prefill_score_with_context(now, req, context, engine),
            reverse=True,
        )
        selected: List[Tuple[Request, int]] = []
        picked = 0
        new_admissions = 0
        local_shadow_budget = shadow_budget
        live_decode_count = len([request for request in engine.active_decode if not request.is_complete])
        first_token_pending_count = len(
            [
                request
                for request in engine.active_decode
                if not request.is_complete and request.generated_tokens == 0
            ]
        )
        decode_load = live_decode_count / max(self._resolved_decode_parallelism(), 1)
        effective_risky_admission_cap, _ = self._guarded_risky_admission_cap(
            first_token_pending_count=first_token_pending_count,
            decode_load=decode_load,
            base_cap=context.risky_admission_cap,
        )
        kv_util = (engine.current_kv_tokens - engine.reclaimable_prefix_tokens) / max(self.config.kv_capacity_tokens, 1)
        shadow_util = self._shadow_commitment(engine, context.shadow_quantile) / max(
            self.config.kv_capacity_tokens * self.config.allow_shadow_oversubscription,
            1.0,
        )

        if context.phase_mode == PhaseMode.DECODE_PRIORITY:
            token_budget = max(0, min(token_budget, self.config.prefill_chunk_size * 2))

        risky_new_admissions = 0
        for request in scored:
            if picked >= self._resolved_prefill_parallelism() or token_budget <= 0 or kv_budget < 0:
                break
            if request.batch_admitted_at_ms is None and new_admissions >= context.new_admission_limit:
                continue
            if (
                request.batch_admitted_at_ms is None
                and request.queue_class == QueueClass.RISKY
                and risky_new_admissions >= effective_risky_admission_cap
            ):
                continue
            extra_shadow = 0
            # A cache owner already holds its prefill allocation. Treat it as
            # admitted for prediction-based gating; physical checks still apply.
            if not request.has_started and engine._prefix_cache_owner_entry(request) is None:
                extra_shadow = self._shadow_reservation_tokens(request, context.shadow_quantile)
                if self._should_defer_new_admission_for_kv_pressure(
                    request,
                    extra_shadow_tokens=extra_shadow,
                    kv_budget=kv_budget,
                    local_shadow_budget=local_shadow_budget,
                    kv_util=kv_util,
                    shadow_util=shadow_util,
                    live_decode_count=live_decode_count,
                    admitted_any_new=new_admissions > 0,
                ):
                    continue
            chunk = min(request.remaining_prefill_tokens, self.config.prefill_chunk_size, token_budget)
            if chunk <= 0:
                continue
            growth = engine.estimate_kv_growth(request, chunk)
            if strict_kv_budget and growth > kv_budget:
                chunk = self._fit_chunk_to_budget(engine, request, chunk, kv_budget)
                if chunk <= 0:
                    continue
                growth = engine.estimate_kv_growth(request, chunk)
            if extra_shadow > local_shadow_budget:
                continue
            selected.append((request, chunk))
            picked += 1
            if request.batch_admitted_at_ms is None:
                new_admissions += 1
                if request.queue_class == QueueClass.RISKY:
                    risky_new_admissions += 1
            token_budget -= chunk
            if strict_kv_budget:
                kv_budget -= growth
            local_shadow_budget -= extra_shadow
        return selected

    def _fit_chunk_to_budget(
        self,
        engine: "LLMSingleNodeSimulator",
        request: Request,
        max_chunk: int,
        kv_budget: int,
    ) -> int:
        if kv_budget < 0 or max_chunk <= 0:
            return 0
        low = 1
        high = max_chunk
        best = 0
        while low <= high:
            mid = (low + high) // 2
            growth = engine.estimate_kv_growth(request, mid)
            if growth <= kv_budget:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best

    def _prefill_score(self, now: float, request: Request) -> float:
        return self._prefill_score_with_bias(
            now,
            request,
            {
                QueueClass.FAST: self.config.queue_bias_fast,
                QueueClass.NORMAL: self.config.queue_bias_normal,
                QueueClass.RISKY: self.config.queue_bias_risky,
            },
        )

    def _prefill_score_with_bias(
        self,
        now: float,
        request: Request,
        queue_bias_map: Dict[QueueClass, float],
    ) -> float:
        age = now - request.arrival_ms
        queue_bias = queue_bias_map[request.queue_class]
        risk = self._request_risk_signal(request)
        return (
            queue_bias
            + self.config.aging_weight * age
            - self.config.prompt_penalty * request.remaining_prefill_tokens
            - self.config.risk_penalty * risk
        )

    def _prefill_score_with_context(
        self,
        now: float,
        request: Request,
        context: ScheduleContext,
        engine: Optional["LLMSingleNodeSimulator"],
    ) -> float:
        score = self._prefill_score_with_bias(now, request, context.admission_bias)
        if request.batch_admitted_at_ms is None:
            score += context.new_admission_bonus
            score -= context.shadow_risk_weight * self._shadow_reservation_tokens(request, context.shadow_quantile)
            score -= self._risky_marginal_first_token_harm_penalty(now, request, context, engine)
            return self.prefill_router.score_prefill(
                now=now,
                request=request,
                base_score=score,
                context=context,
                scheduler=self,
                engine=engine,
            )
        progress_ratio = request.prefill_done_tokens / max(request.active_prefill_target_tokens, 1)
        score += context.started_prefill_boost + 35.0 * progress_ratio
        return self.prefill_router.score_prefill(
            now=now,
            request=request,
            base_score=score,
            context=context,
            scheduler=self,
            engine=engine,
        )

    def _risky_marginal_first_token_harm_penalty(
        self,
        now: float,
        request: Request,
        context: ScheduleContext,
        engine: Optional["LLMSingleNodeSimulator"],
    ) -> float:
        if engine is None or request.queue_class != QueueClass.RISKY or request.batch_admitted_at_ms is not None:
            return 0.0

        live_decodes = [live for live in engine.active_decode if not live.is_complete]
        first_token_pending = [live for live in live_decodes if live.generated_tokens == 0]
        decode_parallelism = max(self._resolved_decode_parallelism(), 1)
        decode_pressure = min(len(live_decodes) / decode_parallelism, 1.0)
        if first_token_pending:
            oldest_first_token_age_s = max(now - live.arrival_ms for live in first_token_pending) / 1000.0
            first_token_pressure = min(
                len(first_token_pending) / decode_parallelism + min(oldest_first_token_age_s / 5.0, 1.0),
                1.0,
            )
        else:
            first_token_pressure = 0.0
        pressure = max(0.5 * decode_pressure, first_token_pressure)
        if pressure <= 0.0:
            return 0.0

        output_tail = request.estimated_total_output_tokens(context.shadow_quantile)
        output_harm = min(
            output_tail / max(float(self.config.risky_marginal_harm_output_scale_tokens), 1.0),
            3.0,
        )
        prompt_harm = min(
            request.remaining_prefill_tokens / max(float(self.config.risky_marginal_harm_prompt_scale_tokens), 1.0),
            2.0,
        )
        shadow_fraction = self._shadow_reservation_tokens(request, context.shadow_quantile) / max(
            float(self.config.kv_capacity_tokens),
            1.0,
        )
        shadow_harm = min(shadow_fraction * 4.0, 2.0)
        request_harm = 0.50 * output_harm + 0.30 * prompt_harm + 0.20 * shadow_harm
        return float(self.config.risky_marginal_first_token_harm_weight) * pressure * request_harm


class MeanOnlyScheduler(DistributionAwareScheduler):
    def __init__(self, config: SimulatorConfig):
        super().__init__(config, use_distribution_features=False, shadow_mode="mean")
