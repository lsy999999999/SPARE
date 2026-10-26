from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple


class QueueClass(str, Enum):
    FAST = "fast"
    NORMAL = "normal"
    RISKY = "risky"


class PhaseMode(str, Enum):
    PREFILL_PRIORITY = "prefill-priority"
    DECODE_PRIORITY = "decode-priority"
    MIXED = "mixed"


class KVCacheScope(str, Enum):
    MODEL = "model"
    QUERY = "query"


@dataclass(slots=True)
class ActionSpaceConfig:
    control_mode: str = "factorized"
    phase_choices: List[str] = field(
        default_factory=lambda: [
            "heuristic",
            PhaseMode.PREFILL_PRIORITY.value,
            PhaseMode.MIXED.value,
            PhaseMode.DECODE_PRIORITY.value,
        ]
    )
    admission_choices: List[str] = field(
        default_factory=lambda: [
            "balanced",
            "ttft-greedy",
            "prefill-drain",
            "tail-relief",
        ]
    )
    new_admission_limit_choices: List[int] = field(default_factory=lambda: [0, 1, 2, 4, 8])
    quantile_choices: List[float] = field(default_factory=lambda: [0.80, 0.90, 0.95, 0.99])
    budget_profile_choices: List[str] = field(
        default_factory=lambda: [
            "balanced",
            "decode-heavy",
            "decode-drain",
            "prefill-heavy",
            "decode-strict",
        ]
    )
    risky_admission_cap_choices: List[int] = field(default_factory=lambda: [0, 1, 2, 4])

    def __post_init__(self) -> None:
        self.control_mode = str(self.control_mode or "factorized")
        self.phase_choices = [str(value) for value in self.phase_choices]
        self.admission_choices = [str(value) for value in self.admission_choices]
        self.new_admission_limit_choices = [int(value) for value in self.new_admission_limit_choices]
        self.quantile_choices = [round(float(value), 2) for value in self.quantile_choices]
        self.budget_profile_choices = [str(value) for value in self.budget_profile_choices]
        self.risky_admission_cap_choices = [int(value) for value in self.risky_admission_cap_choices]

        supported_phases = {
            "heuristic",
            PhaseMode.PREFILL_PRIORITY.value,
            PhaseMode.MIXED.value,
            PhaseMode.DECODE_PRIORITY.value,
        }
        if self.control_mode not in {"factorized", "budget-control"}:
            raise ValueError("action_space.control_mode must be 'factorized' or 'budget-control'")
        if not self.phase_choices:
            raise ValueError("action_space.phase_choices must not be empty")
        if "heuristic" not in self.phase_choices:
            raise ValueError("action_space.phase_choices must include 'heuristic'")
        if any(value not in supported_phases for value in self.phase_choices):
            raise ValueError(f"unsupported phase choice in action_space.phase_choices: {self.phase_choices}")
        if not self.admission_choices:
            raise ValueError("action_space.admission_choices must not be empty")
        if not self.new_admission_limit_choices or min(self.new_admission_limit_choices) < 0:
            raise ValueError("action_space.new_admission_limit_choices must contain non-negative integers")
        if not self.quantile_choices or any(not 0.0 < value < 1.0 for value in self.quantile_choices):
            raise ValueError("action_space.quantile_choices must be in (0, 1)")
        if not self.budget_profile_choices:
            raise ValueError("action_space.budget_profile_choices must not be empty")
        if not self.risky_admission_cap_choices or min(self.risky_admission_cap_choices) < 0:
            raise ValueError("action_space.risky_admission_cap_choices must contain non-negative integers")

    @classmethod
    def from_payload(cls, payload: Dict[str, object] | "ActionSpaceConfig" | None) -> "ActionSpaceConfig":
        if payload is None:
            return cls()
        if isinstance(payload, cls):
            return payload
        defaults = cls()
        return cls(
            control_mode=str(payload.get("control_mode") or defaults.control_mode),
            phase_choices=list(payload.get("phase_choices") or defaults.phase_choices),
            admission_choices=list(payload.get("admission_choices") or defaults.admission_choices),
            new_admission_limit_choices=list(
                payload.get("new_admission_limit_choices") or defaults.new_admission_limit_choices
            ),
            quantile_choices=list(payload.get("quantile_choices") or defaults.quantile_choices),
            budget_profile_choices=list(payload.get("budget_profile_choices") or defaults.budget_profile_choices),
            risky_admission_cap_choices=list(
                payload.get("risky_admission_cap_choices") or defaults.risky_admission_cap_choices
            ),
        )


@dataclass(slots=True)
class DecodeRouterPolicy:
    phase_preference: Optional[PhaseMode] = None
    budget_profile: str = "balanced"
    safety_filter_enabled: Optional[bool] = None


@dataclass(slots=True)
class PrefillRouterPolicy:
    admission_profile: str = "balanced"
    new_admission_limit: Optional[int] = None
    shadow_quantile: Optional[float] = None
    risky_admission_cap: Optional[int] = None
    safety_filter_enabled: Optional[bool] = None


@dataclass(slots=True)
class SchedulerAction:
    phase_preference: Optional[PhaseMode] = None
    admission_profile: str = "balanced"
    new_admission_limit: Optional[int] = None
    shadow_quantile: Optional[float] = None
    budget_profile: str = "balanced"
    risky_admission_cap: Optional[int] = None
    prefill_safety_filter_enabled: Optional[bool] = None
    decode_safety_filter_enabled: Optional[bool] = None

    @classmethod
    def from_router_policies(
        cls,
        prefill_policy: PrefillRouterPolicy,
        decode_policy: DecodeRouterPolicy,
    ) -> "SchedulerAction":
        return cls(
            phase_preference=decode_policy.phase_preference,
            admission_profile=prefill_policy.admission_profile,
            new_admission_limit=prefill_policy.new_admission_limit,
            shadow_quantile=prefill_policy.shadow_quantile,
            budget_profile=decode_policy.budget_profile,
            risky_admission_cap=prefill_policy.risky_admission_cap,
            prefill_safety_filter_enabled=prefill_policy.safety_filter_enabled,
            decode_safety_filter_enabled=decode_policy.safety_filter_enabled,
        )

    def prefill_policy(self) -> PrefillRouterPolicy:
        return PrefillRouterPolicy(
            admission_profile=self.admission_profile,
            new_admission_limit=self.new_admission_limit,
            shadow_quantile=self.shadow_quantile,
            risky_admission_cap=self.risky_admission_cap,
            safety_filter_enabled=self.prefill_safety_filter_enabled,
        )

    def decode_policy(self) -> DecodeRouterPolicy:
        return DecodeRouterPolicy(
            phase_preference=self.phase_preference,
            budget_profile=self.budget_profile,
            safety_filter_enabled=self.decode_safety_filter_enabled,
        )


@dataclass(slots=True)
class ScheduleContext:
    phase_mode: PhaseMode
    shadow_quantile: float
    admission_bias: Dict[QueueClass, float]
    started_prefill_boost: float
    new_admission_bonus: float
    shadow_risk_weight: float
    new_admission_limit: int
    decode_quota_scale: float
    prefill_chunk_cap: int
    risky_admission_cap: int


@dataclass(slots=True)
class LengthDistribution:
    mean: float
    std: float
    p90: float
    p50: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None
    family: str = "lognormal"

    @classmethod
    def from_moments(cls, mean: float, std: float, family: str = "lognormal") -> "LengthDistribution":
        """Construct a coherent parametric distribution (external predictions may differ)."""
        if not math.isfinite(mean) or not math.isfinite(std) or mean <= 0 or std < 0:
            raise ValueError("length moments must be finite, with mean > 0 and std >= 0")
        if family not in {"lognormal", "normal", "deterministic"}:
            raise ValueError(f"unsupported sampling family: {family}")
        if family == "deterministic":
            std = 0.0
        result = cls(float(mean), float(std), 0.0, family=family)
        result.p50, result.p90, result.p95, result.p99 = (
            result.parametric_quantile(q) for q in (0.5, 0.9, 0.95, 0.99)
        )
        return result

    def parametric_quantile(self, q: float) -> float:
        if not 0.0 < q < 1.0:
            raise ValueError("quantile must be in (0, 1)")
        if self.family == "deterministic" or self.std <= 0:
            return max(1.0, self.mean)
        if self.family == "lognormal":
            sigma_sq = math.log1p((self.std / max(self.mean, 1e-9)) ** 2)
            return max(1.0, math.exp(math.log(max(self.mean, 1e-9)) - sigma_sq / 2
                                     + math.sqrt(sigma_sq) * NormalDist().inv_cdf(q)))
        if self.family == "normal":
            return max(1.0, self.mean + self.std * NormalDist().inv_cdf(q))
        raise ValueError(f"unsupported sampling family: {self.family}")

    def conditional_remaining_quantile(self, generated: int, q: float = 0.5, cap: int = 0) -> float:
        """Parametric survival conditioning; never reads a realized output length.

        Uses the rounded-token survival boundary. A cap is a censoring atom,
        not a renormalization of the distribution below the cap.
        """
        if generated < 0 or not 0 < q < 1:
            raise ValueError("invalid progress or conditional quantile")
        if cap and generated >= cap:
            return 0.0
        if self.family == "deterministic" or self.std <= 0:
            total = max(float(generated + 1), self.mean)
        else:
            boundary = max(0.5, generated + 0.5)
            if self.family == "lognormal":
                sigma2 = math.log1p((self.std / max(self.mean, 1e-9)) ** 2)
                z = (math.log(boundary) - math.log(max(self.mean, 1e-9)) + sigma2 / 2) / math.sqrt(sigma2)
            elif self.family == "normal":
                z = (boundary - self.mean) / self.std
            else:
                raise ValueError(f"unsupported sampling family: {self.family}")
            survival = 0.5 * math.erfc(z / math.sqrt(2)) if generated else 1.0
            probability = min(1 - 1e-15, max(1e-15, 1 - (1 - q) * survival))
            total = max(float(generated + 1), self.parametric_quantile(probability))
        if cap:
            total = min(total, float(cap))
        return max(0.0, total - generated)

    @property
    def cv(self) -> float:
        return self.std / max(self.mean, 1.0)

    @property
    def tail_ratio(self) -> float:
        return self.p90 / max(self.mean, 1.0)

    def quantile(self, q: float) -> float:
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {q}")
        rounded = q * 100
        known = {
            50: self.p50,
            90: self.p90,
            95: self.p95,
            99: self.p99,
        }
        if rounded in known and known[rounded] is not None:
            return float(known[rounded])
        return self.parametric_quantile(q)

    def sample(self, rng: random.Random) -> int:
        if self.family == "deterministic" or self.std <= 0:
            return max(1, int(round(self.mean)))
        if self.family == "normal":
            return max(1, int(round(rng.gauss(self.mean, self.std))))
        if self.family == "lognormal":
            sigma_sq = math.log1p((self.std / max(self.mean, 1e-9)) ** 2)
            mu = math.log(max(self.mean, 1e-9)) - sigma_sq / 2.0
            return max(1, int(round(rng.lognormvariate(mu, math.sqrt(sigma_sq)))))
        raise ValueError(f"unsupported sampling family: {self.family}")


@dataclass(slots=True)
class Request:
    request_id: int
    arrival_ms: float
    prompt_tokens: int
    length_distribution: LengthDistribution
    model_id: str = "default"
    priority: float = 0.0
    template_id: str = ""
    template_family: str = ""
    template_version: str = "v1"
    max_new_tokens: int = 0
    stop_strings: List[str] = field(default_factory=list)
    stop_at_eos: bool = True
    structured_output: bool = False
    shared_prefix_id: str = ""
    shared_prefix_tokens: int = 0
    private_prompt_tokens: int = 0
    actual_output_tokens: int = 0
    queue_class: QueueClass = QueueClass.NORMAL
    prefill_target_tokens: int = 0
    prefill_done_tokens: int = 0
    generated_tokens: int = 0
    kv_tokens: int = 0
    shared_kv_tokens: int = 0
    online_length_distribution: Optional[LengthDistribution] = None
    routed_length_distribution: Optional[LengthDistribution] = None
    query_distribution_source: str = "static"
    query_distribution_adapter_name: str = "query-distribution-static"
    query_distribution_metadata: Dict[str, object] = field(default_factory=dict)
    distribution_source: str = "query"
    prefill_route_name: str = "default"
    prefill_route_metadata: Dict[str, object] = field(default_factory=dict)
    model_route_name: str = "default"
    model_route_metadata: Dict[str, object] = field(default_factory=dict)
    decode_route_name: str = "default"
    decode_route_metadata: Dict[str, object] = field(default_factory=dict)
    decode_priority: float = 0.0
    output_length_source: str = "trace"
    output_length_sampled_at_ms: Optional[float] = None
    prefix_cache_status: str = "uncacheable"
    batch_admitted_at_ms: Optional[float] = None
    prefill_started_at_ms: Optional[float] = None
    prefill_completed_at_ms: Optional[float] = None
    first_decode_started_at_ms: Optional[float] = None
    first_token_at_ms: Optional[float] = None
    finished_at_ms: Optional[float] = None
    last_service_at_ms: float = 0.0
    eviction_count: int = 0
    recompute_count: int = 0
    recompute_tokens: int = 0
    wasted_kv_tokens: int = 0
    # Environment truth, separate from length_distribution and all routed predictions.
    true_length_distribution: Optional[LengthDistribution] = None

    def __post_init__(self) -> None:
        self.request_id = int(self.request_id)
        self.arrival_ms = float(self.arrival_ms)
        self.model_id = str(self.model_id or "default")
        self.prompt_tokens = max(1, int(self.prompt_tokens))
        self.template_id = str(self.template_id or "")
        self.template_family = str(self.template_family or "")
        self.template_version = str(self.template_version or "v1")
        self.shared_prefix_id = str(self.shared_prefix_id or "")
        self.shared_prefix_tokens = max(0, int(self.shared_prefix_tokens))
        self.private_prompt_tokens = max(0, int(self.private_prompt_tokens))
        split_prompt_tokens = self.shared_prefix_tokens + self.private_prompt_tokens
        if split_prompt_tokens <= 0:
            self.shared_prefix_tokens = 0
            self.private_prompt_tokens = self.prompt_tokens
        else:
            self.prompt_tokens = max(1, split_prompt_tokens)
        if self.shared_prefix_tokens <= 0:
            self.shared_prefix_id = ""
        elif not self.shared_prefix_id:
            if self.template_id:
                self.shared_prefix_id = f"{self.model_id}:template:{self.template_id}:{self.template_version}"
            else:
                self.shared_prefix_id = f"{self.model_id}:shared:{self.shared_prefix_tokens}"
        if not self.template_id and self.shared_prefix_id:
            self.template_id = self.shared_prefix_id
        self.max_new_tokens = max(0, int(self.max_new_tokens or 0))
        self.stop_strings = [str(value) for value in self.stop_strings if str(value)]
        self.stop_at_eos = bool(self.stop_at_eos)
        self.structured_output = bool(self.structured_output)
        self.actual_output_tokens = max(0, int(self.actual_output_tokens or 0))
        self.kv_tokens = max(0, int(self.kv_tokens or 0))
        self.shared_kv_tokens = min(self.shared_prefix_tokens, max(0, int(self.shared_kv_tokens or 0)))
        self.prefix_cache_status = str(self.prefix_cache_status or "uncacheable")
        if self.max_new_tokens > 0 and self.actual_output_tokens > 0:
            self.actual_output_tokens = min(self.actual_output_tokens, self.max_new_tokens)

    def _bounded_output_estimate(self, value: float) -> int:
        bounded = max(1, int(math.ceil(float(value))))
        if self.max_new_tokens > 0:
            bounded = min(self.max_new_tokens, bounded)
        return bounded

    @property
    def output_length_distribution(self) -> LengthDistribution:
        return self.length_distribution

    @property
    def remaining_prompt_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.prefill_done_tokens)

    @property
    def logical_kv_tokens(self) -> int:
        return self.shared_kv_tokens + self.kv_tokens

    @property
    def active_prefill_target_tokens(self) -> int:
        return self.prefill_target_tokens or self.prompt_tokens

    @property
    def query_length_distribution(self) -> LengthDistribution:
        return self.online_length_distribution or self.output_length_distribution

    @property
    def effective_length_distribution(self) -> LengthDistribution:
        return self.routed_length_distribution or self.online_length_distribution or self.output_length_distribution

    @property
    def remaining_prefill_tokens(self) -> int:
        return max(0, self.active_prefill_target_tokens - self.prefill_done_tokens)

    @property
    def remaining_output_tokens(self) -> int:
        return max(0, self.actual_output_tokens - self.generated_tokens)

    def cacheable_shared_prefix_tokens(self, block_size_tokens: int) -> int:
        if block_size_tokens <= 0 or self.shared_prefix_tokens <= 0:
            return 0
        return (self.shared_prefix_tokens // int(block_size_tokens)) * int(block_size_tokens)

    def uncacheable_shared_prefix_tokens(self, block_size_tokens: int) -> int:
        return max(0, self.shared_prefix_tokens - self.cacheable_shared_prefix_tokens(block_size_tokens))

    def estimated_total_output_tokens(self, quantile: Optional[float] = None) -> int:
        estimate = (
            self.effective_length_distribution.quantile(quantile)
            if quantile is not None
            else self.effective_length_distribution.mean
        )
        return max(self.generated_tokens + 1, self._bounded_output_estimate(estimate))

    def estimated_remaining_output_tokens(self, quantile: Optional[float] = None) -> int:
        return max(1, self.estimated_total_output_tokens(quantile) - self.generated_tokens)

    @property
    def is_complete(self) -> bool:
        return self.finished_at_ms is not None

    @property
    def has_started(self) -> bool:
        return self.prefill_started_at_ms is not None

    @property
    def ttft_ms(self) -> Optional[float]:
        if self.first_token_at_ms is None:
            return None
        return self.first_token_at_ms - self.arrival_ms

    @property
    def e2e_ms(self) -> Optional[float]:
        if self.finished_at_ms is None:
            return None
        return self.finished_at_ms - self.arrival_ms

    @property
    def queue_wait_ms(self) -> Optional[float]:
        return self.admission_wait_ms

    @property
    def admission_wait_ms(self) -> Optional[float]:
        if self.batch_admitted_at_ms is None:
            return None
        return self.batch_admitted_at_ms - self.arrival_ms

    @property
    def prefill_compute_ms(self) -> Optional[float]:
        if self.prefill_started_at_ms is None or self.prefill_completed_at_ms is None:
            return None
        return self.prefill_completed_at_ms - self.prefill_started_at_ms

    @property
    def prefill_queue_wait_ms(self) -> Optional[float]:
        if self.batch_admitted_at_ms is None or self.prefill_started_at_ms is None:
            return None
        return self.prefill_started_at_ms - self.batch_admitted_at_ms

    @property
    def prefill_service_ms(self) -> Optional[float]:
        return self.prefill_compute_ms

    @property
    def ready_to_decode_wait_ms(self) -> Optional[float]:
        if self.prefill_completed_at_ms is None or self.first_decode_started_at_ms is None:
            return None
        return self.first_decode_started_at_ms - self.prefill_completed_at_ms

    @property
    def first_decode_service_ms(self) -> Optional[float]:
        if self.first_decode_started_at_ms is None or self.first_token_at_ms is None:
            return None
        return self.first_token_at_ms - self.first_decode_started_at_ms

    @property
    def decode_to_first_token_ms(self) -> Optional[float]:
        if self.prefill_completed_at_ms is None or self.first_token_at_ms is None:
            return None
        return self.first_token_at_ms - self.prefill_completed_at_ms

    def shadow_kv_target(self, quantile: float) -> int:
        future_decode = self._bounded_output_estimate(self.effective_length_distribution.quantile(quantile))
        return self.prompt_tokens + int(math.ceil(future_decode))

    def mean_kv_target(self) -> int:
        return self.prompt_tokens + self._bounded_output_estimate(self.effective_length_distribution.mean)


@dataclass(slots=True)
class SimulatorConfig:
    step_base_ms: float = 0.25
    prefill_ms_per_token: float = 0.03
    decode_ms_per_token: float = 0.40
    time_unit: str = "ms"
    simulation_duration_ms: float = 0.0
    unfinished_age_grace_ms: float = 0.0
    prefill_decision_interval_ms: float = 0.0
    decode_decision_interval_ms: float = 0.0
    event_record_mode: str = "dense"
    token_budget_per_step: int = 384
    prefill_chunk_size: int = 128
    decode_chunk_size_tokens: int = 1
    kv_capacity_tokens: int = 16000
    kv_cache_scope: str = KVCacheScope.MODEL.value
    kv_allocator_mode: str = "token"
    kv_block_size_tokens: int = 16
    prefix_cache_enabled: bool = True
    prefix_cache_block_size_tokens: int = 0
    prefix_cache_enable_lru_eviction: bool = True
    prefix_cache_max_resident_entries: int = 0
    max_decode_batch: int = 96
    max_prefill_batch: int = 8
    decode_parallelism: Optional[int] = None
    prefill_parallelism: Optional[int] = None
    aging_weight: float = 0.05
    prompt_penalty: float = 0.06
    risk_penalty: float = 0.20
    first_token_boost: float = 400.0
    release_boost: float = 40.0
    queue_bias_fast: float = 140.0
    queue_bias_normal: float = 60.0
    queue_bias_risky: float = 0.0
    kv_pressure_threshold: float = 0.85
    kv_reservation_quantile: float = 0.90
    allow_shadow_oversubscription: float = 1.0
    query_distribution_mode: str = "online-predictor"
    query_distribution_jitter: float = 0.06
    query_distribution_load_weight: float = 0.15
    query_distribution_tail_load_weight: float = 0.25
    query_distribution_mean_floor: float = 0.50
    query_distribution_mean_ceiling: float = 1.35
    query_distribution_tail_ceiling: float = 1.60
    prefill_padding_weight: float = 1.0
    enforce_decode_kv_budget: bool = True
    overflow_policy: str = "block"
    enable_deadlock_recovery: bool = True
    enable_action_safety_filter: bool = True
    action_safety_kv_util_threshold: float = 0.65
    action_safety_first_token_load_threshold: float = 0.01
    action_safety_pending_prompt_load_threshold: float = 0.25
    action_safety_max_new_admissions: int = 2
    action_safety_decode_guard_load_threshold: float = 0.50
    action_safety_drain_kv_util_threshold: float = 0.90
    action_safety_drain_on_first_token_pending: bool = True
    action_safety_first_token_max_risky_admissions: int = 0
    action_safety_decode_guard_max_risky_admissions: int = 1
    risky_marginal_first_token_harm_weight: float = 220.0
    risky_marginal_harm_output_scale_tokens: float = 1024.0
    risky_marginal_harm_prompt_scale_tokens: float = 2048.0
    router_boundary_mode: str = "current"
    action_space: ActionSpaceConfig = field(default_factory=ActionSpaceConfig)
    force_output_length_sampling: bool = False
    runtime_sampling_seed: int = 0
    execution_mode: str = "scheduled-tokens"
    first_token_mode: str = "prefill"
    decision_overhead_ms: float = 0.0
    timing_profile: Optional[Dict[str, object]] = None

    def __post_init__(self) -> None:
        if self.first_token_mode not in {"prefill", "decode"}:
            raise ValueError("first_token_mode must be 'prefill' or legacy 'decode'")
        if not math.isfinite(self.decision_overhead_ms) or self.decision_overhead_ms < 0:
            raise ValueError("decision_overhead_ms must be finite and nonnegative")
        if self.execution_mode not in {"boundary", "scheduled-tokens"}:
            raise ValueError("execution_mode must be 'boundary' or 'scheduled-tokens'")
        if self.timing_profile is not None:
            if self.execution_mode != "scheduled-tokens" or self.time_unit != "ms":
                raise ValueError("timing profiles require scheduled-tokens execution and ms units")
            from .timing import build_timing_model
            build_timing_model(self.timing_profile)
        self.action_space = ActionSpaceConfig.from_payload(self.action_space)
        self.kv_block_size_tokens = max(1, int(self.kv_block_size_tokens))
        self.prefix_cache_enabled = bool(self.prefix_cache_enabled)
        self.prefix_cache_block_size_tokens = max(
            1,
            int(self.prefix_cache_block_size_tokens or self.kv_block_size_tokens),
        )
        self.prefix_cache_enable_lru_eviction = bool(self.prefix_cache_enable_lru_eviction)
        self.prefix_cache_max_resident_entries = max(0, int(self.prefix_cache_max_resident_entries or 0))
        self.router_boundary_mode = str(self.router_boundary_mode or "current").strip() or "current"
        if self.router_boundary_mode not in {
            "current",
            "actual_prefill_conditioned_decode",
            "strict_env_only",
        }:
            raise ValueError(
                "router_boundary_mode must be one of "
                "'current', 'actual_prefill_conditioned_decode', or 'strict_env_only'"
            )


@dataclass(slots=True)
class ModelRuntimeConfig:
    model_id: str
    simulator_config: SimulatorConfig
    routing_weight: float = 1.0
    routing_bias: float = 0.0
    prompt_token_scale: float = 1.0
    output_mean_scale: float = 1.0
    output_std_scale: float = 1.0
    output_tail_scale: float = 1.0
    actual_output_scale: float = 1.0


@dataclass(slots=True)
class MultiModelSimulatorConfig:
    default_model_id: str = "default"
    model_runtimes: List[ModelRuntimeConfig] = field(default_factory=list)


@dataclass(slots=True)
class ScheduleDecision:
    phase_mode: PhaseMode
    decode_requests: List[Tuple[Request, int]] = field(default_factory=list)
    prefill_requests: List[Tuple[Request, int]] = field(default_factory=list)
    scheduler_action: Optional[SchedulerAction] = None
    requested_scheduler_action: Optional[SchedulerAction] = None
    action_safety_triggered: bool = False
    action_safety_reasons: List[str] = field(default_factory=list)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile() requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
