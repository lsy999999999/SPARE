from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, TypeVar

from .types import LengthDistribution, Request

T = TypeVar("T")


@dataclass(slots=True)
class DynamicWorkloadConfig:
    enabled: bool = False
    regime_count: int = 4
    arrival_jitter: float = 0.45
    prompt_jitter: float = 0.30
    output_jitter: float = 0.40
    tail_jitter: float = 0.60
    template_mix_jitter: float = 0.85
    template_locality_reuse_probability: float = 0.72
    template_locality_window: int = 12
    template_locality_recency_decay: float = 0.80
    template_version_churn_probability: float = 0.12
    template_version_churn_scale: float = 0.08
    per_request_jitter: float = 0.08
    burst_probability: float = 0.15
    burst_strength: float = 0.35
    resample_actual_output: bool = True


@dataclass(frozen=True, slots=True)
class DynamicTemplateProfile:
    model_id: str
    template_id: str
    template_family: str
    template_version: str
    shared_prefix_id: str
    shared_prefix_tokens: int
    private_prompt_tokens: int
    length_distribution: LengthDistribution
    structured_output: bool
    stop_strings: tuple[str, ...]
    stop_at_eos: bool
    weight: float


@dataclass(frozen=True, slots=True)
class SyntheticTemplateSpec:
    template_id: str
    template_family: str
    shared_prefix_tokens: int
    private_prompt_min_tokens: int
    private_prompt_max_tokens: int
    output_mean_low: float
    output_mean_high: float
    output_cv_low: float
    output_cv_high: float
    output_prompt_correlation_low: float
    output_prompt_correlation_high: float
    structured_output_rate: float = 0.0
    stop_strings: tuple[str, ...] = ()
    stop_at_eos: bool = True
    model_id: str = "default"
    popularity: float = 1.0
    max_new_token_headroom: float = 1.10
    template_version: str = "v1"


@dataclass(slots=True)
class SyntheticTemplateCatalogConfig:
    template_count: int = 12
    popularity_zipf_exponent: float = 1.10
    locality_reuse_probability: float = 0.62
    locality_window: int = 8
    locality_recency_decay: float = 0.72


_TEMPLATE_PROFILE_LIBRARY = (
    {
        "family": "chat-lite",
        "shared": (48, 128),
        "private": (18, 96),
        "output_mean": (20.0, 90.0),
        "output_cv": (0.18, 0.55),
        "prompt_corr": (0.02, 0.10),
        "structured_output_rate": 0.0,
        "stop_strings": (),
        "stop_at_eos": True,
        "headroom": 1.08,
    },
    {
        "family": "chat-reasoning",
        "shared": (96, 224),
        "private": (24, 192),
        "output_mean": (80.0, 280.0),
        "output_cv": (0.35, 0.90),
        "prompt_corr": (0.05, 0.14),
        "structured_output_rate": 0.0,
        "stop_strings": (),
        "stop_at_eos": True,
        "headroom": 1.12,
    },
    {
        "family": "rag-answer",
        "shared": (128, 320),
        "private": (96, 448),
        "output_mean": (90.0, 260.0),
        "output_cv": (0.30, 0.75),
        "prompt_corr": (0.04, 0.12),
        "structured_output_rate": 0.0,
        "stop_strings": (),
        "stop_at_eos": True,
        "headroom": 1.10,
    },
    {
        "family": "rag-summary",
        "shared": (160, 384),
        "private": (192, 960),
        "output_mean": (140.0, 440.0),
        "output_cv": (0.28, 0.70),
        "prompt_corr": (0.03, 0.09),
        "structured_output_rate": 0.0,
        "stop_strings": (),
        "stop_at_eos": True,
        "headroom": 1.10,
    },
    {
        "family": "tool-call",
        "shared": (256, 640),
        "private": (20, 144),
        "output_mean": (18.0, 80.0),
        "output_cv": (0.12, 0.45),
        "prompt_corr": (0.00, 0.04),
        "structured_output_rate": 0.85,
        "stop_strings": ("</tool_call>",),
        "stop_at_eos": False,
        "headroom": 1.05,
    },
    {
        "family": "json-schema",
        "shared": (320, 896),
        "private": (24, 180),
        "output_mean": (40.0, 140.0),
        "output_cv": (0.10, 0.35),
        "prompt_corr": (0.01, 0.05),
        "structured_output_rate": 0.95,
        "stop_strings": ("</json>",),
        "stop_at_eos": False,
        "headroom": 1.04,
    },
    {
        "family": "code-edit",
        "shared": (192, 576),
        "private": (80, 520),
        "output_mean": (120.0, 420.0),
        "output_cv": (0.28, 0.80),
        "prompt_corr": (0.03, 0.08),
        "structured_output_rate": 0.0,
        "stop_strings": (),
        "stop_at_eos": True,
        "headroom": 1.10,
    },
    {
        "family": "classification",
        "shared": (64, 192),
        "private": (12, 64),
        "output_mean": (6.0, 24.0),
        "output_cv": (0.08, 0.22),
        "prompt_corr": (0.00, 0.02),
        "structured_output_rate": 0.35,
        "stop_strings": ("</label>",),
        "stop_at_eos": False,
        "headroom": 1.02,
    },
)


def add_dynamic_workload_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dynamic-query-stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply deterministic per-seed regime drift so each episode replays a dynamic query stream instead of a fixed trace.",
    )
    parser.add_argument(
        "--dynamic-query-regimes",
        type=int,
        default=4,
        help="Number of regime anchors used to drift arrivals, prompts, and output distributions over time.",
    )
    parser.add_argument(
        "--dynamic-arrival-jitter",
        type=float,
        default=0.45,
        help="Max regime-level inter-arrival multiplier drift. Lower values keep traffic closer to the source trace.",
    )
    parser.add_argument(
        "--dynamic-prompt-jitter",
        type=float,
        default=0.30,
        help="Max regime-level prompt-length drift.",
    )
    parser.add_argument(
        "--dynamic-output-jitter",
        type=float,
        default=0.40,
        help="Max regime-level output-mean drift.",
    )
    parser.add_argument(
        "--dynamic-tail-jitter",
        type=float,
        default=0.60,
        help="Max regime-level output-tail drift.",
    )
    parser.add_argument(
        "--dynamic-template-mix-jitter",
        type=float,
        default=0.85,
        help="How strongly the active template / application mix drifts across regimes.",
    )
    parser.add_argument(
        "--dynamic-template-locality-reuse-probability",
        type=float,
        default=0.72,
        help="Chance that nearby requests reuse a recently hot template instead of redrawing from the long-run pool.",
    )
    parser.add_argument(
        "--dynamic-template-locality-window",
        type=int,
        default=12,
        help="How many recent template selections stay in the locality memory for dynamic template reuse.",
    )
    parser.add_argument(
        "--dynamic-template-version-churn-probability",
        type=float,
        default=0.12,
        help="Base probability that a hot template family rolls to a new version and invalidates exact prefix hits.",
    )
    parser.add_argument(
        "--dynamic-template-version-churn-scale",
        type=float,
        default=0.08,
        help="Small token-scale perturbation applied when a template version rolls forward.",
    )
    parser.add_argument(
        "--dynamic-per-request-jitter",
        type=float,
        default=0.08,
        help="Small request-level perturbation applied on top of regime drift.",
    )
    parser.add_argument(
        "--dynamic-burst-probability",
        type=float,
        default=0.15,
        help="Chance that a request lands in a short burst and sees compressed inter-arrival spacing.",
    )
    parser.add_argument(
        "--dynamic-burst-strength",
        type=float,
        default=0.35,
        help="How aggressively bursty regions shrink inter-arrival gaps.",
    )


def dynamic_workload_config_from_args(args: object) -> DynamicWorkloadConfig:
    return DynamicWorkloadConfig(
        enabled=bool(getattr(args, "dynamic_query_stream", False)),
        regime_count=max(1, int(getattr(args, "dynamic_query_regimes", 4) or 4)),
        arrival_jitter=max(0.0, float(getattr(args, "dynamic_arrival_jitter", 0.45) or 0.0)),
        prompt_jitter=max(0.0, float(getattr(args, "dynamic_prompt_jitter", 0.30) or 0.0)),
        output_jitter=max(0.0, float(getattr(args, "dynamic_output_jitter", 0.40) or 0.0)),
        tail_jitter=max(0.0, float(getattr(args, "dynamic_tail_jitter", 0.60) or 0.0)),
        template_mix_jitter=max(0.0, float(getattr(args, "dynamic_template_mix_jitter", 0.85) or 0.0)),
        template_locality_reuse_probability=min(
            1.0,
            max(0.0, float(getattr(args, "dynamic_template_locality_reuse_probability", 0.72) or 0.0)),
        ),
        template_locality_window=max(1, int(getattr(args, "dynamic_template_locality_window", 12) or 12)),
        template_locality_recency_decay=0.80,
        template_version_churn_probability=min(
            1.0,
            max(0.0, float(getattr(args, "dynamic_template_version_churn_probability", 0.12) or 0.0)),
        ),
        template_version_churn_scale=min(
            0.40,
            max(0.0, float(getattr(args, "dynamic_template_version_churn_scale", 0.08) or 0.0)),
        ),
        per_request_jitter=max(0.0, float(getattr(args, "dynamic_per_request_jitter", 0.08) or 0.0)),
        burst_probability=min(1.0, max(0.0, float(getattr(args, "dynamic_burst_probability", 0.15) or 0.0))),
        burst_strength=min(0.95, max(0.0, float(getattr(args, "dynamic_burst_strength", 0.35) or 0.0))),
        resample_actual_output=True,
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _weighted_choice(rng: random.Random, items: Sequence[T], weights: Sequence[float]) -> T:
    total = sum(max(0.0, float(weight)) for weight in weights)
    if total <= 0.0:
        return items[0]
    threshold = rng.uniform(0.0, total)
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += max(0.0, float(weight))
        if cumulative >= threshold:
            return item
    return items[-1]


def _strip_numeric_suffix(value: str) -> str:
    head = value
    while head and head[-1].isdigit():
        head = head[:-1]
    if head.endswith(("-", "_")):
        head = head[:-1]
    return head or value


def _canonical_template_family(request: Request) -> str:
    if request.template_family:
        return str(request.template_family)
    for candidate in (request.template_id, request.shared_prefix_id):
        text = str(candidate or "")
        if not text:
            continue
        if ":template:" in text:
            suffix = text.split(":template:", 1)[1]
            parts = [segment for segment in suffix.split(":") if segment]
            if parts:
                if len(parts) >= 2 and parts[-1].startswith("v"):
                    return _strip_numeric_suffix(parts[-2])
                return _strip_numeric_suffix(parts[-1])
        parts = [segment for segment in text.replace("/", ":").split(":") if segment]
        if parts:
            token = parts[-1]
            if token.startswith("v") and len(parts) >= 2:
                token = parts[-2]
            return _strip_numeric_suffix(token)
    return "generic"


def _blend_length_distributions(
    base: LengthDistribution,
    template: LengthDistribution,
    template_weight: float,
) -> LengthDistribution:
    weight = _clamp(float(template_weight), 0.0, 1.0)
    inv = 1.0 - weight
    mean = max(1.0, inv * float(base.mean) + weight * float(template.mean))
    # Moment-match a mixture, including between-component variance. This is a
    # parametric approximation, not an exact representation of a multimodal mix.
    variance = inv * ((0 if base.family == "deterministic" else base.std ** 2) + (base.mean - mean) ** 2)
    variance += weight * ((0 if template.family == "deterministic" else template.std ** 2) + (template.mean - mean) ** 2)
    family = template.family if weight >= 0.5 else base.family
    if family == "deterministic" and variance > 0:
        family = "lognormal"
    return LengthDistribution.from_moments(mean, math.sqrt(variance), family)


def _build_dynamic_shared_prefix_id(
    profile: DynamicTemplateProfile,
    template_version: str,
) -> str:
    shared_prefix_id = str(profile.shared_prefix_id or "")
    if not shared_prefix_id:
        return f"{profile.model_id}:template:{profile.template_id}:{template_version}"
    if template_version == profile.template_version:
        return shared_prefix_id
    if ":template:" in shared_prefix_id:
        base, _, suffix = shared_prefix_id.partition(":template:")
        parts = [segment for segment in suffix.split(":") if segment]
        if parts and parts[-1] == profile.template_version:
            parts[-1] = template_version
            return f"{base}:template:{':'.join(parts)}"
    return f"{shared_prefix_id}::{template_version}"


def _next_template_version(version: str) -> str:
    text = str(version or "v1")
    if text.startswith("v") and text[1:].isdigit():
        return f"v{int(text[1:]) + 1}"
    return f"{text}-next"


def _build_dynamic_template_profiles(requests: Sequence[Request]) -> dict[str, list[DynamicTemplateProfile]]:
    grouped: dict[tuple[str, str], list[Request]] = {}
    for request in requests:
        template_key = str(request.template_id or request.shared_prefix_id or f"request-{request.request_id}")
        grouped.setdefault((request.model_id, template_key), []).append(request)

    profiles_by_model: dict[str, list[DynamicTemplateProfile]] = {}
    for (model_id, template_key), members in grouped.items():
        reference = members[0]
        family = _canonical_template_family(reference)
        shared_prefix_tokens = max(0, int(round(sum(item.shared_prefix_tokens for item in members) / len(members))))
        private_prompt_tokens = max(1, int(round(sum(item.private_prompt_tokens for item in members) / len(members))))
        mean = sum(float(item.length_distribution.mean) for item in members) / len(members)
        variance = sum((0 if item.length_distribution.family == "deterministic" else item.length_distribution.std ** 2)
                       + (item.length_distribution.mean - mean) ** 2 for item in members) / len(members)
        family_name = reference.length_distribution.family
        if family_name == "deterministic" and variance > 0:
            family_name = "lognormal"
        profile = DynamicTemplateProfile(
            model_id=model_id,
            template_id=template_key,
            template_family=family,
            template_version=str(reference.template_version or "v1"),
            shared_prefix_id=str(reference.shared_prefix_id or ""),
            shared_prefix_tokens=shared_prefix_tokens,
            private_prompt_tokens=private_prompt_tokens,
            length_distribution=LengthDistribution.from_moments(mean, math.sqrt(variance), family_name),
            structured_output=bool(sum(1 for item in members if item.structured_output) >= math.ceil(len(members) / 2.0)),
            stop_strings=tuple(reference.stop_strings),
            stop_at_eos=bool(reference.stop_at_eos),
            weight=float(len(members)),
        )
        profiles_by_model.setdefault(model_id, []).append(profile)
    return profiles_by_model


def _build_family_mix_anchors(
    rng: random.Random,
    profiles_by_model: dict[str, list[DynamicTemplateProfile]],
    regime_count: int,
    jitter: float,
) -> dict[str, dict[str, list[float]]]:
    anchors_by_model: dict[str, dict[str, list[float]]] = {}
    for model_id, profiles in profiles_by_model.items():
        families = sorted({profile.template_family for profile in profiles})
        anchors_by_model[model_id] = {
            family: _sample_regime_anchors(rng, regime_count, jitter, low=0.20, high=3.40)
            for family in families
        }
    return anchors_by_model


def _sample_dynamic_template_profile(
    rng: random.Random,
    profiles: Sequence[DynamicTemplateProfile],
    family_anchors: dict[str, list[float]],
    position: float,
    config: DynamicWorkloadConfig,
    recent_template_indices: List[int],
) -> DynamicTemplateProfile:
    if recent_template_indices and rng.random() < config.template_locality_reuse_probability:
        recent_candidates = list(reversed(recent_template_indices[-max(1, config.template_locality_window):]))
        recency_weights = []
        for rank, index in enumerate(recent_candidates):
            profile = profiles[index]
            family_multiplier = _interpolate_anchor(
                family_anchors.get(profile.template_family, [1.0]),
                position,
            )
            recency_weights.append((config.template_locality_recency_decay ** rank) * profile.weight * family_multiplier)
        recent_index = _weighted_choice(rng, recent_candidates, recency_weights)
        return profiles[recent_index]

    weights = []
    for profile in profiles:
        family_multiplier = _interpolate_anchor(
            family_anchors.get(profile.template_family, [1.0]),
            position,
        )
        weights.append(profile.weight * family_multiplier)
    return _weighted_choice(rng, profiles, weights)


def build_default_template_catalog(
    seed: int,
    config: SyntheticTemplateCatalogConfig | None = None,
) -> List[SyntheticTemplateSpec]:
    resolved = config or SyntheticTemplateCatalogConfig()
    rng = random.Random(seed)
    template_count = max(1, int(resolved.template_count))
    templates: List[SyntheticTemplateSpec] = []
    for index in range(template_count):
        profile = _TEMPLATE_PROFILE_LIBRARY[index % len(_TEMPLATE_PROFILE_LIBRARY)]
        family = str(profile["family"])
        popularity = 1.0 / float((index + 1) ** max(0.01, float(resolved.popularity_zipf_exponent)))
        popularity *= rng.uniform(0.92, 1.08)
        shared_low, shared_high = profile["shared"]
        private_low, private_high = profile["private"]
        output_mean_low, output_mean_high = profile["output_mean"]
        output_cv_low, output_cv_high = profile["output_cv"]
        prompt_corr_low, prompt_corr_high = profile["prompt_corr"]
        private_min_tokens = max(1, int(round(rng.uniform(private_low * 0.9, private_low * 1.1))))
        private_max_tokens = max(
            private_min_tokens,
            int(round(rng.uniform(private_high * 0.9, private_high * 1.1))),
        )
        mean_low = max(1.0, rng.uniform(output_mean_low * 0.9, output_mean_low * 1.1))
        mean_high = max(mean_low, rng.uniform(output_mean_high * 0.9, output_mean_high * 1.1))
        cv_low = max(0.05, rng.uniform(output_cv_low * 0.9, output_cv_low * 1.1))
        cv_high = max(cv_low, rng.uniform(output_cv_high * 0.9, output_cv_high * 1.1))
        corr_low = max(0.0, rng.uniform(prompt_corr_low * 0.9, prompt_corr_low * 1.1))
        corr_high = max(corr_low, rng.uniform(prompt_corr_high * 0.9, prompt_corr_high * 1.1))
        templates.append(
            SyntheticTemplateSpec(
                template_id=f"{family}-{index:02d}",
                template_family=family,
                shared_prefix_tokens=max(0, int(round(rng.uniform(shared_low, shared_high)))),
                private_prompt_min_tokens=private_min_tokens,
                private_prompt_max_tokens=private_max_tokens,
                output_mean_low=mean_low,
                output_mean_high=mean_high,
                output_cv_low=cv_low,
                output_cv_high=cv_high,
                output_prompt_correlation_low=corr_low,
                output_prompt_correlation_high=corr_high,
                structured_output_rate=float(profile["structured_output_rate"]),
                stop_strings=tuple(str(item) for item in profile["stop_strings"]),
                stop_at_eos=bool(profile["stop_at_eos"]),
                popularity=max(0.01, popularity),
                max_new_token_headroom=max(1.01, float(profile["headroom"])),
            )
        )
    return templates


def _sample_template_spec(
    rng: random.Random,
    template_catalog: Sequence[SyntheticTemplateSpec],
    config: SyntheticTemplateCatalogConfig,
    recent_template_indices: List[int],
) -> SyntheticTemplateSpec:
    if recent_template_indices and rng.random() < max(0.0, min(1.0, config.locality_reuse_probability)):
        reversed_indices = list(reversed(recent_template_indices[-max(1, config.locality_window):]))
        recency_weights = [
            (config.locality_recency_decay ** rank) * template_catalog[index].popularity
            for rank, index in enumerate(reversed_indices)
        ]
        chosen_index = _weighted_choice(rng, reversed_indices, recency_weights)
        return template_catalog[chosen_index]
    base_indices = list(range(len(template_catalog)))
    base_weights = [spec.popularity for spec in template_catalog]
    chosen_index = _weighted_choice(rng, base_indices, base_weights)
    return template_catalog[chosen_index]


def _parse_bool(value: object, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_stop_strings(value: object) -> List[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item)]
    return [text]


def _infer_max_new_tokens(dist: LengthDistribution) -> int:
    p99_value = dist.p99 if dist.p99 is not None else dist.quantile(0.99)
    return max(
        1,
        int(
            math.ceil(
                max(
                    float(dist.mean),
                    float(dist.p90),
                    float(p99_value),
                )
            )
        ),
    )


def _scale_length_distribution(
    distribution: LengthDistribution,
    mean_scale: float,
    tail_scale: float,
) -> LengthDistribution:
    mean_value = max(1.0, float(distribution.mean) * mean_scale)
    # tail_scale changes dispersion; recompute all quantiles from that model.
    std_value = 0.0 if distribution.family == "deterministic" else max(0.0, distribution.std * tail_scale)
    return LengthDistribution.from_moments(mean_value, std_value, distribution.family)


def _sample_regime_anchors(
    rng: random.Random,
    count: int,
    jitter: float,
    *,
    low: float,
    high: float,
) -> List[float]:
    if count <= 1:
        return [1.0]
    anchors = [_clamp(1.0 + rng.uniform(-0.35 * jitter, 0.35 * jitter), low, high)]
    for _ in range(1, count):
        target = 1.0 + rng.uniform(-jitter, jitter)
        anchors.append(_clamp(0.65 * anchors[-1] + 0.35 * target, low, high))
    return anchors


def _interpolate_anchor(anchors: List[float], position: float) -> float:
    if len(anchors) == 1:
        return float(anchors[0])
    scaled = _clamp(position, 0.0, 1.0) * float(len(anchors) - 1)
    lower = int(scaled)
    upper = min(len(anchors) - 1, lower + 1)
    mix = scaled - float(lower)
    return float((1.0 - mix) * anchors[lower] + mix * anchors[upper])


def apply_dynamic_query_stream(
    requests: List[Request],
    seed: int,
    config: DynamicWorkloadConfig,
) -> List[Request]:
    if not config.enabled or not requests:
        return requests

    rng = random.Random(seed)
    ordered = sorted(requests, key=lambda request: request.arrival_ms)
    first_arrival = float(ordered[0].arrival_ms)
    last_arrival = float(ordered[-1].arrival_ms)
    arrival_span = max(last_arrival - first_arrival, 1.0)
    regime_count = max(1, int(config.regime_count))
    template_profiles_by_model = _build_dynamic_template_profiles(ordered)
    family_mix_anchors = _build_family_mix_anchors(
        rng,
        template_profiles_by_model,
        regime_count,
        config.template_mix_jitter,
    )
    version_rollout_anchor_by_family: dict[tuple[str, str], float] = {}
    recent_template_indices_by_model: dict[str, List[int]] = {}
    profile_index_by_template: dict[str, dict[str, int]] = {}
    for model_id, profiles in template_profiles_by_model.items():
        recent_template_indices_by_model[model_id] = []
        profile_index_by_template[model_id] = {
            profile.template_id: index
            for index, profile in enumerate(profiles)
        }
        for family in {profile.template_family for profile in profiles}:
            version_rollout_anchor_by_family[(model_id, family)] = rng.uniform(0.20, 0.85)

    arrival_anchors = _sample_regime_anchors(
        rng,
        regime_count,
        config.arrival_jitter,
        low=0.35,
        high=1.85,
    )
    prompt_anchors = _sample_regime_anchors(
        rng,
        regime_count,
        config.prompt_jitter,
        low=0.55,
        high=1.85,
    )
    output_anchors = _sample_regime_anchors(
        rng,
        regime_count,
        config.output_jitter,
        low=0.45,
        high=2.10,
    )
    tail_anchors = []
    for anchor in _sample_regime_anchors(
        rng,
        regime_count,
        config.tail_jitter,
        low=0.60,
        high=2.60,
    ):
        tail_anchors.append(max(anchor, output_anchors[len(tail_anchors)]))

    transformed: List[Request] = []
    transformed_arrival = 0.0
    previous_base_arrival = 0.0
    for index, request in enumerate(ordered):
        position = 0.0 if len(ordered) == 1 else (float(request.arrival_ms) - first_arrival) / arrival_span
        arrival_scale = _interpolate_anchor(arrival_anchors, position)
        prompt_scale = _interpolate_anchor(prompt_anchors, position)
        mean_scale = _interpolate_anchor(output_anchors, position)
        tail_scale = _interpolate_anchor(tail_anchors, position)

        request_noise = 1.0 + rng.uniform(-config.per_request_jitter, config.per_request_jitter)
        tail_noise = 1.0 + rng.uniform(-config.per_request_jitter, config.per_request_jitter)
        arrival_noise = 1.0 + rng.uniform(-0.5 * config.per_request_jitter, 0.5 * config.per_request_jitter)
        if rng.random() < config.burst_probability:
            arrival_noise *= max(0.15, 1.0 - config.burst_strength)

        prompt_scale = _clamp(prompt_scale * request_noise, 0.50, 1.95)
        mean_scale = _clamp(mean_scale * request_noise, 0.40, 2.25)
        tail_scale = _clamp(max(mean_scale, tail_scale * tail_noise), mean_scale, 2.80)

        if index == 0:
            transformed_arrival = max(0.0, float(request.arrival_ms) * arrival_scale * arrival_noise)
        else:
            base_delta = max(0.0, float(request.arrival_ms) - previous_base_arrival)
            transformed_arrival += max(0.0, base_delta * arrival_scale * arrival_noise)
        previous_base_arrival = float(request.arrival_ms)

        profiles = template_profiles_by_model.get(request.model_id)
        selected_profile: DynamicTemplateProfile | None = None
        if profiles:
            selected_profile = _sample_dynamic_template_profile(
                rng,
                profiles=profiles,
                family_anchors=family_mix_anchors.get(request.model_id, {}),
                position=position,
                config=config,
                recent_template_indices=recent_template_indices_by_model[request.model_id],
            )
            selected_index = profile_index_by_template[request.model_id][selected_profile.template_id]
            recent_template_indices_by_model[request.model_id].append(selected_index)
            if len(recent_template_indices_by_model[request.model_id]) > max(1, config.template_locality_window):
                recent_template_indices_by_model[request.model_id].pop(0)

        selected_profile = selected_profile or DynamicTemplateProfile(
            model_id=request.model_id,
            template_id=str(request.template_id or request.shared_prefix_id or f"request-{request.request_id}"),
            template_family=_canonical_template_family(request),
            template_version=str(request.template_version or "v1"),
            shared_prefix_id=str(request.shared_prefix_id or ""),
            shared_prefix_tokens=int(request.shared_prefix_tokens),
            private_prompt_tokens=max(1, int(request.private_prompt_tokens)),
            length_distribution=request.length_distribution,
            structured_output=bool(request.structured_output),
            stop_strings=tuple(request.stop_strings),
            stop_at_eos=bool(request.stop_at_eos),
            weight=1.0,
        )

        shared_prefix_tokens = max(0, int(selected_profile.shared_prefix_tokens))
        private_prompt_seed = 0.65 * float(request.private_prompt_tokens) + 0.35 * float(selected_profile.private_prompt_tokens)
        private_prompt_tokens = max(1, int(round(private_prompt_seed * prompt_scale)))
        prompt_tokens = max(1, shared_prefix_tokens + private_prompt_tokens)
        blended_distribution = _blend_length_distributions(
            request.length_distribution,
            selected_profile.length_distribution,
            template_weight=0.55,
        )
        length_distribution = _scale_length_distribution(blended_distribution, mean_scale, tail_scale)
        max_new_seed = request.max_new_tokens or _infer_max_new_tokens(selected_profile.length_distribution)
        max_new_tokens = max(1, int(round(float(max_new_seed) * tail_scale)))
        template_version = selected_profile.template_version
        rollout_center = version_rollout_anchor_by_family.get(
            (request.model_id, selected_profile.template_family),
            2.0,
        )
        rollout_pressure = _clamp((position - rollout_center) / 0.20, 0.0, 1.0)
        churn_probability = config.template_version_churn_probability * rollout_pressure
        if shared_prefix_tokens > 0 and rng.random() < churn_probability:
            template_version = _next_template_version(template_version)
            version_scale = 1.0 + rng.uniform(-config.template_version_churn_scale, config.template_version_churn_scale)
            shared_prefix_tokens = max(0, int(round(shared_prefix_tokens * version_scale)))
            prompt_tokens = max(1, shared_prefix_tokens + private_prompt_tokens)
        shared_prefix_id = _build_dynamic_shared_prefix_id(selected_profile, template_version)
        if config.resample_actual_output or request.actual_output_tokens <= 0:
            actual_output_tokens = min(max_new_tokens, length_distribution.sample(rng))
        else:
            actual_output_tokens = min(
                max_new_tokens,
                max(1, int(round(float(request.actual_output_tokens) * tail_scale))),
            )

        transformed.append(
            Request(
                request_id=request.request_id,
                arrival_ms=round(transformed_arrival, 6),
                prompt_tokens=prompt_tokens,
                length_distribution=length_distribution,
                model_id=request.model_id,
                priority=request.priority,
                template_id=selected_profile.template_id,
                template_family=selected_profile.template_family,
                template_version=template_version,
                max_new_tokens=max_new_tokens,
                stop_strings=list(selected_profile.stop_strings),
                stop_at_eos=selected_profile.stop_at_eos,
                structured_output=selected_profile.structured_output,
                shared_prefix_id=shared_prefix_id,
                shared_prefix_tokens=shared_prefix_tokens,
                private_prompt_tokens=private_prompt_tokens,
                actual_output_tokens=max(1, actual_output_tokens),
            )
        )
    return transformed


def generate_synthetic_requests(count: int, seed: int) -> List[Request]:
    rng = random.Random(seed)
    requests: List[Request] = []
    arrival_ms = 0.0
    catalog_config = SyntheticTemplateCatalogConfig()
    template_catalog = build_default_template_catalog(seed, config=catalog_config)
    recent_template_indices: List[int] = []
    template_index_by_id = {
        spec.template_id: index
        for index, spec in enumerate(template_catalog)
    }
    for request_id in range(count):
        arrival_ms += rng.expovariate(1.0 / 18.0)
        template = _sample_template_spec(
            rng,
            template_catalog=template_catalog,
            config=catalog_config,
            recent_template_indices=recent_template_indices,
        )
        recent_template_indices.append(template_index_by_id[template.template_id])
        if len(recent_template_indices) > max(1, catalog_config.locality_window):
            recent_template_indices.pop(0)
        private_prompt_tokens = rng.randint(
            template.private_prompt_min_tokens,
            max(template.private_prompt_min_tokens, template.private_prompt_max_tokens),
        )
        prompt_tokens = max(1, template.shared_prefix_tokens + private_prompt_tokens)
        mean = rng.uniform(template.output_mean_low, max(template.output_mean_low, template.output_mean_high))
        mean += private_prompt_tokens * rng.uniform(
            template.output_prompt_correlation_low,
            max(template.output_prompt_correlation_low, template.output_prompt_correlation_high),
        )
        std = max(2.0, mean * rng.uniform(template.output_cv_low, max(template.output_cv_low, template.output_cv_high)))
        dist = LengthDistribution.from_moments(mean, std)
        shared_prefix_id = f"{template.model_id}:template:{template.template_id}:{template.template_version}"
        max_new_tokens = max(8, int(math.ceil((dist.p99 or dist.quantile(0.99)) * template.max_new_token_headroom)))
        actual_output = min(max_new_tokens, dist.sample(rng))
        requests.append(
            Request(
                request_id=request_id,
                arrival_ms=arrival_ms,
                prompt_tokens=prompt_tokens,
                length_distribution=dist,
                model_id=template.model_id,
                priority=0.0,
                template_id=template.template_id,
                template_family=template.template_family,
                template_version=template.template_version,
                max_new_tokens=max_new_tokens,
                stop_strings=list(template.stop_strings),
                stop_at_eos=template.stop_at_eos,
                structured_output=rng.random() < template.structured_output_rate,
                shared_prefix_id=shared_prefix_id,
                shared_prefix_tokens=template.shared_prefix_tokens,
                private_prompt_tokens=private_prompt_tokens,
                actual_output_tokens=actual_output,
            )
        )
    return requests


def load_trace(
    path: Path,
    seed: int,
    dynamic_config: DynamicWorkloadConfig | None = None,
) -> List[Request]:
    rng = random.Random(seed)
    suffix = path.suffix.lower()
    requests: List[Request] = []
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif suffix == ".json":
        rows = json.loads(path.read_text())
    elif suffix == ".csv":
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError(f"unsupported trace format: {path}")
    for index, row in enumerate(rows):
        mean = float(row["output_mean"])
        std = float(row.get("output_std", max(1.0, mean * 0.35)))
        family = str(row.get("output_family") or "lognormal")
        parametric = LengthDistribution.from_moments(mean, std, family)
        p90 = float(row.get("output_p90") or parametric.p90)
        p50 = row.get("output_p50")
        p95 = row.get("output_p95")
        p99 = row.get("output_p99")
        dist = LengthDistribution(
            mean=mean,
            std=std,
            p90=p90,
            p50=float(p50) if p50 not in (None, "") else None,
            p95=float(p95) if p95 not in (None, "") else None,
            p99=float(p99) if p99 not in (None, "") else None,
            family=family,
        )
        shared_prefix_tokens = int(row.get("shared_prefix_tokens") or 0)
        private_prompt_tokens = int(row.get("private_prompt_tokens") or 0)
        prompt_tokens_raw = row.get("prompt_tokens")
        prompt_tokens = (
            int(prompt_tokens_raw)
            if prompt_tokens_raw not in (None, "")
            else max(1, shared_prefix_tokens + private_prompt_tokens)
        )
        actual_output_raw = row.get("actual_output_tokens")
        actual_output = int(actual_output_raw) if actual_output_raw not in (None, "") else 0
        max_new_raw = row.get("max_new_tokens")
        # An absent request cap is unknown. Inferring it from the realized output
        # would expose replay-only information to scheduling policies.
        max_new_tokens = int(max_new_raw) if max_new_raw not in (None, "") else 0
        stop_strings = _parse_stop_strings(row.get("stop_strings") or row.get("stop_sequences"))
        structured_output = _parse_bool(
            row.get("structured_output", row.get("is_structured_output")),
            default=False,
        )
        stop_at_eos = _parse_bool(row.get("stop_at_eos"), default=True)
        template_id = str(
            row.get("template_id")
            or row.get("prompt_template_id")
            or row.get("shared_prefix_id")
            or ""
        )
        template_family = str(row.get("template_family") or row.get("prompt_template_family") or "")
        template_version = str(row.get("template_version") or row.get("prompt_template_version") or "v1")
        shared_prefix_id = str(
            row.get("shared_prefix_id")
            or row.get("prompt_template_id")
            or row.get("shared_prefix_key")
            or ""
        )
        requests.append(
            Request(
                request_id=int(row.get("request_id", index)),
                arrival_ms=float(row["arrival_ms"]),
                prompt_tokens=prompt_tokens,
                length_distribution=dist,
                model_id=str(row.get("model_id") or row.get("model") or "default"),
                priority=float(row.get("priority", 0.0)),
                template_id=template_id,
                template_family=template_family,
                template_version=template_version,
                max_new_tokens=max_new_tokens,
                stop_strings=stop_strings,
                stop_at_eos=stop_at_eos,
                structured_output=structured_output,
                shared_prefix_id=shared_prefix_id,
                shared_prefix_tokens=shared_prefix_tokens,
                private_prompt_tokens=private_prompt_tokens,
                actual_output_tokens=max(0, actual_output),
            )
        )
    if dynamic_config is not None and dynamic_config.enabled:
        return apply_dynamic_query_stream(requests, seed, dynamic_config)
    return requests
