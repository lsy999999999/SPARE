from __future__ import annotations

import copy
import json
import math
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from .engine import LLMSingleNodeSimulator
from .metrics import OutputMetricsCollector
from .types import LengthDistribution, ModelRuntimeConfig, MultiModelSimulatorConfig, Request, SimulatorConfig


class RequestModelRouter:
    name = "prefill-model-router"

    def resolve_model(self, request: Request, simulator: "LLMMultiModelSimulator | None") -> str:
        raise NotImplementedError


class FixedRequestModelRouter(RequestModelRouter):
    name = "fixed-request-model-router"

    def __init__(self, default_model_id: str = "default"):
        self.default_model_id = default_model_id

    def resolve_model(self, request: Request, simulator: "LLMMultiModelSimulator | None") -> str:
        if request.model_id and request.model_id != "default":
            return str(request.model_id)
        return self.default_model_id


class RuntimeAwareRequestModelRouter(RequestModelRouter):
    name = "runtime-aware-prefill-model-router"

    def __init__(self, default_model_id: str = "default"):
        self.default_model_id = default_model_id

    def resolve_model(self, request: Request, simulator: "LLMMultiModelSimulator | None") -> str:
        if request.model_id and request.model_id != "default":
            return str(request.model_id)
        if simulator is None or not simulator.model_runtime_configs:
            return self.default_model_id

        best_model_id = self.default_model_id
        best_score = float("-inf")
        for model_id, runtime in simulator.model_runtime_configs.items():
            config = runtime.simulator_config
            predicted_prompt = max(1.0, request.prompt_tokens * runtime.prompt_token_scale)
            predicted_output = max(1.0, request.length_distribution.mean * runtime.output_mean_scale)
            predicted_kv = predicted_prompt + predicted_output
            kv_capacity = max(float(config.kv_capacity_tokens), 1.0)
            kv_pressure = predicted_kv / kv_capacity
            prefill_parallelism = int(config.prefill_parallelism or config.max_prefill_batch)
            decode_parallelism = int(config.decode_parallelism or config.max_decode_batch)
            score = (
                float(runtime.routing_bias)
                + math.log(max(float(runtime.routing_weight), 1e-6))
                + 0.08 * prefill_parallelism
                + 0.05 * decode_parallelism
                + 0.15 * math.log1p(kv_capacity / 1024.0)
                - 2.0 * kv_pressure
                - 0.10 * runtime.output_tail_scale
            )
            if score > best_score:
                best_score = score
                best_model_id = model_id
        return best_model_id


class MultiModelEnvironmentAdapter:
    name = "multi-model-environment"

    def build_state(self, simulator: "LLMMultiModelSimulator") -> Dict[str, object]:
        raise NotImplementedError


class DefaultMultiModelEnvironmentAdapter(MultiModelEnvironmentAdapter):
    name = "default-multi-model-environment"

    def build_state(self, simulator: "LLMMultiModelSimulator") -> Dict[str, object]:
        models: Dict[str, Dict[str, object]] = {}
        for model_id, child in simulator.model_simulators.items():
            models[model_id] = {
                "requests": len(child.requests),
                "current_kv_tokens": child.current_kv_tokens,
                "current_logical_kv_tokens": child.current_logical_kv_tokens,
                "prefill_parallelism": int(child.config.prefill_parallelism or child.config.max_prefill_batch),
                "decode_parallelism": int(child.config.decode_parallelism or child.config.max_decode_batch),
                "kv_capacity_tokens": int(child.config.kv_capacity_tokens),
                "kv_cache_scope": child.config.kv_cache_scope,
                "kv_allocator_mode": child.config.kv_allocator_mode,
                "model_profile": {
                    "routing_weight": simulator.model_runtime_configs.get(model_id).routing_weight
                    if model_id in simulator.model_runtime_configs
                    else 1.0,
                    "routing_bias": simulator.model_runtime_configs.get(model_id).routing_bias
                    if model_id in simulator.model_runtime_configs
                    else 0.0,
                    "prompt_token_scale": simulator.model_runtime_configs.get(model_id).prompt_token_scale
                    if model_id in simulator.model_runtime_configs
                    else 1.0,
                    "output_mean_scale": simulator.model_runtime_configs.get(model_id).output_mean_scale
                    if model_id in simulator.model_runtime_configs
                    else 1.0,
                    "output_std_scale": simulator.model_runtime_configs.get(model_id).output_std_scale
                    if model_id in simulator.model_runtime_configs
                    else 1.0,
                    "output_tail_scale": simulator.model_runtime_configs.get(model_id).output_tail_scale
                    if model_id in simulator.model_runtime_configs
                    else 1.0,
                },
                "action_space": {
                    "phase_choices": list(child.config.action_space.phase_choices),
                    "admission_choices": list(child.config.action_space.admission_choices),
                    "new_admission_limit_choices": list(child.config.action_space.new_admission_limit_choices),
                    "quantile_choices": list(child.config.action_space.quantile_choices),
                    "budget_profile_choices": list(child.config.action_space.budget_profile_choices),
                    "risky_admission_cap_choices": list(child.config.action_space.risky_admission_cap_choices),
                },
                "queue_sizes": {
                    queue.value: len(child.prefill_queues[queue])
                    for queue in child.prefill_queues
                },
                "active_decode": len([request for request in child.active_decode if not request.is_complete]),
            }
        return {
            "model_count": len(models),
            "models": models,
        }


def build_model_simulator_config(
    base_config: SimulatorConfig,
    overrides: Dict[str, object],
) -> SimulatorConfig:
    payload = {field.name: copy.deepcopy(getattr(base_config, field.name)) for field in fields(SimulatorConfig)}
    valid_keys = set(payload.keys())
    for key, value in overrides.items():
        if key not in valid_keys:
            raise ValueError(f"unsupported SimulatorConfig override for multi-model runtime: {key}")
        payload[key] = value
    return SimulatorConfig(**payload)


def load_multi_model_simulator_config(
    path: Path,
    base_config: SimulatorConfig,
) -> MultiModelSimulatorConfig:
    payload = json.loads(path.read_text())
    model_rows = payload.get("models")
    if not isinstance(model_rows, list):
        raise ValueError("multi-model config JSON must contain a 'models' list")
    runtimes: List[ModelRuntimeConfig] = []
    for row in model_rows:
        if not isinstance(row, dict):
            raise ValueError("each multi-model runtime config entry must be an object")
        model_id = str(row.get("model_id") or "").strip()
        if not model_id:
            raise ValueError("each multi-model runtime config entry must define model_id")
        overrides = dict(row)
        overrides.pop("model_id", None)
        routing_weight = float(overrides.pop("routing_weight", 1.0))
        routing_bias = float(overrides.pop("routing_bias", 0.0))
        prompt_token_scale = float(overrides.pop("prompt_token_scale", 1.0))
        output_mean_scale = float(overrides.pop("output_mean_scale", 1.0))
        output_std_scale = float(overrides.pop("output_std_scale", 1.0))
        output_tail_scale = float(overrides.pop("output_tail_scale", 1.0))
        actual_output_scale = float(overrides.pop("actual_output_scale", output_mean_scale))
        runtimes.append(
            ModelRuntimeConfig(
                model_id=model_id,
                simulator_config=build_model_simulator_config(base_config, overrides),
                routing_weight=routing_weight,
                routing_bias=routing_bias,
                prompt_token_scale=prompt_token_scale,
                output_mean_scale=output_mean_scale,
                output_std_scale=output_std_scale,
                output_tail_scale=output_tail_scale,
                actual_output_scale=actual_output_scale,
            )
        )
    return MultiModelSimulatorConfig(
        default_model_id=str(payload.get("default_model_id") or "default"),
        model_runtimes=runtimes,
    )


class LLMMultiModelSimulator:
    def __init__(
        self,
        requests: Iterable[Request],
        default_config: SimulatorConfig,
        multi_model_config: Optional[MultiModelSimulatorConfig] = None,
        scheduler_factories: Optional[Dict[str, Callable[[SimulatorConfig], object]]] = None,
        metrics_collectors: Optional[Dict[str, OutputMetricsCollector]] = None,
        model_router: Optional[RequestModelRouter] = None,
        environment_adapter: Optional[MultiModelEnvironmentAdapter] = None,
    ):
        self.default_config = default_config
        self.multi_model_config = multi_model_config or MultiModelSimulatorConfig()
        self.scheduler_factories = scheduler_factories or {}
        self.metrics_collectors = metrics_collectors or {}
        self.model_router = model_router or RuntimeAwareRequestModelRouter(
            default_model_id=self.multi_model_config.default_model_id
        )
        self.environment_adapter = environment_adapter or DefaultMultiModelEnvironmentAdapter()
        self.requests = list(requests)
        self.model_runtime_configs: Dict[str, ModelRuntimeConfig] = {
            runtime.model_id: runtime
            for runtime in self.multi_model_config.model_runtimes
        }
        self.model_configs: Dict[str, SimulatorConfig] = {
            runtime.model_id: runtime.simulator_config
            for runtime in self.multi_model_config.model_runtimes
        }
        self.model_request_counts: Counter[str] = Counter()
        grouped_requests: Dict[str, List[Request]] = {}
        for request in self.requests:
            model_id = self.model_router.resolve_model(request, self)
            self._assign_request_to_model(request, model_id)
            self.model_request_counts[model_id] += 1
            grouped_requests.setdefault(model_id, []).append(request)
        for model_id in self.model_configs:
            grouped_requests.setdefault(model_id, [])
        if not grouped_requests:
            grouped_requests[self.multi_model_config.default_model_id] = []
        self.model_simulators: Dict[str, LLMSingleNodeSimulator] = {}
        self.model_summaries: Dict[str, Dict[str, object]] = {}
        self.events: List[Dict[str, object]] = []
        for model_id, model_requests in grouped_requests.items():
            config = copy.deepcopy(self.model_configs.get(model_id, self.default_config))
            scheduler_factory = self.scheduler_factories.get(model_id)
            scheduler = scheduler_factory(config) if scheduler_factory is not None else None
            metrics_collector = self.metrics_collectors.get(model_id)
            child = LLMSingleNodeSimulator(
                requests=model_requests,
                config=config,
                scheduler=scheduler,
                metrics_collector=metrics_collector,
            )
            self.model_simulators[model_id] = child

    @staticmethod
    def _scale_distribution(
        distribution: LengthDistribution,
        *,
        mean_scale: float,
        std_scale: float,
        tail_scale: float,
    ) -> LengthDistribution:
        mean = max(1.0, float(distribution.mean) * max(float(mean_scale), 1e-6))
        std = max(1.0, float(distribution.std) * max(float(std_scale), 1e-6))
        p50 = (
            max(1.0, float(distribution.p50) * max(float(mean_scale), 1e-6))
            if distribution.p50 is not None
            else None
        )
        p90 = max(mean, float(distribution.p90) * max(float(tail_scale), 1e-6))
        p95 = (
            max(p90, float(distribution.p95) * max(float(tail_scale), 1e-6))
            if distribution.p95 is not None
            else None
        )
        p99 = (
            max(p95 or p90, float(distribution.p99) * max(float(tail_scale), 1e-6))
            if distribution.p99 is not None
            else None
        )
        return LengthDistribution(
            mean=mean,
            std=std,
            p50=p50,
            p90=p90,
            p95=p95,
            p99=p99,
            family=distribution.family,
        )

    def _assign_request_to_model(self, request: Request, model_id: str) -> None:
        original_model_id = request.model_id
        runtime = self.model_runtime_configs.get(model_id)
        request.model_id = model_id
        request.model_route_name = self.model_router.name
        request.model_route_metadata = {
            "original_model_id": original_model_id,
            "selected_model_id": model_id,
            "stage": "before-prefill",
        }
        if runtime is None:
            return

        if runtime.prompt_token_scale != 1.0:
            request.prompt_tokens = max(1, int(round(request.prompt_tokens * runtime.prompt_token_scale)))
            request.shared_prefix_tokens = max(0, int(round(request.shared_prefix_tokens * runtime.prompt_token_scale)))
            request.private_prompt_tokens = max(0, request.prompt_tokens - request.shared_prefix_tokens)
        if request.shared_prefix_tokens > 0 and request.template_id:
            request.shared_prefix_id = f"{model_id}:template:{request.template_id}:{request.template_version}"

        request.length_distribution = self._scale_distribution(
            request.length_distribution,
            mean_scale=runtime.output_mean_scale,
            std_scale=runtime.output_std_scale,
            tail_scale=runtime.output_tail_scale,
        )
        if request.actual_output_tokens > 0 and runtime.actual_output_scale != 1.0:
            request.actual_output_tokens = max(1, int(round(request.actual_output_tokens * runtime.actual_output_scale)))
            if request.max_new_tokens > 0:
                request.actual_output_tokens = min(request.actual_output_tokens, request.max_new_tokens)
        request.model_route_metadata.update(
            {
                "routing_weight": runtime.routing_weight,
                "routing_bias": runtime.routing_bias,
                "prompt_token_scale": runtime.prompt_token_scale,
                "output_mean_scale": runtime.output_mean_scale,
                "output_std_scale": runtime.output_std_scale,
                "output_tail_scale": runtime.output_tail_scale,
                "actual_output_scale": runtime.actual_output_scale,
            }
        )

    def build_environment_state(self) -> Dict[str, object]:
        return self.environment_adapter.build_state(self)

    def run(self) -> Dict[str, object]:
        self.events = []
        self.model_summaries = {}
        for model_id, child in self.model_simulators.items():
            summary = child.run()
            summary["model_id"] = model_id
            self.model_summaries[model_id] = summary
            for event in child.events:
                annotated_event = dict(event)
                annotated_event["model_id"] = model_id
                self.events.append(annotated_event)
        self.events.sort(
            key=lambda event: (
                float(event.get("time_ms", 0.0)),
                LLMSingleNodeSimulator.EVENT_ORDER.get(str(event.get("event", "")), 999),
                int(event.get("request_id", -1)),
                str(event.get("model_id", "")),
            )
        )
        total_requests = sum(int(summary["requests"]) for summary in self.model_summaries.values())
        completed = sum(int(summary["completed"]) for summary in self.model_summaries.values())
        unfinished = sum(int(summary["unfinished_requests"]) for summary in self.model_summaries.values())
        horizon_unfinished = sum(int(summary.get("horizon_unfinished_requests", 0)) for summary in self.model_summaries.values())
        aged_unfinished = sum(int(summary.get("aged_unfinished_requests", 0)) for summary in self.model_summaries.values())
        liveness_unfinished = sum(int(summary.get("liveness_unfinished_requests", 0)) for summary in self.model_summaries.values())
        punishable_unfinished = sum(
            int(summary.get("punishable_unfinished_requests", 0)) for summary in self.model_summaries.values()
        )
        unfinished_ids: List[str] = []
        termination_reason_counts: Counter[str] = Counter()
        completed_request_metrics = 0
        total_output_tokens = 0
        total_released_logical_kv_tokens = 0
        total_released_physical_kv_tokens = 0
        output_length_sources: Counter[str] = Counter()
        for model_id, summary in self.model_summaries.items():
            termination_reason_counts[str(summary.get("termination_reason", "unknown"))] += 1
            unfinished_ids.extend(f"{model_id}:{request_id}" for request_id in summary.get("unfinished_request_ids", []))
            output_metrics = summary.get("output_metrics", {})
            completed_request_metrics += int(output_metrics.get("completed_requests", 0))
            total_output_tokens += int(output_metrics.get("total_output_tokens", 0))
            total_released_logical_kv_tokens += int(output_metrics.get("total_released_logical_kv_tokens", 0))
            total_released_physical_kv_tokens += int(output_metrics.get("total_released_physical_kv_tokens", 0))
            for source_name, count in (output_metrics.get("output_length_sources") or {}).items():
                output_length_sources[str(source_name)] += int(count)
        return {
            "requests": total_requests,
            "completed": completed,
            "unfinished_requests": unfinished,
            "horizon_unfinished_requests": horizon_unfinished,
            "aged_unfinished_requests": aged_unfinished,
            "liveness_unfinished_requests": liveness_unfinished,
            "punishable_unfinished_requests": punishable_unfinished,
            "unfinished_request_ids": unfinished_ids,
            "terminated_early": unfinished > 0,
            "model_count": len(self.model_simulators),
            "default_model_id": self.multi_model_config.default_model_id,
            "model_request_counts": dict(self.model_request_counts),
            "models": sorted(self.model_simulators.keys()),
            "total_runtime_ms": max(
                (float(summary.get("total_runtime_ms", 0.0)) for summary in self.model_summaries.values()),
                default=0.0,
            ),
            "peak_kv_tokens": sum(
                float(summary.get("peak_kv_tokens", 0.0)) for summary in self.model_summaries.values()
            ),
            "peak_logical_kv_tokens": sum(
                float(summary.get("peak_logical_kv_tokens", 0.0)) for summary in self.model_summaries.values()
            ),
            "evictions_total": sum(
                int(summary.get("evictions_total", 0)) for summary in self.model_summaries.values()
            ),
            "recompute_tokens_total": sum(
                int(summary.get("recompute_tokens_total", 0)) for summary in self.model_summaries.values()
            ),
            "wasted_kv_tokens_total": sum(
                int(summary.get("wasted_kv_tokens_total", 0)) for summary in self.model_summaries.values()
            ),
            "termination_reason_counts": dict(termination_reason_counts),
            "environment_adapter": self.environment_adapter.name,
            "environment_state": self.build_environment_state(),
            "output_metrics": {
                "completed_requests": completed_request_metrics,
                "total_output_tokens": total_output_tokens,
                "total_released_logical_kv_tokens": total_released_logical_kv_tokens,
                "total_released_physical_kv_tokens": total_released_physical_kv_tokens,
                "output_length_sources": dict(output_length_sources),
            },
            "model_summaries": self.model_summaries,
        }
