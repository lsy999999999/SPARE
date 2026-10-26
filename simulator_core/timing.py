"""Opt-in phase-specific timing profiles with explicit measurement-domain checks."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TimingBatch:
    prefill_chunks: tuple[int, ...] = ()
    prefill_contexts: tuple[int, ...] = ()
    decode_contexts: tuple[int, ...] = ()


class ResourceTiming:
    """Synthetic roofline-inspired cost family, NOT a measured GPU predictor.

    Shared weight traffic, context traffic and compute scale separately. Parameters
    can be varied per episode. Real deployment claims require profiling/calibration.
    """
    PARAMETERS = ("launch_ms", "weight_read_ms", "prefill_ms_per_token",
                  "prefill_context_ms_per_token2", "decode_ms_per_request",
                  "decode_context_ms_per_token")

    def __init__(self, payload: dict):
        self.parameters = {name: float(payload["parameters"][name]) for name in self.PARAMETERS}
        if any(not math.isfinite(v) or v < 0 for v in self.parameters.values()):
            raise ValueError("resource timing parameters must be finite and nonnegative")
        if self.parameters["launch_ms"] <= 0:
            raise ValueError("positive launch_ms is required")
        if payload.get("measurement_scope") != "synthetic-uncalibrated":
            raise ValueError("resource surrogate must be explicitly labelled synthetic-uncalibrated")

    def estimate(self, batch: TimingBatch) -> tuple[float, tuple[str, ...]]:
        if len(batch.prefill_chunks) != len(batch.prefill_contexts):
            raise ValueError("prefill chunks and contexts must match")
        if any(not isinstance(v, int) or v < 0 for v in
               batch.prefill_chunks + batch.prefill_contexts + batch.decode_contexts):
            raise ValueError("token counts must be nonnegative integers")
        if any(v == 0 for v in batch.prefill_chunks):
            raise ValueError("prefill chunks must be positive")
        if not batch.prefill_chunks and not batch.decode_contexts:
            return 0.0, ("synthetic_uncalibrated",)
        p = self.parameters
        attention_work = sum(n * context + n * (n + 1) / 2
                             for n, context in zip(batch.prefill_chunks, batch.prefill_contexts))
        compute = (sum(batch.prefill_chunks) * p["prefill_ms_per_token"]
                   + attention_work * p["prefill_context_ms_per_token2"]
                   + len(batch.decode_contexts) * p["decode_ms_per_request"])
        memory = p["weight_read_ms"] + sum(batch.decode_contexts) * p["decode_context_ms_per_token"]
        return p["launch_ms"] + max(compute, memory), ("synthetic_uncalibrated",)


def build_timing_model(payload: dict):
    if payload.get("kind") == "resource-surrogate-v1":
        return ResourceTiming(payload)
    return PhaseAffineTiming(payload)


class PhaseAffineTiming:
    PHASE_PARAMETERS = ("prefill_base_ms", "decode_base_ms", "prefill_ms_per_token",
                        "decode_ms_per_token", "decode_ms_per_context_token")
    PHASE_RANGES = ("prefill_chunk_tokens", "prefill_batch_size", "decode_batch_size",
                    "decode_context_tokens")
    MIXED_PARAMETERS = ("batch_base_ms", "decode_ms_per_request",
                        "decode_ms_per_context_token", "prefill_ms_per_token")
    MIXED_RANGES = ("batch_token_budget", "prefill_tokens", "decode_batch_size",
                    "decode_context_tokens")

    def __init__(self, payload: dict):
        self.kind = payload.get("kind")
        if self.kind == "phase-affine-v1":
            parameter_names, range_names = self.PHASE_PARAMETERS, self.PHASE_RANGES
        elif self.kind == "mixed-context-affine-v1":
            parameter_names, range_names = self.MIXED_PARAMETERS, self.MIXED_RANGES
        else:
            raise ValueError("unsupported timing profile kind")
        if any(not payload.get(key) for key in ("source", "model", "hardware", "measurement_scope")):
            raise ValueError("timing profile needs source, model, hardware and measurement_scope")
        self.parameters = {key: float(payload["parameters"][key]) for key in parameter_names}
        if any(not math.isfinite(v) or v < 0 for v in self.parameters.values()):
            raise ValueError("timing parameters must be finite and nonnegative")
        if self.kind == "phase-affine-v1":
            if self.parameters["prefill_base_ms"] + self.parameters["prefill_ms_per_token"] <= 0:
                raise ValueError("prefill timing must be positive")
            if self.parameters["decode_base_ms"] + self.parameters["decode_ms_per_token"] <= 0:
                raise ValueError("decode timing must be positive even at zero context")
        elif self.parameters["batch_base_ms"] <= 0:
            raise ValueError("mixed timing batch base must be positive")
        self.ranges = {key: tuple(float(v) for v in payload["calibration_domain"][key])
                       for key in range_names}
        for bounds in self.ranges.values():
            if len(bounds) != 2 or any(not math.isfinite(v) or v < 0 for v in bounds) or bounds[0] > bounds[1]:
                raise ValueError("invalid timing calibration domain")

    def estimate(self, batch: TimingBatch) -> tuple[float, tuple[str, ...]]:
        if len(batch.prefill_chunks) != len(batch.prefill_contexts):
            raise ValueError("one prefill context is required per executed chunk")
        if any(not isinstance(v, int) for v in batch.prefill_chunks + batch.prefill_contexts + batch.decode_contexts):
            raise ValueError("timing batch token counts must be integers")
        if any(v <= 0 for v in batch.prefill_chunks) or any(
            v < 0 for v in batch.prefill_contexts + batch.decode_contexts
        ):
            raise ValueError("invalid timing batch")
        reasons = set()

        def check(name, values):
            lo, hi = self.ranges[name]
            if any(not lo <= v <= hi for v in values):
                reasons.add(name + "_outside_calibration")

        p = self.parameters
        if self.kind == "phase-affine-v1":
            duration = 0.0
            if batch.prefill_chunks:
                duration += p["prefill_base_ms"] + p["prefill_ms_per_token"] * sum(batch.prefill_chunks)
                check("prefill_chunk_tokens", batch.prefill_chunks)
                check("prefill_batch_size", (len(batch.prefill_chunks),))
                if any(batch.prefill_contexts):
                    reasons.add("chunked_or_cached_prefill_unmeasured")
            if batch.decode_contexts:
                duration += (p["decode_base_ms"] + p["decode_ms_per_token"] * len(batch.decode_contexts)
                             + p["decode_ms_per_context_token"] * sum(batch.decode_contexts))
                check("decode_batch_size", (len(batch.decode_contexts),))
                check("decode_context_tokens", batch.decode_contexts)
            if batch.prefill_chunks and batch.decode_contexts:
                reasons.add("mixed_batch_serial_assumption")
            return duration, tuple(sorted(reasons))

        prefill_tokens = sum(batch.prefill_chunks)
        decode_batch = len(batch.decode_contexts)
        token_budget = prefill_tokens + decode_batch
        duration = (p["batch_base_ms"] + p["decode_ms_per_request"] * decode_batch
                    + p["decode_ms_per_context_token"] * sum(batch.decode_contexts)
                    + p["prefill_ms_per_token"] * prefill_tokens)
        check("batch_token_budget", (token_budget,))
        check("prefill_tokens", (prefill_tokens,))
        if decode_batch:
            check("decode_batch_size", (decode_batch,))
            check("decode_context_tokens", batch.decode_contexts)
        else:
            reasons.add("pure_prefill_composition_unmeasured")
        if any(batch.prefill_contexts):
            reasons.add("prefill_context_effect_unmodeled")
        return duration, tuple(sorted(reasons))
