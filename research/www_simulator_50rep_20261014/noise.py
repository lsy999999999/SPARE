"""Empirical, phase-stratified timing perturbations from GPU fit training rounds."""
from __future__ import annotations

import gzip
import json
import random
import statistics
from collections import defaultdict

from research.www_evaluation_completion_20261003.common import ROOT, read, write, digest
from research.www_evaluation_completion_20261003.timing import batch_row


def phase(batch) -> str:
    return ("mixed" if batch.prefill_chunks and batch.decode_contexts else
            "prefill" if batch.prefill_chunks else "decode")


def residuals(old, out) -> dict:
    target = out / "timing_residuals.json"
    if target.exists():
        return read(target)
    profile = read(old / "timing_profile.json")
    coefficients = profile["coefficients"]
    groups = defaultdict(list)
    sources = []
    for source in profile["sources"]:
        path = ROOT / source["path"]
        if digest(path) != source["sha256"]:
            raise RuntimeError(f"Timing source changed: {path}")
        with gzip.open(path, "rt") as handle:
            rounds = json.load(handle)
        for row in rounds:
            if not row["allocations"]:
                continue
            features = batch_row(row)
            predicted = max(.1, sum(a * b for a, b in zip(features, coefficients)))
            observed = float(row["step_ms"])
            if observed <= 0:
                continue
            bucket = ("mixed" if features[6] and features[3] else
                      "prefill" if features[6] else "decode")
            groups[bucket].append(observed / predicted)
        sources.append(source)
    payload = {}
    for bucket, values in sorted(groups.items()):
        values.sort()
        lo, hi = values[len(values) // 100], values[len(values) * 99 // 100]
        clipped = [min(hi, max(lo, x)) for x in values]
        payload[bucket] = dict(ratios=clipped, mean=statistics.fmean(clipped),
                               count=len(values), clip_p01=lo, clip_p99=hi)
    if set(payload) != {"prefill", "decode", "mixed"}:
        raise RuntimeError(f"Missing timing phase residuals: {set(payload)}")
    result = dict(method="GPU fresh0 training-round observed/fitted ratios, phase stratified; "
                         "1st/99th percentile winsorized and mean normalized per phase",
                  source_profile_sha256=digest(old / "timing_profile.json"),
                  sources=sources, phases=payload)
    write(target, result)
    return result


class PerturbedTiming:
    def __init__(self, base, payload: dict, seed: int):
        self.base = base
        self.phases = payload["phases"]
        self.rng = random.Random(seed)

    def estimate(self, batch):
        duration, _ = self.base.estimate(batch)
        bucket = self.phases[phase(batch)]
        multiplier = self.rng.choice(bucket["ratios"]) / bucket["mean"]
        return max(.1, duration * multiplier), ("empirical_gpu_training_residual",)


class StaircaseTiming:
    def __init__(self, base):
        self.base = base

    def estimate(self, batch):
        import math
        duration, _ = self.base.estimate(batch)
        tokens = sum(batch.prefill_chunks) + len(batch.decode_contexts)
        return duration + 5 * math.ceil(tokens / 256), ("synthetic_staircase",)
