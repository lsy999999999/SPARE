"""Freeze a common-QPS policy ranking check on held-out Qwen3-8B windows."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/www_policy_ranking_fidelity_20261023_r1"
OLD = ROOT / "artifacts/www_cross_model_20261021_r1/runs"
ARMS = ("arrival", "srp", "sarathi")
WINDOWS = ("d1_w02", "d1_w20", "d2_w02", "d2_w20")
RATES = {
    "code": (0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5),
    "conversation": (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0),
}


def frozen_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"frozen design differs: {path}")
        return
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def label(sample: str, arm: str, rate: float) -> str:
    return f"qwen3_8b_{sample}_c2048_{arm}_r{rate:g}"


def main() -> None:
    design = dict(
        model="qwen3_8b", kv_blocks=2048, arms=list(ARMS),
        sources={k: list(v) for k, v in RATES.items()}, windows=list(WINDOWS),
        slo=dict(mean_ttft_ms=2000, p99_tbt_ms=200,
                 backlog_limit="max(2, rate_qps)", arrivals_ms=[0, 60000],
                 observed_arrivals_ms=[20000, 60000], backlog_ms=[40000, 60000]),
        comparison="same sample, model, KV and common QPS grid for all policies",
        selection="Qwen3-8B 2048 KV: Conversation predicted separation; Code tie control",
        gpu_existing="artifacts/www_cross_model_20261021_r1/runs",
        timing_profile="artifacts/www_cross_model_timing_20261022_r1/profiles/qwen3_8b.json",
    )
    frozen_write(OUT / "design.json", design)
    total = reused = 0
    for source, rates in RATES.items():
        for window in WINDOWS:
            sample = f"{source}_{window}"
            tasks = []
            for arm in ARMS:
                for rate in rates:
                    total += 1
                    old = OLD / "{}.json".format(label(sample, arm, rate))
                    if old.exists():
                        row = json.loads(old.read_text())
                        if row.get("status") != "complete" or not row.get("valid"):
                            raise RuntimeError(f"invalid existing run: {old}")
                        reused += 1
                        continue
                    tasks.append(dict(sample=sample, arm=arm, rate_qps=rate))
            frozen_write(OUT / "tasks" / f"{sample}.json", dict(
                model="qwen3_8b", kv_blocks=2048, source=source,
                sample=sample, tasks=tasks))
    print(f"Frozen {total} GPU cells; {reused} reuse completed GPU runs; {total-reused} new runs")


if __name__ == "__main__":
    main()
