"""Refine Qwen2.5-7B policy capacity brackets on the original four windows."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from research.www_policy_ranking_20261023 import bisect_capacity as bisect


ROOT = bisect.ROOT
OLD = ROOT / "artifacts/www_policy_ranking_bisect_20261024_r1"
OUT = ROOT / "artifacts/www_policy_ranking_qwen7b_refine_20261025_r1"
MODEL = "qwen25_7b"
ARMS = bisect.ARMS
TARGETS = (("code", 4096), ("code", 6144), ("code", 7424),
           ("conversation", 4096), ("conversation", 7424))


def refine(source: str, kv: int, arm: str, side: str, engine=None,
           scheduler=None, gpu: int | None = None) -> dict:
    label = f"{MODEL}_{source}_c{kv}_{arm}"
    original = json.loads((OLD / side / "search" / f"{label}.json").read_text())
    if original["state"] != "complete":
        raise RuntimeError(f"original search is incomplete: {label} {side}")
    path = OUT / side / "search" / f"{label}.json"
    saved = json.loads(path.read_text()) if path.exists() else None
    points = saved["new_points"] if saved else {}
    low = round(original["lower_qps"] * 1000)
    high = round(original["upper_qps"] * 1000)

    def publish(state: str) -> dict:
        merged = dict(original["points"])
        merged.update(points)
        rows = sorted(merged.values(), key=lambda row: row["rate_qps"])
        labels = [row["passed"] for row in rows]
        result = dict(state=state, model=MODEL, source=source, kv_blocks=kv,
                      arm=arm, side=side, updated=time.time(),
                      original_lower_qps=original["lower_qps"],
                      original_upper_qps=original["upper_qps"],
                      lower_qps=low / 1000, upper_qps=high / 1000,
                      resolution_qps=(high-low) / 1000,
                      nonmonotonic=labels != sorted(labels, reverse=True),
                      new_points=points)
        bisect.write(path, result)
        return result

    while high-low > 1:
        middle = (low + high) // 2
        rate = middle / 1000
        key = f"{rate:.3f}"
        if key not in points:
            point = (bisect.simulator_point(MODEL, source, kv, arm, rate)
                     if side == "sim" else
                     bisect.gpu_point(MODEL, source, kv, arm, rate,
                                      engine, scheduler, gpu))
            points[key] = point
            publish("running")
        if points[key]["passed"]:
            low = middle
        else:
            high = middle
    return publish("complete")


def run(source: str, kv: int, side: str, gpu: int | None) -> None:
    bisect.OUT = OUT
    if side == "sim":
        for arm in ARMS:
            row = refine(source, kv, arm, side)
            print(source, kv, arm, row["lower_qps"], row["upper_qps"], flush=True)
        return

    from research.www_cross_model_20261021 import run_gpu as base
    from research.www_evaluation_gpu_completion_20261003 import run as gpu_run
    from research.www_idea1_gpu_pilot_20261016 import run as pilot

    if gpu is None:
        raise ValueError("--gpu is required for GPU runs")
    base.OUT = OUT / "gpu"
    gpu_run.OUT = base.OUT
    for workload in ("code", "conversation"):
        source_path = bisect.SOURCE / "history" / f"{workload}.json"
        target = base.OUT / "history" / f"{workload}.json"
        data = json.loads(source_path.read_text())
        if target.exists() and json.loads(target.read_text()) != data:
            raise RuntimeError(f"history changed: {target}")
        bisect.write(target, data)
    base.MODELS[MODEL] = Path(
        "models/Qwen2.5-7B-Instruct")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if not pilot.idle(gpu):
        raise RuntimeError(f"GPU {gpu} is busy")
    with pilot.acquire_gpu(gpu):
        engine, scheduler = base.engine_for(base.MODELS[MODEL], kv)
        warm = [dict(id=f"warm{i}", prompt=256, output=16, arrival_ms=0.)
                for i in range(4)]
        result = gpu_run.run_episode(engine, scheduler, warm, "native",
                                     {"budget": 1024, "chunk": 256,
                                      "history_file": "history/code.json"}, 120)
        if result["status"] != "complete":
            raise RuntimeError(f"warmup failed: {result['error']}")
        for arm in ARMS:
            row = refine(source, kv, arm, side, engine, scheduler, gpu)
            print(source, kv, arm, row["lower_qps"], row["upper_qps"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("gpu", "sim"), required=True)
    parser.add_argument("--source", choices=("code", "conversation"), required=True)
    parser.add_argument("--kv", type=int, required=True)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if (args.source, args.kv) not in TARGETS:
        raise ValueError("group is not in the prespecified refinement set")
    run(args.source, args.kv, args.side, args.gpu)


if __name__ == "__main__":
    main()
