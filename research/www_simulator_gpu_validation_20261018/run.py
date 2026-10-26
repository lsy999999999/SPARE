"""Run the frozen capacity brackets on an idle RTX 4090."""
from __future__ import annotations

import argparse
import fcntl
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from research.www_idea1_gpu_pilot_20261016 import run as pilot
from research.www_evaluation_gpu_completion_20261003 import run as gpu_run

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/www_simulator_gpu_validation_20261018_r2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

gpu_run.OUT = OUT
pilot.OUT = OUT


def design():
    frozen = json.loads((OUT / "design.json").read_text())
    for rel, digest in frozen["source_hashes"].items():
        if sha(ROOT / rel) != digest:
            raise RuntimeError(f"source changed: {rel}")
    for rel, digest in frozen["input_hashes"].items():
        if sha(OUT / rel) != digest:
            raise RuntimeError(f"input changed: {rel}")
    for source, digest in frozen["history_hashes"].items():
        if sha(OUT / "history" / f"{source}.json") != digest:
            raise RuntimeError(f"history changed: {source}")
    return frozen


def gpu_free(gpu):
    import torch

    if (os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu)
            or not torch.cuda.is_available()
            or "RTX 4090" not in torch.cuda.get_device_name(0)):
        return False
    free, total = torch.cuda.mem_get_info(0)
    return total >= 23_000 * 1024**2 and free >= 22_000 * 1024**2


def lock_gpu(gpu):
    handle = open(f"/tmp/query-capacity-gpu{gpu}.lock", "a")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not gpu_free(gpu):
        handle.close()
        raise RuntimeError(f"GPU {gpu} is not idle")
    return handle


def metrics(result, rate, frozen):
    if result["status"] != "complete":
        raise RuntimeError(result["error"])
    records = result["requests"]
    warm, horizon = frozen["warm_ms"], frozen["horizon_ms"]
    observed = [r for r in records if warm <= r["arrival_ms"] < horizon]
    assert observed and result["metrics"]["completed"] == len(records)
    gaps = [right - left for r in observed
            for left, right in zip(result["clocks"][r["request_id"]]["token_times_ms"],
                                   result["clocks"][r["request_id"]]["token_times_ms"][1:])]

    def outstanding(at):
        return sum(r["arrival_ms"] <= at < r["finish_ms"] for r in records)

    drift = outstanding(60000) - outstanding(40000)
    ttft = statistics.fmean(r["ttft_ms"] for r in observed)
    tbt = gpu_run.quantile(gaps, .99) if gaps else 0.0
    submission_delay = [r["submitted_ms"] - r["arrival_ms"] for r in observed]
    passed = (ttft <= frozen["mean_ttft_limit_ms"]
              and tbt <= frozen["p99_tbt_limit_ms"]
              and drift <= max(2, rate))
    return dict(valid=True, passed=passed, measured_requests=len(observed),
                mean_ttft_ms=ttft, p99_tbt_ms=tbt, backlog_drift=drift,
                pending_40s=outstanding(40000), pending_60s=outstanding(60000),
                drift_limit=max(2, rate), submission_delay_p95_ms=gpu_run.quantile(
                    submission_delay, .95), preemptions=result["metrics"]["preemptions"],
                recomputed_tokens=result["metrics"]["recomputed_tokens"],
                total_requests=len(records), elapsed_ms=result["elapsed_ms"])


def run_task(engine, scheduler, task, frozen, gpu):
    sample, kv, arm, rate, repeat = task
    name = f"{sample['name']}_c{kv}_{arm}_r{rate:g}_rep{repeat}"
    target = OUT / "runs" / f"{name}.json"
    if target.exists():
        return json.loads(target.read_text())
    rel = f"inputs/{sample['name']}_r{rate:g}.json"
    requests = json.loads((OUT / rel).read_text())
    cell = dict(budget=frozen["budget"], chunk=frozen["chunk"],
                history_file=f"history/{sample['source']}.json")
    write(OUT / "current.json", dict(task=name, gpu=gpu, started=time.time()))
    result = gpu_run.run_episode(engine, scheduler, requests, arm, cell, 600)
    result.pop("rounds", None)
    raw = OUT / "raw" / f"{name}.json.gz"
    raw.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw, "wt", compresslevel=2) as stream:
        json.dump(result, stream, separators=(",", ":"), allow_nan=False)
    row = dict(sample=sample["name"], source=sample["source"], kv_blocks=kv,
               arm=arm, rate_qps=rate, repeat=repeat, gpu=gpu,
               input_sha256=frozen["input_hashes"][rel],
               raw_path=str(raw.relative_to(OUT)), raw_sha256=sha(raw),
               **metrics(result, rate, frozen))
    write(target, row)
    print(name, "passed", row["passed"], "ttft", round(row["mean_ttft_ms"], 1), flush=True)
    return row


def tasks(frozen):
    samples = frozen["samples"]
    for kv_string, arms in frozen["methods_by_kv"].items():
        kv = int(kv_string)
        for source in ("code", "conversation"):
            for sample in (s for s in samples if s["source"] == source):
                for repeat in range(frozen["repeats"]):
                    for arm in (arms if repeat % 2 == 0 else list(reversed(arms))):
                        plan = next(p for p in frozen["plans"]
                                    if (p["source"], p["kv_blocks"], p["arm"])
                                    == (source, kv, arm))
                        for point in plan["simulator_bracket"]:
                            yield sample, kv, arm, point["rate_qps"], repeat


def preflight(gpu):
    frozen = design()
    handle = lock_gpu(gpu)
    try:
        import torch

        torch.cuda.set_device(0)
        engine, scheduler = pilot.engine_for(2048)
        requests = [dict(id=f"probe{i}", prompt=256 + i * 64,
                         output=4, arrival_ms=float(i * 30)) for i in range(8)]
        cell = dict(budget=1024, chunk=256, history_file="history/code.json")
        for arm in ("arrival", "s3_proxy", "apt_serve_kv_rule"):
            result = gpu_run.run_episode(engine, scheduler, requests, arm, cell, 120)
            if result["status"] != "complete":
                raise RuntimeError(f"{arm}: {result['error']}")
        write(OUT / "preflight.json", dict(passed=True, gpu=gpu,
                                            design_sha256=sha(OUT / "design.json"),
                                            finished=time.time()))
    finally:
        handle.close()


def run(gpu):
    frozen = design()
    ready = json.loads((OUT / "preflight.json").read_text())
    if not ready["passed"] or ready["design_sha256"] != sha(OUT / "design.json"):
        raise RuntimeError("preflight missing or stale")
    handle = lock_gpu(gpu)
    completed = 0
    try:
        import torch

        torch.cuda.set_device(0)
        planned = list(tasks(frozen))
        for kv in frozen["methods_by_kv"]:
            kv = int(kv)
            pending = [task for task in planned if task[1] == kv]
            if all((OUT / "runs" / f"{s['name']}_c{k}_{a}_r{r:g}_rep{rep}.json").exists()
                   for s, k, a, r, rep in pending):
                completed += len(pending)
                continue
            engine, scheduler = pilot.engine_for(kv)
            warm = [dict(id=f"warm{i}", prompt=256, output=16, arrival_ms=0.0)
                    for i in range(4)]
            cell = dict(budget=1024, chunk=256, history_file="history/code.json")
            warmup = gpu_run.run_episode(engine, scheduler, warm, "native", cell, 120)
            if warmup["status"] != "complete":
                raise RuntimeError(f"warmup: {warmup['error']}")
            for task in pending:
                run_task(engine, scheduler, task, frozen, gpu)
                completed += 1
                write(OUT / "status.json", dict(state="running", completed=completed,
                                                  planned=len(planned), current=task,
                                                  updated=time.time()))
            del engine, scheduler
            gc.collect()
            torch.cuda.empty_cache()
        write(OUT / "status.json", dict(state="complete", completed=completed,
                                          planned=len(planned), updated=time.time()))
    finally:
        handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight(args.gpu)
    else:
        run(args.gpu)
