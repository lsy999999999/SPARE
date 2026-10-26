"""Replay held-out cross-model GPU episodes with pilot-fitted timing."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


ROOT = Path(__file__).resolve().parents[2]
GPU = ROOT / "artifacts/www_cross_model_20261021_r1"
TIMING = ROOT / "artifacts/www_cross_model_timing_20261022_r1"
OUT = ROOT / "artifacts/www_cross_model_simulator_validation_20261022_r3"
MODELS = ("qwen25_3b", "qwen3_8b", "mistral_7b")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def install_fixed256() -> None:
    from research.www_evaluation_completion_20261003 import simulate as sim
    from research.www_evaluation_gpu_completion_20261003.scheduler import GPUBaselineScheduler

    if "fixed256" in sim.ARMS:
        return
    sim.FormulaScheduler = GPUBaselineScheduler
    original = sim.make_scheduler

    def make_scheduler(arm="arrival", *args, **kwargs):
        if arm == "fixed256":
            kwargs["policy_options"] = {"mode": "fixed", "fixed_budget": 256}
            return original("arrival", *args, **kwargs)
        return original(arm, *args, **kwargs)

    sim.make_scheduler = make_scheduler
    sim.ARMS = (*sim.ARMS, "fixed256")
    sim.ADAPTATION["fixed256"] = ("ablation", ())


def metric(result: dict, rate: float) -> dict:
    from research.www_evaluation_gpu_completion_20261003 import run as gpu_run

    records = result["probe_records"]
    observed = [r for r in records if 20000 <= r["arrival_ms"] < 60000]
    if not observed:
        raise RuntimeError("no requests in 20-60 s measurement interval")
    gaps = [gap for r in observed
            for gap in result["probe_intervals_ms"][r["request_id"]]]

    def outstanding(at: float) -> int:
        return sum(r["arrival_ms"] <= at < r["finish_ms"] for r in records)

    ttft = statistics.fmean(r["ttft_ms"] for r in observed)
    tbt = gpu_run.quantile(gaps, .99) if gaps else 0.
    drift = outstanding(60000) - outstanding(40000)
    return dict(mean_ttft_ms=ttft, p99_tbt_ms=tbt, backlog_drift=drift,
                passed=ttft <= 2000 and tbt <= 200 and drift <= max(2, rate),
                measured_requests=len(observed),
                preemptions=result["metrics"]["preemptions"],
                recomputed_tokens=result["metrics"]["recomputed_tokens"])


def replay(gpu_path: str) -> dict:
    from research.www_evaluation_completion_20261003 import simulate as sim
    from research.www_evaluation_completion_20261003.timing import FittedTiming

    path = Path(gpu_path)
    row = json.loads(path.read_text())
    label = path.stem
    target = OUT / "runs" / f"{label}.json"
    profile_path = TIMING / "profiles" / f"{row['model']}.json"
    adapter_hash = digest(Path(__file__))
    simulator_hash = digest(Path(sim.__file__))
    if target.exists():
        saved = json.loads(target.read_text())
        if (saved["gpu_run_sha256"] == digest(path) and
                saved["timing_profile_sha256"] == digest(profile_path) and
                saved["adapter_sha256"] == adapter_hash and
                saved["simulator_sha256"] == simulator_hash):
            return dict(skipped=True, label=label)
        raise RuntimeError(f"existing replay has different sources: {target}")
    profile = json.loads(profile_path.read_text())
    input_path = GPU / "inputs" / f"{row['sample']}_r{row['rate_qps']:g}.json"
    if digest(input_path) != row["input_sha256"]:
        raise RuntimeError(f"input hash mismatch: {input_path}")
    history = json.loads((GPU / "history" / f"{row['source']}.json").read_text())
    result = sim.simulate(
        json.loads(input_path.read_text()), arm=row["arm"],
        capacity=row["kv_blocks"], budget=1024, chunk=256,
        history=history["lengths"], seed=0,
        fitted_timing=FittedTiming(profile), capture_probe=True,
    )
    prediction = metric(result, row["rate_qps"])
    saved = dict(model=row["model"], sample=row["sample"], source=row["source"],
                 kv_blocks=row["kv_blocks"], arm=row["arm"], rate_qps=row["rate_qps"],
                 gpu_run_sha256=digest(path), input_sha256=row["input_sha256"],
                 timing_profile_sha256=digest(profile_path),
                 adapter_sha256=adapter_hash, simulator_sha256=simulator_hash,
                 gpu={k: row[k] for k in prediction if k in row},
                 simulator=prediction, trajectory_sha256=result["trajectory_sha256"])
    write(target, saved)
    return dict(skipped=False, label=label)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    install_fixed256()
    paths = sorted((GPU / "runs").glob("*.json"))
    paths = [p for p in paths if (r := json.loads(p.read_text()))["model"] in MODELS
             and not r["sample"].endswith("d1_w08")
             and (args.model is None or r["model"] == args.model)]
    if args.limit is not None:
        paths = paths[:args.limit]
    failures = []
    completed = 0
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=install_fixed256) as pool:
        futures = {pool.submit(replay, str(path)): path.name for path in paths}
        for future in as_completed(futures):
            try:
                future.result()
                completed += 1
            except Exception as exc:
                failures.append(dict(run=futures[future], error=repr(exc)))
            if (completed + len(failures)) % 10 == 0 or failures:
                write(OUT / "status.json", dict(state="running", started=started,
                    updated=time.time(), completed=completed, total=len(paths),
                    failures=failures))
    write(OUT / "status.json", dict(state="complete" if not failures else "finished_with_failures",
          started=started, updated=time.time(), completed=completed,
          total=len(paths), failures=failures))
    print(f"{completed}/{len(paths)} replayed, {len(failures)} failures", flush=True)
    if failures:
        print(failures[:5], flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
