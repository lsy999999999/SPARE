"""Resumable real-GPU KV/capacity pilot using the frozen idea1 workloads."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/www_idea1_gpu_pilot_20261016_r1"
sys.path.insert(0, str(ROOT))
os.environ["WWW_GPU_OUT"] = str(OUT.relative_to(ROOT))

from research.www_evaluation_gpu_completion_20261003 import run as gpu_run  # noqa: E402


METHOD = dict(arrival="threshold", native="native", sarathi="sarathi_v1",
              sola_rule="sola_rule", uniboost_static="uniboost_static",
              past_future="past_future_v1", s3_proxy="s3_proxy",
              apt_serve_kv_rule="apt_serve_kv_rule")
gpu_run.METHOD.update(METHOD)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def design():
    value = json.loads((OUT / "design.json").read_text())
    for rel, sha in value["source_hashes"].items():
        path = OUT / rel if rel.startswith("source/") else ROOT / rel
        if digest(path) != sha:
            raise RuntimeError(f"source changed: {rel}")
    for rel in (
        "research/www_evaluation_gpu_completion_20261003/scheduler.py",
        "research/www_evaluation_gpu_completion_20261003/policy.py",
        "research/www_legacy_baselines_20261006/scheduler.py",
        "research/www_legacy_baselines_20261006/policy.py",
        "research/v1_paper_comparison_20260922/scheduler.py",
        "research/v1_paper_comparison_20260922/rules.py",
    ):
        if digest(ROOT / rel) != value["source_hashes"]["source/" + rel]:
            raise RuntimeError(f"runtime policy differs from frozen source: {rel}")
    for rel, sha in value["input_hashes"].items():
        if digest(OUT / rel) != sha:
            raise RuntimeError(f"input changed: {rel}")
    for source, sha in value["history_hashes"].items():
        if digest(OUT / "history" / f"{source}.json") != sha:
            raise RuntimeError(f"history changed: {source}")
    return value


def memory_free_mib(gpu):
    command = [sys.executable, "-c",
               "import torch; f,t=torch.cuda.mem_get_info(0); "
               "print(torch.cuda.get_device_name(0)); print(f//1048576); print(t//1048576)"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    result = subprocess.check_output(command, text=True, env=env,
                                     stderr=subprocess.DEVNULL, timeout=20).splitlines()
    name, free, total = result[-3], int(result[-2]), int(result[-1])
    return name, free, total


def idle(gpu):
    try:
        name, free, total = memory_free_mib(gpu)
        return "RTX 4090" in name and total >= 23000 and free >= 22000
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def acquire_gpu(gpu):
    handle = open(f"/tmp/query-capacity-gpu{gpu}.lock", "a")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def engine_for(kv):
    import torch
    import vllm.platforms as platforms
    from vllm.platforms.cuda import CudaPlatform

    if platforms.current_platform.device_type != "cuda":
        if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
            raise RuntimeError("CUDA platform unavailable")
        # vLLM provides a CUDA platform backed by torch when NVML is unavailable.
        platforms._current_platform = CudaPlatform()
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    model = "models/Qwen2.5-7B-Instruct"
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=model, dtype="half", max_model_len=8176,
        max_num_seqs=64, max_num_batched_tokens=2048,
        enable_chunked_prefill=True, enable_prefix_caching=False,
        block_size=16, gpu_memory_utilization=.9, enforce_eager=True,
        disable_log_stats=True, seed=0, num_gpu_blocks_override=kv,
        scheduler_cls="research.www_idea1_gpu_pilot_20261016.scheduler.IdeaScheduler"))
    scheduler = engine.engine_core.engine_core.scheduler
    if scheduler.kv_cache_config.num_blocks != kv:
        raise RuntimeError("KV override was not honored")
    return engine, scheduler


def metrics(result, rate, spec):
    if result["status"] != "complete":
        return dict(valid=False, error=result["error"])
    records = result["requests"]
    observed = [r for r in records if spec["warm_ms"] <= r["arrival_ms"] < spec["horizon_ms"]]
    if not observed:
        return dict(valid=False, error="no measured requests")
    gaps = [b - a for r in observed
            for a, b in zip(result["clocks"][r["request_id"]]["token_times_ms"],
                            result["clocks"][r["request_id"]]["token_times_ms"][1:])]

    def outstanding(at):
        return sum(r["arrival_ms"] <= at < r["finish_ms"] for r in records)

    ttft = [r["ttft_ms"] for r in observed]
    p95 = gpu_run.quantile(ttft, .95)
    p99_tbt = gpu_run.quantile(gaps, .99) if gaps else 0.
    drift = outstanding(spec["horizon_ms"]) - outstanding(spec["backlog_check_ms"])
    return dict(valid=True, measured_requests=len(observed),
                mean_ttft_ms=sum(ttft) / len(ttft), p95_ttft_ms=p95,
                p99_tbt_ms=p99_tbt, backlog_drift=drift,
                drift_limit=max(2., rate),
                passed=p95 <= spec["p95_ttft_limit_ms"]
                       and p99_tbt <= spec["p99_tbt_limit_ms"]
                       and drift <= max(2., rate),
                preemptions=result["metrics"]["preemptions"],
                recomputed_tokens=result["metrics"]["recomputed_tokens"],
                elapsed_ms=result["elapsed_ms"])


def episode(engine, scheduler, source, kv, arm, rate, spec):
    rel = f"inputs/{source}_r{rate:g}.json"
    requests = json.loads((OUT / rel).read_text())
    cell = dict(budget=spec["budget"], chunk=spec["chunk"],
                history_file=f"history/{source}.json")
    result = gpu_run.run_episode(engine, scheduler, requests, arm, cell, 900)
    summary = metrics(result, rate, spec)
    summary.update(source=source, kv=kv, method=arm, rate=rate,
                   input_sha256=spec["input_hashes"][rel], gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    return summary


def worker(source, kv, gpu):
    spec = design()
    lock = acquire_gpu(gpu)
    try:
        if not idle(gpu):
            raise RuntimeError(f"GPU {gpu} is occupied")
        import torch
        torch.cuda.set_device(0)
        engine, scheduler = engine_for(kv)
        warm = [dict(id=f"warm{i}", prompt=256, output=16, arrival_ms=0.) for i in range(4)]
        cell = dict(budget=1024, chunk=256, history_file=f"history/{source}.json")
        result = gpu_run.run_episode(engine, scheduler, warm, "native", cell, 120)
        if result["status"] != "complete":
            raise RuntimeError(f"GPU warmup failed: {result['error']}")
        for arm in spec["methods"]:
            for rate in spec["rates"]:
                path = OUT / "rates" / f"{source}_c{kv}_{arm}_r{rate:g}.json"
                if path.exists():
                    row = json.loads(path.read_text())
                else:
                    write(OUT / "current" / f"gpu{gpu}.json",
                          dict(source=source, kv=kv, method=arm, rate=rate, started=time.time()))
                    row = episode(engine, scheduler, source, kv, arm, rate, spec)
                    if not row["valid"]:
                        raise RuntimeError(f"{source} c{kv} {arm} r{rate:g}: {row['error']}")
                    write(path, row)
                    print(f"{source} c{kv} {arm} r{rate:g} pass={row['passed']} "
                          f"TTFT={row['mean_ttft_ms']:.1f}", flush=True)
                if not row["passed"]:
                    break
        write(OUT / "groups" / f"{source}_c{kv}.json",
              dict(state="complete", source=source, kv=kv, gpu=gpu, updated=time.time()))
    finally:
        lock.close()


def preflight(gpu):
    spec = design()
    lock = acquire_gpu(gpu)
    try:
        if not idle(gpu):
            raise RuntimeError(f"GPU {gpu} is occupied")
        engine, scheduler = engine_for(max(spec["capacities"]))
        requests = [dict(id=f"probe{i}", prompt=256 + i * 128,
                         output=4, arrival_ms=float(i * 30)) for i in range(8)]
        cell = dict(budget=1024, chunk=256, history_file="history/code.json")
        for arm in ("arrival", "s3_proxy"):
            result = gpu_run.run_episode(engine, scheduler, requests, arm, cell, 120)
            if result["status"] != "complete":
                raise RuntimeError(f"{arm}: {result['error']}")
            if result["metrics"]["completed"] != len(requests):
                raise RuntimeError(f"{arm}: incomplete probe")
            print(f"GPU {gpu} {arm} completed {len(requests)} requests", flush=True)
        write(OUT / "preflight.json", dict(passed=True, gpu=gpu,
                                            methods=["arrival", "s3_proxy"],
                                            time=time.time()))
    finally:
        lock.close()


def controller():
    spec = design()
    if not json.loads((OUT / "preflight.json").read_text())["passed"]:
        raise RuntimeError("GPU preflight is required")
    tasks = [(s["source"], kv) for kv in spec["capacities"] for s in spec["source_samples"]]
    pending = [task for task in tasks if not (OUT / "groups" / f"{task[0]}_c{task[1]}.json").exists()]
    mutex = threading.Lock()
    active = {}
    failures = []

    def publish(state="running"):
        write(OUT / "status.json", dict(state=state, updated=time.time(),
              pending=[f"{s}_c{k}" for s, k in pending], active=active,
              failures=failures, completed_groups=len(tasks) - len(pending) - len(active) - len(failures),
              total_groups=len(tasks)))

    def consume(gpu):
        while True:
            with mutex:
                if not pending:
                    return
            if not idle(gpu):
                time.sleep(20)
                continue
            with mutex:
                if not pending:
                    return
                source, kv = pending.pop(0)
                active[str(gpu)] = f"{source}_c{kv}"
                publish()
            log_path = OUT / "logs" / f"{source}_c{kv}_gpu{gpu}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-B", "-u", str(Path(__file__)),
                       "--worker", source, str(kv), "--gpu", str(gpu)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       VLLM_USE_V1="1", VLLM_ENABLE_V1_MULTIPROCESSING="0",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1")
            with log_path.open("a") as log:
                process = subprocess.run(command, cwd=ROOT, env=env,
                                         stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT)
            with mutex:
                active.pop(str(gpu), None)
                if process.returncode:
                    failures.append(dict(source=source, kv=kv, gpu=gpu,
                                         log=str(log_path.relative_to(OUT)),
                                         returncode=process.returncode))
                publish()

    publish()
    with ThreadPoolExecutor(max_workers=len(spec["gpu_candidates"])) as pool:
        futures = [pool.submit(consume, gpu) for gpu in spec["gpu_candidates"]]
        for future in futures:
            future.result()
    publish("complete" if not failures else "finished_with_failures")


def launch():
    design()
    if (OUT / "launch.json").exists():
        raise RuntimeError("already launched")
    log = OUT / "controller.log"
    with log.open("a") as handle:
        process = subprocess.Popen([sys.executable, "-B", "-u", str(Path(__file__))],
                                   cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True,
                                   env=dict(os.environ, VLLM_USE_V1="1",
                                            VLLM_ENABLE_V1_MULTIPROCESSING="0"))
    write(OUT / "launch.json", dict(pid=process.pid, started=time.time()))
    print(f"launched controller PID {process.pid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker", nargs=2)
    parser.add_argument("--gpu", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.preflight:
            preflight(args.gpu)
        elif args.launch:
            launch()
        elif args.worker:
            worker(args.worker[0], int(args.worker[1]), args.gpu)
        else:
            controller()
    except BaseException:
        write(OUT / "last_error.json", dict(error=traceback.format_exc(), time=time.time()))
        raise
