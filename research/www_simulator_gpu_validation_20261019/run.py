"""Idle-only, resumable 4090 controller for the broad validation matrix."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gc
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from research.www_simulator_gpu_validation_20261018 import run as previous
from research.www_idea1_gpu_pilot_20261016 import run as pilot
from research.www_evaluation_gpu_completion_20261003 import run as gpu_run
from research.www_simulator_gpu_validation_20261019 import prepare

ROOT, OUT = prepare.ROOT, prepare.OUT
previous.OUT = pilot.OUT = gpu_run.OUT = OUT


def write(path, obj):
    prepare.write(path, obj)


def design():
    spec = json.loads((OUT / "design.json").read_text())
    for rel, expected in spec["source_hashes"].items():
        if prepare.sha(ROOT / rel) != expected:
            raise RuntimeError(f"source changed: {rel}")
    for rel, expected in spec["input_hashes"].items():
        if prepare.sha(OUT / rel) != expected:
            raise RuntimeError(f"input changed: {rel}")
    for kind, expected in spec["history_hashes"].items():
        if prepare.sha(OUT / "history" / f"{kind}.json") != expected:
            raise RuntimeError(f"history changed: {kind}")
    return spec


def tasks(spec, kind, kv):
    for sample in (s for s in spec["samples"] if s["source"] == kind):
        for arm in spec["methods"]:
            plan = next(p for p in spec["plans"] if
                        (p["source"], p["kv_blocks"], p["arm"]) == (kind, kv, arm))
            for point in plan["simulator_points"]:
                yield sample, arm, point["rate_qps"]


def name(sample, kv, arm, rate):
    return f"{sample['name']}_c{kv}_{arm}_r{rate:g}"


def run_task(engine, scheduler, spec, sample, kv, arm, rate, gpu):
    label = name(sample, kv, arm, rate)
    target = OUT / "runs" / f"{label}.json"
    if target.exists():
        return
    rel = f"inputs/{sample['name']}_r{rate:g}.json"
    requests = json.loads((OUT / rel).read_text())
    cell = dict(budget=1024, chunk=256,
                history_file=f"history/{sample['source']}.json")
    write(OUT / "current" / f"gpu{gpu}.json",
          dict(task=label, started=time.time()))
    result = gpu_run.run_episode(engine, scheduler, requests, arm, cell, 900)
    result.pop("rounds", None)
    raw = OUT / "raw" / f"{label}.json.gz"
    raw.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw, "wt", compresslevel=2) as stream:
        json.dump(result, stream, separators=(",", ":"), allow_nan=False)
    row = dict(sample=sample["name"], source=sample["source"], kv_blocks=kv,
               arm=arm, rate_qps=rate, gpu=gpu, input_sha256=spec["input_hashes"][rel],
               raw_path=str(raw.relative_to(OUT)), raw_sha256=prepare.sha(raw),
               **previous.metrics(result, rate, spec))
    write(target, row)
    print(label, "passed", row["passed"], "ttft", round(row["mean_ttft_ms"], 1), flush=True)


def worker(kind, kv, gpu):
    spec = design()
    lock = previous.lock_gpu(gpu)
    try:
        import torch
        torch.cuda.set_device(0)
        engine, scheduler = pilot.engine_for(kv)
        warm = [dict(id=f"warm{i}", prompt=256, output=16, arrival_ms=0.)
                for i in range(4)]
        cell = dict(budget=1024, chunk=256, history_file="history/code.json")
        warmup = gpu_run.run_episode(engine, scheduler, warm, "native", cell, 120)
        if warmup["status"] != "complete":
            raise RuntimeError(f"warmup failed: {warmup['error']}")
        for sample, arm, rate in tasks(spec, kind, kv):
            run_task(engine, scheduler, spec, sample, kv, arm, rate, gpu)
        write(OUT / "groups" / f"{kind}_c{kv}.json",
              dict(state="complete", gpu=gpu, updated=time.time()))
        del engine, scheduler
        gc.collect()
        torch.cuda.empty_cache()
    finally:
        lock.close()


def controller():
    spec = design()
    jobs = [(kind, kv) for kv in spec["kv_blocks"]
            for kind in ("code", "conversation")]
    pending = [job for job in jobs if not (OUT / "groups" / f"{job[0]}_c{job[1]}.json").exists()]
    mutex = threading.Lock()
    active, failures = {}, []

    def publish(state="running"):
        write(OUT / "status.json", dict(state=state, updated=time.time(),
              completed_runs=len(list((OUT / "runs").glob("*.json"))),
              planned_runs=len(spec["plans"]) * 16,
              completed_groups=len(jobs) - len(pending) - len(active) - len(failures),
              total_groups=len(jobs), pending=[f"{s}_c{k}" for s, k in pending],
              active=active, failures=failures))

    def consume(gpu):
        while True:
            with mutex:
                if not pending:
                    return
            if not pilot.idle(gpu):
                time.sleep(30)
                continue
            with mutex:
                if not pending:
                    return
                kind, kv = pending.pop(0)
                active[str(gpu)] = f"{kind}_c{kv}"
                publish()
            log = OUT / "logs" / f"{kind}_c{kv}_gpu{gpu}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            command = [sys.executable, "-B", "-u", str(Path(__file__)),
                       "--worker", kind, str(kv), "--gpu", str(gpu)]
            with log.open("a") as stream:
                result = subprocess.run(command, cwd=ROOT, env=env,
                                        stdin=subprocess.DEVNULL, stdout=stream,
                                        stderr=subprocess.STDOUT)
            with mutex:
                active.pop(str(gpu), None)
                if result.returncode:
                    failures.append(dict(group=f"{kind}_c{kv}", gpu=gpu,
                                         log=str(log.relative_to(OUT)),
                                         exit_code=result.returncode))
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
    with (OUT / "controller.log").open("a") as stream:
        process = subprocess.Popen([sys.executable, "-B", "-u", str(Path(__file__))],
                                   cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    write(OUT / "launch.json", dict(pid=process.pid, started=time.time()))
    print("controller PID", process.pid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker", nargs=2)
    parser.add_argument("--gpu", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.preflight:
            previous.preflight(args.gpu)
        elif args.launch:
            launch()
        elif args.worker:
            worker(args.worker[0], int(args.worker[1]), args.gpu)
        else:
            controller()
    except BaseException:
        write(OUT / "last_error.json", dict(error=traceback.format_exc(),
                                             time=time.time()))
        raise
