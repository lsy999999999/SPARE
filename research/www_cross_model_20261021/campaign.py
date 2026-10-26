"""Run model-specific GPU boundary scans and held-out-window comparisons."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import random
import subprocess
import sys
import threading
import time


ROOT = Path(os.environ.get("WWW_CROSS_MODEL_ROOT", Path.cwd())).resolve()
OUT = ROOT / "artifacts/www_cross_model_20261021_r1"
MODEL_ROOT = Path("models")
MODELS = {"qwen3_8b": (1536, 1792, 2048),
          "qwen25_3b": (2048, 4096, 6144),
          "mistral_7b": (1536, 2048, 2560)}
ARMS = ("arrival", "srp", "fixed256", "sarathi")
RATES = (.5, 1., 1.5, 2., 2.5, 3., 4., 5., 6., 8., 10., 12., 16.)
GPU_CANDIDATES = tuple(range(6))
PYTHON = ".venv/bin/python"
RUNNER = str(ROOT / "research/www_cross_model_20261021/run_gpu.py")


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def label(model: str, sample: str, kv: int, arm: str, rate: float) -> str:
    return f"{model}_{sample}_c{kv}_{arm}_r{rate:g}"


def run(args: list[str], group: str, phase: str) -> None:
    log = OUT / "logs" / f"{group}_{phase}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, WWW_CROSS_MODEL_ROOT=str(ROOT), PYTHONPATH=str(ROOT))
    with log.open("a") as stream:
        result = subprocess.run([PYTHON, "-B", "-u", RUNNER, *args],
                                cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{phase} failed (exit {result.returncode}); see {log}")


def execute(model: str, source: str, kv: int, gpu: int) -> None:
    group = f"{model}_{source}_c{kv}"
    pilot_sample = f"{source}_d1_w08"
    base = ["--model", model, "--kv", str(kv), "--gpu", str(gpu)]
    run([*base, "--samples", pilot_sample, "--rates",
         *(f"{rate:g}" for rate in RATES), "--arms", *ARMS, "--adaptive"],
        group, "pilot")

    brackets = {}
    for arm in ARMS:
        observations = []
        for rate in RATES:
            path = OUT / "runs" / f"{label(model, pilot_sample, kv, arm, rate)}.json"
            if not path.exists():
                break
            row = json.loads(path.read_text())
            if row["status"] != "complete":
                raise RuntimeError(f"failed pilot row: {path}")
            observations.append(row)
            if not row["passed"]:
                break
        passing = [row["rate_qps"] for row in observations if row["passed"]]
        failing = [row["rate_qps"] for row in observations if not row["passed"]]
        if not passing or not failing:
            raise RuntimeError(f"capacity not bracketed: {group} {arm}")
        brackets[arm] = [passing[-1], failing[0]]

    tasks = []
    for sample in (f"{source}_d1_w02", f"{source}_d1_w20",
                   f"{source}_d2_w02", f"{source}_d2_w20"):
        for arm in ARMS:
            for rate in sorted(set([*brackets[arm], brackets["arrival"][0]])):
                tasks.append(dict(sample=sample, arm=arm, rate_qps=rate))
    random.Random(20261021 + kv + sum(map(ord, group))).shuffle(tasks)
    design = OUT / "design" / f"{group}.json"
    if design.exists():
        if json.loads(design.read_text())["tasks"] != tasks:
            raise RuntimeError(f"frozen formal task list changed: {design}")
    else:
        write(design, dict(model=model, source=source, kv_blocks=kv, gpu_pilot=gpu,
                           pilot_sample=pilot_sample, brackets=brackets,
                           tasks=tasks, fixed_parameters={"budget": 1024,
                           "chunk": 256, "concurrency": 32},
                           protocol="research/www_cross_model_20261021/PROTOCOL_CN.md"))
    time.sleep(10)
    run([*base, "--task-file", str(design)], group, "formal")
    write(OUT / "groups" / f"{group}.json",
          dict(state="complete", model=model, source=source, kv_blocks=kv,
               gpu=gpu, formal_runs=len(tasks), completed_at=time.time()))


def main() -> None:
    from research.www_idea1_gpu_pilot_20261016 import run as pilot

    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "launch.json", dict(pid=os.getpid(), started=time.time()))
    pending = queue.Queue()
    for model, kvs in MODELS.items():
        for kv in kvs:
            for source in ("code", "conversation"):
                group = f"{model}_{source}_c{kv}"
                if not (OUT / "groups" / f"{group}.json").exists():
                    pending.put((model, source, kv))
    initial = sum(2 * len(kvs) for kvs in MODELS.values())
    mutex = threading.Lock()
    active: dict[int, str] = {}
    failures: list[dict] = []
    finished = initial - pending.qsize()

    def publish(state: str) -> None:
        write(OUT / "status.json", dict(state=state, updated=time.time(),
              total_groups=initial, completed_groups=finished,
              active=active, pending_groups=pending.qsize(), failures=failures,
              completed_runs=len(list((OUT / "runs").glob("*.json")))))

    def worker(gpu: int) -> None:
        nonlocal finished
        while True:
            with mutex:
                if pending.empty():
                    return
            if not pilot.idle(gpu):
                time.sleep(30)
                continue
            with mutex:
                try:
                    model, source, kv = pending.get_nowait()
                except queue.Empty:
                    return
                group = f"{model}_{source}_c{kv}"
                active[gpu] = group
                publish("running")
            try:
                metadata = MODEL_ROOT / model / "download.json"
                while not metadata.exists():
                    status_path = MODEL_ROOT.parent / "download_status.json"
                    if status_path.exists():
                        status = json.loads(status_path.read_text())
                        if status["state"] == "failed":
                            raise RuntimeError(f"model download failed: {status['detail']}")
                    time.sleep(30)
                execute(model, source, kv, gpu)
                with mutex:
                    finished += 1
            except Exception as exc:
                with mutex:
                    failures.append(dict(group=group, gpu=gpu, error=str(exc)))
            finally:
                with mutex:
                    active.pop(gpu, None)
                    publish("running")
                pending.task_done()

    publish("running")
    with ThreadPoolExecutor(max_workers=len(GPU_CANDIDATES)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in GPU_CANDIDATES]
        for future in futures:
            future.result()
    publish("complete" if not failures else "finished_with_failures")


if __name__ == "__main__":
    main()
