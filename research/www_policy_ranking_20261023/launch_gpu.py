"""Distribute frozen ranking tasks over idle remote RTX 4090 cards."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time


ROOT = Path(os.environ.get("WWW_CROSS_MODEL_ROOT", Path.cwd())).resolve()
OUT = ROOT / "artifacts/www_policy_ranking_fidelity_20261023_r1"
PYTHON = ".venv/bin/python"
RUNNER = ROOT / "research/www_policy_ranking_20261023/run_gpu.py"
GPUS = (0, 1, 2, 3, 6, 7)


def write_status(value: dict, name: str) -> None:
    path = OUT / name
    temp = path.with_name(f"status.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("code", "conversation"))
    parser.add_argument("--gpus", default=",".join(map(str, GPUS)))
    parser.add_argument("--task-dir", default="tasks")
    args = parser.parse_args()
    gpu_pool = tuple(int(value) for value in args.gpus.split(","))
    design = json.loads((OUT / "design.json").read_text())
    assert design["model"] == "qwen3_8b" and design["kv_blocks"] == 2048
    tasks = sorted((OUT / args.task_dir).glob("*.json"))
    if args.source:
        tasks = [task for task in tasks if task.stem.startswith(args.source + "_")]
    assert len(tasks) == (4 if args.source else 8)
    status_name = (f"status_{args.task_dir}.json" if args.task_dir != "tasks"
                   else f"status_{args.source}.json" if args.source else "status.json")
    free = list(gpu_pool)
    lock = threading.Lock()
    active: dict[int, str] = {}
    complete: list[str] = []
    failures: list[dict] = []
    started = time.time()

    def publish(state: str) -> None:
        write_status(dict(state=state, started=started, updated=time.time(),
                          active=active, completed=complete, failures=failures,
                          total_groups=len(tasks)), status_name)

    def run(task_file: Path) -> tuple[str, int]:
        while True:
            with lock:
                if free:
                    gpu = free.pop(0)
                    active[gpu] = task_file.stem
                    publish("running")
                    break
            time.sleep(2)
        try:
            log = OUT / "logs" / f"{task_file.stem}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, PYTHONPATH=str(ROOT),
                       WWW_CROSS_MODEL_ROOT=str(ROOT), CUDA_DEVICE_ORDER="PCI_BUS_ID")
            with log.open("a") as stream:
                result = subprocess.run([
                    PYTHON, "-B", "-u", str(RUNNER), "--model", "qwen3_8b",
                    "--kv", "2048", "--gpu", str(gpu), "--task-file", str(task_file),
                ], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            return task_file.stem, result.returncode
        finally:
            with lock:
                active.pop(gpu)
                free.append(gpu)
                publish("running")

    publish("running")
    with ThreadPoolExecutor(max_workers=len(gpu_pool)) as pool:
        futures = [pool.submit(run, task) for task in tasks]
        for future in as_completed(futures):
            name, code = future.result()
            with lock:
                if code:
                    failures.append(dict(group=name, exit_code=code))
                else:
                    complete.append(name)
                publish("running")
    publish("complete" if not failures else "finished_with_failures")
    if failures:
        raise SystemExit(failures)


if __name__ == "__main__":
    main()
