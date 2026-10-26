"""Resume the frozen Fixed512 matrix after its original controller exits."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

script_path = Path(__file__).resolve()
project_root = next(parent for parent in script_path.parents
                    if (parent / "research/www_fixed512_capacity_20261014/common.py").exists())
sys.path.insert(0, str(project_root))
from research.www_fixed512_capacity_20261014 import run as core


OUT = core.OUT
HERE = core.HERE
ROOT = core.ROOT


def original_controller_alive() -> bool:
    launch = core.read(OUT / "launch.json")
    pid = launch["pid"]
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return False
    return b"www_fixed512_capacity_20261014/run.py" in command


def accepted_or_pending(design: dict) -> tuple[list[dict], list[dict]]:
    accepted = []
    pending = []
    for cell in design["cells"]:
        key = cell["key"]
        group_file = OUT / "groups" / f"{key}.json"
        group = core.read(group_file) if group_file.exists() else None
        if group and group["state"] == "complete":
            folder = OUT / group["folder"]
            status = core.read(folder / "status.json")
            if status.get("state") != "complete" or status.get("cell") != key:
                raise RuntimeError(f"Accepted group has invalid status: {key}")
            accepted.append(group)
            continue
        # A worker can finish while the original controller is stopped, before
        # it writes the group record. Recover only a fully closed group.
        candidates = sorted((OUT / "attempts" / key).glob("*/status.json"))
        complete = []
        for path in candidates:
            status = core.read(path)
            if status.get("state") == "complete" and status.get("cell") == key:
                folder = path.parent
                aa = core.read(folder / "aa.json")
                if aa.get("passed") and all(
                    (folder / "runs" / f"{arm}_r{repeat}.json").exists()
                    for repeat in range(design["repeats"]) for arm in core.ARMS
                ):
                    complete.append(folder)
        if complete:
            if len(complete) != 1:
                raise RuntimeError(f"Multiple unregistered complete groups: {key}")
            folder = complete[0]
            record = dict(cell=key, folder=str(folder.relative_to(OUT)),
                          state="complete", recovered_group_record=True,
                          ended=time.time())
            core.write(group_file, record)
            accepted.append(record)
            continue
        prior = list((OUT / "attempts" / key).glob("*/attempt_status.json"))
        failed_attempts = [core.read(path)["attempt"] for path in prior
                           if core.read(path).get("state") == "failed"]
        attempt = max(failed_attempts, default=-1) + 1
        if attempt > design["max_group_retries"]:
            raise RuntimeError(f"Retry budget exhausted before resume: {key}")
        pending.append(dict(cell=cell, attempt=attempt))
    return accepted, pending


def controller() -> None:
    lock = core.acquire(OUT / "controller.lock")
    try:
        if original_controller_alive():
            raise RuntimeError("Original controller still exists; resume it or stop it first")
        design = core.verify_design()
        manifest = core.read(OUT / "manifest.json")
        accepted, pending = accepted_or_pending(design)
        deadline = time.time() + design["deadline_hours"] * 3600
        run_id = time.strftime("%Y%m%d_%H%M%S")
        active = {}
        finished = list(accepted)
        mutex = threading.Lock()

        def publish(state="running"):
            core.write(OUT / "resume_status.json", dict(state=state, run_id=run_id,
                updated=time.time(), deadline=deadline, accepted_before_resume=len(accepted),
                finished=finished, active=active,
                pending=[task["cell"]["key"] for task in pending]))

        def consume(gpu):
            while time.time() < deadline:
                with mutex:
                    if not pending:
                        return
                current = core.hardware()[gpu]
                if not core.idle(current):
                    time.sleep(15)
                    continue
                with mutex:
                    if not pending:
                        return
                    task = pending.pop(0)
                    cell, attempt = task["cell"], task["attempt"]
                    folder_rel = f"attempts/{cell['key']}/resume_{run_id}_a{attempt}"
                    active[str(gpu)] = dict(cell=cell["key"], attempt=attempt,
                                             folder=folder_rel)
                    publish()
                folder = OUT / folder_rel
                folder.mkdir(parents=True, exist_ok=False)
                core.write(folder / "task.json", task)
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                    CUDA_DEVICE_ORDER="PCI_BUS_ID", VLLM_USE_V1="1",
                    VLLM_ENABLE_V1_MULTIPROCESSING="0", OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                    TOKENIZERS_PARALLELISM="false")
                command = [sys.executable, "-B", "-u", str(HERE / "run.py"),
                           "--worker", cell["key"], "--gpu", str(gpu),
                           "--folder", folder_rel]
                started = time.time()
                error = None
                with (folder / "worker.log").open("a") as log:
                    process = subprocess.Popen(command, cwd=ROOT, env=environment,
                        stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    while process.poll() is None:
                        if time.time() >= deadline:
                            error = "global_deadline"
                            break
                        external = (core.compute_pids(manifest["hardware"][str(gpu)]["uuid"])
                                    - core.descendants(process.pid))
                        if external:
                            error = f"external_gpu_process:{sorted(external)}"
                            core.write(folder / "contamination.json", dict(
                                pids=sorted(external), updated=time.time()))
                            break
                        if time.time() - started > 12 * 3600:
                            error = "group_watchdog_12h"
                            break
                        time.sleep(3)
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=10)
                    if error is None and process.returncode != 0:
                        error = f"worker_exit_{process.returncode}"
                status_file = folder / "status.json"
                status = core.read(status_file) if status_file.exists() else {}
                complete = error is None and status.get("state") == "complete"
                record = dict(cell=cell["key"], gpu=gpu, attempt=attempt,
                    folder=folder_rel, state="complete" if complete else "failed",
                    error=error or status.get("error"), started=started, ended=time.time())
                core.write(folder / "attempt_status.json", record)
                with mutex:
                    active.pop(str(gpu), None)
                    if complete:
                        finished.append(record)
                        core.write(OUT / "groups" / f"{cell['key']}.json", record)
                    elif attempt < design["max_group_retries"] and time.time() < deadline:
                        pending.append(dict(cell=cell, attempt=attempt + 1))
                    else:
                        finished.append(record)
                        core.write(OUT / "groups" / f"{cell['key']}.json", record)
                    publish()

        publish()
        with ThreadPoolExecutor(max_workers=len(manifest["gpu_candidates"])) as pool:
            futures = [pool.submit(consume, gpu) for gpu in manifest["gpu_candidates"]]
            for future in futures:
                future.result()
        completed = [record for record in finished if record["state"] == "complete"]
        final = ("complete" if len(completed) == len(design["cells"]) else
                 "deadline_reached" if time.time() >= deadline else "finished_with_failures")
        publish(final)
        if final == "complete":
            subprocess.run([sys.executable, "-B", str(HERE / "analyze.py")],
                           cwd=ROOT, check=True)
    finally:
        lock.close()


def launch() -> None:
    if original_controller_alive():
        raise RuntimeError("Original controller still exists")
    if (OUT / "resume_launch.json").exists():
        raise RuntimeError("resume_launch.json exists; inspect the previous recovery first")
    with (OUT / "resume_controller.log").open("a") as log:
        process = subprocess.Popen([sys.executable, "-B", "-u", str(Path(__file__))],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, VLLM_USE_V1="1", VLLM_ENABLE_V1_MULTIPROCESSING="0"))
    core.write(OUT / "resume_launch.json", dict(pid=process.pid, started=time.time(),
               command=str(Path(__file__)), detached=True))
    print(json.dumps(core.read(OUT / "resume_launch.json"), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.launch:
        launch()
    else:
        controller()


if __name__ == "__main__":
    main()
