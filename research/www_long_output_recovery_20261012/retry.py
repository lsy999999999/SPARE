"""One user-authorized whole-group retry; never alters the frozen experiment."""
import argparse
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.www_long_output_gpu_20261010.run import core

CELL = "conversation_output1025_1500_20240512_burst_background_c768"
ATTEMPT = int(os.environ.get("WWW_RECOVERY_ATTEMPT", "4"))
if ATTEMPT not in (4, 5):
    raise RuntimeError("only the documented a4 and a5 attempts are supported")
EXTENDED = ATTEMPT == 5
REC = core.OUT / ("supplemental_extended_20261012" if EXTENDED else "recovery_20261012_a4")
REL = f"attempts/{CELL}/a{ATTEMPT}"
FOLDER = core.OUT / REL
WORKER = (ROOT / "research/www_long_output_recovery_20261012/extended_worker.py"
          if EXTENDED else core.HERE / "run.py")


def publish(state, **fields):
    record = dict(state=state, cell=CELL, attempt=ATTEMPT, updated=time.time(), **fields)
    core.write(REC / "status.json", record)
    return record


def check():
    design = core.verify_design()
    manifest = core.read(core.OUT / "manifest.json")
    if core.selected_versions() != manifest["versions"]:
        raise RuntimeError("Python dependency versions differ from original manifest")
    import importlib.metadata
    dist = importlib.metadata.distribution("vllm")
    for name, expected in manifest["native_vllm_hashes"].items():
        if core.digest(dist.locate_file("vllm/" + name)) != expected:
            raise RuntimeError(f"native vLLM source changed: {name}")
    for name, expected in manifest["model_hashes"].items():
        if core.digest(core.MODEL / name) != expected:
            raise RuntimeError(f"model metadata changed: {name}")
    hardware = core.hardware()
    available = []
    for gpu in design["gpu_candidates"]:
        current, original = hardware[gpu], manifest["hardware"][str(gpu)]
        if current["uuid"] != original["uuid"] or current["driver_version"] != original["driver_version"]:
            raise RuntimeError(f"GPU {gpu} identity or driver changed")
        if core.idle(current):
            available.append(gpu)
    if not available:
        raise RuntimeError("no idle original GPU; rerun preflight later")
    return design, manifest, available[-1]


def run():
    with core.acquire(core.OUT / "controller.lock"):
        design, manifest, gpu = check()
        if EXTENDED:
            for _ in range(100):
                if (REC / "launch.json").exists():
                    break
                time.sleep(.1)
        deadline = (core.read(REC / "launch.json")["deadline"] if EXTENDED
                    else core.read(core.OUT / "launch.json")["deadline"])
        if not EXTENDED and time.time() >= deadline:
            raise RuntimeError("original batch deadline expired")
        if FOLDER.exists():
            raise RuntimeError(f"a{ATTEMPT} already exists; refusing overwrite or extra retry")
        original_group = core.read(core.OUT / "groups" / f"{CELL}.json")
        if original_group.get("state") != "failed":
            raise RuntimeError("target is no longer a failed group")
        original_status = core.read(core.OUT / "status.json")
        core.write(REC / "original_group.json", original_group)
        core.write(REC / "original_status.json", original_status)
        cell = next(c for c in design["cells"] if c["key"] == CELL)
        FOLDER.mkdir(parents=True)
        core.write(FOLDER / "task.json", dict(cell=cell, attempt=ATTEMPT))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
            VLLM_USE_V1="1", VLLM_ENABLE_V1_MULTIPROCESSING="0", OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
        command = [sys.executable, "-B", "-u", str(WORKER),
                   "--worker", CELL, "--gpu", str(gpu), "--folder", REL]
        started = time.time()
        error = None
        with (FOLDER / "worker.log").open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            publish("running", gpu=gpu, worker_pid=process.pid, folder=REL, started=started,
                    deadline=deadline, controller_pid=os.getpid())
            try:
                while process.poll() is None:
                    if time.time() >= deadline:
                        error = "original_global_deadline"
                        break
                    if time.time() - started > (48 if EXTENDED else 12) * 3600:
                        error = "group_watchdog_48h" if EXTENDED else "group_watchdog_12h"
                        break
                    external = core.compute_pids(manifest["hardware"][str(gpu)]["uuid"]) - core.descendants(process.pid)
                    if external:
                        error = f"external_gpu_process:{sorted(external)}"
                        core.write(FOLDER / "contamination.json", dict(pids=sorted(external), updated=time.time()))
                        break
                    time.sleep(3)
            except BaseException:
                error = traceback.format_exc()
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
            if error is None and process.returncode:
                error = f"worker_exit_{process.returncode}"
        status = core.read(FOLDER / "status.json") if (FOLDER / "status.json").exists() else {}
        complete = error is None and status.get("state") == "complete"
        record = dict(cell=CELL, gpu=gpu, attempt=ATTEMPT, folder=REL,
            state="complete" if complete else "failed", error=error or status.get("error"),
            started=started, ended=time.time(), recovery_authorization="user request 2026-10-12")
        core.write(FOLDER / "attempt_status.json", record)
        if not complete:
            publish("failed", result=record, detail=status.get("error"), original_group_retained=True)
            return
        if EXTENDED:
            publish("auditing", result=record)
            script = ROOT / "research/www_long_output_recovery_20261012/extended_analyze.py"
            subprocess.run([sys.executable, "-B", str(script)], cwd=ROOT, env=env, check=True)
            publish("complete_supplemental", result=record,
                audit="supplemental_extended_20261012/analysis/summary.json",
                report="research/www_long_output_recovery_20261012/SUPPLEMENTAL_RESULTS_CN.md",
                original_group_retained=True)
            return
        core.write(core.OUT / "groups" / f"{CELL}.json", record)
        updated = dict(original_status)
        updated["finished"] = [record if x["cell"] == CELL else x for x in original_status["finished"]]
        updated.update(state="complete_pending_audit", updated=time.time(), recovery="recovery_20261012_a4/status.json")
        core.write(core.OUT / "status.json", updated)
        publish("auditing", result=record)
        subprocess.run([sys.executable, "-B", str(core.HERE / "analyze.py")], cwd=ROOT, check=True)
        updated.update(state="complete", updated=time.time())
        core.write(core.OUT / "status.json", updated)
        publish("complete", result=record, audit="analysis/summary.json", report=str(core.HERE / "RESULTS_CN.md"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    REC.mkdir(parents=True, exist_ok=True)
    if not args.launch:
        try:
            run()
        except BaseException:
            publish("blocked_or_error", error=traceback.format_exc())
            raise
        return
    with core.acquire(REC / "launch.lock"):
        if (REC / "launch.json").exists() or FOLDER.exists():
            raise RuntimeError("recovery launch already exists")
        check()
        with (REC / "controller.log").open("x") as log:
            process = subprocess.Popen([sys.executable, "-B", "-u", str(Path(__file__).resolve())],
                cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        core.write(REC / "launch.json", dict(pid=process.pid, started=time.time(),
            script_sha256=core.digest(__file__), detached=True, folder=REL,
            deadline=time.time() + 48 * 3600 if EXTENDED else core.read(core.OUT / "launch.json")["deadline"],
            episode_timeout_s=21600 if EXTENDED else core.read(core.OUT / "design.json")["episode_timeout_s"],
            ld_library_path=os.environ.get("LD_LIBRARY_PATH")))
        print(core.read(REC / "launch.json"))


if __name__ == "__main__":
    main()
