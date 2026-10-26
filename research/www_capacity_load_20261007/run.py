"""Detached, idle-only RTX 4090 controller for the full WWW baseline matrix."""
from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import importlib.metadata
import json
import math
import os
import platform
import random
import signal
import statistics as st
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.www_capacity_load_20261007.common import *

# Load policy dependencies from the immutable experiment snapshot.  ``common``
# is imported first so ROOT/HERE still refer to the live launcher location.
SNAPSHOT_SOURCE = OUT / "source"
if SNAPSHOT_SOURCE.exists():
    sys.path.insert(0, str(SNAPSHOT_SOURCE))


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def distribution(values, prefix):
    values = list(values)
    return {f"{prefix}_{name}": (function(values) if values else None)
            for name, function in (
                ("mean", st.fmean), ("p50", lambda x: quantile(x, .50)),
                ("p95", lambda x: quantile(x, .95)),
                ("p99", lambda x: quantile(x, .99)), ("max", max))}


def peak_kv_fraction(rounds):
    values = []
    for row in rounds:
        usable = row.get("usable_blocks", row["total_blocks"] - 1)
        if usable <= 0:
            raise ValueError("KV capacity must contain at least one usable block")
        values.append(1 - row["free_after"] / usable)
    return max(values, default=0.)


def hardware():
    fields = ("index,name,uuid,pci.bus_id,memory.total,memory.used,"
              "utilization.gpu,temperature.gpu,power.draw,power.limit,driver_version")
    raw = subprocess.check_output(["nvidia-smi", f"--query-gpu={fields}",
                                   "--format=csv,noheader,nounits"], text=True)
    result = {}
    for line in raw.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        index = int(parts[0])
        result[index] = dict(index=index, name=parts[1], uuid=parts[2],
            pci_bus_id=parts[3], memory_total_mib=int(parts[4]),
            memory_used_mib=int(parts[5]), utilization_gpu_pct=int(parts[6]),
            temperature_c=int(parts[7]), power_draw_w=float(parts[8]),
            power_limit_w=float(parts[9]), driver_version=parts[10])
    return result


def compute_pids(uuid):
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                   "--format=csv,noheader,nounits"], text=True)
    return {int(pid.strip()) for line in raw.splitlines() if "," in line
            for gpu_uuid, pid in [line.split(",", 1)] if gpu_uuid.strip() == uuid}


def _process_parents(proc_root="/proc"):
    """Read a best-effort PID/PPID snapshot despite normal /proc races."""
    result = []
    try:
        entries = os.scandir(proc_root)
        with entries:
            while True:
                try:
                    entry = next(entries)
                except StopIteration:
                    break
                except OSError:
                    break
                if not entry.name.isdigit():
                    continue
                try:
                    raw = Path(entry.path, "stat").read_text()
                    prefix, suffix = raw.rsplit(")", 1)
                    child = int(prefix.split("(", 1)[0].strip())
                    parent = int(suffix.split()[1])
                except (OSError, ValueError, IndexError):
                    continue
                result.append((child, parent))
    except OSError:
        pass
    return result


def descendants(pid, proc_root="/proc"):
    result = {pid}
    changed = True
    while changed:
        changed = False
        for child, parent in _process_parents(proc_root):
            if parent in result and child not in result:
                result.add(child)
                changed = True
    return result


def idle(info):
    return (info["memory_used_mib"] < 256 and info["utilization_gpu_pct"] == 0
            and not compute_pids(info["uuid"]))


def acquire(path):
    handle = open(path, "a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        handle.close()
        raise
    return handle


def cpu_assign(gpus):
    topology = []
    raw = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET"], text=True)
    seen = set()
    for line in raw.splitlines():
        if line.startswith("#"):
            continue
        cpu, core, socket = map(int, line.split(","))
        identity = (core, socket)
        if cpu in os.sched_getaffinity(0) and identity not in seen:
            seen.add(identity)
            topology.append(cpu)
    if len(topology) < 4 * len(gpus):
        raise RuntimeError("insufficient distinct physical CPU cores")
    return {str(gpu): sorted(topology[i * 4:i * 4 + 4])
            for i, gpu in enumerate(gpus)}


def config(arm, cell):
    result = dict(budget=cell["budget"], chunk=cell["chunk"], concurrency=32)
    if arm in OPTIONS:
        from research.v1_broad12h_20260925.policy import formula
        result["threshold"] = formula(1)
        result["evaluation_options"] = OPTIONS[arm]
    return result


def verify_design():
    design = read(OUT / "design.json")
    if design["source_hashes"] != source_hashes():
        raise RuntimeError("source changed after design freeze")
    if tuple(design["methods"]) != ARMS or design["expected_formal_runs"] != len(design["cells"]) * len(ARMS) * design["repeats"]:
        raise RuntimeError("unexpected frozen matrix")
    for cell in design["cells"]:
        if digest(OUT / cell["input"]) != cell["input_sha256"]:
            raise RuntimeError(f"input changed: {cell['key']}")
        history = read(OUT / cell["history_file"])
        if history["source_sha256"] != cell["history_sha256"]:
            raise RuntimeError(f"history changed: {cell['key']}")
    return design


def selected_versions():
    names = ("vllm", "torch", "transformers", "numpy", "safetensors")
    return {name: importlib.metadata.version(name) for name in names}


def create_manifest(design):
    if (OUT / "manifest.json").exists():
        raise RuntimeError("manifest already exists; refusing a second controller")
    validation = read(OUT / "validation.json")
    if not validation.get("passed") or validation.get("source_hashes") != source_hashes():
        raise RuntimeError("frozen validation is missing or stale")
    all_hardware = hardware()
    gpus = [gpu for gpu in design["gpu_candidates"]
            if gpu in all_hardware and "RTX 4090" in all_hardware[gpu]["name"]]
    if not gpus:
        raise RuntimeError("no candidate RTX 4090 found")
    versions = selected_versions()
    if versions["vllm"] != "0.9.2":
        raise RuntimeError(f"expected vLLM 0.9.2, found {versions['vllm']}")
    distribution = importlib.metadata.distribution("vllm")
    native_files = ("v1/core/sched/scheduler.py", "v1/core/block_pool.py",
                    "v1/core/kv_cache_manager.py",
                    "v1/core/single_type_kv_cache_manager.py")
    model_files = [name for name in ("config.json", "generation_config.json",
        "tokenizer_config.json", "model.safetensors.index.json") if (MODEL / name).exists()]
    model_config = read(MODEL / "config.json")
    manifest = dict(
        schema=1, created=time.time(), controller_pid=os.getpid(),
        hostname=platform.node(), platform=platform.platform(),
        python=sys.version, executable=sys.executable, model=str(MODEL),
        model_hashes={name: digest(MODEL / name) for name in model_files},
        model_config={key: model_config.get(key) for key in (
            "model_type", "architectures", "hidden_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "vocab_size",
            "max_position_embeddings", "torch_dtype")},
        versions=versions, torch_cuda=__import__("torch").version.cuda,
        cudnn=__import__("torch").backends.cudnn.version(),
        cpu_lscpu=subprocess.check_output(["lscpu"], text=True),
        memory_meminfo=Path("/proc/meminfo").read_text(),
        gpu_candidates=gpus,
        hardware={str(gpu): all_hardware[gpu] for gpu in gpus},
        affinity=cpu_assign(gpus), design_sha256=digest(OUT / "design.json"),
        source_hashes=design["source_hashes"],
        native_vllm_hashes={name: digest(distribution.locate_file("vllm/" + name))
                            for name in native_files},
        engine=dict(model=str(MODEL), dtype="half", max_model_len=8176,
            max_num_seqs=64, max_num_batched_tokens=2048,
            enable_chunked_prefill=True, enable_prefix_caching=False,
            block_size=16, gpu_memory_utilization=.9, enforce_eager=True,
            disable_log_stats=True, seed=0, tensor_parallel_size=1,
            pipeline_parallel_size=1, VLLM_USE_V1="1",
            VLLM_ENABLE_V1_MULTIPROCESSING="0"),
    )
    write(OUT / "manifest.json", manifest)
    return manifest


def gpu_snapshot(gpu):
    current = hardware()[gpu]
    current["compute_pids"] = sorted(compute_pids(current["uuid"]))
    current["time"] = time.time()
    return current


def run_episode(engine, scheduler, requests, arm, cell, timeout):
    from vllm import SamplingParams
    scheduler.history_context = read(OUT / cell["history_file"])["lengths"]
    scheduler.seed_context = 20260929
    scheduler.oracle_lengths_context = ({row["id"]: row["output"] for row in requests}
                                        if arm in ORACLE_ARMS else {})
    scheduler.configure(METHOD[arm], **config(arm, cell))
    clocks = {row["id"]: dict(arrival_ms=row["arrival_ms"], submitted_ms=None,
        first_ms=None, finish_ms=None, tokens=0, token_times_ms=[])
        for row in requests}
    cursor = 0
    error = None
    start = time.perf_counter()
    try:
        while cursor < len(requests) or engine.has_unfinished_requests():
            now = (time.perf_counter() - start) * 1000
            if now > timeout * 1000:
                raise TimeoutError("predeclared episode deadline exceeded")
            while cursor < len(requests) and requests[cursor]["arrival_ms"] <= now:
                row = requests[cursor]
                clocks[row["id"]]["submitted_ms"] = now
                engine.add_request(row["id"], {"prompt_token_ids": [100] * row["prompt"]},
                    SamplingParams(temperature=0., max_tokens=row["output"],
                                   ignore_eos=True, detokenize=False))
                cursor += 1
            if engine.has_unfinished_requests():
                step_start = time.perf_counter()
                outputs = engine.step()
                now = (time.perf_counter() - start) * 1000
                scheduler.round_rows[-1]["step_ms"] = (time.perf_counter() - step_start) * 1000
                scheduler.round_rows[-1]["step_end_ms"] = now
                for output in outputs:
                    clock = clocks[output.request_id]
                    count = len(output.outputs[0].token_ids) if output.outputs else 0
                    if count > clock["tokens"]:
                        if count != clock["tokens"] + 1:
                            raise RuntimeError("expected synchronous one-token decode")
                        clock["tokens"] = count
                        clock["token_times_ms"].append(now)
                        if clock["first_ms"] is None:
                            clock["first_ms"] = now
                    if output.finished:
                        clock["finish_ms"] = now
            else:
                delay = max(0., requests[cursor]["arrival_ms"] - now) / 1000
                time.sleep(min(.005, delay))
        for row in requests:
            clock = clocks[row["id"]]
            if clock["tokens"] != row["output"] or clock["finish_ms"] is None:
                raise RuntimeError(f"incomplete request {row['id']}")
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    elapsed = (time.perf_counter() - start) * 1000
    rounds = list(scheduler.round_rows)
    request_rows = []
    all_itl = []
    for request in requests:
        clock = clocks[request["id"]]
        token_times = clock["token_times_ms"]
        itl = [right - left for left, right in zip(token_times, token_times[1:])]
        all_itl.extend(itl)
        request_rows.append(dict(request_id=request["id"],
            prompt_tokens=request["prompt"], output_tokens=request["output"],
            arrival_ms=request["arrival_ms"], submitted_ms=clock["submitted_ms"],
            first_ms=clock["first_ms"], finish_ms=clock["finish_ms"],
            ttft_ms=(clock["first_ms"] - request["arrival_ms"]
                     if clock["first_ms"] is not None else None),
            e2e_ms=(clock["finish_ms"] - request["arrival_ms"]
                    if clock["finish_ms"] is not None else None),
            tpot_ms=((token_times[-1] - token_times[0]) / (len(token_times) - 1)
                     if len(token_times) > 1 else None),
            max_itl_ms=max(itl) if itl else None))
    complete = error is None
    metrics = dict(requests=len(requests), completed=sum(row["finish_ms"] is not None
        for row in request_rows), elapsed_ms=elapsed, rounds=len(rounds),
        request_throughput_rps=(len(requests) * 1000 / elapsed if complete else None),
        output_throughput_tps=(sum(row["output"] for row in requests) * 1000 / elapsed
                               if complete else None),
        preemptions=sum(row.get("preemptions", 0) for row in rounds),
        recomputed_tokens=sum(row.get("recomputed_tokens", 0) for row in rounds),
        interventions=sum(row.get("intervened", 0) for row in rounds),
        peak_kv_util=peak_kv_fraction(rounds))
    if complete:
        for field in ("ttft_ms", "e2e_ms", "tpot_ms", "max_itl_ms"):
            metrics.update(distribution((row[field] for row in request_rows
                if row[field] is not None), field))
        metrics.update(distribution(all_itl, "token_itl_ms"))
        metrics.update(distribution((row["schedule_ms"] for row in rounds
            if "schedule_ms" in row), "schedule_ms"))
        good = sum(row["ttft_ms"] <= 5000 and (row["max_itl_ms"] or 0) <= 200
                   for row in request_rows)
        metrics["slo_ttft5000_maxitl200_attainment"] = good / len(requests)
        metrics["slo_ttft5000_maxitl200_goodput_rps"] = good * 1000 / elapsed
    return dict(status="complete" if complete else "failed", error=error,
        arm=arm, method=METHOD[arm], config=config(arm, cell),
        uses_realized_output_lengths=arm in ORACLE_ARMS,
        deployable=arm not in ORACLE_ARMS, elapsed_ms=elapsed,
        metrics=metrics, requests=request_rows, clocks=clocks, rounds=rounds)


def save_run(engine, scheduler, requests, arm, cell, folder, name, metadata, timeout):
    target = folder / "runs" / f"{name}.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite {target}")
    write(folder / "current.json", dict(state="running", name=name,
                                          updated=time.time()))
    result = run_episode(engine, scheduler, requests, arm, cell, timeout)
    rounds = result.pop("rounds")
    round_path = folder / "rounds" / f"{name}.json.gz"
    round_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = round_path.with_name(round_path.name + ".tmp")
    with gzip.open(temporary, "wt", compresslevel=1) as handle:
        json.dump(rounds, handle, separators=(",", ":"), allow_nan=False)
    temporary.replace(round_path)
    result.update(metadata, name=name,
        rounds_path=str(round_path.relative_to(folder)),
        rounds_sha256=digest(round_path), recorded_at=time.time())
    write(target, result)
    write(folder / "current.json", dict(state="saved", name=name,
                                          status=result["status"], updated=time.time()))
    print(name, result["status"], result["metrics"].get("ttft_ms_mean"), flush=True)
    if result["status"] != "complete":
        raise RuntimeError(result["error"])
    return result


def order_for(cell_key, repeat):
    pair = repeat // 2
    order = list(ARMS)
    seed = int(object_digest([cell_key, pair])[:16], 16)
    random.Random(seed).shuffle(order)
    return order if repeat % 2 == 0 else list(reversed(order))


def worker(cell_key, gpu, folder_relative):
    lock = acquire(f"/tmp/query-capacity-gpu{gpu}.lock")
    design = verify_design()
    manifest = read(OUT / "manifest.json")
    cell = next(cell for cell in design["cells"] if cell["key"] == cell_key)
    folder = OUT / folder_relative
    folder.mkdir(parents=True, exist_ok=True)
    if not idle(hardware()[gpu]):
        raise RuntimeError(f"GPU {gpu} is no longer idle")
    os.sched_setaffinity(0, manifest["affinity"][str(gpu)])
    write(folder / "gpu_start.json", gpu_snapshot(gpu))
    try:
        import torch
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        expected_uuid = manifest["hardware"][str(gpu)]["uuid"].removeprefix("GPU-").lower()
        if str(torch.cuda.get_device_properties(0).uuid).lower() != expected_uuid:
            raise RuntimeError("CUDA/NVML GPU identity mismatch")
        engine = LLMEngine.from_engine_args(EngineArgs(
            model=str(MODEL), dtype="half", max_model_len=design["max_model_len"],
            max_num_seqs=design["max_num_seqs"],
            max_num_batched_tokens=design["max_num_batched_tokens"],
            enable_chunked_prefill=True, enable_prefix_caching=False,
            block_size=design["block_size"], gpu_memory_utilization=.9,
            enforce_eager=True, disable_log_stats=True, seed=0,
            num_gpu_blocks_override=cell["capacity"],
            scheduler_cls="research.www_capacity_load_20261007.scheduler.GPUBaselineScheduler"))
        scheduler = engine.engine_core.engine_core.scheduler
        if scheduler.kv_cache_config.num_blocks != cell["capacity"]:
            raise RuntimeError("KV block override was not honored")
        requests = read(OUT / cell["input"])
        warm = [dict(id=f"warm{i}", prompt=256, output=16, arrival_ms=0.)
                for i in range(4)]
        run_episode(engine, scheduler, warm, "native", cell, 120)
        aa = [save_run(engine, scheduler, requests, "arrival", cell, folder,
                       f"aa_arrival_r{repeat}", dict(phase="aa", repeat=repeat,
                       cell=cell_key, gpu=gpu), design["episode_timeout_s"])
              for repeat in range(design["aa_repeats"])]
        stability = {}
        for metric in ("ttft_ms_mean", "elapsed_ms"):
            values = [row["metrics"][metric] for row in aa]
            stability[metric] = dict(values=values,
                range_over_mean=(max(values) - min(values)) / st.fmean(values))
        passed = max(value["range_over_mean"] for value in stability.values()) <= .05
        write(folder / "aa.json", dict(passed=passed, threshold=.05,
                                         metrics=stability))
        if not passed:
            raise RuntimeError("A/A stability gate exceeded 5%")
        for repeat in range(design["repeats"]):
            for position, arm in enumerate(order_for(cell_key, repeat)):
                save_run(engine, scheduler, requests, arm, cell, folder,
                    f"{arm}_r{repeat}", dict(phase="formal", repeat=repeat,
                    order_position=position, cell=cell_key, gpu=gpu),
                    design["episode_timeout_s"])
        write(folder / "gpu_end.json", gpu_snapshot(gpu))
        write(folder / "status.json", dict(state="complete", cell=cell_key,
            gpu=gpu, formal_runs=len(ARMS) * design["repeats"], updated=time.time()))
    except BaseException:
        write(folder / "status.json", dict(state="failed", cell=cell_key,
            gpu=gpu, error=traceback.format_exc(), updated=time.time()))
        raise
    finally:
        lock.close()


def controller():
    controller_lock = acquire(OUT / "controller.lock")
    design = verify_design()
    manifest = create_manifest(design)
    for _ in range(100):
        if (OUT / "launch.json").exists():
            break
        time.sleep(.1)
    launch = read(OUT / "launch.json")
    deadline = launch["deadline"]
    pending = [dict(cell=cell, attempt=0) for cell in design["cells"]]
    random.Random(20260929).shuffle(pending)
    active = {}
    finished = []
    mutex = threading.Lock()

    def publish(state="running"):
        write(OUT / "status.json", dict(state=state, updated=time.time(),
            deadline=deadline, planned_groups=len(design["cells"]),
            planned_formal_runs=design["expected_formal_runs"],
            finished=finished, active=active,
            pending=[task["cell"]["key"] for task in pending]))

    def consume(gpu):
        while time.time() < deadline:
            with mutex:
                if not pending:
                    return
            current = hardware()[gpu]
            if not idle(current):
                write(OUT / f"worker_g{gpu}.json", dict(state="waiting_for_idle_gpu",
                    memory_used_mib=current["memory_used_mib"],
                    compute_pids=sorted(compute_pids(current["uuid"])),
                    updated=time.time()))
                time.sleep(15)
                continue
            with mutex:
                if not pending:
                    return
                task = pending.pop(0)
                cell = task["cell"]
                attempt = task["attempt"]
                relative = f"attempts/{cell['key']}/a{attempt}"
                active[str(gpu)] = dict(cell=cell["key"], attempt=attempt,
                                        folder=relative)
                publish()
            folder = OUT / relative
            folder.mkdir(parents=True, exist_ok=True)
            write(folder / "task.json", task)
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                CUDA_DEVICE_ORDER="PCI_BUS_ID", VLLM_USE_V1="1",
                VLLM_ENABLE_V1_MULTIPROCESSING="0", OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                TOKENIZERS_PARALLELISM="false")
            command = [sys.executable, "-B", "-u", str(HERE / "run.py"),
                       "--worker", cell["key"], "--gpu", str(gpu),
                       "--folder", relative]
            started = time.time()
            error = None
            with (folder / "worker.log").open("a") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=environment,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                while process.poll() is None:
                    if time.time() >= deadline:
                        error = "global_deadline"
                        break
                    external = compute_pids(manifest["hardware"][str(gpu)]["uuid"]) - descendants(process.pid)
                    if external:
                        error = f"external_gpu_process:{sorted(external)}"
                        write(folder / "contamination.json", dict(pids=sorted(external),
                                                                   updated=time.time()))
                        break
                    if time.time() - started > 60 * 60 * 12:
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
            status = read(folder / "status.json") if (folder / "status.json").exists() else {}
            complete = error is None and status.get("state") == "complete"
            record = dict(cell=cell["key"], gpu=gpu, attempt=attempt,
                folder=relative, state="complete" if complete else "failed",
                error=error or status.get("error"), started=started, ended=time.time())
            write(folder / "attempt_status.json", record)
            with mutex:
                active.pop(str(gpu), None)
                if complete:
                    finished.append(record)
                    write(OUT / "groups" / f"{cell['key']}.json", record)
                elif attempt < design["max_group_retries"] and time.time() < deadline:
                    pending.append(dict(cell=cell, attempt=attempt + 1))
                else:
                    finished.append(record)
                    write(OUT / "groups" / f"{cell['key']}.json", record)
                publish()
            write(OUT / f"worker_g{gpu}.json", dict(state="idle", updated=time.time()))

    publish()
    with ThreadPoolExecutor(max_workers=len(manifest["gpu_candidates"])) as pool:
        futures = [pool.submit(consume, gpu) for gpu in manifest["gpu_candidates"]]
        for future in futures:
            future.result()
    completed = [record for record in finished if record["state"] == "complete"]
    final = ("complete" if len(completed) == len(design["cells"])
             else "deadline_reached" if time.time() >= deadline
             else "finished_with_failures")
    publish(final)
    if final == "complete":
        subprocess.run([sys.executable, "-B", str(HERE / "analyze.py")],
                       cwd=ROOT, check=True)
    controller_lock.close()


def launch():
    verify_design()
    if (OUT / "launch.json").exists():
        raise RuntimeError("launch.json already exists; refusing duplicate launch")
    log_path = OUT / "controller.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(PYTHON), "-B", "-u", str(HERE / "run.py")]
    with log_path.open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, VLLM_USE_V1="1",
                     VLLM_ENABLE_V1_MULTIPROCESSING="0"))
    now = time.time()
    design = read(OUT / "design.json")
    write(OUT / "launch.json", dict(pid=process.pid, started=now,
        deadline=now + design["deadline_hours"] * 3600,
        command=command, cwd=str(ROOT), detached=True,
        stdin="DEVNULL", stdout=str(log_path), session_leader=True))
    print(json.dumps(read(OUT / "launch.json"), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--folder")
    arguments = parser.parse_args()
    if arguments.launch:
        launch()
    elif arguments.worker:
        worker(arguments.worker, arguments.gpu, arguments.folder)
    else:
        try:
            controller()
        except BaseException:
            write(OUT / "status.json", dict(state="failed",
                error=traceback.format_exc(), updated=time.time()))
            raise
