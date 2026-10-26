"""Resume four-window GPU or simulator capacity searches for policy ranking."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time


ROOT = Path(os.environ.get("WWW_CROSS_MODEL_ROOT", Path(__file__).resolve().parents[2]))
OUT = ROOT / "artifacts/www_policy_ranking_bisect_20261024_r1"
SOURCE = ROOT / "artifacts/www_simulator_gpu_validation_20261019_r1"
PROFILE_ROOT = ROOT / "artifacts/www_cross_model_timing_20261022_r1/profiles"
REMOTE_PYTHON = ".venv/bin/python"
WINDOWS = ("d1_w02", "d1_w20", "d2_w02", "d2_w20")
ARMS = ("arrival", "srp", "sarathi")
KVS = {
    "qwen25_7b": (2048, 4096, 6144, 7424),
    "qwen25_3b": (2048, 4096, 6144),
    "qwen3_8b": (1536, 1792, 2048),
    "mistral_7b": (1536, 2048, 2560),
}
MODELS = tuple(KVS)
GPUS = (0, 1, 2, 3, 6, 7)
QPS_STEP = .01
MAX_QPS = 16.
QUICK_GROUPS = (
    ("qwen25_7b", "code", 2048),
    ("qwen25_3b", "code", 2048),
    ("qwen3_8b", "conversation", 2048),
    ("mistral_7b", "code", 2048),
)


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def groups() -> list[tuple[str, str, int]]:
    # Start with cross-model coverage, then fill out each model's KV range.
    result = [(m, s, kvs[0]) for m, kvs in KVS.items()
              for s in ("code", "conversation")]
    result += [(m, s, kv) for m, kvs in KVS.items() for kv in kvs[1:]
               for s in ("code", "conversation")]
    return result


def name(model: str, source: str, kv: int) -> str:
    return f"{model}_{source}_c{kv}"


def sample_for(source: str, window: str) -> dict:
    spec = json.loads((SOURCE / "design.json").read_text())
    return next(s for s in spec["samples"] if s["name"] == f"{source}_{window}")


def rate_record(model: str, source: str, kv: int, arm: str, rate: float,
                side: str) -> Path:
    return OUT / side / "runs" / f"{model}_{source}_c{kv}_{arm}_r{rate:g}.json"


def simulator_point(model: str, source: str, kv: int, arm: str,
                    rate: float) -> dict:
    from research.www_cross_model_20261021 import run_gpu as base
    from research.www_cross_model_20261021.validate_simulator import metric
    from research.www_evaluation_completion_20261003 import simulate as sim
    from research.www_evaluation_completion_20261003.timing import FittedTiming

    profile_path = (ROOT / "artifacts/www_pareto_20261014_r1/timing_profile.json"
                    if model == "qwen25_7b" else PROFILE_ROOT / f"{model}.json")
    profile = FittedTiming(json.loads(profile_path.read_text()))
    history = json.loads((SOURCE / "history" / f"{source}.json").read_text())
    metrics = []
    for window in WINDOWS:
        path = rate_record(model, f"{source}_{window}", kv, arm, rate, "sim")
        sample = sample_for(source, window)
        requests = base.generate(sample, rate)
        input_sha = hashlib.sha256((json.dumps(requests, indent=2, allow_nan=False)
                                    + "\n").encode()).hexdigest()
        if path.exists():
            row = json.loads(path.read_text())
            if row["input_sha256"] != input_sha or row["profile_sha256"] != hashlib.sha256(profile_path.read_bytes()).hexdigest():
                raise RuntimeError(f"stale simulator point: {path}")
        else:
            result = sim.simulate(requests, arm=arm, capacity=kv, budget=1024,
                                  chunk=256, history=history["lengths"], seed=0,
                                  fitted_timing=profile, capture_probe=True)
            row = dict(model=model, sample=sample["name"], source=source,
                       kv_blocks=kv, arm=arm, rate_qps=rate,
                       input_sha256=input_sha,
                       profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                       metrics=metric(result, rate))
            write(path, row)
        metrics.append(row["metrics"])
    return pooled(metrics, rate)


def gpu_point(model: str, source: str, kv: int, arm: str,
              rate: float, engine, scheduler, gpu: int) -> dict:
    from research.www_cross_model_20261021 import run_gpu as base
    spec = json.loads((SOURCE / "design.json").read_text())
    metrics = []
    for window in WINDOWS:
        sample = sample_for(source, window)
        path = rate_record(model, sample["name"], kv, arm, rate, "gpu")
        if path.exists():
            row = json.loads(path.read_text())
            if row["status"] != "complete" or not row["valid"]:
                raise RuntimeError(f"invalid GPU point: {path}")
        else:
            row = base.one_run(engine, scheduler, sample, model, kv,
                               arm, rate, gpu, spec)
        metrics.append(row)
    return pooled(metrics, rate)


def pooled(rows: list[dict], rate: float) -> dict:
    ttft = statistics.fmean(r["mean_ttft_ms"] for r in rows)
    tbt = statistics.fmean(r["p99_tbt_ms"] for r in rows)
    drift = statistics.fmean(r["backlog_drift"] for r in rows)
    return dict(rate_qps=rate, passed=ttft <= 2000 and tbt <= 200
                and drift <= max(2, rate), mean_ttft_ms=ttft,
                p99_tbt_ms=tbt, backlog_drift=drift)


def search(model: str, source: str, kv: int, arm: str, side: str,
           engine=None, scheduler=None, gpu: int | None = None) -> dict:
    group = name(model, source, kv)
    path = OUT / side / "search" / f"{group}_{arm}.json"
    points = json.loads(path.read_text())["points"] if path.exists() else {}

    def check(rate: float) -> bool:
        key = f"{rate:.2f}"
        if key not in points:
            row = (simulator_point(model, source, kv, arm, rate) if side == "sim"
                   else gpu_point(model, source, kv, arm, rate, engine, scheduler, gpu))
            points[key] = row
            publish("running")
        return points[key]["passed"]

    def publish(state: str, **extra) -> None:
        ordered = sorted(points.values(), key=lambda p: p["rate_qps"])
        labels = [r["passed"] for r in ordered]
        write(path, dict(protocol_version=2, state=state, model=model, source=source, kv_blocks=kv,
                         arm=arm, side=side, updated=time.time(),
                         nonmonotonic=labels != sorted(labels, reverse=True),
                         points=points, **extra))

    low = .50 if source == "code" else 1.
    step = .25 if source == "code" else .50
    if not check(low):
        publish("unresolved", reason="starting_rate_fails")
        return json.loads(path.read_text())
    high = None
    rate = round(low + step, 2)
    while rate <= MAX_QPS:
        if not check(rate):
            high = rate
            break
        low = rate
        rate = round(rate + step, 2)
    if high is None:
        publish("unresolved", reason="max_rate_passes")
        return json.loads(path.read_text())
    # Keep a short post-boundary audit so a later pass is visible in the record.
    for offset in (1, 2, 3, 4):
        followup = round(high + offset * step, 2)
        if followup <= MAX_QPS:
            check(followup)
    while high - low > QPS_STEP + 1e-8:
        middle = round((low + high) / 2, 2)
        if middle <= low or middle >= high:
            break
        if check(middle):
            low = middle
        else:
            high = middle
    publish("complete", lower_qps=low, upper_qps=high,
            resolution_qps=round(high-low, 2))
    return json.loads(path.read_text())


def run_group(model: str, source: str, kv: int, side: str,
              gpu: int | None = None) -> None:
    from research.www_cross_model_20261021 import run_gpu as base
    from research.www_idea1_gpu_pilot_20261016 import run as pilot
    engine = scheduler = None
    if side == "gpu":
        from research.www_evaluation_gpu_completion_20261003 import run as gpu_run
        base.OUT = OUT / "gpu"
        gpu_run.OUT = base.OUT
        for history_source in ("code", "conversation"):
            history_path = base.OUT / "history" / f"{history_source}.json"
            source_path = SOURCE / "history" / f"{history_source}.json"
            if history_path.exists():
                if json.loads(history_path.read_text()) != json.loads(source_path.read_text()):
                    raise RuntimeError(f"history changed: {history_path}")
            else:
                write(history_path, json.loads(source_path.read_text()))
        base.MODELS["qwen25_7b"] = Path(
            "models/Qwen2.5-7B-Instruct")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if not pilot.idle(gpu):
            raise RuntimeError(f"GPU {gpu} became busy")
        lock = pilot.acquire_gpu(gpu)
        try:
            engine, scheduler = base.engine_for(base.MODELS[model], kv)
            warm = [dict(id=f"warm{i}", prompt=256, output=16, arrival_ms=0.)
                    for i in range(4)]
            result = gpu_run.run_episode(engine, scheduler, warm, "native",
                                         {"budget": 1024, "chunk": 256,
                                          "history_file": "history/code.json"}, 120)
            if result["status"] != "complete":
                raise RuntimeError(f"warmup failed: {result['error']}")
            for arm in ARMS:
                search(model, source, kv, arm, side, engine, scheduler, gpu)
        finally:
            lock.close()
    else:
        for arm in ARMS:
            search(model, source, kv, arm, side)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("gpu", "sim"), required=True)
    parser.add_argument("--group", nargs=3, metavar=("MODEL", "SOURCE", "KV"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpus", default=",".join(map(str, GPUS)))
    parser.add_argument("--quick", action="store_true",
                        help="Prioritize one discriminative KV/workload condition per model")
    parser.add_argument("--exclude-group", action="append", default=[],
                        help="Do not queue a group already running in another process")
    args = parser.parse_args()
    if args.group:
        model, source, kv = args.group
        run_group(model, source, int(kv), args.side,
                  int(os.environ["WWW_GPU"]) if args.side == "gpu" else None)
        return
    queue = [group for group in (QUICK_GROUPS if args.quick else groups())
             if name(*group) not in args.exclude_group]
    lock = threading.Lock()
    completed = set()
    failures = []
    active = {}
    candidates = tuple(int(x) for x in args.gpus.split(",")) if args.side == "gpu" else range(args.workers)
    total = len(queue)

    def publish() -> None:
        write(OUT / args.side / "status.json",
              dict(state="complete" if len(completed) == total else "running",
                   updated=time.time(), total_groups=total,
                   completed_groups=len(completed), active=active,
                   pending_groups=len(queue),
                   failures=failures))

    def worker(device: int) -> None:
        from research.www_idea1_gpu_pilot_20261016 import run as pilot
        while True:
            with lock:
                if not queue:
                    return
            if args.side == "gpu" and not pilot.idle(device):
                time.sleep(30)
                continue
            with lock:
                if not queue:
                    return
                model, source, kv = queue.pop(0)
                label = name(model, source, kv)
                active[device] = label
                publish()
            try:
                env = dict(os.environ, PYTHONPATH=str(ROOT), WWW_CROSS_MODEL_ROOT=str(ROOT),
                           WWW_GPU=str(device), CUDA_DEVICE_ORDER="PCI_BUS_ID")
                log = OUT / args.side / "logs" / f"{label}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                command = ([REMOTE_PYTHON] if args.side == "gpu" else [os.sys.executable])
                command += ["-B", "-u", str(Path(__file__).resolve()),
                            "--side", args.side, "--group", model, source, str(kv)]
                with log.open("a") as stream:
                    code = subprocess.run(command, cwd=ROOT, env=env,
                                          stdout=stream, stderr=subprocess.STDOUT).returncode
                with lock:
                    if code:
                        failures.append(dict(group=label, gpu=device, exit_code=code))
                    else:
                        completed.add(label)
            finally:
                with lock:
                    active.pop(device, None)
                    publish()

    # GPU workers wait for available cards. The queue remains resumable across launches.
    if args.side == "gpu":
        queue = [g for g in queue if not all(
            (OUT / "gpu" / "search" / f"{name(*g)}_{a}.json").exists()
            and (saved := json.loads((OUT / "gpu" / "search" / f"{name(*g)}_{a}.json").read_text()))["state"] == "complete"
            and saved.get("protocol_version") == 2
            for a in ARMS)]
    else:
        queue = [g for g in queue if not all(
            (OUT / "sim" / "search" / f"{name(*g)}_{a}.json").exists()
            and (saved := json.loads((OUT / "sim" / "search" / f"{name(*g)}_{a}.json").read_text()))["state"] == "complete"
            and saved.get("protocol_version") == 2
            for a in ARMS)]
    total = len(queue)
    def publish_final() -> None:
        write(OUT / args.side / "status.json",
              dict(state="complete" if not failures else "finished_with_failures",
                   updated=time.time(), total_groups=total,
                   completed_groups=len(completed), active=active,
                   pending_groups=0, failures=failures))
    publish()
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        futures = [pool.submit(worker, device) for device in candidates]
        for future in futures:
            future.result()
    publish_final()


if __name__ == "__main__":
    main()
