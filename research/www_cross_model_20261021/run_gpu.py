"""Resumable cross-model SPARE GPU replay using the frozen V1 scheduler."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(os.environ.get("WWW_CROSS_MODEL_ROOT", Path.cwd())).resolve()
OUT = ROOT / "artifacts/www_cross_model_20261021_r1"
SOURCE = ROOT / "artifacts/www_simulator_gpu_validation_20261019_r1"
MODEL_ROOT = Path("models")
MODELS = {"qwen25_3b": MODEL_ROOT / "qwen25_3b",
          "qwen3_8b": MODEL_ROOT / "qwen3_8b",
          "mistral_7b": MODEL_ROOT / "mistral_7b",
          "qwen3_4b_preflight": Path(
              "/mnt/disk1/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/"
              "1cfa9a7208912126459214e8b04321603b3df60c")}
ARMS = ("arrival", "srp", "fixed256", "sarathi")
WARM_MS, HORIZON_MS = 20000, 60000


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def engine_for(model: Path, kv: int):
    import torch
    import vllm.platforms as platforms
    from vllm.platforms.cuda import CudaPlatform

    if platforms.current_platform.device_type != "cuda":
        if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
            raise RuntimeError("RTX 4090 CUDA platform unavailable")
        platforms._current_platform = CudaPlatform()
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    engine = LLMEngine.from_engine_args(EngineArgs(
        model=str(model), dtype="half", max_model_len=8176,
        max_num_seqs=64, max_num_batched_tokens=2048,
        enable_chunked_prefill=True, enable_prefix_caching=False,
        block_size=16, gpu_memory_utilization=.9, enforce_eager=True,
        disable_log_stats=True, seed=0, num_gpu_blocks_override=kv,
        scheduler_cls="research.www_idea1_gpu_pilot_20261016.scheduler.IdeaScheduler"))
    scheduler = engine.engine_core.engine_core.scheduler
    if scheduler.kv_cache_config.num_blocks != kv:
        raise RuntimeError("KV override was not honored")
    return engine, scheduler


def generate(sample: dict, rate: float) -> list[dict]:
    pool = json.loads((ROOT / sample["input"]).read_text())
    seed = int.from_bytes(hashlib.sha256(sample["name"].encode()).digest()[:8],
                          "big")
    arrival_rng = np.random.default_rng(seed)
    length_rng = np.random.default_rng(seed ^ 0x9E3779B97F4A7C15)
    times = np.cumsum(arrival_rng.exponential(1000., 4096)) / rate
    if times[-1] < HORIZON_MS:
        raise RuntimeError("arrival stream too short")
    picks = length_rng.integers(0, len(pool), size=int(np.searchsorted(times, HORIZON_MS)))
    return [dict(id=f"{sample['name']}:{i}", prompt=pool[int(j)]["prompt"],
                 output=pool[int(j)]["output"], arrival_ms=float(times[i]))
            for i, j in enumerate(picks)]


def one_run(engine, scheduler, sample: dict, model: str, kv: int,
            arm: str, rate: float, gpu: int, spec: dict) -> dict:
    from research.www_evaluation_gpu_completion_20261003 import run as gpu_run
    from research.www_simulator_gpu_validation_20261018 import run as validation

    version_file = MODELS[model] / "download.json"
    revision = (json.loads(version_file.read_text())["revision"]
                if version_file.exists() else MODELS[model].name)

    label = f"{model}_{sample['name']}_c{kv}_{arm}_r{rate:g}"
    target = OUT / "runs" / f"{label}.json"
    if target.exists():
        row = json.loads(target.read_text())
        if row["status"] == "complete":
            print(label, "already complete", flush=True)
            return row
        raise RuntimeError(f"existing failed run: {target}")
    requests = generate(sample, rate)
    input_path = OUT / "inputs" / f"{sample['name']}_r{rate:g}.json"
    if not input_path.exists():
        write(input_path, requests)
    if json.loads(input_path.read_text()) != requests:
        raise RuntimeError(f"input changed: {input_path}")
    write(OUT / "current" / f"gpu{gpu}.json",
          {"task": label, "gpu": gpu, "started": time.time()})
    result = gpu_run.run_episode(
        engine, scheduler, requests, arm,
        {"budget": 1024, "chunk": 256,
         "history_file": f"history/{sample['source']}.json"}, 900)
    rounds = result.pop("rounds")
    round_path = OUT / "rounds" / f"{label}.json.gz"
    round_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(round_path, "wt", compresslevel=2) as stream:
        json.dump(rounds, stream, separators=(", ", ":"), allow_nan=False)
    raw_path = OUT / "raw" / f"{label}.json.gz"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw_path, "wt", compresslevel=2) as stream:
        json.dump(result, stream, separators=(",", ":"), allow_nan=False)
    metrics = validation.metrics(result, rate, spec) if result["status"] == "complete" else {}
    row = {"status": result["status"], "error": result["error"],
           "model": model, "model_revision": revision,
           "runner_sha256": digest(Path(__file__)),
           "sample": sample["name"], "source": sample["source"],
           "kv_blocks": kv, "arm": arm, "rate_qps": rate, "gpu": gpu,
           "input_sha256": digest(input_path),
           "source_pool_sha256": digest(ROOT / sample["input"]),
           "raw_path": str(raw_path.relative_to(OUT)),
           "raw_sha256": digest(raw_path),
           "round_path": str(round_path.relative_to(OUT)),
           "round_sha256": digest(round_path), **metrics}
    write(target, row)
    print(label, result["status"], metrics.get("passed"), flush=True)
    if result["status"] != "complete":
        raise RuntimeError(result["error"])
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--kv", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--samples", nargs="+")
    parser.add_argument("--rates", nargs="+", type=float)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--task-file", type=Path)
    args = parser.parse_args()
    model = MODELS[args.model]
    if not (model / "config.json").exists() or not list(model.glob("*.safetensors")):
        raise RuntimeError(f"model incomplete: {model}")
    spec = json.loads((SOURCE / "design.json").read_text())
    if args.task_file and (args.samples or args.rates or args.adaptive):
        raise RuntimeError("task file cannot be combined with scan arguments")
    if not args.task_file and (not args.samples or not args.rates):
        raise RuntimeError("samples and rates are required for a scan")
    samples = [next(s for s in spec["samples"] if s["name"] == name)
               for name in (args.samples or [])]
    if args.adaptive and len(samples) != 1:
        raise RuntimeError("adaptive QPS scan requires one source window")
    OUT.mkdir(parents=True, exist_ok=True)
    for source in ("code", "conversation"):
        history = SOURCE / "history" / f"{source}.json"
        target = OUT / "history" / f"{source}.json"
        if not target.exists():
            write(target, json.loads(history.read_text()))
        if digest(target) != digest(history):
            raise RuntimeError("history differs from original GPU validation")

    from research.www_idea1_gpu_pilot_20261016 import run as pilot
    from research.www_evaluation_gpu_completion_20261003 import run as gpu_run
    gpu_run.OUT = OUT
    if not pilot.idle(args.gpu):
        raise RuntimeError(f"GPU {args.gpu} is not idle")
    with pilot.acquire_gpu(args.gpu):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        engine, scheduler = engine_for(model, args.kv)
        warm = [dict(id=f"warm{i}", prompt=256, output=16, arrival_ms=0.)
                for i in range(4)]
        result = gpu_run.run_episode(
            engine, scheduler, warm, "native",
            {"budget": 1024, "chunk": 256,
             "history_file": "history/code.json"}, 120)
        if result["status"] != "complete":
            raise RuntimeError(f"warmup failed: {result['error']}")
        if args.task_file:
            tasks = json.loads(args.task_file.read_text())["tasks"]
            for task in tasks:
                sample = next(s for s in spec["samples"]
                              if s["name"] == task["sample"])
                one_run(engine, scheduler, sample, args.model, args.kv,
                        task["arm"], task["rate_qps"], args.gpu, spec)
        elif args.adaptive:
            for arm in args.arms:
                for rate in args.rates:
                    row = one_run(engine, scheduler, samples[0], args.model,
                                  args.kv, arm, rate, args.gpu, spec)
                    if not row["passed"]:
                        break
        else:
            for sample in samples:
                for rate in args.rates:
                    for arm in args.arms:
                        one_run(engine, scheduler, sample, args.model, args.kv,
                                arm, rate, args.gpu, spec)


if __name__ == "__main__":
    main()
