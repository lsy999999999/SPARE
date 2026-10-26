"""Run the paired fixed-256 explanation matrix on idle RTX 4090 GPUs."""
import argparse
import statistics as st
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.www_fixed256_explanation_20261014 import common as spec
from research.www_evaluation_gpu_completion_20261003 import run as core

for name in ("OUT", "HERE", "ARMS", "OPTIONS", "METHOD", "ORACLE_ARMS",
             "source_hashes", "source_paths"):
    setattr(core, name, getattr(spec, name))


def worker(cell_key, gpu, folder_relative):
    lock = core.acquire(f"/tmp/query-capacity-gpu{gpu}.lock")
    design = core.verify_design()
    manifest = spec.read(spec.OUT / "manifest.json")
    cell = next(c for c in design["cells"] if c["key"] == cell_key)
    folder = spec.OUT / folder_relative
    folder.mkdir(parents=True, exist_ok=True)
    if not core.idle(core.hardware()[gpu]):
        raise RuntimeError(f"GPU {gpu} is no longer idle")
    core.os.sched_setaffinity(0, manifest["affinity"][str(gpu)])
    spec.write(folder / "gpu_start.json", core.gpu_snapshot(gpu))
    try:
        import torch
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine

        expected_uuid = manifest["hardware"][str(gpu)]["uuid"].removeprefix("GPU-").lower()
        if str(torch.cuda.get_device_properties(0).uuid).lower() != expected_uuid:
            raise RuntimeError("CUDA/NVML GPU identity mismatch")
        engine = LLMEngine.from_engine_args(EngineArgs(
            model=str(spec.MODEL), dtype="half", max_model_len=design["max_model_len"],
            max_num_seqs=design["max_num_seqs"],
            max_num_batched_tokens=design["max_num_batched_tokens"],
            enable_chunked_prefill=True, enable_prefix_caching=False,
            block_size=design["block_size"], gpu_memory_utilization=.9,
            enforce_eager=True, disable_log_stats=True, seed=0,
            num_gpu_blocks_override=cell["capacity"],
            scheduler_cls="research.www_fixed256_explanation_20261014.scheduler.ExplanationScheduler"))
        scheduler = engine.engine_core.engine_core.scheduler
        if scheduler.kv_cache_config.num_blocks != cell["capacity"]:
            raise RuntimeError("KV block override was not honored")
        requests = spec.read(spec.OUT / cell["input"])
        warm = [dict(id=f"warm{i}", prompt=256, output=16, arrival_ms=0.)
                for i in range(4)]
        core.run_episode(engine, scheduler, warm, "native", cell, 120)
        timeout = design["episode_timeout_s"]
        aa = [core.save_run(engine, scheduler, requests, "full", cell, folder,
                           f"aa_full_r{repeat}", dict(phase="aa", repeat=repeat,
                           cell=cell_key, gpu=gpu), timeout)
              for repeat in range(design["aa_repeats"])]
        aa_values = [run["metrics"]["ttft_ms_mean"] for run in aa]
        aa_gap = (max(aa_values) - min(aa_values)) / st.fmean(aa_values)
        spec.write(folder / "aa.json", dict(passed=aa_gap <= .05,
            threshold=.05, ttft_ms_mean=aa_values, range_over_mean=aa_gap))
        if aa_gap > .05:
            raise RuntimeError("A/A TTFT stability gate exceeded 5%")
        before = core.save_run(engine, scheduler, requests, "full", cell, folder,
            "bookend_before", dict(phase="bookend", position="before",
                                   cell=cell_key, gpu=gpu), timeout)
        for repeat in range(design["repeats"]):
            for position, arm in enumerate(core.order_for(cell_key, repeat)):
                core.save_run(engine, scheduler, requests, arm, cell, folder,
                    f"{arm}_r{repeat}", dict(phase="formal", repeat=repeat,
                    order_position=position, cell=cell_key, gpu=gpu), timeout)
        after = core.save_run(engine, scheduler, requests, "full", cell, folder,
            "bookend_after", dict(phase="bookend", position="after",
                                  cell=cell_key, gpu=gpu), timeout)
        pair = [row["metrics"]["ttft_ms_mean"] for row in (before, after)]
        drift = abs(pair[0] - pair[1]) / st.fmean(pair)
        spec.write(folder / "bookend.json", dict(passed=drift <= .05,
            threshold=.05, ttft_ms_mean=pair, relative_drift=drift))
        if drift > .05:
            raise RuntimeError("bookend TTFT drift exceeded 5%")
        spec.write(folder / "gpu_end.json", core.gpu_snapshot(gpu))
        spec.write(folder / "status.json", dict(state="complete", cell=cell_key,
            gpu=gpu, formal_runs=len(spec.ARMS) * design["repeats"],
            updated=time.time()))
    except BaseException:
        spec.write(folder / "status.json", dict(state="failed", cell=cell_key,
            gpu=gpu, error=traceback.format_exc(), updated=time.time()))
        raise
    finally:
        lock.close()


core.worker = worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--folder")
    args = parser.parse_args()
    if args.launch:
        core.launch()
    elif args.worker:
        worker(args.worker, args.gpu, args.folder)
    else:
        try:
            core.controller()
        except BaseException:
            spec.write(spec.OUT / "status.json", dict(state="failed",
                error=traceback.format_exc(), updated=time.time()))
            raise


if __name__ == "__main__":
    main()
