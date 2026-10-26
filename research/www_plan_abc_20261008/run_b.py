"""Frozen same-state, one-action G2 SPARE GPU branches."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import shutil
import statistics as st
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
G2 = ROOT / "artifacts/www_evaluation_gpu_completion_20261003_r1"
OUT = ROOT / "artifacts/www_plan_abc_20261008_r1/b"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(G2 / "source"))
sys.path.insert(0, str(ROOT))
from research.www_evaluation_gpu_completion_20261003.common import read, write, digest, MODEL

SOURCE = ("branch_policy.py", "probe_scheduler.py", "run_b.py")
SPEC = {"kind": "linear", "intercept": 8,
        "features": ["arrival_adjusted_ratio", "decoded_kv_headroom"],
        "coefficients": [128, 0], "tau_s": 1}


def freeze():
    assert not (OUT / "design.json").exists()
    old = read(G2 / "design.json")
    cells = []
    for window in (f"{source}_new{i}" for source in ("code", "conversation") for i in range(3)):
        old_cell = next(c for c in old["cells"] if c["window"] == window
                        and c["shape"] == "burst_stop" and c["capacity"] == 768)
        input_path = OUT / "inputs" / (window + ".json")
        history_path = OUT / "history" / (window.split("_")[0] + ".json")
        input_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(G2 / old_cell["input"], input_path)
        shutil.copyfile(G2 / old_cell["history_file"], history_path)
        requests = read(input_path)
        assert len(requests) == 128 and all(r["arrival_ms"] == 0 for r in requests)
        cells.append(dict(window=window, input=str(input_path.relative_to(OUT)),
            input_sha256=digest(input_path), history=str(history_path.relative_to(OUT)),
            history_sha256=digest(history_path), capacities=[768, 2048]))
    design = dict(schema=1, created=time.time(), cells=cells, max_states_per_window=4,
        strata="one seeded state from each capacity x early/late half of eligible interventions",
        seed=20261008, repeats=4, order="alternating Base/SPARE action within each state",
        branch_followup="full SPARE", gpu_candidates=[1, 2, 6],
        aa_threshold=.05, episode_timeout_s=900,
        objective="remaining first-token waiting area from action start / 128",
        g2_design_sha256=digest(G2 / "design.json"),
        g2_source_hashes=old["source_hashes"],
        source_hashes={name: digest(HERE / name) for name in SOURCE})
    write(OUT / "design.json", design)
    return design


def verify():
    design = read(OUT / "design.json")
    assert design["g2_design_sha256"] == digest(G2 / "design.json")
    for name, value in design["source_hashes"].items():
        assert digest(HERE / name) == value, name
    for c in design["cells"]:
        assert digest(OUT / c["input"]) == c["input_sha256"]
        assert digest(OUT / c["history"]) == c["history_sha256"]
    return design


def select_targets(rounds, window, capacity):
    eligible = [r for r in rounds if r["intervened"] and r["verified"]
        and r["baseline"] is not None and r["baseline_growth_blocks"] > r["actual_growth_blocks"]
        and r["reductions"] and not r["preemptions"]]
    targets = []
    for stage, pool in (("early", eligible[:len(eligible)//2]),
                        ("late", eligible[len(eligible)//2:])):
        if not pool:
            continue
        seed = int(hashlib.sha256(f"20261008:{window}:{capacity}:{stage}".encode()).hexdigest()[:16], 16)
        r = random.Random(seed).choice(pool)
        targets.append(dict(round=r["round"], stage=stage, capacity=capacity,
            fingerprint=r["state_fingerprint"], physical_state_sha256=r["physical_state_sha256"],
            free_before=r["free_before"], base_growth=r["baseline_growth_blocks"],
            selected_growth=r["actual_growth_blocks"], threshold=r["threshold"],
            base_tokens=r["baseline"], selected_tokens=r["scheduled"],
            spare_budget=r["budget"], spare_chunk=r["chunk"],
            pending=sum(x["outputs"] == 0 for x in r["state"])))
    return dict(eligible=len(eligible), selected=targets)


def episode_save(engine, scheduler, requests, config, folder, name, meta, anchor=None):
    from research.v1_fixed_order_20260927.episode import episode
    from research.v1_candidate_audit_20260928.audit import measure
    path = folder / "runs" / f"{name}.json"
    assert not path.exists(), path
    result = episode(engine, scheduler, requests, "threshold", config, timeout=900)
    rounds = result.pop("rounds")
    assert result["status"] == "complete", result["error"]
    assert result["completed"] == len(requests)
    branch = config.get("branch")
    if branch:
        index = branch["round"]
        assert scheduler.branch_seen and sum(bool(r["branch_applied"]) for r in rounds) == 1
        assert all(r["state_fingerprint"] == anchor[i]["state_fingerprint"] for i, r in enumerate(rounds[:index+1]))
        assert all(r["physical_state_sha256"] == anchor[i]["physical_state_sha256"] for i, r in enumerate(rounds[:index+1]))
        assert all(r["scheduled"] == anchor[i]["scheduled"] for i, r in enumerate(rounds[:index]))
        assert rounds[index]["scheduled"] == meta["expected_tokens"]
        result["label"] = measure(result, rounds, index)
        assert result["label"]["pending"] == meta["pending"]
    result.update(meta, name=name, config=config)
    round_path = folder / "rounds" / f"{name}.json.gz"
    round_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(round_path, "wt", compresslevel=1) as handle:
        json.dump(rounds, handle, separators=(",", ":"), allow_nan=False)
    result.update(rounds_path=str(round_path.relative_to(OUT)), rounds_sha256=digest(round_path))
    write(path, result)
    print(name, result.get("label", {}).get("area_per_request_ms", result["metrics"]["mean_ttft_ms"]), flush=True)
    return result, rounds


def worker(window, gpu):
    from research.www_evaluation_gpu_completion_20261003.run import acquire, hardware, idle
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    from research.www_plan_abc_20261008.probe_scheduler import ProbeScheduler
    lock = acquire(f"/tmp/query-capacity-gpu{gpu}.lock")
    assert idle(hardware()[gpu]), f"GPU {gpu} busy"
    design = verify()
    cell = next(c for c in design["cells"] if c["window"] == window)
    requests = read(OUT / cell["input"])
    history = read(OUT / cell["history"])["lengths"]
    folder = OUT / "groups" / window
    assert not (folder / "status.json").exists()
    write(folder / "status.json", dict(state="running", gpu=gpu, started=time.time()))
    try:
        for capacity in cell["capacities"]:
            engine = LLMEngine.from_engine_args(EngineArgs(model=str(MODEL), dtype="half",
                max_model_len=8176, max_num_seqs=64, max_num_batched_tokens=2048,
                enable_chunked_prefill=True, enable_prefix_caching=False, block_size=16,
                gpu_memory_utilization=.9, enforce_eager=True, disable_log_stats=True, seed=0,
                num_gpu_blocks_override=capacity,
                scheduler_cls="research.www_plan_abc_20261008.probe_scheduler.ProbeScheduler"))
            scheduler = engine.engine_core.engine_core.scheduler
            assert isinstance(scheduler, ProbeScheduler)
            assert scheduler.kv_cache_config.num_blocks == capacity
            scheduler.history_context = history
            scheduler.seed_context = 20260929
            config = dict(budget=1024, chunk=256, concurrency=32, threshold=SPEC)
            scheduler.probe_times_context = None
            scheduler.probe_target_context = None
            capfolder = folder / f"c{capacity}"
            warm = [dict(id=f"warm{i}", prompt=256, output=16, arrival_ms=0.) for i in range(4)]
            from research.v1_fixed_order_20260927.episode import episode
            assert episode(engine, scheduler, warm, "threshold", config, timeout=120)["status"] == "complete"
            aa = [episode_save(engine, scheduler, requests, config, capfolder,
                f"aa_r{i}", dict(phase="aa", repeat=i))[0] for i in range(2)]
            ttft = [r["metrics"]["mean_ttft_ms"] for r in aa]
            spread = (max(ttft) - min(ttft)) / st.fmean(ttft)
            write(capfolder / "aa.json", dict(passed=spread <= design["aa_threshold"],
                relative_spread=spread, values=ttft))
            assert spread <= design["aa_threshold"], "A/A failed"
            _, anchor = episode_save(engine, scheduler, requests, config, capfolder,
                "anchor", dict(phase="anchor"))
            selected = select_targets(anchor, window, capacity)
            write(capfolder / "selected.json", selected)
            times = [r["features"]["arrival_time_s"] for r in anchor]
            for target in selected["selected"]:
                index = target["round"]
                scheduler.probe_times_context = times
                scheduler.probe_target_context = index
                for repeat in range(design["repeats"]):
                    order = ("base", "spare") if repeat % 2 == 0 else ("spare", "base")
                    for arm in order:
                        action = (dict(budget=1024, chunk=256) if arm == "base" else
                                  dict(budget=target["spare_budget"], chunk=target["spare_chunk"]))
                        expected = target["base_tokens"] if arm == "base" else target["selected_tokens"]
                        branch = dict(round=index, fingerprint=target["fingerprint"],
                            threshold=dict(SPEC, forced_action=action))
                        meta = dict(phase="branch", target=index, arm=arm, repeat=repeat,
                            capacity=capacity, pending=target["pending"], expected_tokens=expected)
                        episode_save(engine, scheduler, requests, dict(config, branch=branch),
                            capfolder, f"s{index}_{arm}_r{repeat}", meta, anchor)
            scheduler.probe_times_context = None
            scheduler.probe_target_context = None
            del engine
        write(folder / "status.json", dict(state="complete", gpu=gpu, ended=time.time()))
    except BaseException:
        write(folder / "status.json", dict(state="failed", gpu=gpu,
            error=traceback.format_exc(), ended=time.time()))
        raise
    finally:
        lock.close()


def launch():
    design = verify()
    assert not (OUT / "launch.json").exists()
    import importlib.metadata
    from research.www_evaluation_gpu_completion_20261003.run import hardware, idle
    assert importlib.metadata.version("vllm") == "0.9.2"
    gpus = [g for g in design["gpu_candidates"] if idle(hardware()[g])]
    assert gpus, "No idle candidate GPU"
    env = dict(os.environ, VLLM_USE_V1="1", VLLM_ENABLE_V1_MULTIPROCESSING="0",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    p = subprocess.Popen([sys.executable, "-B", "-u", str(HERE / "run_b.py"), "--controller"],
        cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
        stdout=(OUT / "controller.log").open("x"), stderr=subprocess.STDOUT,
        start_new_session=True)
    write(OUT / "launch.json", dict(pid=p.pid, started=time.time(),
        gpus=gpus, detached=True))
    print(p.pid)


def controller():
    from research.www_evaluation_gpu_completion_20261003.run import hardware, idle, compute_pids, descendants
    design = verify()
    launch_info = read(OUT / "launch.json")
    pending = [c["window"] for c in design["cells"]]
    active = {}
    finished = []
    deadline = time.time() + 12 * 3600
    while pending or active:
        for gpu, task in list(active.items()):
            process = task["process"]
            if process.poll() is None:
                foreign = compute_pids(hardware()[gpu]["uuid"]) - descendants(process.pid)
                if foreign:
                    write(OUT / "groups" / task["window"] / "contamination.json",
                          dict(pids=sorted(foreign), at=time.time()))
                    process.terminate()
                elif time.time() - task["started"] > 3 * 3600:
                    process.terminate()
            if process.poll() is not None:
                task["log"].close()
                status = read(OUT / "groups" / task["window"] / "status.json")
                finished.append(dict(window=task["window"], gpu=gpu,
                    exit_code=process.returncode, state=status["state"]))
                del active[gpu]
        if time.time() >= deadline:
            break
        for gpu in launch_info["gpus"]:
            if not pending or gpu in active or not idle(hardware()[gpu]):
                continue
            window = pending.pop(0)
            folder = OUT / "groups" / window
            folder.mkdir(parents=True, exist_ok=True)
            log = (folder / "worker.log").open("x")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID")
            process = subprocess.Popen([sys.executable, "-B", "-u", str(HERE / "run_b.py"),
                "--worker", window, "--gpu", str(gpu)], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            active[gpu] = dict(process=process, window=window, log=log, started=time.time())
        write(OUT / "status.json", dict(state="running", pending=pending,
            active={str(g): dict(window=t["window"], pid=t["process"].pid)
                    for g, t in active.items()}, finished=finished, updated=time.time()))
        time.sleep(10)
    state = "complete" if len(finished) == len(design["cells"]) and all(
        x["state"] == "complete" and x["exit_code"] == 0 for x in finished) else "finished_with_gaps"
    write(OUT / "status.json", dict(state=state, pending=pending, active={},
        finished=finished, updated=time.time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--controller", action="store_true")
    parser.add_argument("--worker")
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.freeze:
        print(json.dumps(freeze()))
    elif args.launch:
        launch()
    elif args.controller:
        controller()
    else:
        worker(args.worker, args.gpu)
