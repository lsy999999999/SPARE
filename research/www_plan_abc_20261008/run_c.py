"""GPU calibration and held-out comparison of fixed and selective controls."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import statistics as st
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
G2 = ROOT / "artifacts/www_evaluation_gpu_completion_20261003_r1"
OUT = ROOT / "artifacts/www_plan_abc_20261008_r1/c"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(G2 / "source"))
sys.path.insert(0, str(ROOT))
from research.www_evaluation_gpu_completion_20261003.common import read, write, digest, MODEL


def verify():
    design = read(OUT / "design.json")
    assert digest(G2 / "design.json") == design["old_design_sha256"]
    for name, value in design["source_hashes"].items():
        assert digest(HERE / name) == value, name
    for cell in design["cells"]:
        assert digest(OUT / cell["input"]) == cell["input_sha256"]
        assert digest(OUT / cell["history"]) == cell["history_sha256"]
    return design


def methods(stage):
    design = verify()
    if stage == "cal":
        options = {}
        for budget in design["calibration_budgets"]:
            options[f"fixed_{budget}"] = dict(mode="fixed", fixed_budget=budget)
            options[f"guarded_{budget}"] = dict(mode="fixed_guarded", fixed_budget=budget)
        return options
    selection = read(OUT / "selection.json")
    assert selection["passed"]
    return dict(srp=None, fixed_selected=dict(mode="fixed", fixed_budget=selection["fixed"]),
        guarded_selected=dict(mode="fixed_guarded", fixed_budget=selection["guarded"]),
        budget_only=dict(candidate_axes="budget"), spare={})


def configure_runner(stage):
    from research.www_evaluation_gpu_completion_20261003 import run as g2_run
    options = methods(stage)
    g2_run.OUT = OUT
    g2_run.OPTIONS = {name: value for name, value in options.items() if value is not None}
    g2_run.METHOD = {name: ("srp" if value is None else "threshold")
                     for name, value in options.items()}
    g2_run.METHOD["srp"] = "srp"
    return g2_run, options


def run_cell(stage, key, attempt, gpu):
    from research.www_evaluation_gpu_completion_20261003.run import acquire, hardware, idle
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    from research.www_plan_abc_20261008.c_scheduler import CScheduler
    lock = acquire(f"/tmp/query-capacity-gpu{gpu}.lock")
    assert idle(hardware()[gpu]), f"GPU {gpu} busy"
    design = verify()
    cell = next(c for c in design["cells"] if c["key"] == key and c["stage"] == stage)
    folder = OUT / "attempts" / key / f"a{attempt}"
    assert not (folder / "status.json").exists()
    write(folder / "status.json", dict(state="running", gpu=gpu, started=time.time()))
    try:
        os.sched_setaffinity(0, read(OUT / "manifest.json")["affinity"][str(gpu)])
        g2_run, options = configure_runner(stage)
        engine = LLMEngine.from_engine_args(EngineArgs(model=str(MODEL), dtype="half",
            max_model_len=8176, max_num_seqs=64, max_num_batched_tokens=2048,
            enable_chunked_prefill=True, enable_prefix_caching=False, block_size=16,
            gpu_memory_utilization=.9, enforce_eager=True, disable_log_stats=True,
            seed=0, num_gpu_blocks_override=cell["capacity"],
            scheduler_cls="research.www_plan_abc_20261008.c_scheduler.CScheduler"))
        scheduler = engine.engine_core.engine_core.scheduler
        assert isinstance(scheduler, CScheduler)
        assert scheduler.kv_cache_config.num_blocks == cell["capacity"]
        requests = read(OUT / cell["input"])
        g2_cell = dict(cell, history_file=cell["history"])
        warm = [dict(id=f"warm{i}", prompt=256, output=16, arrival_ms=0.) for i in range(4)]
        assert g2_run.run_episode(engine, scheduler, warm, "srp", g2_cell, 120)["status"] == "complete"
        aa = [g2_run.save_run(engine, scheduler, requests, "srp", g2_cell, folder,
            f"aa_srp_r{i}", dict(phase="aa", cell=key, repeat=i, gpu=gpu),
            design["episode_timeout_s"]) for i in range(design["aa_repeats"])]
        spread = {}
        for metric in ("ttft_ms_mean", "elapsed_ms"):
            values = [row["metrics"][metric] for row in aa]
            spread[metric] = (max(values) - min(values)) / st.fmean(values)
        passed = max(spread.values()) <= design["aa_threshold"]
        write(folder / "aa.json", dict(passed=passed, relative_spread=spread,
                                        threshold=design["aa_threshold"]))
        assert passed, "A/A stability gate failed"
        repeats = design["cal_repeats"] if stage == "cal" else design["test_repeats"]
        arms = list(options)
        for repeat in range(repeats):
            order = list(arms)
            seed = int(hashlib.sha256(f"20261008:{key}:{repeat//2}".encode()).hexdigest()[:16], 16)
            random.Random(seed).shuffle(order)
            if repeat % 2:
                order.reverse()
            for position, arm in enumerate(order):
                g2_run.save_run(engine, scheduler, requests, arm, g2_cell, folder,
                    f"{arm}_r{repeat}", dict(phase=stage, cell=key, arm=arm, repeat=repeat,
                    order_position=position, gpu=gpu), design["episode_timeout_s"])
        write(folder / "status.json", dict(state="complete", gpu=gpu, stage=stage,
            cell=key, attempt=attempt, formal_runs=len(arms)*repeats, ended=time.time()))
    except BaseException:
        write(folder / "status.json", dict(state="failed", gpu=gpu, stage=stage,
            cell=key, attempt=attempt, error=traceback.format_exc(), ended=time.time()))
        raise
    finally:
        lock.close()


def choose_budgets():
    design = verify()
    assert not (OUT / "selection.json").exists()
    cal_cells = [c for c in design["cells"] if c["stage"] == "cal"]
    values = {arm: [] for arm in design["cal_methods"]}
    audit = []
    for cell in cal_cells:
        group = read(OUT / "groups" / f"{cell['key']}.json")
        assert group["state"] == "complete"
        folder = OUT / group["folder"]
        assert read(folder / "aa.json")["passed"]
        for arm in values:
            runs = []
            for repeat in range(design["cal_repeats"]):
                path = folder / "runs" / f"{arm}_r{repeat}.json"
                result = read(path)
                assert result["status"] == "complete" and result["metrics"]["completed"] == 128
                round_path = folder / result["rounds_path"]
                assert digest(round_path) == result["rounds_sha256"]
                runs.append(result["metrics"]["ttft_ms_mean"])
            mean = st.fmean(runs)
            values[arm].append(mean)
            audit.append(dict(cell=cell["key"], arm=arm, mean_ttft_ms=mean,
                              repeats=design["cal_repeats"]))
    score = {arm: st.fmean(vals) for arm, vals in values.items()}
    selected = {}
    for family in ("fixed", "guarded"):
        arm = min((f"{family}_{budget}" for budget in design["calibration_budgets"]),
                  key=lambda name: (score[name], int(name.split("_")[-1])))
        selected[family] = int(arm.split("_")[-1])
    write(OUT / "selection.json", dict(passed=True, created=time.time(),
        fixed=selected["fixed"], guarded=selected["guarded"], scores=score,
        calibration_audit=audit, test_cells=[c["key"] for c in design["cells"] if c["stage"] == "test"],
        selection_rule=design["selection"]))
    return selected


def run_stage(stage, design, gpus, deadline):
    from research.www_evaluation_gpu_completion_20261003.run import hardware, idle, compute_pids, descendants
    pending = [dict(cell=c["key"], attempt=0) for c in design["cells"] if c["stage"] == stage]
    active = {}
    finished = []
    while pending or active:
        now = time.time()
        for gpu, task in list(active.items()):
            process = task["process"]
            folder = OUT / "attempts" / task["cell"] / f"a{task['attempt']}"
            if process.poll() is None:
                foreign = compute_pids(hardware()[gpu]["uuid"]) - descendants(process.pid)
                if foreign:
                    write(folder / "contamination.json", dict(pids=sorted(foreign), at=now))
                    process.terminate()
                elif now - task["started"] > 2 * 3600 or now >= deadline:
                    write(folder / "timeout.json", dict(at=now))
                    process.terminate()
            if process.poll() is not None:
                task["log"].close()
                status = read(folder / "status.json") if (folder / "status.json").exists() else {}
                passed = (process.returncode == 0 and status.get("state") == "complete"
                          and not (folder / "contamination.json").exists()
                          and not (folder / "timeout.json").exists())
                if passed:
                    write(OUT / "groups" / f"{task['cell']}.json",
                          dict(state="complete", stage=stage, cell=task["cell"],
                               folder=str(folder.relative_to(OUT)), gpu=gpu,
                               attempt=task["attempt"]))
                elif task["attempt"] < 2 and now < deadline:
                    pending.append(dict(cell=task["cell"], attempt=task["attempt"]+1))
                finished.append(dict(cell=task["cell"], attempt=task["attempt"],
                    passed=passed, gpu=gpu, exit_code=process.returncode))
                del active[gpu]
        if now >= deadline and not active:
            break
        for gpu in gpus:
            if not pending or gpu in active or not idle(hardware()[gpu]):
                continue
            task = pending.pop(0)
            folder = OUT / "attempts" / task["cell"] / f"a{task['attempt']}"
            folder.mkdir(parents=True, exist_ok=True)
            log = (folder / "worker.log").open("x")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID")
            process = subprocess.Popen([sys.executable, "-B", "-u", str(HERE / "run_c.py"),
                "--worker", task["cell"], "--stage", stage, "--attempt", str(task["attempt"]),
                "--gpu", str(gpu)], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            active[gpu] = dict(task, process=process, log=log, started=time.time())
        write(OUT / "status.json", dict(state="running", stage=stage, pending=pending,
            active={str(g): dict(cell=t["cell"], attempt=t["attempt"], pid=t["process"].pid)
                    for g, t in active.items()}, finished=finished, updated=time.time()))
        time.sleep(10)
    accepted = sum((OUT / "groups" / f"{c['key']}.json").exists()
                   for c in design["cells"] if c["stage"] == stage)
    required = sum(c["stage"] == stage for c in design["cells"])
    write(OUT / f"{stage}_status.json", dict(passed=accepted == required,
        accepted=accepted, required=required, attempts=finished, ended=time.time()))
    return accepted == required


def launch():
    from research.www_evaluation_gpu_completion_20261003.run import hardware, idle, cpu_assign
    design = verify()
    assert not (OUT / "launch.json").exists()
    assert importlib.metadata.version("vllm") == "0.9.2"
    all_hardware = hardware()
    gpus = [g for g in design["gpu_candidates"] if idle(all_hardware[g])]
    assert gpus, "No idle RTX 4090"
    manifest = dict(created=time.time(), hostname=os.uname().nodename,
        design_sha256=digest(OUT / "design.json"),
        g2_design_sha256=design["old_design_sha256"],
        hardware={str(g): all_hardware[g] for g in gpus},
        affinity=cpu_assign(gpus),
        versions={name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers")})
    write(OUT / "manifest.json", manifest)
    env = dict(os.environ, VLLM_USE_V1="1", VLLM_ENABLE_V1_MULTIPROCESSING="0",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false")
    p = subprocess.Popen([sys.executable, "-B", "-u", str(HERE / "run_c.py"), "--controller"],
        cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
        stdout=(OUT / "controller.log").open("x"), stderr=subprocess.STDOUT,
        start_new_session=True)
    write(OUT / "launch.json", dict(pid=p.pid, started=time.time(), gpus=gpus,
        deadline=time.time()+48*3600, detached=True))
    print(p.pid)


def controller():
    design = verify()
    launch_info = read(OUT / "launch.json")
    if run_stage("cal", design, launch_info["gpus"], launch_info["deadline"]):
        choose_budgets()
        run_stage("test", design, launch_info["gpus"], launch_info["deadline"])
    test = read(OUT / "test_status.json") if (OUT / "test_status.json").exists() else None
    write(OUT / "status.json", dict(state="complete" if test and test["passed"] else "finished_with_gaps",
        calibration=read(OUT / "cal_status.json"), test=test, updated=time.time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--controller", action="store_true")
    parser.add_argument("--worker")
    parser.add_argument("--stage")
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.launch:
        launch()
    elif args.controller:
        controller()
    else:
        run_cell(args.stage, args.worker, args.attempt, args.gpu)
