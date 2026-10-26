"""Bounded high-load replay of frozen capacity configurations."""
from __future__ import annotations

import argparse
import hashlib
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np

from research.www_capacity_pareto_20261015 import run as base
from research.www_pareto_20261014.run import ROOT, digest, read, write

SOURCE = ROOT / "artifacts/www_capacity_pareto_20261015_r1"
OUT = ROOT / "artifacts/www_pressure_extended_20261015"


def select_samples(count, sources):
    samples = read(SOURCE / "design.json")["samples"]
    result = []
    for source in sources:
        group = [s for s in samples if s["split"] == "test" and s["source"] == source]
        result.extend(group[int(i)] for i in np.linspace(0, len(group)-1, count))
    return result


def generate(sample, rate, horizon_ms):
    pool = read(ROOT / sample["input"])
    seed = int.from_bytes(hashlib.sha256(sample["name"].encode()).digest()[:8], "big")
    arrival_rng = np.random.default_rng(seed)
    length_rng = np.random.default_rng(seed ^ 0x9E3779B97F4A7C15)
    gaps = arrival_rng.exponential(1000., max(4096, math.ceil(rate*horizon_ms/1000*2)))
    times = np.cumsum(gaps) / rate
    if times[-1] < horizon_ms:
        raise RuntimeError("arrival stream too short")
    count = int(np.searchsorted(times, horizon_ms))
    picks = length_rng.integers(0, len(pool), size=count)
    return [dict(id=f"{sample['name']}:{i}", prompt=pool[int(j)]["prompt"],
                 output=pool[int(j)]["output"], arrival_ms=float(times[i]))
            for i, j in enumerate(picks)]


def setup(horizon_ms, max_wall_s):
    base.initialize()
    base.MAX_WALL_S = max_wall_s
    base.SIM.simulate = partial(base.SIM.simulate, stop_at_ms=horizon_ms,
                                probe_times_ms=(horizon_ms/2,))


def task_run(task):
    sample, capacity, arm, budget, rate, horizon_ms, output = task
    path = output / "runs" / f"{sample['name']}_c{capacity}_{arm}_t{budget}_r{rate:g}.json"
    if path.exists():
        return read(path)
    requests = generate(sample, rate, horizon_ms)
    seed = int.from_bytes(hashlib.sha256(sample["name"].encode()).digest()[:4], "big")
    try:
        result = base.simulate(requests, arm, capacity, budget, sample["source"], seed)
        windows = result["pressure_windows"]
        mid, end = windows
        assert end["arrivals"] == len(requests)
        data = dict(valid=True, windows=windows,
                    first_half_qps=mid["completed"]*2000/horizon_ms,
                    second_half_qps=(end["completed"]-mid["completed"])*2000/horizon_ms,
                    backlog_growth=end["outstanding"]-mid["outstanding"],
                    cpu_wall_s=result["cpu_wall_s"], rounds=result["rounds"],
                    trajectory_sha256=result["trajectory_sha256"])
    except RuntimeError as exc:
        if str(exc) != "simulation limit; no completed result will be reported":
            raise
        data = dict(valid=False, failure="simulation_limit", arrivals=len(requests))
    row = dict(source=sample["source"], sample=sample["name"], capacity=capacity,
               arm=arm, budget=budget, rate=rate, horizon_ms=horizon_ms, **data)
    write(path, row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--rates", default="8,10,12,16,20,24")
    parser.add_argument("--horizon-s", type=int, default=120)
    parser.add_argument("--samples-per-source", type=int, default=3)
    parser.add_argument("--arms", default="arrival,apt_serve_kv_rule")
    parser.add_argument("--sources", default="code,conversation")
    parser.add_argument("--capacities", default="768,4096")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--max-wall-s", type=int, default=120)
    args = parser.parse_args()
    if args.samples_per_source < 1 or args.samples_per_source > 50:
        parser.error("samples-per-source must be 1..50")
    rates = tuple(float(x) for x in args.rates.split(","))
    arms = tuple(args.arms.split(","))
    sources = tuple(args.sources.split(","))
    capacities = tuple(int(x) for x in args.capacities.split(","))
    if not set(sources) <= {"code", "conversation"} or not set(capacities) <= set(base.CAPACITIES):
        parser.error("unknown source or capacity")
    horizon_ms = args.horizon_s * 1000.
    samples = select_samples(args.samples_per_source, sources)
    selections = read(SOURCE / "selected_test.json")
    tasks = []
    for sample in samples:
        for capacity in capacities:
            for arm in arms:
                config = [r for r in selections if r["source"] == sample["source"]
                          and r["capacity"] == capacity and r["arm"] == arm
                          and r["slo"] == "strict" and r["weight"] == 0.5]
                if len(config) != 1:
                    raise RuntimeError(f"missing frozen configuration: {sample} {capacity} {arm}")
                tasks.extend((sample, capacity, arm, config[0]["budget"], rate,
                              horizon_ms, OUT / args.tag) for rate in rates)
    design = dict(tag=args.tag, rates=rates, horizon_ms=horizon_ms,
                  samples=samples, arms=arms, sources=sources,
                  capacities=capacities, tasks=len(tasks),
                  source_selection_sha256=digest(SOURCE / "selected_test.json"),
                  timing_sha256=digest(ROOT / "artifacts/www_pareto_20261014_r1/timing_profile.json"),
                  simulator_sha256=digest(ROOT / "research/www_evaluation_completion_20261003/simulate.py"),
                  runner_sha256=digest(ROOT / "research/www_pressure_extended_20261015/run.py"),
                  input_sha256={s["input"]: digest(ROOT / s["input"]) for s in samples},
                  max_wall_s=args.max_wall_s, gpu_execution=False)
    output = OUT / args.tag
    frozen = output / "design.json"
    if frozen.exists() and read(frozen) != design:
        raise RuntimeError("design changed for existing tag")
    write(frozen, design)
    failures = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=setup,
                             initargs=(horizon_ms, args.max_wall_s)) as pool:
        futures = [pool.submit(task_run, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            try:
                row = future.result()
                if not row["valid"]:
                    failures.append((row["sample"], row["capacity"], row["arm"], row["rate"]))
            except Exception as exc:
                failures.append(repr(exc))
            if index % 8 == 0 or index == len(tasks):
                write(output / "status.json", dict(completed=index, expected=len(tasks),
                                                    failures=failures))
                print(f"{index}/{len(tasks)} failures={len(failures)}", flush=True)
    if failures:
        print(f"Simulation failures: {failures}", flush=True)


if __name__ == "__main__":
    main()
