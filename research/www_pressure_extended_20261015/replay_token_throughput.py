"""Replay frozen 120-second pressure runs to count every emitted output token."""
from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from research.www_capacity_pareto_20261015 import run as base
from research.www_pareto_20261014.run import ROOT, digest, read, write
from research.www_pressure_extended_20261015.run import generate

SOURCE = ROOT / "artifacts/www_pressure_extended_20261015"
DEST = SOURCE / "token_throughput_120s"
TAGS = ("intro_low_120s", "coarse_120s", "upper_120s", "apt_upper_120s",
        "intro_baselines_120s")


def setup():
    base.initialize()
    base.MAX_WALL_S = 240
    base.SIM.simulate = partial(base.SIM.simulate, stop_at_ms=120000.,
                                probe_times_ms=(60000.,))


def replay(task):
    source_tag, original_path, sample = task
    original = read(original_path)
    target = DEST / "runs" / original_path.name
    if target.exists():
        result = read(target)
        if result["original_trajectory_sha256"] != original["trajectory_sha256"]:
            raise RuntimeError(f"stale replay: {target}")
        return result
    requests = generate(sample, original["rate"], 120000.)
    seed = int.from_bytes(hashlib.sha256(sample["name"].encode()).digest()[:4], "big")
    simulation = base.simulate(requests, original["arm"], original["capacity"],
                               original["budget"], original["source"], seed)
    if simulation["trajectory_sha256"] != original["trajectory_sha256"]:
        raise RuntimeError(f"trajectory changed: {original_path}")
    windows = simulation["pressure_windows"]
    for new, old in zip(windows, original["windows"], strict=True):
        if any(new[key] != value for key, value in old.items()):
            raise RuntimeError(f"checkpoint changed: {original_path}")
        if new["emitted_output_tokens"] < new["completed_output_tokens"]:
            raise RuntimeError(f"token accounting failed: {original_path}")
    result = dict(source_tag=source_tag, source=original["source"],
                  sample=original["sample"], capacity=original["capacity"],
                  arm=original["arm"], budget=original["budget"], rate=original["rate"],
                  valid=True, horizon_ms=120000., windows=windows,
                  original_trajectory_sha256=original["trajectory_sha256"],
                  cpu_wall_s=simulation["cpu_wall_s"])
    write(target, result)
    return result


def prepare():
    tasks = []
    designs = {}
    for tag in TAGS:
        folder = SOURCE / tag
        design = read(folder / "design.json")
        samples = {sample["name"]: sample for sample in design["samples"]}
        for path, sha in design["input_sha256"].items():
            if digest(ROOT / path) != sha:
                raise RuntimeError(f"input changed: {path}")
        files = list((folder / "runs").glob("*.json"))
        if len(files) != design["tasks"]:
            raise RuntimeError(f"incomplete source batch: {tag}")
        for path in files:
            original = read(path)
            if not original["valid"]:
                raise RuntimeError(f"invalid source run: {path}")
            tasks.append((tag, path, samples[original["sample"]], original["cpu_wall_s"]))
        designs[tag] = digest(folder / "design.json")
    manifest = dict(tags=TAGS, source_design_sha256=designs,
                    simulator_sha256=digest(ROOT / "research/www_evaluation_completion_20261003/simulate.py"),
                    runner_sha256=digest(Path(__file__)),
                    token_metric="Output tokens emitted at or before 120000 ms, including unfinished requests",
                    tasks=len(tasks))
    path = DEST / "design.json"
    if path.exists() and read(path) != manifest:
        raise RuntimeError("replay design changed")
    write(path, manifest)
    tasks.sort(key=lambda task: task[3], reverse=True)
    return [task[:3] for task in tasks]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    tasks = prepare()
    errors = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=setup) as pool:
        futures = {pool.submit(replay, task): task[1] for task in tasks}
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                errors.append(dict(path=str(futures[future]), error=repr(exc)))
            if completed % 48 == 0 or completed == len(tasks):
                write(DEST / "status.json", dict(completed=completed, expected=len(tasks),
                                                 errors=errors))
                print(f"{completed}/{len(tasks)} errors={len(errors)}", flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} replay failures; see {DEST / 'status.json'}")


if __name__ == "__main__":
    main()
