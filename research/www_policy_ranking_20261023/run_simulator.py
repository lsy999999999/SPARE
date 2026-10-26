"""Replay the frozen ranking grid with the held-out Qwen3-8B timing profile."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/www_policy_ranking_fidelity_20261023_r1"
OLD = ROOT / "artifacts/www_cross_model_20261021_r1"
SPEC = ROOT / "artifacts/www_simulator_gpu_validation_20261019_r1/design.json"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run_one(task: tuple[str, str, float]) -> str:
    from research.www_cross_model_20261021 import run_gpu as gpu
    from research.www_cross_model_20261021.validate_simulator import metric
    from research.www_evaluation_completion_20261003 import simulate as sim
    from research.www_evaluation_completion_20261003.timing import FittedTiming

    sample_name, arm, rate = task
    label = f"qwen3_8b_{sample_name}_c2048_{arm}_r{rate:g}"
    target = OUT / "sim" / f"{label}.json"
    profile_path = ROOT / "artifacts/www_cross_model_timing_20261022_r1/profiles/qwen3_8b.json"
    profile_hash = digest(profile_path.read_bytes())
    if target.exists():
        saved = json.loads(target.read_text())
        if saved["timing_profile_sha256"] == profile_hash:
            return label
        raise RuntimeError(f"existing replay uses a different profile: {target}")
    frozen = json.loads(SPEC.read_text())
    sample = next(s for s in frozen["samples"] if s["name"] == sample_name)
    requests = gpu.generate(sample, rate)
    input_hash = digest((json.dumps(requests, indent=2, allow_nan=False) + "\n").encode())
    history = json.loads((OLD / "history" / f"{sample['source']}.json").read_text())
    result = sim.simulate(
        requests, arm=arm, capacity=2048, budget=1024, chunk=256,
        history=history["lengths"], seed=0,
        fitted_timing=FittedTiming(json.loads(profile_path.read_text())),
        capture_probe=True,
    )
    row = dict(model="qwen3_8b", sample=sample_name, source=sample["source"],
               kv_blocks=2048, arm=arm, rate_qps=rate,
               input_sha256=input_hash, timing_profile_sha256=profile_hash,
               simulator_sha256=digest(Path(sim.__file__).read_bytes()),
               metrics=metric(result, rate))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    return label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-dir", help="Replay only tasks from this frozen task directory")
    args = parser.parse_args()
    if args.task_dir:
        tasks = [(row["sample"], row["arm"], row["rate_qps"])
                 for path in sorted((OUT / args.task_dir).glob("*.json"))
                 for row in json.loads(path.read_text())["tasks"]]
        if not tasks:
            raise RuntimeError(f"no tasks found in {args.task_dir}")
    else:
        design = json.loads((OUT / "design.json").read_text())
        tasks = [(f"{source}_{window}", arm, rate)
                 for source, rates in design["sources"].items()
                 for window in design["windows"]
                 for arm in design["arms"] for rate in rates]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 20 == 0 or index == len(tasks):
                print(f"simulator {index}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
