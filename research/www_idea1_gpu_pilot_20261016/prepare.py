"""Freeze a small real-GPU version of the idea1 KV/capacity scan."""

import hashlib
import json
from pathlib import Path
import shutil

from research.www_capacity_pareto_20261015 import run as capacity


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/www_idea1_gpu_pilot_20261016_r1"
PAPER = ROOT / "artifacts/www_kv_capacity_paper_20261015_r1"
CAPACITIES = (672, 768, 896) + tuple(range(1024, 7425, 256))
RATES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
METHODS = (
    "arrival", "native", "sarathi", "sola_rule", "uniboost_static",
    "past_future", "s3_proxy", "apt_serve_kv_rule",
)
SOURCE_DIRS = (
    "v1_budget_20260922", "v1_saving_threshold_20260924",
    "v1_broad12h_20260925", "v1_paper_comparison_20260922",
    "www_evaluation_gpu_completion_20261003", "www_legacy_baselines_20261006",
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (OUT / "launch.json").exists():
        raise RuntimeError("cannot change design after launch")
    source = json.loads((PAPER / "design.json").read_text())
    history = json.loads((ROOT / "artifacts/www_pareto_20261014_r1/history.json").read_text())
    samples = [next(s for s in source["samples"] if s["name"] == f"{name}_d1_w00")
               for name in ("code", "conversation")]
    for folder in SOURCE_DIRS:
        source_root = PAPER / "source/research" / folder
        for path in source_root.glob("*.py"):
            target = OUT / "source/research" / folder / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    source_hashes = {}
    for path in (OUT / "source").rglob("*.py"):
        source_hashes[str(path.relative_to(OUT))] = digest(path)
    for name in ("scheduler.py", "run.py", "status.py"):
        path = Path(__file__).with_name(name)
        source_hashes[str(path.relative_to(ROOT))] = digest(path)
    inputs = {}
    for sample in samples:
        key = sample["source"]
        write(OUT / "history" / f"{key}.json",
              {"lengths": [r["output"] for r in history[key]]})
        for rate in RATES:
            path = OUT / "inputs" / f"{key}_r{rate:g}.json"
            write(path, capacity.generate(sample, rate))
            inputs[str(path.relative_to(OUT))] = digest(path)
    design = dict(schema=1, experiment="real 4090 GPU, one window per workload",
                  source_samples=samples, capacities=CAPACITIES, rates=RATES,
                  methods=METHODS, repeats=1, budget=1024, chunk=256,
                  concurrency=32, block_size=16, max_model_len=8176,
                  horizon_ms=60000, warm_ms=20000, backlog_check_ms=40000,
                  p95_ttft_limit_ms=12000, p99_tbt_limit_ms=200,
                  backlog_limit="max(2, offered QPS)",
                  model="Qwen2.5-7B-Instruct", dtype="float16",
                  gpu_candidates=(0, 1, 2, 3, 4, 5),
                  source_hashes=source_hashes, input_hashes=inputs,
                  history_hashes={s["source"]: digest(OUT / "history" / f"{s['source']}.json")
                                  for s in samples},
                  protocol="Paired methods and rate grid within each source/KV; fixed 60 s arrivals, drain to completion. Capacity is largest passing grid point before first failure. No statistical repeat.")
    write(OUT / "design.json", design)
    print(f"prepared {len(samples)} sources, {len(CAPACITIES)} KV, {len(METHODS)} methods")


if __name__ == "__main__":
    main()
