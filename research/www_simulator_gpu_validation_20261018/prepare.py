"""Freeze independent test windows and QPS brackets before GPU measurement."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from research.www_capacity_pareto_20261015 import run as capacity
from research.www_mean_ttft_capacity_20261017 import analyze


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/www_simulator_gpu_validation_20261018_r2"
SOURCE_DESIGN = ROOT / "artifacts/www_kv_capacity_dense_20261016_r1/design.json"
SOURCE_AUDIT = analyze.OUT / "rate_audit.csv"
METHODS = {2048: ("arrival", "s3_proxy"),
           6144: ("arrival", "apt_serve_kv_rule")}
WINDOWS = (7, 17)
REPEATS = 2


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def selected_samples():
    design = json.loads(SOURCE_DESIGN.read_text())
    samples = [s for s in design["samples"] if s["window"] in WINDOWS]
    assert len(samples) == 8
    assert all(sum(s["source"] == source for s in samples) == 4
               for source in ("code", "conversation"))
    assert len({(s["source"], s["day"]) for s in samples}) == 4
    return samples


def bracket(scans):
    last_pass = None
    first_fail = None
    for rate, rows, path in sorted(scans):
        valid = all(r["valid"] for r in rows)
        mean_ttft = (sum(r["mean_ttft_ms"] for r in rows) / len(rows)
                     if valid else None)
        p99_tbt = (sum(r["p99_tbt_ms"] for r in rows) / len(rows)
                   if valid else None)
        drift = (sum(r["backlog_drift"] for r in rows) / len(rows)
                 if valid else None)
        passed = (valid and mean_ttft <= 2000 and p99_tbt <= 200
                  and drift <= max(2, rate))
        entry = dict(rate_qps=rate, passed=passed, valid=valid,
                     mean_ttft_ms=mean_ttft, p99_tbt_ms=p99_tbt,
                     backlog_drift=drift, source_record=str(path.relative_to(ROOT)))
        if passed and first_fail is None:
            last_pass = entry
        elif first_fail is None:
            first_fail = entry
            break
    assert last_pass and first_fail, (last_pass, first_fail)
    return [last_pass, first_fail]


def main():
    if (OUT / "design.json").exists():
        raise RuntimeError("design already frozen")
    samples = selected_samples()
    by_source = defaultdict(set)
    for sample in samples:
        by_source[sample["source"]].add(sample["name"])

    scans = defaultdict(list)
    with SOURCE_AUDIT.open(newline="") as stream:
        for row in csv.DictReader(stream):
            kv = int(row["kv_blocks"])
            arm = row["arm"]
            if kv not in METHODS or arm not in METHODS[kv]:
                continue
            path = ROOT / row["rate_record_path"]
            data = json.loads(path.read_text())
            chosen = [r for r in data["rows"] if r["name"] in by_source[row["source"]]]
            assert len(chosen) == 4 and len({r["name"] for r in chosen}) == 4
            scans[row["source"], kv, arm].append((float(row["rate_qps"]), chosen, path))

    plans = []
    inputs = {}
    simulator = None
    for source in ("code", "conversation"):
        for kv, arms in METHODS.items():
            for arm in arms:
                try:
                    points = bracket(scans[source, kv, arm])
                except AssertionError:
                    if simulator is None:
                        from research.www_kv_extreme_stress_20261016 import run as simulator
                        simulator.initialize()
                    rate = max(entry[0] for entry in scans[source, kv, arm])
                    while rate < 8:
                        rate = round(rate + .25, 2)
                        chosen_samples = [s for s in samples if s["source"] == source]
                        rows = [simulator.simulate_sample(s, kv, arm, rate)
                                for s in chosen_samples]
                        path = OUT / "supplemental_simulation" / f"{source}_c{kv}_{arm}_r{rate:g}.json"
                        write(path, dict(rows=rows, rate_qps=rate))
                        scans[source, kv, arm].append((rate, rows, path))
                        try:
                            points = bracket(scans[source, kv, arm])
                            break
                        except AssertionError:
                            continue
                    else:
                        raise RuntimeError(f"no simulator failure through 8 QPS: {source} {kv} {arm}")
                plans.append(dict(source=source, kv_blocks=kv, arm=arm,
                                  simulator_bracket=points))
                for sample in (s for s in samples if s["source"] == source):
                    for point in points:
                        rate = point["rate_qps"]
                        path = OUT / "inputs" / f"{sample['name']}_r{rate:g}.json"
                        if not path.exists():
                            write(path, capacity.generate(sample, rate))
                        inputs[str(path.relative_to(OUT))] = sha(path)

    history = json.loads((ROOT / "artifacts/www_pareto_20261014_r1/history.json").read_text())
    for source in ("code", "conversation"):
        write(OUT / "history" / f"{source}.json",
              dict(lengths=[r["output"] for r in history[source]]))

    sources = ("research/www_simulator_gpu_validation_20261018/prepare.py",
               "research/www_simulator_gpu_validation_20261018/run.py",
               "research/www_simulator_gpu_validation_20261018/analyze.py",
               "research/www_idea1_gpu_pilot_20261016/run.py",
               "research/www_idea1_gpu_pilot_20261016/scheduler.py",
               "research/www_evaluation_gpu_completion_20261003/run.py",
               "research/www_evaluation_gpu_completion_20261003/scheduler.py",
               "research/www_legacy_baselines_20261006/scheduler.py")
    design = dict(schema=1, purpose="CPU simulator capacity versus independent RTX 4090 replay",
                  gpu_execution=True, model="Qwen2.5-7B-Instruct", gpu="RTX 4090",
                  samples=samples, methods_by_kv={str(k): list(v) for k, v in METHODS.items()},
                  plans=plans, repeats=REPEATS, input_hashes=inputs,
                  history_hashes={source: sha(OUT / "history" / f"{source}.json")
                                  for source in ("code", "conversation")},
                  source_hashes={path: sha(ROOT / path) for path in sources},
                  source_design_sha256=sha(SOURCE_DESIGN),
                  source_audit_sha256=sha(SOURCE_AUDIT),
                  timing_profile_sha256=sha(ROOT / "artifacts/www_pareto_20261014_r1/timing_profile.json"),
                  horizon_ms=60000, warm_ms=20000, backlog_interval_ms=[40000, 60000],
                  mean_ttft_limit_ms=2000, p99_tbt_limit_ms=200,
                  backlog_limit="max(2, offered QPS) requests",
                  budget=1024, chunk=256, concurrency=32, block_size=16,
                  max_num_seqs=64, max_num_batched_tokens=2048,
                  qps_rule="two archived simulator rates bracketing each paired-window 2 s boundary",
                  gpu_candidates=[4])
    write(OUT / "design.json", design)
    for plan in plans:
        print(plan["source"], plan["kv_blocks"], plan["arm"],
              [(p["rate_qps"], p["passed"], round(p["mean_ttft_ms"] or 0))
               for p in plan["simulator_bracket"]])


if __name__ == "__main__":
    main()
