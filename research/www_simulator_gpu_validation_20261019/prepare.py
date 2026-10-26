"""Freeze a broad, balanced simulator versus RTX 4090 replay matrix."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/www_simulator_gpu_validation_20261019_r1"
SOURCE = ROOT / "artifacts/www_kv_capacity_dense_20261016_r1/design.json"
AUDIT = ROOT / "artifacts/www_mean_ttft_capacity_20261017_r2/rate_audit.csv"
KVS = (1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192)
METHODS = ("arrival", "native", "sarathi", "sola_rule", "uniboost_static",
           "past_future", "s3_proxy", "apt_serve_kv_rule")
WINDOWS = (2, 8, 14, 20)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    from research.www_capacity_pareto_20261015 import run as capacity

    if (OUT / "design.json").exists():
        raise RuntimeError("design already frozen")
    source = json.loads(SOURCE.read_text())
    samples = [s for s in source["samples"] if s["window"] in WINDOWS]
    assert len(samples) == 16
    assert all(sum(s["source"] == kind for s in samples) == 8
               for kind in ("code", "conversation"))
    wanted = {kind: {s["name"] for s in samples if s["source"] == kind}
              for kind in ("code", "conversation")}
    scans = defaultdict(list)
    with AUDIT.open(newline="") as stream:
        for row in csv.DictReader(stream):
            kind, kv, arm = row["source"], int(row["kv_blocks"]), row["arm"]
            if kv not in KVS or arm not in METHODS:
                continue
            path = ROOT / row["rate_record_path"]
            data = json.loads(path.read_text())
            rows = [r for r in data["rows"] if r["name"] in wanted[kind]]
            assert {r["name"] for r in rows} == wanted[kind]
            rate = float(row["rate_qps"])
            valid = all(r["valid"] for r in rows)
            mean = sum(r["mean_ttft_ms"] for r in rows) / 8 if valid else None
            tbt = sum(r["p99_tbt_ms"] for r in rows) / 8 if valid else None
            drift = sum(r["backlog_drift"] for r in rows) / 8 if valid else None
            passed = bool(valid and mean <= 2000 and tbt <= 200
                          and drift <= max(2, rate))
            scans[kind, kv, arm].append(dict(rate_qps=rate, passed=passed,
                                             valid=valid, mean_ttft_ms=mean,
                                             p99_tbt_ms=tbt, backlog_drift=drift,
                                             source_record=str(path.relative_to(ROOT))))

    plans = []
    inputs = {}
    for kind in ("code", "conversation"):
        for kv in KVS:
            for arm in METHODS:
                options = sorted(scans[kind, kv, arm], key=lambda x: x["rate_qps"])
                assert len(options) >= 2, (kind, kv, arm)
                boundary = next((i for i, point in enumerate(options)
                                 if not point["passed"]), len(options))
                if 0 < boundary < len(options):
                    selected = options[boundary - 1:boundary + 1]
                elif boundary == 0:
                    selected = options[:2]
                else:
                    selected = options[-2:]
                plans.append(dict(source=kind, kv_blocks=kv, arm=arm,
                                  simulator_points=selected))
                for sample in (s for s in samples if s["source"] == kind):
                    for point in selected:
                        rate = point["rate_qps"]
                        rel = f"inputs/{sample['name']}_r{rate:g}.json"
                        path = OUT / rel
                        if not path.exists():
                            write(path, capacity.generate(sample, rate))
                        inputs[rel] = sha(path)

    history = json.loads((ROOT / "artifacts/www_pareto_20261014_r1/history.json").read_text())
    histories = {}
    for kind in ("code", "conversation"):
        path = OUT / "history" / f"{kind}.json"
        write(path, dict(lengths=[r["output"] for r in history[kind]]))
        histories[kind] = sha(path)
    sources = ("research/www_simulator_gpu_validation_20261019/prepare.py",
               "research/www_simulator_gpu_validation_20261019/run.py",
               "research/www_idea1_gpu_pilot_20261016/run.py",
               "research/www_idea1_gpu_pilot_20261016/scheduler.py",
               "research/www_evaluation_gpu_completion_20261003/run.py",
               "research/www_evaluation_gpu_completion_20261003/scheduler.py",
               "research/www_legacy_baselines_20261006/scheduler.py")
    design = dict(schema=1, purpose="broad paired simulator and RTX 4090 replay",
                  samples=samples, kv_blocks=list(KVS), methods=list(METHODS),
                  plans=plans, repeats=1, input_hashes=inputs,
                  history_hashes=histories,
                  source_hashes={rel: sha(ROOT / rel) for rel in sources},
                  source_design_sha256=sha(SOURCE), rate_audit_sha256=sha(AUDIT),
                  horizon_ms=60000, warm_ms=20000,
                  backlog_interval_ms=[40000, 60000], mean_ttft_limit_ms=2000,
                  p99_tbt_limit_ms=200, backlog_limit="max(2, QPS)",
                  budget=1024, chunk=256, concurrency=32, block_size=16,
                  gpu_candidates=list(range(6)),
                  qps_rule="last simulator pass and first fail for the fixed eight-window cohort; endpoints if unbracketed")
    write(OUT / "design.json", design)
    print(f"Frozen {len(samples)} windows, {len(KVS)} KV levels, "
          f"{len(METHODS)} methods, {len(plans)*8*2} GPU episodes")


if __name__ == "__main__":
    main()
