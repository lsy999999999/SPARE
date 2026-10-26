"""Audit the earlier single-window GPU pilot against paired simulator records."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from research.www_capacity_pareto_20261015.run import generate


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "artifacts/www_idea1_gpu_pilot_20261016_r1"
AUDIT = ROOT / "artifacts/www_mean_ttft_capacity_20261017_r2/rate_audit.csv"
OUT = ROOT / "artifacts/www_simulator_gpu_validation_pilot_analysis_20261019"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def passed(row, rate):
    return (row["valid"] and row["mean_ttft_ms"] <= 2000
            and row["p99_tbt_ms"] <= 200
            and row["backlog_drift"] <= max(2, rate))


def margin(row, rate):
    return max(row["mean_ttft_ms"] / 2000,
               row["p99_tbt_ms"] / 200,
               row["backlog_drift"] / max(2, rate)) - 1


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        if not rows:
            return
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    design = read(PILOT / "design.json")
    assert {sample["name"] for sample in design["source_samples"]} == {
        "code_d1_w00", "conversation_d1_w00"}
    for rel, expected in design["input_hashes"].items():
        path = PILOT / rel
        assert sha(path) == expected, path
    for sample in design["source_samples"]:
        for rate in design["rates"]:
            path = PILOT / f"inputs/{sample['source']}_r{rate:g}.json"
            assert read(path) == generate(sample, rate), path

    lookup = {}
    for row in csv.DictReader(AUDIT.open()):
        lookup[row["source"], int(row["kv_blocks"]), row["arm"],
               float(row["rate_qps"])] = ROOT / row["rate_record_path"]

    archives = [ROOT / f"artifacts/{name}/rates" for name in (
        "www_mean_ttft_capacity_20261017_r2",
        "www_kv_capacity_dense_20261016_r1",
        "www_kv_capacity_50wide_20261016_r1",
        "www_kv_capacity_paper_20261015_r1")]

    pairs = []
    missing = []
    initialized = False
    gpu_keys = set()
    for path in sorted((PILOT / "rates").glob("*.json")):
        gpu = read(path)
        assert gpu["valid"]
        source, kv, arm, rate = (gpu["source"], gpu["kv"], gpu["method"], gpu["rate"])
        assert (source, kv, arm, rate) not in gpu_keys
        gpu_keys.add((source, kv, arm, rate))
        input_path = PILOT / f"inputs/{source}_r{rate:g}.json"
        assert gpu["input_sha256"] == design["input_hashes"][str(input_path.relative_to(PILOT))]
        sim_path = lookup.get((source, kv, arm, rate))
        if sim_path is None:
            filename = f"{source}_c{kv}_{arm}_r{rate:g}.json"
            sim_path = next((root / filename for root in archives
                             if (root / filename).exists()), None)
        if sim_path is None:
            from research.www_kv_extreme_stress_20261016 import run as fresh
            sim_path = OUT / "simulated_missing" / path.name
            if not sim_path.exists():
                if not initialized:
                    fresh.initialize()
                    initialized = True
                sample = next(s for s in design["source_samples"] if s["source"] == source)
                result = fresh.simulate_sample(sample, kv, arm, rate)
                sim_path.parent.mkdir(parents=True, exist_ok=True)
                sim_path.write_text(json.dumps({"rows": [result]}, indent=2) + "\n")
        matched = [r for r in read(sim_path)["rows"] if r["name"] == f"{source}_d1_w00"]
        assert len(matched) == 1 and matched[0]["capacity"] == kv
        sim = matched[0]
        assert sim["rate"] == rate and sim["arm"] == arm
        if not sim["valid"]:
            missing.append(dict(source=source, kv_blocks=kv, arm=arm, rate_qps=rate))
            continue
        sim_pass, gpu_pass = passed(sim, rate), passed(gpu, rate)
        label = "TP" if sim_pass and gpu_pass else "FP" if sim_pass else "FN" if gpu_pass else "TN"
        pairs.append(dict(source=source, kv_blocks=kv, arm=arm, rate_qps=rate,
                          sim_pass=sim_pass, gpu_pass=gpu_pass, outcome=label,
                          sim_margin=margin(sim, rate), gpu_margin=margin(gpu, rate),
                          sim_ttft_ms=sim["mean_ttft_ms"], gpu_ttft_ms=gpu["mean_ttft_ms"],
                          sim_tbt_ms=sim["p99_tbt_ms"], gpu_tbt_ms=gpu["p99_tbt_ms"],
                          sim_drift=sim["backlog_drift"], gpu_drift=gpu["backlog_drift"],
                          sim_preemptions=sim["preemptions"], gpu_preemptions=gpu["preemptions"],
                          sim_record=str(sim_path.relative_to(ROOT)),
                          gpu_record=str(path.relative_to(ROOT))))
    assert len(gpu_keys) == 1836
    write_csv(OUT / "pairs.csv", pairs)
    write_csv(OUT / "unmatched_gpu.csv", missing)

    by_group = defaultdict(list)
    for row in pairs:
        by_group[row["source"], row["kv_blocks"], row["arm"]].append(row)
    brackets = []
    for (source, kv, arm), rows in sorted(by_group.items()):
        record = dict(source=source, kv_blocks=kv, arm=arm, paired_rates=len(rows))
        for side in ("sim", "gpu"):
            ordered = sorted(rows, key=lambda row: row["rate_qps"])
            failed = [row["rate_qps"] for row in ordered if not row[f"{side}_pass"]]
            first_fail = min(failed) if failed else None
            passed_before = [row["rate_qps"] for row in ordered
                             if row[f"{side}_pass"] and
                             (first_fail is None or row["rate_qps"] < first_fail)]
            record[f"{side}_lower_qps"] = max(passed_before) if passed_before else 0.0
            record[f"{side}_upper_qps"] = first_fail if first_fail is not None else ""
            record[f"{side}_nonmonotonic"] = any(
                row[f"{side}_pass"] and first_fail is not None and
                row["rate_qps"] > first_fail for row in ordered)
            record[f"{side}_closed"] = bool(passed_before and first_fail is not None
                                              and not record[f"{side}_nonmonotonic"])
        brackets.append(record)
    write_csv(OUT / "capacity_brackets.csv", brackets)
    by_capacity = {(r["source"], r["kv_blocks"], r["arm"]): r for r in brackets}
    gains = []
    for spare in (r for r in brackets if r["arm"] == "arrival"):
        for baseline in design["methods"]:
            if baseline == "arrival":
                continue
            control = by_capacity.get((spare["source"], spare["kv_blocks"], baseline))
            if control is None:
                continue
            row = dict(source=spare["source"], kv_blocks=spare["kv_blocks"], baseline=baseline)
            for side in ("sim", "gpu"):
                a, b = spare, control
                closed = a[f"{side}_closed"] and b[f"{side}_closed"]
                row[f"{side}_closed"] = closed
                row[f"{side}_gain_lower_pct"] = (
                    100 * (a[f"{side}_lower_qps"] / b[f"{side}_upper_qps"] - 1)
                    if closed else "")
                row[f"{side}_gain_upper_pct"] = (
                    100 * (a[f"{side}_upper_qps"] / b[f"{side}_lower_qps"] - 1)
                    if closed else "")
                row[f"{side}_grid_gain_pct"] = (
                    100 * (a[f"{side}_lower_qps"] / b[f"{side}_lower_qps"] - 1)
                    if closed else "")
            gains.append(row)
    write_csv(OUT / "capacity_gains.csv", gains)

    by_arm = defaultdict(Counter)
    by_kv = defaultdict(Counter)
    for row in pairs:
        by_arm[row["arm"]][row["outcome"]] += 1
        by_kv[row["kv_blocks"]][row["outcome"]] += 1
    near = [r for r in pairs if abs(r["sim_margin"]) <= .2]
    summary = dict(total_gpu=len(gpu_keys), paired=len(pairs), unmatched=len(missing),
                   all=dict(Counter(r["outcome"] for r in pairs)),
                   near_20pct=dict(Counter(r["outcome"] for r in near)),
                   capacity_groups=len(brackets),
                   closed_gpu_brackets=sum(r["gpu_closed"] for r in brackets),
                   closed_sim_brackets=sum(r["sim_closed"] for r in brackets),
                   nonmonotonic_sim_groups=sum(r["sim_nonmonotonic"] for r in brackets),
                   nonmonotonic_gpu_groups=sum(r["gpu_nonmonotonic"] for r in brackets),
                   both_closed_comparisons=sum(r["sim_closed"] and r["gpu_closed"]
                                               for r in gains),
                   by_arm={k: dict(v) for k, v in sorted(by_arm.items())},
                   by_kv={k: dict(v) for k, v in sorted(by_kv.items())},
                   errors=[{k: v for k, v in r.items() if k not in ("sim_record", "gpu_record")}
                           for r in pairs if r["outcome"] in ("FP", "FN")])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("errors", "by_kv")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
