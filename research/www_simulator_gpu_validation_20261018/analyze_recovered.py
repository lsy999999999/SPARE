"""Audit completed simulator/GPU capacity brackets across all methods."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from research.www_simulator_gpu_validation_20261018.analyze import classify


ROOT = Path(__file__).resolve().parents[2]
RUNS = (
    ROOT / "artifacts/www_simulator_gpu_validation_20261018_r2",
    ROOT / "artifacts/www_simulator_gpu_validation_sarathi_20261018_r1",
)
OUT = ROOT / "artifacts/www_simulator_gpu_validation_20261018_analysis_r1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    points = []
    brackets = []
    total = 0
    for directory in RUNS:
        design = json.loads((directory / "design.json").read_text())
        grouped = defaultdict(list)
        for path in (directory / "runs").glob("*.json"):
            row = json.loads(path.read_text())
            key = (row["source"], row["kv_blocks"], row["arm"], row["rate_qps"])
            rel = f"inputs/{row['sample']}_r{row['rate_qps']:g}.json"
            assert row["input_sha256"] == design["input_hashes"][rel]
            assert sha(directory / row["raw_path"]) == row["raw_sha256"]
            grouped[key].append(row)
            total += 1
        assert total <= 192
        for plan in design["plans"]:
            common = dict(source=plan["source"], kv_blocks=plan["kv_blocks"],
                          arm=plan["arm"])
            outcomes = []
            for point in plan["simulator_bracket"]:
                rate = point["rate_qps"]
                rows = grouped[common["source"], common["kv_blocks"],
                               common["arm"], rate]
                expected_samples = {s["name"] for s in design["samples"]
                                    if s["source"] == common["source"]}
                assert {(r["sample"], r["repeat"]) for r in rows} == {
                    (sample, repeat) for sample in expected_samples
                    for repeat in range(design["repeats"])}
                gpu = classify(rows, design["mean_ttft_limit_ms"],
                               design["p99_tbt_limit_ms"])
                outcomes.append(gpu["passed"])
                points.append(dict(**common, rate_qps=rate,
                                   simulator_passed=point["passed"],
                                   gpu_passed=gpu["passed"],
                                   simulator_mean_ttft_ms=point["mean_ttft_ms"],
                                   gpu_mean_ttft_ms=gpu["mean_ttft_ms"],
                                   simulator_p99_tbt_ms=point["p99_tbt_ms"],
                                   gpu_p99_tbt_ms=gpu["p99_tbt_ms"],
                                   simulator_backlog_drift=point["backlog_drift"],
                                   gpu_backlog_drift=gpu["backlog_drift"],
                                   gpu_failure_reason=gpu["failure_reason"],
                                   individual_gpu_passes=sum(r["passed"] for r in rows),
                                   gpu_runs=len(rows)))
            category = ("confirmed" if outcomes == [True, False]
                        else "gpu_higher" if outcomes == [True, True]
                        else "gpu_lower" if outcomes == [False, False]
                        else "nonmonotonic")
            brackets.append(dict(**common,
                                 simulator_last_pass_qps=plan["simulator_bracket"][0]["rate_qps"],
                                 simulator_first_fail_qps=plan["simulator_bracket"][1]["rate_qps"],
                                 gpu_lower_passed=outcomes[0],
                                 gpu_upper_passed=outcomes[1],
                                 category=category))
        expected = (len(design["samples"]) * design["repeats"] * 2
                    * sum(len(arms) for arms in design["methods_by_kv"].values()))
        assert sum(len(v) for v in grouped.values()) == expected
    write_csv(OUT / "points.csv", points)
    write_csv(OUT / "brackets.csv", brackets)
    print(json.dumps(dict(verified_gpu_runs=total,
                          matched_points=sum(p["simulator_passed"] == p["gpu_passed"]
                                             for p in points),
                          total_points=len(points),
                          categories={category: sum(b["category"] == category
                                                    for b in brackets)
                                      for category in sorted({b["category"] for b in brackets})}),
                     indent=2))


if __name__ == "__main__":
    main()
