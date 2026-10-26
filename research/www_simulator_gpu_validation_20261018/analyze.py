"""Compare frozen simulator brackets with paired RTX 4090 measurements."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/www_simulator_gpu_validation_20261018_r2"


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def classify(rows, limit, tbt_limit):
    valid = all(r["valid"] for r in rows)
    mean_ttft = statistics.fmean(r["mean_ttft_ms"] for r in rows) if valid else None
    p99_tbt = statistics.fmean(r["p99_tbt_ms"] for r in rows) if valid else None
    drift = statistics.fmean(r["backlog_drift"] for r in rows) if valid else None
    rate = rows[0]["rate_qps"]
    passed = (valid and mean_ttft <= limit and p99_tbt <= tbt_limit
              and drift <= max(2, rate))
    reason = ("invalid" if not valid else "mean_ttft" if mean_ttft > limit
              else "p99_tbt" if p99_tbt > tbt_limit
              else "backlog" if drift > max(2, rate) else "pass")
    return dict(valid=valid, passed=passed, mean_ttft_ms=mean_ttft,
                p99_tbt_ms=p99_tbt, backlog_drift=drift,
                failure_reason=reason, runs=len(rows))


def main():
    frozen = json.loads((OUT / "design.json").read_text())
    results = defaultdict(list)
    for path in (OUT / "runs").glob("*.json"):
        row = json.loads(path.read_text())
        results[row["source"], row["kv_blocks"], row["arm"], row["rate_qps"]].append(row)
    predicted = []
    measured = []
    brackets = []
    for plan in frozen["plans"]:
        key = (plan["source"], plan["kv_blocks"], plan["arm"])
        records = []
        for point in plan["simulator_bracket"]:
            rate = point["rate_qps"]
            rows = results[key + (rate,)]
            expected = 2 * frozen["repeats"]
            observed = (classify(rows, frozen["mean_ttft_limit_ms"],
                                 frozen["p99_tbt_limit_ms"])
                        if len(rows) == expected else None)
            records.append((rate, observed))
            predicted.append(dict(source=key[0], kv_blocks=key[1], arm=key[2],
                                  rate_qps=rate, simulator_passed=point["passed"],
                                  simulator_mean_ttft_ms=point["mean_ttft_ms"],
                                  simulator_p99_tbt_ms=point["p99_tbt_ms"],
                                  simulator_backlog_drift=point["backlog_drift"],
                                  gpu_complete=observed is not None,
                                  gpu_passed=observed["passed"] if observed else None,
                                  gpu_mean_ttft_ms=observed["mean_ttft_ms"] if observed else None,
                                  gpu_p99_tbt_ms=observed["p99_tbt_ms"] if observed else None,
                                  gpu_backlog_drift=observed["backlog_drift"] if observed else None,
                                  gpu_failure_reason=observed["failure_reason"] if observed else None))
            if observed:
                measured.append(dict(source=key[0], kv_blocks=key[1], arm=key[2],
                                     rate_qps=rate, **observed))
        complete = all(row is not None for _, row in records)
        brackets.append(dict(source=key[0], kv_blocks=key[1], arm=key[2],
                             simulator_last_pass_qps=records[0][0],
                             simulator_first_fail_qps=records[1][0],
                             gpu_lower_passed=(records[0][1]["passed"] if complete else None),
                             gpu_upper_passed=(records[1][1]["passed"] if complete else None),
                             gpu_bracket_confirmed=(records[0][1]["passed"]
                                                    and not records[1][1]["passed"]
                                                    if complete else None)))
    write_csv(OUT / "prediction_vs_gpu.csv", predicted)
    write_csv(OUT / "gpu_rates.csv", measured)
    write_csv(OUT / "brackets.csv", brackets)
    print(json.dumps(dict(expected_runs=len(frozen["samples"]) * 2 * 2
                          * frozen["repeats"] * 2,
                          completed_runs=sum(len(v) for v in results.values()),
                          confirmed_brackets=sum(r["gpu_bracket_confirmed"] is True
                                                 for r in brackets),
                          total_brackets=len(brackets)), indent=2))


if __name__ == "__main__":
    main()
