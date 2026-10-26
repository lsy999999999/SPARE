"""Re-evaluate archived capacity scans with outstanding requests at 60 s."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from research.www_mean_ttft_capacity_20261017 import analyze


ROOT = Path(__file__).resolve().parents[2]
SOURCE = analyze.OUT / "rate_audit.csv"
OUTPUT = ROOT / "artifacts/www_mean_ttft_capacity_backlog_0_60_20261018"


def read_rate_rows():
    rows = []
    with SOURCE.open(newline="") as stream:
        for item in csv.DictReader(stream):
            path = ROOT / item["rate_record_path"]
            record = json.loads(path.read_text())
            windows = record["rows"]
            assert len(windows) == 50 and len({r["name"] for r in windows}) == 50
            assert all(r["valid"] and "pending_end" in r for r in windows), path
            assert all(r["pending_end"] == r["arrivals_by_end"] - r["completed_by_end"]
                       for r in windows), path
            rate = float(item["rate_qps"])
            pending_end = sum(r["pending_end"] for r in windows) / len(windows)
            old_eligible = item["eligible"] == "True"
            eligible = (int(item["failed_samples"]) == 0
                        and float(item["p99_tbt_ms"]) <= 200
                        and pending_end <= max(2, rate))
            rows.append(dict(source=item["source"], kv_blocks=int(item["kv_blocks"]),
                             arm=item["arm"], rate_qps=rate,
                             mean_ttft_ms=float(item["mean_ttft_ms"]),
                             p99_tbt_ms=float(item["p99_tbt_ms"]),
                             old_backlog_drift=float(item["backlog_drift"]),
                             backlog_0_60=pending_end, drift_limit=max(2, rate),
                             old_eligible=old_eligible, eligible=eligible,
                             rate_record_path=item["rate_record_path"]))
    return rows


def first_failure_capacity(rows, limit_ms):
    capacity = 0.0
    first_fail = None
    reason = None
    previous = 0.0
    for row in sorted(rows, key=lambda r: r["rate_qps"]):
        if not row["eligible"] or row["mean_ttft_ms"] > limit_ms:
            first_fail = row["rate_qps"]
            reason = ("backlog" if row["backlog_0_60"] > row["drift_limit"]
                      else "tbt" if row["p99_tbt_ms"] > 200 else "mean_ttft")
            break
        previous = capacity
        capacity = row["rate_qps"]
    return dict(capacity_qps=capacity, first_failed_qps=first_fail,
                gap_qps=first_fail - capacity if first_fail is not None else None,
                failure_reason=reason, prior_rate_qps=previous)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    rate_rows = read_rate_rows()
    groups = defaultdict(list)
    for row in rate_rows:
        groups[row["source"], row["kv_blocks"], row["arm"]].append(row)
    kvs = sorted({key[1] for key in groups})
    assert len(kvs) == 27 and len(groups) == 27 * 2 * len(analyze.METHODS)

    source_rows = []
    comparisons = []
    old_checks = {(int(r["kv_blocks"]), int(r["mean_ttft_limit_ms"])): r
                  for r in csv.DictReader((analyze.OUT / "threshold_checks.csv").open(newline=""))}
    for limit in analyze.CHECK_LIMITS_MS:
        for kv in kvs:
            capacities = {}
            for arm in analyze.METHODS:
                results = {}
                for source in analyze.SOURCES:
                    result = first_failure_capacity(groups[source, kv, arm], limit)
                    source_rows.append(dict(kv_blocks=kv, mean_ttft_limit_ms=limit,
                                            source=source, arm=arm, **result))
                    results[source] = result
                capacities[arm] = sum(results[source]["capacity_qps"]
                                      for source in analyze.SOURCES) / 2
            best = max(capacities[a] for a in analyze.METHODS if a != "arrival")
            old = old_checks[kv, limit]
            comparisons.append(dict(kv_blocks=kv, mean_ttft_limit_ms=limit,
                                    spare_capacity_qps=capacities["arrival"],
                                    best_baseline_capacity_qps=best,
                                    spare_is_best=capacities["arrival"] >= best - 1e-9,
                                    higher_baselines="; ".join(
                                        analyze.LABELS[analyze.METHODS.index(a)]
                                        for a in analyze.METHODS if a != "arrival"
                                        and capacities[a] > capacities["arrival"] + 1e-9),
                                    old_spare_capacity_qps=float(old["spare_capacity_qps"]),
                                    old_best_baseline_capacity_qps=float(
                                        old["best_baseline_capacity_qps"]),
                                    old_spare_is_best=old["spare_is_best"] == "True"))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "rate_audit.csv", rate_rows)
    write_csv(OUTPUT / "source_boundaries.csv", source_rows)
    write_csv(OUTPUT / "threshold_checks.csv", comparisons)
    summary = dict(rate_points=len(rate_rows), groups=len(groups),
                   changed_eligibility=sum(r["old_eligible"] != r["eligible"]
                                           for r in rate_rows),
                   thresholds={str(limit): dict(
                       spare_best=sum(r["spare_is_best"] for r in comparisons
                                      if r["mean_ttft_limit_ms"] == limit),
                       old_spare_best=sum(r["old_spare_is_best"] for r in comparisons
                                          if r["mean_ttft_limit_ms"] == limit))
                       for limit in analyze.CHECK_LIMITS_MS},
                   boundary_gaps_at_2s=sorted({r["gap_qps"] for r in source_rows
                                               if r["mean_ttft_limit_ms"] == 2000
                                               and r["gap_qps"] is not None}),
                   note=("Sensitivity analysis only: arrivals begin after 0 s, so the "
                         "0-60 s delta equals all requests still in service at 60 s, "
                         "including normally running requests. Capacity is the last "
                         "continuously passing archived QPS; no new rates were simulated."))
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
