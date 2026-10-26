"""Summarize paired held-out simulator and GPU cross-model results."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/www_cross_model_simulator_validation_20261022_r3"


def summarize(rows: list[dict]) -> dict:
    labels = [(r["gpu_passed"], r["simulator_passed"]) for r in rows]
    return dict(
        n=len(rows),
        agreement=sum(a == b for a, b in labels),
        gpu_pass=sum(a for a, _ in labels),
        simulator_pass=sum(b for _, b in labels),
        gpu_pass_simulator_fail=sum(a and not b for a, b in labels),
        gpu_fail_simulator_pass=sum(not a and b for a, b in labels),
        mean_ttft_mape_pct=statistics.fmean(r["mean_ttft_ape_pct"] for r in rows),
        median_ttft_ape_pct=statistics.median(r["mean_ttft_ape_pct"] for r in rows),
        mean_tbt_mape_pct=statistics.fmean(r["p99_tbt_ape_pct"] for r in rows),
        mean_backlog_absolute_error=statistics.fmean(
            r["backlog_absolute_error"] for r in rows),
    )


def main() -> None:
    paths = sorted((SOURCE / "runs").glob("*.json"))
    if len(paths) != 596:
        raise RuntimeError(f"expected 596 held-out paired runs, found {len(paths)}")
    rows = []
    for path in paths:
        entry = json.loads(path.read_text())
        gpu, sim = entry["gpu"], entry["simulator"]
        rows.append(dict(
            model=entry["model"], source=entry["source"], sample=entry["sample"],
            kv_blocks=entry["kv_blocks"], arm=entry["arm"],
            rate_qps=entry["rate_qps"], gpu_passed=gpu["passed"],
            simulator_passed=sim["passed"],
            gpu_mean_ttft_ms=gpu["mean_ttft_ms"],
            simulator_mean_ttft_ms=sim["mean_ttft_ms"],
            mean_ttft_ape_pct=100*abs(sim["mean_ttft_ms"]-gpu["mean_ttft_ms"])/gpu["mean_ttft_ms"],
            gpu_p99_tbt_ms=gpu["p99_tbt_ms"],
            simulator_p99_tbt_ms=sim["p99_tbt_ms"],
            p99_tbt_ape_pct=100*abs(sim["p99_tbt_ms"]-gpu["p99_tbt_ms"])/gpu["p99_tbt_ms"],
            gpu_backlog_drift=gpu["backlog_drift"],
            simulator_backlog_drift=sim["backlog_drift"],
            backlog_absolute_error=abs(sim["backlog_drift"]-gpu["backlog_drift"]),
            gpu_preemptions=gpu["preemptions"],
            simulator_preemptions=sim["preemptions"],
            gpu_recomputed_tokens=gpu["recomputed_tokens"],
            simulator_recomputed_tokens=sim["recomputed_tokens"],
        ))
    with (SOURCE / "paired.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        for key in (("model", row["model"]), ("model_source", row["model"], row["source"]),
                    ("arm", row["arm"]), ("model_arm", row["model"], row["arm"]),
                    ("outcome", "gpu_pass" if row["gpu_passed"] else "gpu_fail")):
            groups[key].append(row)
    report = dict(overall=summarize(rows), groups={"/".join(key): summarize(value)
                   for key, value in sorted(groups.items())})
    (SOURCE / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
