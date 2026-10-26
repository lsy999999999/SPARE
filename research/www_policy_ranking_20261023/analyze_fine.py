"""Summarize fixed-QPS SPARE/SRP ordering on the targeted fine grid."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/www_policy_ranking_fidelity_20261023_r1"
RATES = (2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5)
ARMS = ("arrival", "srp")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    design = json.loads((OUT / "fine_design.json").read_text())
    windows = design["windows"]
    missing = []
    paired = []
    for window in windows:
        sample = f"conversation_{window}"
        for rate in RATES:
            for arm in ARMS:
                label = f"qwen3_8b_{sample}_c2048_{arm}_r{rate:g}"
                gpu_path = OUT / "runs" / f"{label}.json"
                sim_path = OUT / "sim" / f"{label}.json"
                if not gpu_path.exists() or not sim_path.exists():
                    missing.append(label)
                    continue
                gpu = json.loads(gpu_path.read_text())
                sim = json.loads(sim_path.read_text())
                if gpu["status"] != "complete" or not gpu["valid"]:
                    raise RuntimeError(f"invalid GPU result: {gpu_path}")
                if gpu["input_sha256"] != sim["input_sha256"]:
                    raise RuntimeError(f"request stream mismatch: {label}")
                sim_metrics = sim["metrics"]
                paired.append(dict(window=window, rate_qps=rate, arm=arm,
                                   new_rate=rate in design["rates_qps"],
                                   gpu_ttft_ms=gpu["mean_ttft_ms"],
                                   sim_ttft_ms=sim_metrics["mean_ttft_ms"],
                                   gpu_tbt_ms=gpu["p99_tbt_ms"],
                                   sim_tbt_ms=sim_metrics["p99_tbt_ms"],
                                   gpu_drift=gpu["backlog_drift"],
                                   sim_drift=sim_metrics["backlog_drift"],
                                   gpu_pass=gpu["passed"],
                                   sim_pass=sim_metrics["passed"]))
    if missing:
        print(f"Missing {len(missing)} of {len(windows) * len(RATES) * len(ARMS)} paired points")
        return
    write_csv(OUT / "fine_paired.csv", paired)

    by_rate = defaultdict(dict)
    for row in paired:
        by_rate[(row["window"], row["rate_qps"])][row["arm"]] = row
    window_comparisons = []
    for (window, rate), arms in sorted(by_rate.items()):
        spare, srp = arms["arrival"], arms["srp"]
        window_comparisons.append(dict(
            window=window, rate_qps=rate, new_rate=spare["new_rate"],
            gpu_spare_ttft_ms=spare["gpu_ttft_ms"],
            gpu_srp_ttft_ms=srp["gpu_ttft_ms"],
            sim_spare_ttft_ms=spare["sim_ttft_ms"],
            sim_srp_ttft_ms=srp["sim_ttft_ms"],
            gpu_spare_faster=spare["gpu_ttft_ms"] < srp["gpu_ttft_ms"],
            sim_spare_faster=spare["sim_ttft_ms"] < srp["sim_ttft_ms"],
            gpu_spare_pass=spare["gpu_pass"],
            gpu_srp_pass=srp["gpu_pass"],
            sim_spare_pass=spare["sim_pass"],
            sim_srp_pass=srp["sim_pass"],
        ))
    write_csv(OUT / "fine_window_order.csv", window_comparisons)

    pooled = []
    for rate in RATES:
        item = dict(rate_qps=rate, new_rate=rate in design["rates_qps"])
        for arm in ARMS:
            rows = [row for row in paired if row["rate_qps"] == rate
                    and row["arm"] == arm]
            for side in ("gpu", "sim"):
                ttft = statistics.fmean(row[f"{side}_ttft_ms"] for row in rows)
                tbt = statistics.fmean(row[f"{side}_tbt_ms"] for row in rows)
                drift = statistics.fmean(row[f"{side}_drift"] for row in rows)
                prefix = f"{side}_{arm}"
                item[f"{prefix}_ttft_ms"] = ttft
                item[f"{prefix}_tbt_ms"] = tbt
                item[f"{prefix}_drift"] = drift
                item[f"{prefix}_pass"] = (ttft <= 2000 and tbt <= 200
                                           and drift <= max(2, rate))
        for side in ("gpu", "sim"):
            item[f"{side}_spare_faster"] = (
                item[f"{side}_arrival_ttft_ms"] < item[f"{side}_srp_ttft_ms"])
            item[f"{side}_spare_only_pass"] = (
                item[f"{side}_arrival_pass"] and not item[f"{side}_srp_pass"])
        pooled.append(item)
    write_csv(OUT / "fine_pooled_order.csv", pooled)

    new_windows = [row for row in window_comparisons if row["new_rate"]]
    new_pooled = [row for row in pooled if row["new_rate"]]
    summary = dict(
        new_paired_points=len(new_windows) * 2,
        new_window_comparisons=len(new_windows),
        new_ttft_order_agreement=sum(row["gpu_spare_faster"] == row["sim_spare_faster"]
                                     for row in new_windows),
        new_pooled_ttft_order_agreement=sum(
            row["gpu_spare_faster"] == row["sim_spare_faster"] for row in new_pooled),
        new_pooled_spare_only_pass_gpu=[row["rate_qps"] for row in new_pooled
                                        if row["gpu_spare_only_pass"]],
        new_pooled_spare_only_pass_sim=[row["rate_qps"] for row in new_pooled
                                        if row["sim_spare_only_pass"]],
    )
    (OUT / "fine_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
