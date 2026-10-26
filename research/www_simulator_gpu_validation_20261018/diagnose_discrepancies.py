"""Inspect the two Code/2048 threshold disagreements per independent window."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/www_simulator_gpu_validation_20261018_analysis_r1"
GPU = ROOT / "artifacts/www_simulator_gpu_validation_20261018_r2"
DESIGN = json.loads((GPU / "design.json").read_text())


def main():
    rows = []
    for arm in ("arrival", "s3_proxy"):
        plan = next(p for p in DESIGN["plans"] if
                    (p["source"], p["kv_blocks"], p["arm"]) == ("code", 2048, arm))
        point = next(p for p in plan["simulator_bracket"] if p["rate_qps"] == 3)
        sim = {r["name"]: r for r in json.loads((ROOT / point["source_record"]).read_text())["rows"]}
        for sample in (s for s in DESIGN["samples"] if s["source"] == "code"):
            name = sample["name"]
            gpu = [json.loads((GPU / "runs" / f"{name}_c2048_{arm}_r3_rep{rep}.json").read_text())
                   for rep in range(2)]
            s = sim[name]
            rows.append(dict(arm=arm, sample=name,
                             sim_ttft_ms=s["mean_ttft_ms"],
                             gpu_ttft_ms=mean(r["mean_ttft_ms"] for r in gpu),
                             sim_drift=s["backlog_drift"],
                             gpu_drift=mean(r["backlog_drift"] for r in gpu),
                             sim_pending_40=s["pending_mid"],
                             gpu_pending_40=mean(r["pending_40s"] for r in gpu),
                             sim_pending_60=s["pending_end"],
                             gpu_pending_60=mean(r["pending_60s"] for r in gpu),
                             sim_preemptions=s["preemptions"],
                             gpu_preemptions=mean(r["preemptions"] for r in gpu),
                             sim_recomputed_tokens=s["recomputed_tokens"],
                             gpu_recomputed_tokens=mean(r["recomputed_tokens"] for r in gpu)))
    with (OUT / "discrepancy_diagnostics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for arm in ("arrival", "s3_proxy"):
        group = [r for r in rows if r["arm"] == arm]
        print(arm, {key: round(mean(r[key] for r in group), 2)
                    for key in ("sim_ttft_ms", "gpu_ttft_ms", "sim_drift",
                                "gpu_drift", "sim_pending_40", "gpu_pending_40",
                                "sim_pending_60", "gpu_pending_60",
                                "sim_preemptions", "gpu_preemptions")})


if __name__ == "__main__":
    main()
