"""Keep G4 capacity and sustained-load applicability results separate."""
from __future__ import annotations

import csv
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
G4 = ROOT / "artifacts/www_capacity_load_20261007_r1"
OUT = ROOT / "artifacts/www_plan_abc_20261008_r1/g4_scope.json"


def ci(values, seed):
    rng = random.Random(seed)
    values = list(values)
    samples = sorted(st.fmean(rng.choice(values) for _ in values) for _ in range(10000))
    return [samples[249], samples[9749]]


def main():
    design = json.loads((G4 / "design.json").read_text())
    published = json.loads((G4 / "analysis/summary.json").read_text())
    assert published["audited"] and published["runs"] == 2640 and published["groups"] == 132
    with (G4 / "analysis/runs.csv").open(newline="") as handle:
        runs = list(csv.DictReader(handle))
    assert len(runs) == 2640
    grouped = defaultdict(list)
    for row in runs:
        grouped[row["cell"], row["arm"]].append(float(row["ttft_ms_mean"]))
    rows = []
    for cell in design["cells"]:
        for reference in ("srp", "fixed256"):
            ours = grouped[cell["key"], "arrival"]
            other = grouped[cell["key"], reference]
            assert len(ours) == len(other) == 4
            a, b = st.fmean(ours), st.fmean(other)
            rows.append(dict(family=cell["family"], window=cell["window"],
                capacity=cell["capacity"], shape=cell["shape"], reference=reference,
                improvement_pct=100*(b-a)/b))
    results = []
    for family in ("capacity", "load"):
        for reference in ("srp", "fixed256"):
            subset = [r for r in rows if r["family"] == family and r["reference"] == reference]
            by_window = defaultdict(list)
            for row in subset:
                by_window[row["window"]].append(row["improvement_pct"])
            window_means = [st.fmean(x) for x in by_window.values()]
            results.append(dict(family=family, reference=reference, cells=len(subset),
                improvement_pct=st.fmean(r["improvement_pct"] for r in subset),
                window_bootstrap_95ci_pct=ci(window_means, 20261008+len(results)),
                wins=sum(r["improvement_pct"] > 0 for r in subset),
                losses=sum(r["improvement_pct"] < 0 for r in subset)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(source="G4 completed audited summary and runs.csv",
        interpretation="Same six G2 source windows; capacity and load cells are separate correlated families",
        effects=results), indent=2) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
