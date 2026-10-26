"""Freeze date-disjoint Azure calibration and test windows before GPU outcomes."""
from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

from research.www_evaluation_completion_20261003.common import digest

ROOT = Path(__file__).resolve().parents[2]
G2 = ROOT / "artifacts/www_evaluation_gpu_completion_20261003_r1"
OUT = ROOT / "artifacts/www_plan_abc_20261008_r1/c"
HERE = Path(__file__).resolve().parent
BUDGETS = (128, 256, 384, 512, 768, 1024)
DATES = {
    "code": {"cal": ("2024-05-15",), "test": ("2024-05-10", "2024-05-12", "2024-05-14")},
    "conversation": {"cal": ("2024-05-17",), "test": ("2024-05-12", "2024-05-14", "2024-05-16")},
}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prior_ids():
    seen = set()
    scanned = []
    for pattern in ("v1*/*inputs/*.json", "www*/*inputs/*.json"):
        for path in sorted((ROOT / "artifacts").glob(pattern)):
            if OUT in path.parents:
                continue
            data = read(path)
            if isinstance(data, list):
                seen.update(str(r["id"]) for r in data if isinstance(r, dict) and "id" in r)
                scanned.append(str(path.relative_to(ROOT)))
    for path in (ROOT / "writing/www/experiments/inputs").glob("*.json"):
        seen.update(str(r["id"]) for r in read(path))
        scanned.append(str(path.relative_to(ROOT)))
    return seen, scanned


def main():
    assert not (OUT / "design.json").exists()
    seen, scanned = prior_ids()
    selected = set()
    windows = []
    provenance = {}
    for source, stages in DATES.items():
        csv_path = ROOT / f"data/azure_{'conv' if source == 'conversation' else 'code'}_1week.csv"
        dates = {date: [] for values in stages.values() for date in values}
        with csv_path.open(newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            for index, row in enumerate(reader):
                date = row[0][:10]
                if date not in dates or row[0][11:16] < "16:37" or len(dates[date]) >= 128:
                    continue
                prompt, output = int(row[1]), int(row[2])
                rid = f"{source}:raw:{index}"
                if rid in seen or rid in selected or not (prompt > 0 and 0 < output <= 512
                    and prompt + output <= 8000):
                    continue
                dates[date].append(dict(id=rid, prompt=prompt, output=output,
                                        raw_timestamp=row[0]))
                selected.add(rid)
        assert all(len(rows) == 128 for rows in dates.values()), (source, {d: len(r) for d, r in dates.items()})
        provenance[source] = dict(path=str(csv_path.relative_to(ROOT)), sha256=digest(csv_path),
                                  dates=stages)
        for stage, stage_dates in stages.items():
            for number, date in enumerate(stage_dates):
                window = f"{source}_{stage}{number}"
                raw_path = OUT / "raw_inputs" / f"{window}.json"
                write(raw_path, dates[date])
                for shape in ("steady", "burst_stop"):
                    path = OUT / "inputs" / f"{window}_{shape}.json"
                    write(path, [dict(id=r["id"], prompt=r["prompt"], output=r["output"],
                        arrival_ms=1000*i/5.25 if shape == "steady" else 0.)
                        for i, r in enumerate(dates[date])])
                    for capacity in (768, 2048):
                        windows.append(dict(key=f"{window}_{shape}_c{capacity}",
                            window=window, stage=stage, source=source, date=date,
                            shape=shape, capacity=capacity, budget=1024, chunk=256,
                            requests=128, input=str(path.relative_to(OUT)),
                            input_sha256=digest(path), history=f"history/{source}.json"))
    assert not selected & seen
    for source in DATES:
        old = next(c for c in read(G2 / "design.json")["cells"] if c["window"].startswith(source))
        target = OUT / "history" / f"{source}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(G2 / old["history_file"], target)
    for cell in windows:
        cell["history_sha256"] = digest(OUT / cell["history"])
    source_names = ("prepare_c.py", "c_policy.py", "c_scheduler.py", "run_c.py")
    design = dict(schema=1, created=time.time(), cells=windows, calibration_budgets=BUDGETS,
        cal_repeats=2, test_repeats=4, aa_repeats=2, aa_threshold=.05,
        cal_methods=[f"{kind}_{budget}" for kind in ("fixed", "guarded") for budget in BUDGETS],
        test_methods=["srp", "fixed_selected", "guarded_selected", "budget_only", "spare"],
        selection="equal cell mean TTFT across eight date-disjoint calibration cells; ties lower budget",
        inference_unit="six test dates/windows, with correlated shape and capacity cells",
        gpu_candidates=[1, 2, 6], episode_timeout_s=900,
        old_design_sha256=digest(G2 / "design.json"),
        provenance=provenance, overlap_audit=dict(prior_ids=len(seen), selected_ids=len(selected),
            overlap=0, scanned_files=len(scanned)),
        source_hashes={name: digest(HERE / name) for name in source_names})
    assert len([x for x in windows if x["stage"] == "cal"]) == 8
    assert len([x for x in windows if x["stage"] == "test"]) == 24
    write(OUT / "design.json", design)
    print(json.dumps(dict(cal_cells=8, test_cells=24, selected_ids=len(selected),
        prior_ids=len(seen), output=str(OUT))))


if __name__ == "__main__":
    main()
