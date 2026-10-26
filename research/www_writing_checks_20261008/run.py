"""Frozen exploratory burst-plus-background replay for writing questions."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
import statistics as st
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OLD = ROOT / "artifacts/www_evaluation_completion_20261003_r1"
OUT = ROOT / "artifacts/www_writing_checks_20261008_r1"
PROFILES = ["resource", "resource_staircase", "gpu_fitted"]
ARMS = ["spare", "srp", "fixed256", "fixed512"]
CAPACITIES = [768, 2048, 4096]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csvwrite(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def execute(task):
    from research.www_evaluation_completion_20261003.simulate import simulate
    from research.www_evaluation_completion_20261003.timing import FittedTiming
    name = f"{task['window']}__c{task['capacity']}__{task['profile']}__{task['arm']}"
    dest = OUT / "runs" / name
    if (dest / "result.json").exists():
        result = read(dest / "result.json")
        if result.get("writing_task") == task and result.get("complete"):
            return dict(task, **result["metrics"])
    try:
        options = {}
        arm = "srp" if task["arm"] == "srp" else "arrival"
        if task["arm"].startswith("fixed"):
            options = dict(mode="fixed", fixed_budget=int(task["arm"][5:]))
        fitted = FittedTiming(read(OLD / "timing_profile.json")) if task["profile"] == "gpu_fitted" else None
        result = simulate(read(OUT / "inputs" / f"{task['window']}.json"),
                          arm=arm, capacity=task["capacity"],
                          profile="resource" if fitted else task["profile"],
                          fitted_timing=fitted, policy_options=options, output=dest)
        result.update(writing_task=task, timing_profile_sha256=digest(OLD / "timing_profile.json"))
        write(dest / "result.json", result)
        return dict(task, **result["metrics"])
    except Exception:
        write(dest / "error.json", dict(task=task, error=traceback.format_exc()))
        return dict(task, failed=True)


def main():
    from research.www_evaluation_completion_20261003.simulate import simulate
    from research.www_evaluation_completion_20261003.timing import FittedTiming
    OUT.mkdir(parents=True, exist_ok=True)
    old = read(OLD / "runs/core__code_new0_burst_stop__c768__gpu_fitted__spare/result.json")
    check = simulate(read(OLD / "inputs/code_new0_burst_stop.json"),
                     fitted_timing=FittedTiming(read(OLD / "timing_profile.json")),
                     arm="arrival", capacity=768, output=OUT / "reproduction_check")
    assert old["trajectory_sha256"] == check["trajectory_sha256"], "Archived trajectory mismatch"
    assert abs(old["metrics"]["ttft_ms_mean"]-check["metrics"]["ttft_ms_mean"]) < 1e-8
    print("Frozen replay matches archived trajectory and TTFT.", flush=True)
    tasks = []
    for path in sorted((OLD / "raw_inputs").glob("*.json")):
        requests = [dict(id=r["id"], prompt=r["prompt"], output=r["output"],
                         arrival_ms=0. if i < 32 else (i-31)*1000/5.25)
                    for i, r in enumerate(read(path)[:128])]
        write(OUT / "inputs" / path.name, requests)
        for cap in CAPACITIES:
            for profile in PROFILES:
                for arm in ARMS:
                    tasks.append(dict(window=path.stem, capacity=cap, profile=profile, arm=arm))
    assert len(tasks) == 216
    write(OUT / "design.json", dict(tasks=tasks, profiles=PROFILES,
          shape="32 requests at zero, then 96 at 5.25 req/s; 128 per window",
          timing_sha256=digest(OLD / "timing_profile.json"),
          protocol_sha256=digest(Path(__file__).with_name("PROTOCOL_CN.md")),
          runner_sha256=digest(Path(__file__)), source_sha256=check["source_sha256"],
          input_sha256={p.name: digest(p) for p in sorted((OUT / "inputs").glob("*.json"))}))
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(execute, t) for t in tasks]
        for f in as_completed(futures):
            row = f.result()
            (failures if row.get("failed") else results).append(row)
            done = len(results)+len(failures)
            write(OUT / "status.json", dict(completed=len(results), failed=len(failures), expected=len(tasks)))
            if done % 12 == 0:
                print(f"{done}/216 processed; {len(failures)} failures", flush=True)
    write(OUT / "failures.json", failures)
    csvwrite(OUT / "metrics.csv", sorted(results, key=lambda r: (r["profile"], r["window"], r["capacity"], r["arm"])))
    by = {(r["window"], r["capacity"], r["profile"], r["arm"]): r for r in results}
    pairs = []
    for window in sorted({r["window"] for r in results}):
        for cap in CAPACITIES:
            for profile in PROFILES:
                for ref in ARMS[1:]:
                    a, c = by.get((window, cap, profile, "spare")), by.get((window, cap, profile, ref))
                    if a and c:
                        pairs.append(dict(window=window, capacity=cap, profile=profile, reference=ref,
                                          reduction_pct=100*(1-a["ttft_ms_mean"]/c["ttft_ms_mean"])))
    csvwrite(OUT / "paired.csv", pairs)
    summary = []
    for profile in PROFILES:
        for ref in ARMS[1:]:
            vals = [r["reduction_pct"] for r in pairs if r["profile"] == profile and r["reference"] == ref]
            summary.append(dict(profile=profile, reference=ref, scenarios=len(vals),
                                mean_reduction_pct=st.fmean(vals), wins=sum(v>1e-9 for v in vals),
                                ties=sum(abs(v)<=1e-9 for v in vals), losses=sum(v < -1e-9 for v in vals),
                                min_reduction_pct=min(vals), max_reduction_pct=max(vals)))
    write(OUT / "summary.json", dict(complete=not failures and len(results)==216,
                                   expected=216, runs=len(results), summary=summary))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
