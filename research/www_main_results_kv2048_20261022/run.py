"""Replay the frozen paper-baseline test windows at 2048 KV blocks."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from research.www_pareto_papers_20261014 import run as papers
from research.www_pareto_20261014.run import ROOT, read


OUT = ROOT / "artifacts/www_main_results_kv2048_20261022_r1"
CAPACITY = 2048
BUDGET = 1024
FIELDS = (
    "ttft_ms_mean", "ttft_ms_p99", "e2e_ms_mean", "e2e_ms_p99",
    "token_itl_ms_mean", "token_itl_ms_p99",
    "slo_ttft5000_maxitl200_attainment", "preemptions",
)


def replay(task):
    sample, arm = task
    requests = read(ROOT / sample["input"])
    seed = int.from_bytes(hashlib.sha256(sample["name"].encode()).digest()[:4], "big")
    result = papers.simulate(requests, arm, CAPACITY, BUDGET,
                             papers.HISTORY[sample["source"]], seed)
    metrics = result["metrics"]
    assert metrics["completed"] == metrics["requests"] == len(requests) == 64
    assert metrics["executed_tokens"] - metrics["recomputed_tokens"] == sum(
        row["prompt"] + row["output"] - 1 for row in requests)
    return dict(source=sample["source"], name=sample["name"], arm=arm,
                capacity=CAPACITY, budget=BUDGET, seed=seed,
                trajectory_sha256=result["trajectory_sha256"], metrics=metrics)


def check_parity(design):
    papers.initialize()
    sample = next(s for s in design["samples"] if s["split"] == "test")
    requests = read(ROOT / sample["input"])
    seed = int.from_bytes(hashlib.sha256(sample["name"].encode()).digest()[:4], "big")
    for arm in papers.MAIN:
        old = next(row for row in read(papers.OUT / "runs" /
                       f"{sample['name']}_c768_{arm}.json") if row["budget"] == BUDGET)
        current = papers.simulate(requests, arm, 768, BUDGET,
                                  papers.HISTORY[sample["source"]], seed)
        assert current["trajectory_sha256"] == old["trajectory_sha256"], arm
        for field in FIELDS:
            assert current["metrics"][field] == old["metrics"][field], (arm, field)


def main(jobs):
    design = read(papers.OUT / "design.json")
    samples = [s for s in design["samples"] if s["split"] == "test"]
    assert len(samples) == 100 and len(papers.MAIN) == 12
    check_parity(design)
    tasks = [(sample, arm) for sample in samples for arm in papers.MAIN]
    results = []
    with ProcessPoolExecutor(max_workers=jobs, initializer=papers.initialize) as pool:
        futures = {pool.submit(replay, task): task for task in tasks}
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 100 == 0:
                print(f"{len(results)}/{len(tasks)}", flush=True)
    assert len(results) == len(tasks)
    results.sort(key=lambda row: (row["name"], papers.MAIN.index(row["arm"])))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "design.json").write_text(json.dumps(dict(
        source_design=str(papers.OUT / "design.json"), capacity=CAPACITY,
        budget=BUDGET, test_windows=len(samples), methods=list(papers.MAIN),
        parity_check="All 12 methods matched archived KV=768 trajectory hashes and table metrics on one test window.",
        timing="Frozen Qwen2.5-7B fitted-time CPU simulation; no GPU execution in this batch.",
    ), indent=2) + "\n")
    with (OUT / "runs.jsonl").open("w") as handle:
        for row in results:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    import statistics
    with (OUT / "means.csv").open("w") as handle:
        import csv
        writer = csv.writer(handle)
        writer.writerow(("arm", "windows", *FIELDS))
        for arm in papers.MAIN:
            rows = [row["metrics"] for row in results if row["arm"] == arm]
            assert len(rows) == len(samples)
            writer.writerow((arm, len(rows), *(statistics.fmean(row[f] for row in rows)
                                               for f in FIELDS)))
    print(OUT / "means.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=12)
    main(parser.parse_args().jobs)
