"""Audit completed runs and summarize paired decision and service effects."""
import csv
import gzip
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.www_fixed256_explanation_20261014.common import *
from research.www_fixed256_explanation_20261014.policy import select
from research.v1_budget_20260922.policy import State


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decision_comparison(round_row, arm):
    states = [State(**row) for row in round_row["state"]]
    args = (states, 1024, 256, round_row["free_before"],
            round_row["total_blocks"], 16, 32, 8176)
    recent = {"arrival_rate": round_row["features"]["arrival_rate"]}
    full = select(*args, recent=recent, options={})
    variant = select(*args, recent=recent,
                     options=OPTIONS[arm])
    full_tokens = None if full["selected"] is None else full["selected"].tokens
    variant_tokens = (None if variant["selected"] is None
                      else variant["selected"].tokens)
    if (round_row["budget"], round_row["chunk"]) != (variant["budget"], variant["chunk"]):
        raise RuntimeError("recorded budget differs from replayed decision")
    different = ((full["budget"], full["chunk"], full_tokens) !=
                 (variant["budget"], variant["chunk"], variant_tokens))
    return different, variant["variant_triggered"]


def main():
    design = read(OUT / "design.json")
    runs = []
    cells = []
    for cell in design["cells"]:
        key = cell["key"]
        group = read(OUT / "groups" / f"{key}.json")
        if group["state"] != "complete":
            raise RuntimeError(f"incomplete cell: {key}")
        folder = OUT / group["folder"]
        if not read(folder / "aa.json")["passed"] or not read(folder / "bookend.json")["passed"]:
            raise RuntimeError(f"stability gate failed: {key}")
        by_arm = {}
        for arm in ARMS:
            by_arm[arm] = []
            for repeat in range(design["repeats"]):
                run = read(folder / "runs" / f"{arm}_r{repeat}.json")
                if run["status"] != "complete" or run["metrics"]["completed"] != 64:
                    raise RuntimeError(f"incomplete run: {key}/{arm}/{repeat}")
                round_file = folder / run["rounds_path"]
                if digest(round_file) != run["rounds_sha256"]:
                    raise RuntimeError(f"round hash mismatch: {round_file}")
                with gzip.open(round_file, "rt") as handle:
                    rounds = json.load(handle)
                actual_over256 = sum(sum(row["scheduled"].values()) > 256
                                     for row in rounds)
                decisions = [decision_comparison(row, arm) for row in rounds]
                decision_changed = sum(different for different, _ in decisions)
                variant_triggered = sum(triggered for _, triggered in decisions)
                metrics = run["metrics"]
                item = dict(cell=key, source=cell["source"],
                    output_band=cell["output_band"], capacity=cell["capacity"],
                    arm=arm, repeat=repeat, gpu=run["gpu"],
                    ttft_ms_mean=metrics["ttft_ms_mean"],
                    ttft_ms_p99=metrics["ttft_ms_p99"],
                    token_itl_ms_p99=metrics["token_itl_ms_p99"],
                    max_itl_ms_p99=metrics["max_itl_ms_p99"],
                    output_throughput_tps=metrics["output_throughput_tps"],
                    goodput_rps=metrics["slo_ttft5000_maxitl200_goodput_rps"],
                    preemptions=metrics["preemptions"],
                    recomputed_tokens=metrics["recomputed_tokens"],
                    rounds=len(rounds), actual_over256=actual_over256,
                    different_from_full_states=decision_changed,
                    variant_triggered=variant_triggered,
                    mean_step_ms=st.fmean(r["step_ms"] for r in rounds),
                    mean_schedule_ms=st.fmean(r["schedule_ms"] for r in rounds),
                    mean_non_schedule_step_ms=st.fmean(
                        r["step_ms"] - r["schedule_ms"] for r in rounds),
                    peak_kv_util=metrics["peak_kv_util"])
                runs.append(item)
                by_arm[arm].append(item)
        full = by_arm["full"]
        for arm in ARMS[1:]:
            effects = [100 * (1 - ref["ttft_ms_mean"] / test["ttft_ms_mean"])
                       for ref, test in zip(full, by_arm[arm])]
            cell_effect = 100 * (1 - st.fmean(r["ttft_ms_mean"] for r in full) /
                                 st.fmean(r["ttft_ms_mean"] for r in by_arm[arm]))
            cells.append(dict(cell=key, source=cell["source"],
                output_band=cell["output_band"], capacity=cell["capacity"],
                arm=arm, mean_full_advantage_pct=cell_effect,
                repeat_effects_pct=json.dumps(effects),
                changed_state_rounds=sum(r["different_from_full_states"]
                                         for r in by_arm[arm]),
                preemption_difference=st.fmean(r["preemptions"] for r in by_arm[arm])
                    - st.fmean(r["preemptions"] for r in full),
                recompute_difference=st.fmean(r["recomputed_tokens"]
                    for r in by_arm[arm]) - st.fmean(r["recomputed_tokens"]
                    for r in full)))
    write_csv(OUT / "analysis" / "runs.csv", runs)
    write_csv(OUT / "analysis" / "cells.csv", cells)
    summary = {arm: dict(cells=len([row for row in cells if row["arm"] == arm]),
                         mean_full_advantage_pct=st.fmean(
                             row["mean_full_advantage_pct"] for row in cells
                             if row["arm"] == arm)) for arm in ARMS[1:]}
    write(OUT / "analysis" / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
