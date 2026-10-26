"""Audit G2 paired traces and describe SPARE's actual scheduling actions."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
G2 = ROOT / "artifacts/www_evaluation_gpu_completion_20261003_r1"
OUT = ROOT / "artifacts/www_plan_abc_20261008_r1/a"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    design = read(G2 / "design.json")
    manifest = read(G2 / "manifest.json")
    assert sha(G2 / "design.json") == manifest["design_sha256"]
    indexed = {}
    with (G2 / "analysis/runs.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["arm"] in ("arrival", "srp"):
                indexed[row["cell"], row["arm"], int(row["repeat"])] = row
    assert len(indexed) == len(design["cells"]) * 2 * design["repeats"]

    run_rows = []
    action_rows = []
    pairing = defaultdict(dict)
    totals = Counter()
    categories = Counter()
    guards = Counter()
    for cell in design["cells"]:
        group = read(G2 / "groups" / (cell["key"] + ".json"))
        assert group["state"] == "complete"
        folder = G2 / group["folder"]
        assert read(folder / "aa.json")["passed"]
        for arm in ("arrival", "srp"):
            for repeat in range(design["repeats"]):
                index = indexed[cell["key"], arm, repeat]
                path = G2 / index["result_path"]
                assert sha(path) == index["sha256"], path
                result = read(path)
                assert result["status"] == "complete" and result["cell"] == cell["key"]
                round_path = folder / result["rounds_path"]
                assert sha(round_path) == result["rounds_sha256"], round_path
                with gzip.open(round_path, "rt") as handle:
                    rounds = json.load(handle)
                assert len(rounds) == result["metrics"]["rounds"]
                assert result["metrics"]["completed"] == cell["requests"]
                assert abs(float(index["ttft_ms_mean"]) - result["metrics"]["ttft_ms_mean"]) < 1e-7
                assert sum(r["preemptions"] for r in rounds) == result["metrics"]["preemptions"]
                assert sum(r["recomputed_tokens"] for r in rounds) == result["metrics"]["recomputed_tokens"]
                assert sum(bool(r.get("intervened")) for r in rounds) == result["metrics"]["interventions"]
                run_rows.append(dict(cell=cell["key"], window=cell["window"],
                    shape=cell["shape"], capacity=cell["capacity"], arm=arm,
                    repeat=repeat, rounds=len(rounds), ttft_ms_mean=result["metrics"]["ttft_ms_mean"],
                    preemptions=result["metrics"]["preemptions"],
                    recomputed_tokens=result["metrics"]["recomputed_tokens"],
                    interventions=result["metrics"]["interventions"],
                    round_sha256=result["rounds_sha256"]))
                pairing[cell["key"], arm][repeat] = result["metrics"]["ttft_ms_mean"]
                if arm != "arrival":
                    continue
                for r in rounds:
                    totals["all_rounds"] += 1
                    base = r["baseline"]
                    eligible = base is not None and r["baseline_growth_blocks"] > 0
                    totals["eligible_rounds"] += eligible
                    totals["intervened_rounds"] += bool(r["intervened"])
                    if eligible:
                        totals["eligible_intervened_rounds"] += bool(r["intervened"])
                    assert not r["intervened"] or eligible
                    if r["intervened"]:
                        assert not r["preemptions"] and r["verified"]
                        saved = r["baseline_growth_blocks"] - r["actual_growth_blocks"]
                        assert saved >= r["threshold"] > 0
                        assert r["reductions"]
                        totals["saved_blocks"] += saved
                        totals["deferred_tokens"] += sum(x["baseline_tokens"] - x["actual_tokens"] for x in r["reductions"])
                        for x in r["reductions"]:
                            state = next(s for s in r["state"] if s["rid"] == x["id"])
                            assert x["outputs"] == state["outputs"]
                            kind = ("unstarted_prefill" if x["outputs"] == 0 and x["computed"] < state["prompt"]
                                else "unstarted_first_token" if x["outputs"] == 0 else "started_recovery")
                            categories[kind + "_requests"] += 1
                            categories[kind + "_tokens"] += x["baseline_tokens"] - x["actual_tokens"]
                        for s in r["state"]:
                            if s["rid"] not in base:
                                continue
                            protected = ((s["outputs"] > 0 and s["tokens"] - s["computed"] == 1)
                                or (s["outputs"] == 0 and base[s["rid"]] == s["tokens"] - s["computed"]))
                            if protected:
                                assert r["scheduled"].get(s["rid"], 0) >= base[s["rid"]]
                        action_rows.append(dict(cell=cell["key"], window=cell["window"],
                            shape=cell["shape"], capacity=cell["capacity"], repeat=repeat,
                            round=r["round"], free_before=r["free_before"],
                            baseline_tokens=sum(base.values()), actual_tokens=sum(r["scheduled"].values()),
                            baseline_growth_blocks=r["baseline_growth_blocks"],
                            actual_growth_blocks=r["actual_growth_blocks"], saved_blocks=saved,
                            threshold=r["threshold"], budget=r["budget"], chunk=r["chunk"],
                            reduced_requests=len(r["reductions"]), step_end_ms=r["step_end_ms"],
                            unstarted_kv_blocks=sum(s["blocks"] for s in r["state"] if s["outputs"] == 0),
                            state_fingerprint=r["state_fingerprint"]))
                    guards["rounds_with_guard_rejection"] += r["rejected_guard"] > 0
                    guards["guard_rejected_candidates"] += r["rejected_guard"]
                    guards["saving_rejected_candidates"] += r["rejected_saving"]
                    guards["infeasible_candidates"] += r["rejected_infeasible"]
                    guards["evaluated_candidates"] += r["candidates"]

    assert len(run_rows) == 288
    paired = []
    for cell in design["cells"]:
        a = st.fmean(pairing[cell["key"], "arrival"].values())
        b = st.fmean(pairing[cell["key"], "srp"].values())
        paired.append(dict(cell=cell["key"], window=cell["window"], shape=cell["shape"],
            capacity=cell["capacity"], arrival_ttft_ms=a, srp_ttft_ms=b,
            improvement_pct=100 * (b - a) / b))
    overall = st.fmean(x["improvement_pct"] for x in paired)
    assert abs(overall - 16.06) < .01
    window_effects = defaultdict(list)
    for row in paired:
        window_effects[row["window"]].append(row["improvement_pct"])
    window_means = [st.fmean(values) for values in window_effects.values()]
    seed = int(hashlib.sha256(json.dumps("srp", sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()[:12], 16)
    rng = random.Random(seed)
    samples = sorted(st.fmean(rng.choice(window_means) for _ in window_means)
                     for _ in range(10000))
    interval = [samples[249], samples[9749]]
    with (G2 / "analysis/relative_to_arrival.csv").open(newline="") as handle:
        published = next(row for row in csv.DictReader(handle) if row["reference"] == "srp")
    assert abs(interval[0] - float(published["bootstrap_window_95ci_low_pct"])) < 1e-7
    assert abs(interval[1] - float(published["bootstrap_window_95ci_high_pct"])) < 1e-7
    burst = [x for x in paired if x["shape"] == "burst_stop" and x["capacity"] == 768]
    median = st.median(x["improvement_pct"] for x in burst)
    case = min(burst, key=lambda x: (abs(x["improvement_pct"] - median), x["cell"]))
    case_actions = [x for x in action_rows if x["cell"] == case["cell"] and x["repeat"] == 0]
    representative = max(case_actions, key=lambda x: (x["saved_blocks"], -x["round"])) if case_actions else None
    case_group = read(G2 / "groups" / (case["cell"] + ".json"))
    case_folder = G2 / case_group["folder"]
    case_result = read(case_folder / "runs" / "arrival_r0.json")
    with gzip.open(case_folder / case_result["rounds_path"], "rt") as handle:
        case_rounds = json.load(handle)
    case_trace = []
    cumulative_preemptions = 0
    cumulative_recompute = 0
    clocks = case_result["clocks"]
    for row in case_rounds:
        cumulative_preemptions += row["preemptions"]
        cumulative_recompute += row["recomputed_tokens"]
        case_trace.append(dict(round=row["round"], time_ms=row["step_end_ms"],
            unstarted_kv_blocks_before=sum(s["blocks"] for s in row["state"] if s["outputs"] == 0),
            unstarted_active_requests_before=sum(s["outputs"] == 0 for s in row["state"]),
            first_tokens_completed=sum(c["first_ms"] is not None and
                c["first_ms"] <= row["step_end_ms"] for c in clocks.values()),
            cumulative_preemptions=cumulative_preemptions,
            cumulative_recomputed_tokens=cumulative_recompute))
    chosen_row = case_rounds[representative["round"]]
    case_action = dict(rule="Closest median burst-stop 768 paired effect; repeat 0; maximum saved-block intervention, earliest tie",
        cell=case["cell"], paired_improvement_pct=case["improvement_pct"],
        round=chosen_row["round"], state_fingerprint=chosen_row["state_fingerprint"],
        free_before=chosen_row["free_before"], threshold=chosen_row["threshold"],
        baseline=chosen_row["baseline"], scheduled=chosen_row["scheduled"],
        reductions=chosen_row["reductions"],
        baseline_growth_blocks=chosen_row["baseline_growth_blocks"],
        actual_growth_blocks=chosen_row["actual_growth_blocks"],
        affected_request_first_ms={r["id"]: clocks[r["id"]]["first_ms"]
            for r in chosen_row["reductions"]},
        trace_note="This is one closed-loop SPARE trace; SRP is paired only at the complete-scene level.")
    summary = dict(source="G2 accepted groups; four repeats per cell", audited_runs=len(run_rows),
        audited_round_files=len(run_rows), paired_cells=len(paired),
        paired_mean_improvement_pct=overall, paired_window_bootstrap_95ci_pct=interval,
        totals=dict(totals), categories=dict(categories),
        guards=dict(guards), intervention_share_all=totals["intervened_rounds"] / totals["all_rounds"],
        intervention_share_eligible=totals["eligible_intervened_rounds"] / totals["eligible_rounds"],
        representative_rule="Burst-stop, 768 blocks: scene closest to median paired improvement; repeat 0, max saved blocks, earliest tie",
        representative_cell=case, representative_action=representative)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("runs.csv", run_rows), ("actions.csv", action_rows),
                       ("paired.csv", paired), ("case_trace.csv", case_trace)):
        with (OUT / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write(OUT / "summary.json", summary)
    write(OUT / "case_action.json", case_action)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
