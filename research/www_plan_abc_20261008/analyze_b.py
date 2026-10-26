"""Audit and summarize paired same-state one-action branches."""
from __future__ import annotations

import csv
import gzip
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.www_evaluation_gpu_completion_20261003.common import digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/www_plan_abc_20261008_r1/b_v2"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ci(window_values, seed=20261008):
    rng = random.Random(seed)
    values = list(window_values)
    samples = sorted(st.fmean(rng.choice(values) for _ in values) for _ in range(10000))
    return [samples[249], samples[9749]]


def main():
    design = read(OUT / "design.json")
    summary_rows = []
    all_runs = []
    aa_spreads = []
    for cell in design["cells"]:
        window = cell["window"]
        group = read(OUT / "groups" / window / "status.json")
        assert group["state"] == "complete", window
        for capacity in cell["capacities"]:
            folder = OUT / "groups" / window / f"c{capacity}"
            aa = read(folder / "aa.json")
            assert aa["passed"]
            aa_spreads.append(aa["relative_spread"])
            selection = read(folder / "selected.json")
            for target in selection["selected"]:
                index = target["round"]
                assert target["base_growth"] > target["selected_growth"]
                assert target["pending"] > 0
                by_arm = defaultdict(list)
                for repeat in range(design["repeats"]):
                    for arm in ("base", "spare"):
                        path = folder / "runs" / f"s{index}_{arm}_r{repeat}.json"
                        result = read(path)
                        assert result["status"] == "complete" and result["target"] == index
                        assert result["arm"] == arm and result["repeat"] == repeat
                        assert result["pending"] == target["pending"]
                        label = result["label"]
                        assert label["prefix_rounds"] == index + 1
                        round_path = OUT / result["rounds_path"]
                        assert digest(round_path) == result["rounds_sha256"]
                        with gzip.open(round_path, "rt") as handle:
                            rounds = json.load(handle)
                        assert rounds[index]["state_fingerprint"] == target["fingerprint"]
                        assert rounds[index]["physical_state_sha256"] == target["physical_state_sha256"]
                        expected = target["base_tokens"] if arm == "base" else target["selected_tokens"]
                        assert rounds[index]["scheduled"] == expected
                        assert label["q"] == sum(expected.values())
                        assert label["growth"] == (target["base_growth"] if arm == "base" else target["selected_growth"])
                        record = dict(window=window, capacity=capacity, stage=target["stage"],
                            round=index, repeat=repeat, arm=arm,
                            area_per_request_ms=label["area_per_request_ms"],
                            first_round_ms=label["first_round_ms"], later_ms=label["later_ms"],
                            q=label["q"], growth=label["growth"], step_ms=label["step_ms"],
                            preemptions=label["preemptions"], recomputed_tokens=label["recomputed_tokens"],
                            block_id_match_rounds=label["block_id_match_rounds"],
                            prefix_rounds=label["prefix_rounds"])
                        all_runs.append(record)
                        by_arm[arm].append(record)
                base, spare = by_arm["base"], by_arm["spare"]
                assert len(base) == len(spare) == design["repeats"]
                differences = [left["area_per_request_ms"] - right["area_per_request_ms"]
                               for left, right in zip(base, spare)]
                summary_rows.append(dict(window=window, capacity=capacity, stage=target["stage"],
                    round=index, pending=target["pending"], free_before=target["free_before"],
                    saved_blocks=target["base_growth"] - target["selected_growth"],
                    threshold=target["threshold"],
                    mean_benefit_ms=st.fmean(differences),
                    min_benefit_ms=min(differences), max_benefit_ms=max(differences),
                    positive_repeats=sum(x > 0 for x in differences),
                    first_round_benefit_ms=st.fmean(
                        left["first_round_ms"] - right["first_round_ms"] for left, right in zip(base, spare)),
                    later_benefit_ms=st.fmean(
                        left["later_ms"] - right["later_ms"] for left, right in zip(base, spare)),
                    preemption_difference=st.fmean(
                        left["preemptions"] - right["preemptions"] for left, right in zip(base, spare)),
                    recompute_difference=st.fmean(
                        left["recomputed_tokens"] - right["recomputed_tokens"] for left, right in zip(base, spare)),
                    block_id_match_fraction=st.fmean(
                        row["block_id_match_rounds"] / row["prefix_rounds"] for row in base+spare)))
    assert len(summary_rows) <= 24 and len(all_runs) == len(summary_rows) * 8
    by_window = defaultdict(list)
    for row in summary_rows:
        by_window[row["window"]].append(row["mean_benefit_ms"])
    window_means = {window: st.fmean(values) for window, values in by_window.items()}
    summary = dict(states=len(summary_rows), branch_runs=len(all_runs), windows=len(window_means),
        mean_benefit_ms=st.fmean(row["mean_benefit_ms"] for row in summary_rows),
        window_equal_mean_benefit_ms=st.fmean(window_means.values()),
        window_bootstrap_95ci_ms=ci(window_means.values()),
        positive_states=sum(row["mean_benefit_ms"] > 0 for row in summary_rows),
        negative_states=sum(row["mean_benefit_ms"] < 0 for row in summary_rows),
        all_four_positive_states=sum(row["positive_repeats"] == 4 for row in summary_rows),
        all_four_negative_states=sum(row["positive_repeats"] == 0 for row in summary_rows),
        first_round_benefit_ms=st.fmean(row["first_round_benefit_ms"] for row in summary_rows),
        later_benefit_ms=st.fmean(row["later_benefit_ms"] for row in summary_rows),
        aa_relative_spread_max=max(aa_spreads), window_means=window_means,
        inference="Four repetitions are paired measurements of one state, not independent workloads")
    csv_write(OUT / "analysis" / "states.csv", summary_rows)
    csv_write(OUT / "analysis" / "runs.csv", all_runs)
    write(OUT / "analysis" / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
