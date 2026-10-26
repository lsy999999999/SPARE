"""Compare GPU and simulator capacity ordering on the frozen common-QPS grid."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/www_policy_ranking_fidelity_20261023_r1"
OLD = ROOT / "artifacts/www_cross_model_20261021_r1/runs"
ARMS = ("arrival", "srp", "sarathi")
NAMES = {"arrival": "SPARE", "srp": "SRP", "sarathi": "Sarathi"}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bracket(rows: list[dict], key: str, *, contiguous: bool = False) -> dict:
    rows = sorted(rows, key=lambda row: row["rate_qps"])
    labels = [row[key] for row in rows]
    if not any(labels):
        return dict(closed=False, reason="all_fail")
    if all(labels):
        return dict(closed=False, reason="all_pass")
    nonmonotonic = labels != sorted(labels, reverse=True)
    if nonmonotonic and not contiguous:
        return dict(closed=False, reason="nonmonotonic")
    first_fail = min(row["rate_qps"] for row in rows if not row[key])
    passed_before = [row["rate_qps"] for row in rows
                     if row[key] and row["rate_qps"] < first_fail]
    if not passed_before:
        return dict(closed=False, reason="first_rate_fail")
    return dict(closed=True, lower=max(passed_before), upper=first_fail,
                nonmonotonic=nonmonotonic)


def relation(a: dict, b: dict) -> int:
    if a["lower"] >= b["upper"]:
        return 1
    if b["lower"] >= a["upper"]:
        return -1
    return 0


def gain(policy: dict, baseline: dict) -> dict:
    return dict(grid_pct=100 * (policy["lower"] / baseline["lower"] - 1),
                min_pct=100 * (policy["lower"] / baseline["upper"] - 1),
                max_pct=100 * (policy["upper"] / baseline["lower"] - 1))


def main() -> None:
    design = json.loads((OUT / "design.json").read_text())
    pairs = []
    missing = []
    for source, rates in design["sources"].items():
        for window in design["windows"]:
            sample = f"{source}_{window}"
            for arm in ARMS:
                for rate in rates:
                    label = f"qwen3_8b_{sample}_c2048_{arm}_r{rate:g}"
                    gpu_path = OUT / "runs" / f"{label}.json"
                    if not gpu_path.exists():
                        gpu_path = OLD / f"{label}.json"
                    sim_path = OUT / "sim" / f"{label}.json"
                    if not gpu_path.exists() or not sim_path.exists():
                        missing.append(label)
                        continue
                    gpu = json.loads(gpu_path.read_text())
                    sim = json.loads(sim_path.read_text())
                    if gpu["status"] != "complete" or not gpu["valid"]:
                        raise RuntimeError(f"bad GPU record: {gpu_path}")
                    if gpu["input_sha256"] != sim["input_sha256"]:
                        raise RuntimeError(f"input mismatch: {label}")
                    m = sim["metrics"]
                    pairs.append(dict(source=source, sample=sample, kv_blocks=2048,
                                      arm=arm, rate_qps=rate, gpu_passed=gpu["passed"],
                                      simulator_passed=m["passed"],
                                      gpu_mean_ttft_ms=gpu["mean_ttft_ms"],
                                      simulator_mean_ttft_ms=m["mean_ttft_ms"],
                                      gpu_p99_tbt_ms=gpu["p99_tbt_ms"],
                                      simulator_p99_tbt_ms=m["p99_tbt_ms"],
                                      gpu_backlog_drift=gpu["backlog_drift"],
                                      simulator_backlog_drift=m["backlog_drift"],
                                      gpu_record=str(gpu_path.relative_to(ROOT)),
                                      simulator_record=str(sim_path.relative_to(ROOT))))
    write_csv(OUT / "paired.csv", pairs)
    groups = defaultdict(list)
    for row in pairs:
        groups[(row["source"], row["sample"], row["arm"])].append(row)
    capacities = []
    lookup = {}
    for (source, sample, arm), rows in sorted(groups.items()):
        gpu = bracket(rows, "gpu_passed")
        sim = bracket(rows, "simulator_passed")
        lookup[(source, sample, arm)] = (gpu, sim)
        capacities.append(dict(source=source, sample=sample, arm=arm,
                               tested_qps=len(rows), gpu_closed=gpu["closed"],
                               gpu_lower_qps=gpu.get("lower", ""),
                               gpu_upper_qps=gpu.get("upper", ""),
                               gpu_reason=gpu.get("reason", ""),
                               simulator_closed=sim["closed"],
                               simulator_lower_qps=sim.get("lower", ""),
                               simulator_upper_qps=sim.get("upper", ""),
                               simulator_reason=sim.get("reason", "")))
    write_csv(OUT / "capacity_brackets.csv", capacities)

    ranking = []
    improvements = []
    for source in design["sources"]:
        for window in design["windows"]:
            sample = f"{source}_{window}"
            for i, left in enumerate(ARMS):
                for right in ARMS[i + 1:]:
                    a = lookup.get((source, sample, left))
                    b = lookup.get((source, sample, right))
                    if not a or not b or not all(v["closed"] for v in (*a, *b)):
                        continue
                    ranking.append(dict(source=source, sample=sample,
                                        policy_a=NAMES[left], policy_b=NAMES[right],
                                        gpu_relation=relation(a[0], b[0]),
                                        simulator_relation=relation(a[1], b[1]),
                                        gpu_lower_a=a[0]["lower"], gpu_lower_b=b[0]["lower"],
                                        simulator_lower_a=a[1]["lower"],
                                        simulator_lower_b=b[1]["lower"]))
            all_brackets = [lookup.get((source, sample, arm)) for arm in ARMS]
            if any(b is None or not b[0]["closed"] or not b[1]["closed"]
                   for b in all_brackets):
                continue
            for side, index in (("gpu", 0), ("simulator", 1)):
                spare = all_brackets[0][index]
                best_index = max((1, 2), key=lambda j: all_brackets[j][index]["lower"])
                best = all_brackets[best_index][index]
                current = gain(spare, best)
                if side == "gpu":
                    gpu_gain = current
                    gpu_best = NAMES[ARMS[best_index]]
                else:
                    sim_gain = current
                    sim_best = NAMES[ARMS[best_index]]
            improvements.append(dict(source=source, sample=sample,
                                     gpu_best_baseline=gpu_best,
                                     simulator_best_baseline=sim_best,
                                     gpu_gain_pct=gpu_gain["grid_pct"],
                                     simulator_gain_pct=sim_gain["grid_pct"],
                                     error_pp=sim_gain["grid_pct"]-gpu_gain["grid_pct"],
                                     gpu_min_pct=gpu_gain["min_pct"],
                                     gpu_max_pct=gpu_gain["max_pct"],
                                     simulator_min_pct=sim_gain["min_pct"],
                                     simulator_max_pct=sim_gain["max_pct"]))
    write_csv(OUT / "ranking_pairs.csv", ranking)
    write_csv(OUT / "improvement_prediction.csv", improvements)

    pooled_rows = []
    by_pool = defaultdict(list)
    for row in pairs:
        by_pool[(row["source"], row["arm"], row["rate_qps"])].append(row)
    for (source, arm, rate), rows in sorted(by_pool.items()):
        if len(rows) != len(design["windows"]):
            continue
        pooled = dict(source=source, arm=arm, rate_qps=rate, windows=len(rows))
        for side in ("gpu", "simulator"):
            ttft = statistics.fmean(r[f"{side}_mean_ttft_ms"] for r in rows)
            tbt = statistics.fmean(r[f"{side}_p99_tbt_ms"] for r in rows)
            drift = statistics.fmean(r[f"{side}_backlog_drift"] for r in rows)
            pooled[f"{side}_mean_ttft_ms"] = ttft
            pooled[f"{side}_p99_tbt_ms"] = tbt
            pooled[f"{side}_backlog_drift"] = drift
            pooled[f"{side}_passed"] = (ttft <= 2000 and tbt <= 200
                                        and drift <= max(2, rate))
        pooled_rows.append(pooled)
    write_csv(OUT / "pooled_points.csv", pooled_rows)
    by_policy = defaultdict(list)
    for row in pooled_rows:
        by_policy[(row["source"], row["arm"])].append(row)
    pooled_brackets = []
    pooled_lookup = {}
    for (source, arm), rows in sorted(by_policy.items()):
        gpu = bracket(rows, "gpu_passed", contiguous=True)
        sim = bracket(rows, "simulator_passed", contiguous=True)
        pooled_lookup[(source, arm)] = (gpu, sim)
        pooled_brackets.append(dict(source=source, arm=arm,
                                    gpu_closed=gpu["closed"],
                                    gpu_lower_qps=gpu.get("lower", ""),
                                    gpu_upper_qps=gpu.get("upper", ""),
                                    gpu_nonmonotonic=gpu.get("nonmonotonic", False),
                                    gpu_reason=gpu.get("reason", ""),
                                    simulator_closed=sim["closed"],
                                    simulator_lower_qps=sim.get("lower", ""),
                                    simulator_upper_qps=sim.get("upper", ""),
                                    simulator_nonmonotonic=sim.get("nonmonotonic", False),
                                    simulator_reason=sim.get("reason", "")))
    write_csv(OUT / "pooled_capacity_brackets.csv", pooled_brackets)
    pooled_ranking = []
    pooled_improvement = []
    for source in design["sources"]:
        arms = [pooled_lookup.get((source, arm)) for arm in ARMS]
        for i, left in enumerate(ARMS):
            for j in range(i + 1, len(ARMS)):
                a, b = arms[i], arms[j]
                if not a or not b or not all(v["closed"] for v in (*a, *b)):
                    continue
                pooled_ranking.append(dict(source=source, policy_a=NAMES[left],
                                           policy_b=NAMES[ARMS[j]],
                                           gpu_relation=relation(a[0], b[0]),
                                           simulator_relation=relation(a[1], b[1])))
        if any(a is None or not a[0]["closed"] or not a[1]["closed"] for a in arms):
            continue
        gpu_best_index = max((1, 2), key=lambda j: arms[j][0]["lower"])
        sim_best_index = max((1, 2), key=lambda j: arms[j][1]["lower"])
        gpu_base = arms[gpu_best_index][0]
        sim_base = arms[sim_best_index][1]
        g, s = gain(arms[0][0], gpu_base), gain(arms[0][1], sim_base)
        pooled_improvement.append(dict(source=source,
                                       gpu_best_baseline=NAMES[ARMS[gpu_best_index]],
                                       simulator_best_baseline=NAMES[ARMS[sim_best_index]],
                                       gpu_gain_pct=g["grid_pct"],
                                       simulator_gain_pct=s["grid_pct"],
                                       error_pp=s["grid_pct"]-g["grid_pct"],
                                       gpu_min_pct=g["min_pct"],
                                       gpu_max_pct=g["max_pct"],
                                       simulator_min_pct=s["min_pct"],
                                       simulator_max_pct=s["max_pct"]))
    write_csv(OUT / "pooled_ranking_pairs.csv", pooled_ranking)
    write_csv(OUT / "pooled_improvement_prediction.csv", pooled_improvement)
    summary = dict(expected_points=sum(len(r) * len(ARMS) * len(design["windows"])
                                       for r in design["sources"].values()),
                   paired_points=len(pairs), missing=missing,
                   point_decision_match=sum(r["gpu_passed"] == r["simulator_passed"]
                                            for r in pairs),
                   bracket_reasons=dict(Counter(r["gpu_reason"] or "closed"
                                                for r in capacities)),
                   bracket_match=sum(r["gpu_closed"] and r["simulator_closed"]
                                     and (r["gpu_lower_qps"], r["gpu_upper_qps"])
                                     == (r["simulator_lower_qps"], r["simulator_upper_qps"])
                                     for r in capacities),
                   gpu_closed_brackets=sum(r["gpu_closed"] for r in capacities),
                   ranking_pairs=len(ranking),
                   gpu_strict_pairs=sum(r["gpu_relation"] != 0 for r in ranking),
                   strict_direction_match=sum(r["gpu_relation"] != 0 and
                                              r["gpu_relation"] == r["simulator_relation"]
                                              for r in ranking),
                   false_spare_win=sum(r["policy_a"] == "SPARE" and
                                       r["simulator_relation"] == 1 and
                                       r["gpu_relation"] != 1 for r in ranking),
                   false_spare_loss=sum(r["policy_a"] == "SPARE" and
                                        r["gpu_relation"] == 1 and
                                        r["simulator_relation"] != 1 for r in ranking),
                   ties_both=sum(r["gpu_relation"] == r["simulator_relation"] == 0
                                 for r in ranking),
                   improvement_groups=len(improvements),
                   improvement_mae_pp=(statistics.fmean(abs(r["error_pp"])
                                                       for r in improvements)
                                       if improvements else None),
                   improvement_bias_pp=(statistics.fmean(r["error_pp"]
                                                        for r in improvements)
                                        if improvements else None),
                   pooled_ranking_pairs=len(pooled_ranking),
                   pooled_gpu_strict_pairs=sum(r["gpu_relation"] != 0
                                               for r in pooled_ranking),
                   pooled_strict_direction_match=sum(r["gpu_relation"] != 0 and
                                                     r["gpu_relation"] == r["simulator_relation"]
                                                     for r in pooled_ranking),
                   pooled_false_spare_win=sum(r["policy_a"] == "SPARE" and
                                              r["simulator_relation"] == 1 and
                                              r["gpu_relation"] != 1
                                              for r in pooled_ranking),
                   pooled_improvement_groups=len(pooled_improvement))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
