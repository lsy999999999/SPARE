"""Audit multi-policy GPU/simulator rankings over the shared wide-KV pilot."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/www_simulator_gpu_validation_pilot_analysis_20261019"
OUT = ROOT / "artifacts/www_policy_ranking_wide_kv_20261024_r1"
ARMS = ("arrival", "native", "sarathi", "uniboost_static", "past_future",
        "s3_proxy", "apt_serve_kv_rule")
NAMES = {"arrival": "SPARE", "native": "Default", "sarathi": "Sarathi",
         "uniboost_static": "UniBoost rule", "past_future": "Past-Future rule",
         "s3_proxy": "S3 proxy", "apt_serve_kv_rule": "APT-Serve rule"}
MIN_KV, MAX_KV = 2048, 7424


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, data: list[dict]) -> None:
    if not data:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def signature(values: dict[str, float], *, higher: bool) -> str:
    ordered = sorted(set(values.values()), reverse=higher)
    return " > ".join("=".join(NAMES[arm] for arm in ARMS if values[arm] == value)
                      for value in ordered)


def capacity_relation(left: dict, right: dict, side: str) -> int:
    lower_a, upper_a = (float(left[f"{side}_{part}_qps"])
                        for part in ("lower", "upper"))
    lower_b, upper_b = (float(right[f"{side}_{part}_qps"])
                        for part in ("lower", "upper"))
    if lower_a >= upper_b:
        return 1
    if lower_b >= upper_a:
        return -1
    return 0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    brackets = defaultdict(dict)
    for row in rows(SOURCE / "capacity_brackets.csv"):
        kv = int(row["kv_blocks"])
        if MIN_KV <= kv <= MAX_KV and row["arm"] in ARMS:
            brackets[(row["source"], kv)][row["arm"]] = row

    capacity = []
    capacity_pairs = []
    for (source, kv), methods in sorted(brackets.items()):
        complete = len(methods) == len(ARMS) and all(
            methods[arm][f"{side}_closed"] == "True"
            for arm in ARMS for side in ("gpu", "sim"))
        row = dict(source=source, kv_blocks=kv, method_count=len(methods),
                   both_closed=complete, gpu_rank="", simulator_rank="",
                   rank_exact="", gpu_strict_pairs="", matched_strict_pairs="",
                   simulated_strict_gpu_unresolved="")
        if complete:
            gpu_values = {arm: float(methods[arm]["gpu_lower_qps"]) for arm in ARMS}
            sim_values = {arm: float(methods[arm]["sim_lower_qps"]) for arm in ARMS}
            row["gpu_rank"] = signature(gpu_values, higher=True)
            row["simulator_rank"] = signature(sim_values, higher=True)
            row["rank_exact"] = row["gpu_rank"] == row["simulator_rank"]
            strict = matched = false_strict = 0
            for left, right in itertools.combinations(ARMS, 2):
                gpu_rel = capacity_relation(methods[left], methods[right], "gpu")
                sim_rel = capacity_relation(methods[left], methods[right], "sim")
                strict += gpu_rel != 0
                matched += gpu_rel != 0 and gpu_rel == sim_rel
                false_strict += gpu_rel == 0 and sim_rel != 0
                capacity_pairs.append(dict(source=source, kv_blocks=kv,
                                           policy_a=NAMES[left], policy_b=NAMES[right],
                                           gpu_relation=gpu_rel, simulator_relation=sim_rel))
            row["gpu_strict_pairs"] = strict
            row["matched_strict_pairs"] = matched
            row["simulated_strict_gpu_unresolved"] = false_strict
        capacity.append(row)
    write_csv(OUT / "capacity_rankings.csv", capacity)
    write_csv(OUT / "capacity_pairs.csv", capacity_pairs)

    paired = defaultdict(dict)
    for row in rows(SOURCE / "pairs.csv"):
        kv = int(row["kv_blocks"])
        if MIN_KV <= kv <= MAX_KV and row["arm"] in ARMS:
            key = (row["source"], kv, float(row["rate_qps"]))
            paired[key][row["arm"]] = row
    ttft = []
    ttft_pairs = []
    for (source, kv, rate), methods in sorted(paired.items()):
        if len(methods) != len(ARMS):
            continue
        gpu_values = {arm: float(methods[arm]["gpu_ttft_ms"]) for arm in ARMS}
        sim_values = {arm: float(methods[arm]["sim_ttft_ms"]) for arm in ARMS}
        gpu_rank = signature(gpu_values, higher=False)
        sim_rank = signature(sim_values, higher=False)
        strict = matched = material = material_matched = 0
        for left, right in itertools.combinations(ARMS, 2):
            g = gpu_values[left] - gpu_values[right]
            s = sim_values[left] - sim_values[right]
            gpu_rel = (g > 0) - (g < 0)
            sim_rel = (s > 0) - (s < 0)
            is_material = abs(g) / min(gpu_values[left], gpu_values[right]) >= .05
            strict += gpu_rel != 0
            matched += gpu_rel != 0 and gpu_rel == sim_rel
            material += is_material
            material_matched += is_material and gpu_rel == sim_rel
            ttft_pairs.append(dict(source=source, kv_blocks=kv, rate_qps=rate,
                                   policy_a=NAMES[left], policy_b=NAMES[right],
                                   gpu_a_ms=gpu_values[left], gpu_b_ms=gpu_values[right],
                                   sim_a_ms=sim_values[left], sim_b_ms=sim_values[right],
                                   gpu_relation=gpu_rel, simulator_relation=sim_rel,
                                   gpu_gap_at_least_5pct=is_material))
        ttft.append(dict(source=source, kv_blocks=kv, rate_qps=rate,
                         method_count=len(ARMS), gpu_rank=gpu_rank,
                         simulator_rank=sim_rank, rank_exact=gpu_rank == sim_rank,
                         gpu_strict_pairs=strict, matched_strict_pairs=matched,
                         gpu_material_pairs=material,
                         matched_material_pairs=material_matched))
    write_csv(OUT / "ttft_rankings.csv", ttft)
    write_csv(OUT / "ttft_pairs.csv", ttft_pairs)

    complete = [row for row in capacity if row["both_closed"]]
    kvs = sorted({row["kv_blocks"] for row in capacity})
    summary = dict(kv_range=[MIN_KV, MAX_KV], kv_levels=kvs,
                   methods=[NAMES[arm] for arm in ARMS],
                   source_kv_conditions=len(capacity),
                   capacity_complete_conditions=len(complete),
                   capacity_exact_tie_aware_rankings=sum(row["rank_exact"] for row in complete),
                   capacity_all_tied_conditions=sum(row["gpu_strict_pairs"] == 0
                                                    for row in complete),
                   capacity_gpu_strict_pairs=sum(row["gpu_strict_pairs"] for row in complete),
                   capacity_matched_strict_pairs=sum(row["matched_strict_pairs"] for row in complete),
                   capacity_sim_strict_gpu_unresolved=sum(
                       row["simulated_strict_gpu_unresolved"] for row in complete),
                   ttft_common_qps_conditions=len(ttft),
                   ttft_exact_rankings=sum(row["rank_exact"] for row in ttft),
                   ttft_gpu_strict_pairs=sum(row["gpu_strict_pairs"] for row in ttft),
                   ttft_matched_strict_pairs=sum(row["matched_strict_pairs"] for row in ttft),
                   ttft_gpu_gap_at_least_5pct_pairs=sum(row["gpu_material_pairs"] for row in ttft),
                   ttft_matched_gap_at_least_5pct_pairs=sum(
                       row["matched_material_pairs"] for row in ttft),
                   ttft_conditions_by_source=dict(Counter(row["source"] for row in ttft)))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
