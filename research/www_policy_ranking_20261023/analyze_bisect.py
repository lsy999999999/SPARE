"""Compare interval-aware policy rankings from paired capacity searches."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from research.www_policy_ranking_20261023.bisect_capacity import ARMS, OUT, groups, name


NAMES = {"arrival": "SPARE", "srp": "SRP", "sarathi": "Sarathi"}


def relation(a: dict, b: dict) -> int:
    if a["lower_qps"] >= b["upper_qps"]:
        return 1
    if b["lower_qps"] >= a["upper_qps"]:
        return -1
    return 0


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    capacities = []
    rankings = []
    simulator_orders = []
    by_condition = {}
    for model, source, kv in groups():
        condition = name(model, source, kv)
        lookup = {}
        for arm in ARMS:
            for side in ("gpu", "sim"):
                path = OUT / side / "search" / f"{condition}_{arm}.json"
                row = json.loads(path.read_text()) if path.exists() else None
                lookup[(side, arm)] = row
                capacities.append(dict(model=model, source=source, kv_blocks=kv,
                                       policy=NAMES[arm], side=side,
                                       state=row["state"] if row else "missing",
                                       lower_qps=row.get("lower_qps", "") if row else "",
                                       upper_qps=row.get("upper_qps", "") if row else "",
                                       nonmonotonic=row.get("nonmonotonic", "") if row else "",
                                       tested_rates=len(row["points"]) if row else 0))
        sim_rows = [lookup[("sim", arm)] for arm in ARMS]
        if all(r is not None and r["state"] == "complete" for r in sim_rows):
            ordered = sorted(zip(ARMS, sim_rows),
                             key=lambda pair: pair[1]["lower_qps"], reverse=True)
            strict_pairs = sum(relation(sim_rows[i], sim_rows[j]) != 0
                               for i in range(len(ARMS))
                               for j in range(i+1, len(ARMS)))
            simulator_orders.append(dict(
                model=model, source=source, kv_blocks=kv,
                descending_lower_bound_order=";".join(NAMES[arm] for arm, _ in ordered),
                strict_pairs=strict_pairs, all_pairs_strict=strict_pairs == 3,
                nonmonotonic_policies=";".join(NAMES[arm] for arm, row in zip(ARMS, sim_rows)
                                                if row["nonmonotonic"]),
                spare_interval=f"{sim_rows[0]['lower_qps']:.2f}-{sim_rows[0]['upper_qps']:.2f}",
                srp_interval=f"{sim_rows[1]['lower_qps']:.2f}-{sim_rows[1]['upper_qps']:.2f}",
                sarathi_interval=f"{sim_rows[2]['lower_qps']:.2f}-{sim_rows[2]['upper_qps']:.2f}"))
        exact = 0
        compared = 0
        for i, left in enumerate(ARMS):
            for right in ARMS[i+1:]:
                records = [lookup[(side, arm)] for side in ("gpu", "sim")
                           for arm in (left, right)]
                if any(r is None or r["state"] != "complete" for r in records):
                    continue
                gpu_relation = relation(records[0], records[1])
                sim_relation = relation(records[2], records[3])
                compared += 1
                exact += gpu_relation == sim_relation
                rankings.append(dict(model=model, source=source, kv_blocks=kv,
                                     policy_a=NAMES[left], policy_b=NAMES[right],
                                     gpu_relation=gpu_relation,
                                     simulator_relation=sim_relation,
                                     agreement=gpu_relation == sim_relation,
                                     gpu_strict=gpu_relation != 0,
                                     gpu_nonmonotonic=records[0]["nonmonotonic"] or records[1]["nonmonotonic"],
                                     simulator_nonmonotonic=records[2]["nonmonotonic"] or records[3]["nonmonotonic"]))
        by_condition[condition] = dict(compared_pairs=compared, matching_pairs=exact)
    save_csv(OUT / "capacity_brackets.csv", capacities)
    save_csv(OUT / "ranking_pairs.csv", rankings)
    save_csv(OUT / "simulator_rankings.csv", simulator_orders)
    strict = [r for r in rankings if r["gpu_strict"]]
    clean_strict = [r for r in strict if not r["gpu_nonmonotonic"]
                    and not r["simulator_nonmonotonic"]]
    summary = dict(groups=len(groups()), simulator_groups=len(simulator_orders),
                   simulator_fully_ordered_groups=sum(r["all_pairs_strict"] for r in simulator_orders),
                   complete_pairs=len(rankings),
                   matching_pairs=sum(r["agreement"] for r in rankings),
                   gpu_strict_pairs=len(strict),
                   matching_gpu_strict_pairs=sum(r["agreement"] for r in strict),
                   clean_gpu_strict_pairs=len(clean_strict),
                   matching_clean_gpu_strict_pairs=sum(r["agreement"] for r in clean_strict),
                   conditions=by_condition)
    (OUT / "ranking_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "conditions"}, indent=2))


if __name__ == "__main__":
    main()
