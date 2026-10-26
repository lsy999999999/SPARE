"""Compare Qwen2.5-7B GPU brackets with refined simulator brackets."""
from __future__ import annotations

import csv
import json

from research.www_policy_ranking_20261023 import bisect_capacity as bisect
from research.www_policy_ranking_20261023.analyze_bisect import relation
from research.www_policy_ranking_20261023.refine_qwen7b import OUT, OLD, TARGETS


NAMES = {"arrival": "SPARE", "srp": "SRP", "sarathi": "Sarathi"}


def load(source: str, kv: int, arm: str, side: str) -> dict:
    label = f"qwen25_7b_{source}_c{kv}_{arm}.json"
    refined = OUT / side / "search" / label
    path = refined if side == "sim" and (source, kv) in TARGETS else OLD / side / "search" / label
    result = json.loads(path.read_text())
    if result["state"] != "complete":
        raise RuntimeError(f"incomplete search: {path}")
    return result


def main() -> None:
    rows = []
    complete_groups = 0
    for source in ("code", "conversation"):
        for kv in bisect.KVS["qwen25_7b"]:
            records = {(side, arm): load(source, kv, arm, side)
                       for side in ("gpu", "sim") for arm in bisect.ARMS}
            matches = []
            for i, left in enumerate(bisect.ARMS):
                for right in bisect.ARMS[i+1:]:
                    gpu = relation(records["gpu", left], records["gpu", right])
                    sim = relation(records["sim", left], records["sim", right])
                    matches.append(gpu == sim)
                    rows.append(dict(source=source, kv_blocks=kv,
                                     policy_a=NAMES[left], policy_b=NAMES[right],
                                     gpu_relation=gpu, simulator_relation=sim,
                                     agreement=gpu == sim,
                                     gpu_a_lower=records["gpu", left]["lower_qps"],
                                     gpu_b_lower=records["gpu", right]["lower_qps"],
                                     simulator_a_lower=records["sim", left]["lower_qps"],
                                     simulator_b_lower=records["sim", right]["lower_qps"],
                                     simulator_resolution=records["sim", left]["resolution_qps"]))
            complete_groups += all(matches)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "ranking_pairs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(model="Qwen2.5-7B", groups=8, refined_groups=len(TARGETS),
                   complete_ranking_groups=complete_groups,
                   matching_pairs=sum(row["agreement"] for row in rows),
                   compared_pairs=len(rows), refinement_qps=.001)
    bisect.write(OUT / "ranking_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
