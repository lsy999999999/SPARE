"""Audit three-policy TTFT ordering using completed 4090 paired runs."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/www_cross_model_simulator_validation_20261022_r3/paired_no_fixed256.csv"
OUT = ROOT / "artifacts/www_policy_ranking_quick_existing_20261024_r1"
ARMS = ("arrival", "srp", "sarathi")


def save(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(rows: list[dict]) -> dict:
    material = [r for r in rows if r["material_gpu_gap"]]
    return dict(points=len({(r["model"], r["source"], r["kv_blocks"],
                            r["sample"], r["rate_qps"]) for r in rows}),
                pairs=len(rows), pair_matches=sum(r["match"] for r in rows),
                material_pairs=len(material),
                material_matches=sum(r["match"] for r in material))


def main() -> None:
    groups = defaultdict(dict)
    with SOURCE.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["model"], row["source"], int(row["kv_blocks"]),
                   row["sample"], float(row["rate_qps"]))
            groups[key][row["arm"]] = row
    pairs = []
    orders = []
    for (model, source, kv, sample, rate), methods in sorted(groups.items()):
        if any(arm not in methods for arm in ARMS):
            continue
        gpu = {arm: float(methods[arm]["gpu_mean_ttft_ms"]) for arm in ARMS}
        sim = {arm: float(methods[arm]["simulator_mean_ttft_ms"]) for arm in ARMS}
        point_pairs = []
        for i, left in enumerate(ARMS):
            for right in ARMS[i+1:]:
                g = gpu[left] - gpu[right]
                s = sim[left] - sim[right]
                material = abs(g) / min(gpu[left], gpu[right]) >= .05
                matched = (g > 0) == (s > 0) if g and s else g == s
                item = dict(model=model, source=source, kv_blocks=kv,
                            sample=sample, rate_qps=rate,
                            policy_a=left, policy_b=right,
                            gpu_a_ms=gpu[left], gpu_b_ms=gpu[right],
                            simulator_a_ms=sim[left], simulator_b_ms=sim[right],
                            gpu_gap_pct=100 * abs(g) / min(gpu[left], gpu[right]),
                            material_gpu_gap=material, match=matched)
                pairs.append(item)
                point_pairs.append(item)
        gpu_order = sorted(ARMS, key=lambda a: gpu[a])
        sim_order = sorted(ARMS, key=lambda a: sim[a])
        orders.append(dict(model=model, source=source, kv_blocks=kv,
                           sample=sample, rate_qps=rate,
                           gpu_ttft_order=";".join(gpu_order),
                           simulator_ttft_order=";".join(sim_order),
                           exact_order=gpu_order == sim_order,
                           material_pairs=sum(r["material_gpu_gap"] for r in point_pairs),
                           matched_material_pairs=sum(r["material_gpu_gap"] and r["match"]
                                                      for r in point_pairs)))
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / "pairs.csv", pairs)
    save(OUT / "orders.csv", orders)
    by_model = {model: summary([r for r in pairs if r["model"] == model])
                for model in sorted({r["model"] for r in pairs})}
    result = dict(overall=summary(pairs), by_model=by_model,
                  full_order_matches=sum(r["exact_order"] for r in orders),
                  full_order_points=len(orders))
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
