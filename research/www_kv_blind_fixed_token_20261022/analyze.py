"""Point-estimate capacity and mechanism audit for fixed-token KV-blind control."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "artifacts/www_component_ablation_mean_20261020_r4"
BLIND = ROOT / "artifacts/www_kv_blind_fixed_token_20261022_r1"
OUT = BLIND / "analysis"
ARM = "kv_blind_fixed_token"
THETAS = (1000, 2000, 4000, 8000)


def read(path):
    return json.loads(path.read_text())


def records(root, source, kv, arm, rate):
    path = root / "rates" / f"{source}_c{kv}_{arm}_r{rate:g}.json"
    rows = read(path)["rows"]
    assert len(rows) == 24 and len({row["name"] for row in rows}) == 24
    return rows


def full_interventions(samples, source, kv, archived):
    from research.www_component_ablation_mean_20261019 import run as pilot

    original = {row["name"]: row for row in archived}
    replayed = [pilot.old.prior.simulate_one(sample, kv, "spare", 2.)
                for sample in samples if sample["source"] == source]
    assert len(replayed) == 24
    assert all(row["trajectory_sha256"] == original[row["name"]]
               ["trajectory_sha256"] for row in replayed)
    return mean(row["interventions"] for row in replayed)


def main():
    from research.www_component_ablation_mean_20261019 import run as pilot

    reference = read(FULL / "design.json")
    design = read(BLIND / "design.json")
    assert design["samples"] == reference["samples"]
    status = read(BLIND / "status.json")
    assert status["completed"] == status["total"] == 6 and not status["errors"]
    pilot.initialize()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in ("code", "conversation"):
        for kv in (1024, 2048, 4096):
            full = read(FULL / "groups" / f"{source}_c{kv}_spare.json")
            blind = read(BLIND / "groups" / f"{source}_c{kv}_{ARM}.json")
            full_at_2 = records(FULL, source, kv, "spare", 2)
            blind_at_2 = records(BLIND, source, kv, ARM, 2)
            assert {r["name"] for r in full_at_2} == {
                r["name"] for r in blind_at_2}
            full_interventions_at_2 = full_interventions(
                design["samples"], source, kv, full_at_2)
            for theta in THETAS:
                f = full["capacities"][str(theta)]
                b = blind["capacities"][str(theta)]
                assert 0 < f["last_pass_qps"] < f["first_fail_qps"]
                assert 0 < b["last_pass_qps"] < b["first_fail_qps"]
                rows.append(dict(source=source, kv_blocks=kv,
                                 theta_s=theta / 1000,
                                 full_capacity_qps=f["last_pass_qps"],
                                 fixed_blind_capacity_qps=b["last_pass_qps"],
                                 full_minus_blind_qps=round(
                                     f["last_pass_qps"]-b["last_pass_qps"], 2),
                                 full_first_fail=";".join(f["failure_causes"]),
                                 fixed_blind_first_fail=";".join(b["failure_causes"]),
                                 full_mean_ttft_at_2qps_ms=round(mean(
                                     r["mean_ttft_ms"] for r in full_at_2), 2),
                                 fixed_blind_mean_ttft_at_2qps_ms=round(mean(
                                     r["mean_ttft_ms"] for r in blind_at_2), 2),
                                 full_p99_tbt_at_2qps_ms=round(mean(
                                     r["p99_tbt_ms"] for r in full_at_2), 2),
                                 fixed_blind_p99_tbt_at_2qps_ms=round(mean(
                                     r["p99_tbt_ms"] for r in blind_at_2), 2),
                                 full_preemptions_at_2qps=round(mean(
                                     r["preemptions"] for r in full_at_2), 2),
                                 fixed_blind_preemptions_at_2qps=round(mean(
                                     r["preemptions"] for r in blind_at_2), 2),
                                 full_interventions_at_2qps=round(
                                     full_interventions_at_2, 2),
                                 fixed_blind_interventions_at_2qps=round(mean(
                                     r["interventions"] for r in blind_at_2), 2)))
    with (OUT / "capacity_by_limit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Fixed-token KV-blind 对照", "",
             "决策器不读取物理 KV 状态、块大小或到达率；四个候选和请求级输出保护保留，"
             "候选相对影子基础计划至少减少 256 token 才接受。原生执行器仍负责 KV 分配。",
             "Code/Conversation 各 24 个相同窗口，KV=1,024/2,048/4,096；"
             "容量规则与 SPARE 相同。下表均为点估计，不使用配对区间。", "",
             "|负载|KV|2 s 容量 SPARE/Blind (QPS)|SPARE−Blind (QPS)|"
             "TTFT@2 QPS SPARE/Blind (ms)|干预/窗口 SPARE/Blind|",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["theta_s"] != 2:
            continue
        lines.append(f"|{row['source'].title()}|{row['kv_blocks']:,}|"
                     f"{row['full_capacity_qps']:.2f}/"
                     f"{row['fixed_blind_capacity_qps']:.2f}|"
                     f"{row['full_minus_blind_qps']:+.2f}|"
                     f"{row['full_mean_ttft_at_2qps_ms']:.0f}/"
                     f"{row['fixed_blind_mean_ttft_at_2qps_ms']:.0f}|"
                     f"{row['full_interventions_at_2qps']:.0f}/"
                     f"{row['fixed_blind_interventions_at_2qps']:.0f}|")
    lines += ["", "在 2 s 约束下，六组容量点估计中 SPARE 与固定 Blind 各领先三组。"
              "固定门槛虽然不读取 KV 信息，仍会在 SPARE 不暂缓的条件下频繁缩减预算；"
              "因此这组结果不能支持 KV-aware 决策带来一致容量提升。",
              "", "四档 TTFT 上限、首次失败原因和共同 QPS 下的 TBT/抢占见"
              " [capacity_by_limit.csv](capacity_by_limit.csv)。"
              "以上为 CPU 拟合时钟模拟，不是 GPU 容量测量。", ""]
    (OUT / "REPORT_CN.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
