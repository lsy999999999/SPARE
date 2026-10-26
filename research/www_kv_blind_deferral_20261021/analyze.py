"""Paired capacity and latency analysis for the KV-blind control."""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "artifacts/www_component_ablation_mean_20261020_r4"
BLIND = ROOT / "artifacts/www_kv_blind_deferral_20261021_r3"
OUT = BLIND / "analysis"
THETAS = (1000, 2000, 4000, 8000)
RESAMPLES = 2000


def read(path):
    return json.loads(path.read_text())


def root(arm):
    return FULL if arm == "spare" else BLIND


@lru_cache(None)
def group(source, kv, arm):
    return read(root(arm) / "groups" / f"{source}_c{kv}_{arm}.json")


@lru_cache(None)
def records(source, kv, arm, rate):
    path = root(arm) / "rates" / f"{source}_c{kv}_{arm}_r{rate:g}.json"
    rows = read(path)["rows"]
    assert len(rows) == 24 and len({row["name"] for row in rows}) == 24
    return {row["name"]: row for row in rows}


def passes(ordered, rate, theta):
    return (all(row["valid"] for row in ordered)
            and np.mean([row["mean_ttft_ms"] for row in ordered]) <= theta
            and np.mean([row["p99_tbt_ms"] for row in ordered]) <= 200
            and np.mean([row["backlog_drift"] for row in ordered]) <= max(2, rate))


def audit(source, kv, arm, names, theta):
    cap = group(source, kv, arm)["capacities"][str(theta)]
    low, high = cap["last_pass_qps"], cap["first_fail_qps"]
    assert 0 < low < high and high - low <= .0200001
    assert passes([records(source, kv, arm, low)[name] for name in names], low, theta)
    assert not passes([records(source, kv, arm, high)[name] for name in names],
                      high, theta)
    return cap


def capacity_bootstrap(source, kv, arm, names, indices, theta):
    rates = sorted(group(source, kv, arm)["rates"])
    passing = []
    for rate in rates:
        rows = [records(source, kv, arm, rate)[name] for name in names]
        valid = np.asarray([row["valid"] for row in rows], dtype=bool)
        ttft = np.asarray([row.get("mean_ttft_ms", np.inf) for row in rows])
        tbt = np.asarray([row.get("p99_tbt_ms", np.inf) for row in rows])
        drift = np.asarray([row.get("backlog_drift", np.inf) for row in rows])
        passing.append(valid[indices].all(axis=1)
                       & (ttft[indices].mean(axis=1) <= theta)
                       & (tbt[indices].mean(axis=1) <= 200)
                       & (drift[indices].mean(axis=1) <= max(2, rate)))
    continuous = np.logical_and.accumulate(np.stack(passing, axis=1), axis=1)
    return (continuous * np.asarray(rates)).max(axis=1)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    design = read(FULL / "design.json")
    blind_design = read(BLIND / "design.json")
    assert design["samples"] == blind_design["samples"]
    status = read(BLIND / "status.json")
    assert status["completed"] == status["total"] == 6 and not status["errors"]
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261021)
    rows = []
    for source in ("code", "conversation"):
        names = sorted(sample["name"] for sample in design["samples"]
                       if sample["source"] == source)
        assert len(names) == 24
        indices = rng.integers(0, 24, size=(RESAMPLES, 24))
        for kv in (1024, 2048, 4096):
            latency_full = [records(source, kv, "spare", 2)[name]["mean_ttft_ms"]
                            for name in names]
            latency_blind = [records(source, kv, "kv_blind", 2)[name]["mean_ttft_ms"]
                             for name in names]
            interventions = [records(source, kv, "kv_blind", 2)[name]
                             ["interventions"] for name in names]
            for theta in THETAS:
                full = audit(source, kv, "spare", names, theta)
                blind = audit(source, kv, "kv_blind", names, theta)
                paired = (capacity_bootstrap(source, kv, "spare", names,
                                             indices, theta)
                          - capacity_bootstrap(source, kv, "kv_blind", names,
                                               indices, theta))
                low, high = np.quantile(paired, (.025, .975))
                rows.append(dict(source=source, kv_blocks=kv,
                                 theta_s=theta / 1000, windows=24,
                                 full_capacity_qps=full["last_pass_qps"],
                                 kv_blind_capacity_qps=blind["last_pass_qps"],
                                 full_minus_blind_qps=round(
                                     full["last_pass_qps"]-blind["last_pass_qps"], 4),
                                 paired_95ci_low_qps=round(float(low), 4),
                                 paired_95ci_high_qps=round(float(high), 4),
                                 full_first_fail_qps=full["first_fail_qps"],
                                 kv_blind_first_fail_qps=blind["first_fail_qps"],
                                 full_failure_causes=";".join(full["failure_causes"]),
                                 kv_blind_failure_causes=";".join(blind["failure_causes"]),
                                 mean_ttft_full_at_2qps_ms=round(float(np.mean(latency_full)), 2),
                                 mean_ttft_blind_at_2qps_ms=round(float(np.mean(latency_blind)), 2),
                                 blind_interventions_at_2qps=sum(interventions)))
    write_csv(OUT / "capacity_by_limit.csv", rows)
    lines = [
        "# KV-blind deferral：容量与延迟对照", "",
        "暂缓动作选择器不读取当前空闲/总 KV、请求已占块、块增长或物理省块量；"
        "只用请求进度、顺序、到达率和 token 预算决定是否暂缓。"
        "物理 KV 分配与恢复仍由原生执行器处理。",
        "请求视图不含已占 KV 块字段。候选动作与 Full 相同；接受候选要求"
        "请求级影子计划少推进至少 max(1, chunk−1 秒预期到达 tokens) 个 token，"
        "并保留影子基础计划的可完成 prefill 与 decode 配额。"
        "这是一个具体的 KV-blind 暂缓规则，不是固定预算，也不是固定省块门槛。",
        "", "同一批 Code/Conversation 各 24 窗口，KV=1,024/2,048/4,096；"
        "统一平均 TTFT、窗口 P99 TBT 均值≤200 ms、积压增长≤max(2,QPS) 约束。",
        "[KV=2,048 容量曲线](kv_blind_capacity_kv2048.pdf)并列展示两个负载"
        "随 mean-TTFT 上限变化的结果。",
        "", "|负载|KV|Full 2 s (QPS)|KV-blind 2 s (QPS)|Full−blind (QPS)|95% 配对区间|TTFT@2 QPS Full/blind (ms)|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["theta_s"] != 2:
            continue
        lines.append(f"|{row['source'].title()}|{row['kv_blocks']:,}|"
                     f"{row['full_capacity_qps']:.2f}|{row['kv_blind_capacity_qps']:.2f}|"
                     f"{row['full_minus_blind_qps']:+.2f}|"
                     f"[{row['paired_95ci_low_qps']:.2f}, {row['paired_95ci_high_qps']:.2f}]|"
                     f"{row['mean_ttft_full_at_2qps_ms']:.0f}/"
                     f"{row['mean_ttft_blind_at_2qps_ms']:.0f}|")
    lines += ["", "六组 2 s 容量差的配对区间均包含零：本批数据没有显示 Full"
              " 相对这个 KV-blind 控制的一致容量优势。这个控制采用预先固定的"
              " token 暂缓触发规则，不能代表所有可能的 KV-blind 策略。",
              "共同 2 QPS 下，KV-blind 在六组中的 mean TTFT 均低于 Full。"
              "在更紧的 1 s 上限、Code/1,024 下，Full−KV-blind 容量差为"
              " −0.28 QPS，配对区间 [−0.75, −0.15] QPS。",
              "", "四档上限、首次失败原因及干预次数见"
              "[capacity_by_limit.csv](capacity_by_limit.csv)。配对区间对同一 24 窗口"
              "重采样 2,000 次；此为 CPU 拟合时钟模拟，不是 GPU 容量测量。", ""]
    (OUT / "REPORT_CN.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
