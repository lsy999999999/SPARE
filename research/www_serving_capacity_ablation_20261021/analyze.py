"""Re-evaluate component ablations under common serving-capacity limits."""
from __future__ import annotations

import csv
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ROOT / "artifacts/www_component_ablation_mean_20261020_r4"
FIXED512 = ROOT / "artifacts/www_serving_capacity_ablation_20261021/fixed512"
OUT = ROOT / "artifacts/www_serving_capacity_ablation_20261021/analysis"
SOURCES = ("code", "conversation")
KVS = (1024, 2048, 4096)
THETAS = (1000, 2000, 4000, 8000)
ARMS = ("spare", "srp", "drop_candidate1", "no_finish_guard", "no_arrival",
        )
RESAMPLES = 2000
SEED = 20261020


def archive(arm):
    return FIXED512 if arm == "fixed512" else PRIMARY


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


@lru_cache(None)
def group(source, kv, arm):
    if arm == "fixed512":
        assert (source, kv) == ("conversation", 2048)
        return read_json(FIXED512 / "group.json")
    return read_json(PRIMARY / "groups" / f"{source}_c{kv}_{arm}.json")


@lru_cache(None)
def records(source, kv, arm, rate):
    path = archive(arm) / "rates" / f"{source}_c{kv}_{arm}_r{rate:g}.json"
    rows = read_json(path)["rows"]
    assert len(rows) == 24 and len({r["name"] for r in rows}) == 24, path
    return {row["name"]: row for row in rows}


def passes(rows, rate, theta):
    return (all(row["valid"] for row in rows)
            and np.mean([row["mean_ttft_ms"] for row in rows]) <= theta
            and np.mean([row["p99_tbt_ms"] for row in rows]) <= 200
            and np.mean([row["backlog_drift"] for row in rows]) <= max(2, rate))


def bootstrap_capacity(source, kv, arm, names, indices, theta):
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


def audit_boundary(source, kv, arm, names, theta):
    cap = group(source, kv, arm)["capacities"][str(theta)]
    low, high = cap["last_pass_qps"], cap["first_fail_qps"]
    assert low < high
    assert passes([records(source, kv, arm, low)[name] for name in names],
                  low, theta), (source, kv, arm, theta, "last pass")
    assert not passes([records(source, kv, arm, high)[name] for name in names],
                      high, theta), (source, kv, arm, theta, "first failure")
    return cap


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_existing_2s(rows):
    path = PRIMARY / "analysis/paired_capacity_bootstrap_2s.csv"
    with path.open(newline="") as handle:
        existing = {(r["source"], int(r["kv_blocks"]), r["arm"]): r
                    for r in csv.DictReader(handle)}
    for row in rows:
        if row["theta_s"] != 2 or row["arm"] in ("spare", "fixed512"):
            continue
        original = existing[row["source"], row["kv_blocks"], row["arm"]]
        assert abs(row["full_minus_variant_qps"]
                   + float(original["delta_capacity_qps"])) < 1e-9
        assert abs(row["paired_95ci_low_qps"]
                   + float(original["bootstrap_high_qps"])) < 1e-4
        assert abs(row["paired_95ci_high_qps"]
                   + float(original["bootstrap_low_qps"])) < 1e-4


def main():
    design = read_json(PRIMARY / "design.json")
    assert len(design["samples"]) == 48
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    output = []
    for source in SOURCES:
        names = sorted(s["name"] for s in design["samples"] if s["source"] == source)
        assert len(names) == 24
        indices = rng.integers(0, len(names), size=(RESAMPLES, len(names)))
        for kv in KVS:
            for theta in THETAS:
                full_cap = audit_boundary(source, kv, "spare", names, theta)
                full = full_cap["last_pass_qps"]
                full_boot = bootstrap_capacity(source, kv, "spare", names,
                                               indices, theta)
                arms = ARMS + (("fixed512",) if (source, kv) ==
                               ("conversation", 2048) else ())
                for arm in arms:
                    cap = audit_boundary(source, kv, arm, names, theta)
                    variant = cap["last_pass_qps"]
                    if arm == "spare":
                        low = high = None
                    else:
                        paired = full_boot - bootstrap_capacity(
                            source, kv, arm, names, indices, theta)
                        low, high = (round(float(v), 4)
                                     for v in np.quantile(paired, (.025, .975)))
                    output.append(dict(
                        source=source, kv_blocks=kv, arm=arm, theta_s=theta / 1000,
                        windows=24, full_capacity_qps=full,
                        variant_capacity_qps=variant,
                        full_minus_variant_qps=round(full - variant, 4),
                        gain_over_variant_pct=round(100 * (full / variant - 1), 2),
                        paired_95ci_low_qps=low, paired_95ci_high_qps=high,
                        full_first_failure=";".join(full_cap["failure_causes"]),
                        variant_first_failure=";".join(cap["failure_causes"]),
                        full_scan_step_qps=round(full_cap["first_fail_qps"] - full, 4),
                        variant_scan_step_qps=round(
                            cap["first_fail_qps"] - variant, 4),
                        archive=str(archive(arm).relative_to(ROOT)),
                    ))
    verify_existing_2s(output)
    write_csv(OUT / "capacity_by_limit.csv", output)

    lines = [
        "# Serving-capacity 消融：统一 TTFT 约束",
        "",
        "主图采用 Conversation 的 24 个配对窗口、KV=2,048 块；完整组件消融"
        "另覆盖 Code/Conversation 与 KV=1,024/2,048/4,096。"
        "采用 CPU 拟合时钟模拟，逐 QPS 扫描到首次失败；每档 QPS 须同时满足"
        "平均 TTFT≤指定上限、窗口 P99 TBT 的均值≤200 ms、"
        "平均积压增长≤max(2,QPS)。容量是最后通过的输入 QPS。",
        "",
        "**固定 512** 指每轮执行预算固定为 512 tokens，是简单控制策略，"
        "不是将省块接受门槛设为 512。它与组件消融共用相同 24 窗口；"
        "原有记录可复用的直接复用，额外模拟 264 个窗口以细化容量边界。",
        "",
        "[Conversation/2,048 容量曲线](capacity_curve_conversation_2048.pdf)"
        "以 mean-TTFT 上限为横轴、最大可服务输入 QPS 为纵轴。",
        "",
        "|mean-TTFT 上限 (s)|Full|SRP|去候选 1|去完成保护|去到达项|固定预算 512|",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_key = {(r["source"], r["kv_blocks"], r["theta_s"], r["arm"]): r
              for r in output}
    for theta in THETAS:
        cells = [f"{by_key['conversation', 2048, theta / 1000, arm]['variant_capacity_qps']:.2f}"
                 for arm in ARMS + ("fixed512",)]
        lines.append(f"|{theta / 1000:g}|" + "|".join(cells) + "|")
    supported = [r for r in output if r["arm"] not in ("spare", "fixed512")
                 and r["paired_95ci_low_qps"] is not None
                 and r["paired_95ci_low_qps"] > 0]
    assert {(r["source"], r["kv_blocks"], r["arm"]) for r in supported
            if r["theta_s"] == 2} == {("conversation", 2048, "srp")}
    lines += [
        "",
        "完整组件扫描、配对 95% 区间、各容量点首次失败原因见"
        "[capacity_by_limit.csv](capacity_by_limit.csv)。对 24 个窗口配对"
        "重采样 2,000 次；容量由离散 QPS 扫描确定。主图画绝对容量，"
        "配对区间保留在 CSV 中。",
        "",
        "## 结论边界",
        "",
        "在 2 s 上限下，Full、SRP、去候选 1、去完成保护、去到达项、"
        "固定 512 的容量分别为 3.88、2.72、3.18、3.92、3.80、3.25 QPS。"
        "Full 对 SRP 的容量差为 +1.16 QPS，配对区间 [0.43, 1.45]；"
        "对固定 512 的差为 +0.63 QPS，配对区间 [−0.12, 1.25]，"
        "尚未严格排除零。固定 512 是简单策略对照，"
        "不能把它的差额解释为省块门槛的单独贡献。",
        "",
        "在这个配置下，Full 的容量随上限从 1 s 到 4 s 由 3.14 增至"
        "4.39 QPS；固定 512 由 2.76 增至 3.38 QPS。4 s 与 8 s"
        " 的容量相同，是积压约束先触发，不表示 TTFT 不再随负载变化。",
        "",
        "去候选 1 的 2 s 容量差为 +0.70 QPS，区间 [0, 1.25]，"
        "尚不能称为严格排除零；去完成保护与去到达项的容量变化更小。"
        "完成保护的直接作用应同时看输出机会冲突率。",
        "",
        "已核验 120 组原组件容量及 4 组固定 512 容量的通过/失败端点，"
        "并复现既有 2 s 组件配对区间。新增模拟仅用于固定 512 的"
        " Conversation/2,048 容量边界细化，没有新增 GPU 实机测量。",
        "",
    ]
    (OUT / "REPORT_CN.md").write_text("\n".join(lines))
    print("\n".join(lines[:18]))


if __name__ == "__main__":
    main()
