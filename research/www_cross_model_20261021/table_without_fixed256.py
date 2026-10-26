"""Export cross-model simulator fidelity tables without fixed256."""
from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/www_cross_model_simulator_validation_20261022_r3"
SOURCE = OUT / "paired.csv"
ARMS = {"arrival": "SPARE", "srp": "SRP", "sarathi": "Sarathi"}
MODELS = {"qwen25_3b": "Qwen2.5-3B", "qwen3_8b": "Qwen3-8B",
          "mistral_7b": "Mistral-7B"}
SOURCES = {"code": "Code", "conversation": "Conversation"}


def read_rows() -> list[dict]:
    with SOURCE.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["arm"] in ARMS]
    if len(rows) != 452:
        raise RuntimeError(f"expected 452 non-fixed256 held-out pairs, got {len(rows)}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], **labels) -> dict:
    matched = sum(r["gpu_passed"] == r["simulator_passed"] for r in rows)
    mean = lambda name: statistics.fmean(float(r[name]) for r in rows)
    return dict(**labels, pairs=len(rows), matched=matched,
                agreement_pct=100*matched/len(rows),
                gpu_passed=sum(r["gpu_passed"] == "True" for r in rows),
                simulator_passed=sum(r["simulator_passed"] == "True" for r in rows),
                gpu_mean_ttft_ms=mean("gpu_mean_ttft_ms"),
                simulator_mean_ttft_ms=mean("simulator_mean_ttft_ms"),
                gpu_p99_tbt_ms=mean("gpu_p99_tbt_ms"),
                simulator_p99_tbt_ms=mean("simulator_p99_tbt_ms"),
                gpu_backlog_drift=mean("gpu_backlog_drift"),
                simulator_backlog_drift=mean("simulator_backlog_drift"),
                ttft_mape_pct=mean("mean_ttft_ape_pct"),
                tbt_mape_pct=mean("p99_tbt_ape_pct"))


def table(rows: list[dict]) -> str:
    header = ("| 模型 | 负载 | 方法 | 配对数 | 判定一致 | GPU通过 | 模拟通过 | "
              "mean TTFT GPU/模拟 (ms) | P99 TBT GPU/模拟 (ms) |")
    lines = [header, "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {MODELS[r['model']]} | {SOURCES[r['source']]} | {ARMS[r['arm']]} | "
            f"{r['pairs']} | {r['matched']}/{r['pairs']} ({r['agreement_pct']:.1f}%) | "
            f"{r['gpu_passed']} | {r['simulator_passed']} | "
            f"{r['gpu_mean_ttft_ms']:.1f} / {r['simulator_mean_ttft_ms']:.1f} | "
            f"{r['gpu_p99_tbt_ms']:.1f} / {r['simulator_p99_tbt_ms']:.1f} |")
    return "\n".join(lines)


def main() -> None:
    rows = read_rows()
    write_csv(OUT / "paired_no_fixed256.csv", rows)
    by_cell = defaultdict(list)
    for row in rows:
        by_cell[(row["model"], row["source"], row["arm"])].append(row)
    cell_rows = [aggregate(by_cell[(model, source, arm)], model=model, source=source, arm=arm)
                 for model in MODELS for source in SOURCES for arm in ARMS]
    write_csv(OUT / "summary_no_fixed256_by_model_workload.csv", cell_rows)
    by_arm = [aggregate([r for r in rows if r["arm"] == arm], arm=arm)
              for arm in ARMS]
    overall = aggregate(rows, arm="all")
    write_csv(OUT / "summary_no_fixed256_by_method.csv", [*by_arm, overall])
    macro = statistics.fmean(r["agreement_pct"] for r in by_arm)
    method_lines = ["| 方法 | 判定一致 | GPU通过 | 模拟通过 | TTFT MAPE | TBT MAPE |",
                    "|---|---:|---:|---:|---:|---:|"]
    for row in [*by_arm, overall]:
        label = ARMS.get(row["arm"], "全部（按回放数加权）")
        method_lines.append(
            f"| {label} | {row['matched']}/{row['pairs']} ({row['agreement_pct']:.2f}%) | "
            f"{row['gpu_passed']} | {row['simulator_passed']} | "
            f"{row['ttft_mape_pct']:.2f}% | {row['tbt_mape_pct']:.2f}% |")
    report = [
        "# 跨模型模拟器验证：不含 fixed256",
        "",
        "判定一致指同一输入在 GPU 和模拟器上是否同时满足 mean TTFT <= 2 s、"
        "P99 TBT <= 200 ms、40–60 s 积压增长 <= max(2,QPS)。"
        "这是通过/失败的一致率，不是延迟数值预测准确率。",
        "",
        "## 各方法与平均",
        "",
        *method_lines,
        "",
        f"三个方法等权的平均一致率为 **{macro:.2f}%**；按全部 452 条回放加权为 "
        f"**{overall['matched']}/{overall['pairs']}（{overall['agreement_pct']:.2f}%）**。",
        "",
        "## 三个模型的 GPU 与模拟器数据",
        "",
        table(cell_rows),
        "",
        "表内 TTFT 和 TBT 是该模型、负载、方法全部配对回放的算术平均；"
        "它们包含三个 KV 档及实验设计指定的多个 QPS，用于展示预测值与实测值，"
        "不应用来直接比较方法收益。每个 KV、QPS、窗口的原始数值、积压及通过判定"
        "见 `paired_no_fixed256.csv`；机器可读汇总见两个 `summary_no_fixed256_*.csv`。",
        "",
        "SPARE 的两处不一致均为 Qwen2.5-3B Conversation、4096 KV、8 QPS："
        "GPU 积压增长为 11，模拟器分别预测为 3、4，因此模拟器误判通过。",
    ]
    (OUT / "TABLE_NO_FIXED256_CN.md").write_text("\n".join(report) + "\n")
    print(f"wrote {len(rows)} paired points and {len(cell_rows)} model/workload/method rows")


if __name__ == "__main__":
    main()
