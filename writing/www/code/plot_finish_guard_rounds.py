"""Plot the decision and conflict counts for the finish-guard ablation."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/www_component_behavior_20261020_r1/analysis/behavior_summary.csv"
OUT = ROOT / "writing/www/figures/protection_guard_20261022"


def load() -> dict[str, tuple[list[int], float]]:
    with SOURCE.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for workload in ("code", "conversation"):
        def row(arm):
            return next(r for r in rows if r["source"] == workload
                        and r["kv_blocks"] == "1024" and r["rate_qps"] == "2.0"
                        and r["arm"] == arm)

        full, unguarded = row("spare"), row("no_finish_guard")
        assert int(full["windows"]) == int(unguarded["windows"]) == 24
        assert int(full["finish_conflict_rounds_total"]) == 0
        counts = [int(full["finish_potential_rounds_total"]),
                  int(full["finish_toggle_changed_rounds_total"]),
                  int(unguarded["finish_conflict_rounds_total"])]
        result[workload] = counts, float(unguarded["finish_conflict_pct"])
    return result


def main() -> None:
    data = load()
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "Noto Sans CJK JP"],
        "font.size": 11,
        "axes.linewidth": 1.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    labels = ("达门槛的威胁轮", "guard 改变计划轮", "关 guard 后冲突轮")
    colors = ("#A9ADB0", "#0F4D92", "#B64342")
    hatches = ("", "//", "..")
    positions = (2, 1, 0)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharex=True)
    for ax, workload, title in zip(axes, ("code", "conversation"),
                                   ("Code", "Conversation")):
        counts, pct = data[workload]
        for i, (position, count) in enumerate(zip(positions, counts)):
            ax.barh(position, count, height=.57, color=colors[i],
                    edgecolor="#242424", linewidth=.7, hatch=hatches[i])
            annotation = f"{count} ({pct:.2f}%)" if i == 2 else str(count)
            ax.text(count + 10, position, annotation, va="center", fontsize=11)
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=10)
        ax.set_yticks(positions, labels)
        ax.set_xlim(0, 590)
        ax.set_xticks((0, 150, 300, 450))
        ax.grid(axis="x", color="#D9DCDE", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=7)
    fig.suptitle("完成保护的三层计数 | 1,024 KV 块，2 QPS", fontsize=15, y=.965)
    fig.supxlabel("决策轮数（每类负载 24 窗口合计）", fontsize=11, y=.14)
    fig.text(.5, .055,
             "前两项来自完整 SPARE 的同状态检查；红色来自关闭 guard 后的闭环轨迹。完整 SPARE 的实际冲突为 0。",
             ha="center", fontsize=9.5, color="#444444")
    fig.subplots_adjust(left=.165, right=.985, top=.77, bottom=.245, wspace=.29)
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"finish_guard_rounds.{suffix}", dpi=300,
                    facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
