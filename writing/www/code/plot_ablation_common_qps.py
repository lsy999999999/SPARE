"""Reproduce the three-KV ablation panels with mean TTFT at a common 2 QPS."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PRIMARY = ROOT / "artifacts/www_component_ablation_mean_20261020_r4"
SUPPLEMENT = ROOT / "artifacts/www_ablation_shared_settings_20261023_r1"
OUTPUT = ROOT / "writing/www/nowlatex/www(3)/figures"
DATA = SUPPLEMENT / "analysis/ablation_common_2qps.csv"
KVS = (1024, 3584, 5120)
ARMS = ("spare", "srp", "constant8", "no_arrival")
LABELS = ("SPARE (Ours)", "w/o Deferral", "w/o Adaptive Threshold",
          "w/o Arrival Extrapolation")
COLORS = ("#378ec3", "#76acd0", "#a0c6e0", "#cddff0")
RATE = 2.0


def load_rows():
    expected = {sample["name"] for sample in json.loads(
        (SUPPLEMENT / "design.json").read_text())["samples"]}
    assert len(expected) == 24
    records = []
    for kv in KVS:
        archive = PRIMARY if kv == 1024 else SUPPLEMENT
        for arm, label in zip(ARMS, LABELS):
            group = json.loads((archive / "groups" /
                                f"conversation_c{kv}_{arm}.json").read_text())
            source = archive / "rates" / f"conversation_c{kv}_{arm}_r2.json"
            rows = json.loads(source.read_text())["rows"]
            assert {row["name"] for row in rows} == expected
            assert all(row["valid"] for row in rows)
            means = {key: sum(row[key] for row in rows) / len(rows) for key in
                     ("mean_ttft_ms", "p99_tbt_ms", "backlog_drift")}
            capacity = group["capacities"]["4000"]["last_pass_qps"]
            serviceable = (RATE <= capacity and means["mean_ttft_ms"] <= 4000
                           and means["p99_tbt_ms"] <= 200
                           and means["backlog_drift"] <= max(2, RATE))
            records.append(dict(kv_blocks=kv, method=label, arm=arm,
                                capacity_qps_4s=capacity, offered_qps=RATE,
                                **means, serviceable_at_2_qps=serviceable,
                                windows=len(rows), source=str(source.relative_to(ROOT))))
    assert [(r["kv_blocks"], r["arm"]) for r in records if not r["serviceable_at_2_qps"]] == [
        (1024, "srp")]
    DATA.parent.mkdir(parents=True, exist_ok=True)
    with DATA.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def style_axis(axis):
    axis.set_facecolor("#f9fafc")
    axis.set_axisbelow(True)
    axis.grid(axis="y", linestyle=":", color="#bdc5cd", linewidth=0.55)
    for spine in axis.spines.values():
        spine.set_color("#30333a")
        spine.set_linewidth(0.8)
    axis.tick_params(labelsize=7, width=0.8, length=3)


def main():
    rows = load_rows()
    by_key = {(row["kv_blocks"], row["arm"]): row for row in rows}
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    figure = plt.figure(figsize=(7.4, 3.05), facecolor="white")
    grid = figure.add_gridspec(2, 2, left=.085, right=.985, bottom=.16, top=.82,
                              width_ratios=(1, 1), height_ratios=(.23, .77),
                              wspace=.255, hspace=.075)
    capacity_ax = figure.add_subplot(grid[:, 0])
    ttft_top = figure.add_subplot(grid[0, 1])
    ttft_ax = figure.add_subplot(grid[1, 1], sharex=ttft_top)
    for axis in (capacity_ax, ttft_top, ttft_ax):
        style_axis(axis)

    centers = np.arange(3)
    offsets = np.array((-.285, -.095, .095, .285))
    width = .18
    for i, arm in enumerate(ARMS):
        xs = centers + offsets[i]
        capacities = [by_key[kv, arm]["capacity_qps_4s"] for kv in KVS]
        latencies = [by_key[kv, arm]["mean_ttft_ms"] for kv in KVS]
        appearance = dict(width=width, color=COLORS[i], edgecolor="white", linewidth=.55)
        capacity_ax.bar(xs, capacities, **appearance)
        ttft_ax.bar(xs, latencies, **appearance)
        ttft_top.bar(xs, latencies, **appearance)

    capacity_ax.set_xlim(-.52, 2.72)
    capacity_ax.set_ylim(0, 7.15)
    capacity_ax.set_yticks(range(8))
    capacity_ax.set_ylabel("Serving Capacity (QPS)", fontsize=8)
    capacity_ax.set_xticks(centers, ("Low KV", "Medium KV", "High KV"))
    ttft_ax.set_xlim(-.52, 2.72)
    ttft_ax.set_ylim(0, 1800)
    ttft_ax.set_yticks(range(0, 1801, 300))
    ttft_ax.set_ylabel("Mean TTFT (ms)", fontsize=8)
    ttft_ax.set_xticks(centers, ("Low KV", "Medium KV", "High KV"))
    ttft_top.set_ylim(4400, 5250)
    ttft_top.set_yticks((4500, 5000))
    ttft_top.tick_params(axis="x", bottom=False, labelbottom=False)
    ttft_top.spines["bottom"].set_visible(False)
    ttft_ax.spines["top"].set_visible(False)
    for axis, y, slope in ((ttft_top, 0, 1), (ttft_ax, 1, -1)):
        for x in (0, 1):
            axis.plot((x-.012, x+.012), (y-.025*slope, y+.025*slope),
                      transform=axis.transAxes, color="#30333a", lw=.8,
                      clip_on=False)

    outlier_x = offsets[1]
    outlier = by_key[1024, "srp"]["mean_ttft_ms"]
    ttft_top.plot(outlier_x, outlier, marker="o", color="#30333a", ms=2.3)
    ttft_top.text(outlier_x+.09, outlier+60, "4,899*", fontsize=7.3,
                  va="bottom", color="#22262b")
    ttft_top.text(2.68, 5150, "* above capacity", ha="right", fontsize=6,
                  va="top", color="#343943")

    figure.text(.285, .973, "(a) Serving Capacity", ha="center", va="top",
                fontsize=9.4, fontweight="bold")
    figure.text(.762, .973, "(b) Mean TTFT at 2 QPS", ha="center", va="top",
                fontsize=9.4, fontweight="bold")
    handles = [Patch(facecolor=color, edgecolor="white", label=label)
               for color, label in zip(COLORS, LABELS)]
    for x in (.09, .56):
        figure.legend(handles=handles, ncol=2, loc="upper left",
                      bbox_to_anchor=(x, .94), frameon=False, fontsize=5.7,
                      handlelength=1.2, handleheight=.8, columnspacing=1.2,
                      labelspacing=.45, borderaxespad=0)

    capacity_ax.annotate("Better", xy=(2.58, 6.86), xytext=(2.58, 6.17),
                         arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#22262b"),
                         ha="center", va="center", rotation=90, fontsize=6.8)
    ttft_ax.annotate("Better", xy=(2.58, 1050), xytext=(2.58, 1550),
                     arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#22262b"),
                     ha="center", va="center", rotation=90, fontsize=6.8)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        figure.savefig(OUTPUT / f"ablation_common_qps.{extension}", dpi=300,
                       facecolor="white")
    plt.close(figure)
    print(f"Wrote {len(rows)} cells to {DATA} and figure to {OUTPUT}")


if __name__ == "__main__":
    main()
