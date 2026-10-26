"""Single-column introduction figures: scheduling decision and paired GPU results."""

from collections import defaultdict
from pathlib import Path
import csv
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "artifacts/www_evaluation_gpu_completion_20261003_r1/analysis/runs.csv"
OUT = ROOT / "writing/www/figures/introduction_v2"

BLUE = "#0F4D92"
TEAL = "#42949E"
RED = "#B64342"
GREEN = "#3D8056"
INK = "#202020"
RULE = "#B8B8B8"


def paired_improvements():
    grouped = defaultdict(list)
    with RUNS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["arm"] in {"arrival", "srp"}:
                grouped[row["cell"], row["arm"]].append(float(row["ttft_ms_mean"]))

    results = []
    for cell, arm in grouped:
        if arm != "arrival":
            continue
        spare = grouped[cell, "arrival"]
        control = grouped[cell, "srp"]
        assert len(spare) == len(control) == 4, cell
        baseline = statistics.mean(control)
        improvement = 100 * (baseline - statistics.mean(spare)) / baseline
        results.append((int(cell.rsplit("_c", 1)[1]), cell.split("_", 1)[0], cell, improvement))
    assert len(results) == 36
    return results


def export(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def decision_figure():
    fig, ax = plt.subplots(figsize=(3.35, 2.42))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.985, bottom=0.04)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.965, "Same order, one scheduling round", ha="left", va="top",
            color=INK, weight="bold", fontsize=9.2)
    ax.plot([0.02, 0.98], [0.87, 0.87], color=INK, lw=0.75)

    left, right = (0.02, 0.495), (0.525, 0.98)
    for x0, x1 in (left, right):
        ax.add_patch(Rectangle((x0, 0.47), x1 - x0, 0.32, facecolor="white",
                               edgecolor=RULE, lw=0.8))
    ax.plot([left[0], left[1]], [0.79, 0.79], color=GREEN, lw=2.2)
    ax.plot([right[0], right[1]], [0.79, 0.79], color=RED, lw=2.2)

    ax.text(0.05, 0.73, "Prefill finishes", color=INK, weight="bold", fontsize=8.2)
    ax.text(0.05, 0.625, "First token this round", color=GREEN, fontsize=7.8)
    ax.text(0.05, 0.52, "Keep its token quota", color=INK, fontsize=7.6)

    ax.text(0.555, 0.73, "Prefill remains partial", color=INK, weight="bold", fontsize=8.2)
    ax.text(0.555, 0.625, "KV grows now", color=RED, fontsize=7.8)
    ax.text(0.555, 0.52, "First token comes later", color=INK, fontsize=7.6)

    for x in (0.255, 0.755):
        ax.add_patch(FancyArrowPatch((x, 0.46), (x, 0.36), arrowstyle="-|>",
                                     mutation_scale=8, lw=0.9, color=INK))

    ax.text(0.50, 0.30, "SPARE keeps decode and finishing prefill;",
            ha="center", color=BLUE, weight="bold", fontsize=8.0)
    ax.text(0.50, 0.20, "it may defer partial prefill when the candidate",
            ha="center", color=INK, fontsize=7.8)
    ax.text(0.50, 0.10, "saves enough new physical KV blocks.",
            ha="center", color=INK, fontsize=7.8)
    export(fig, "fig1a_prefill_decision")


def gpu_figure(rows):
    fig, ax = plt.subplots(figsize=(3.35, 2.74))
    fig.subplots_adjust(left=0.19, right=0.98, top=0.80, bottom=0.19)
    ax.set_title("Same-order GPU comparison", loc="left", fontsize=9.2,
                 weight="bold", color=INK, pad=22)

    capacities = (768, 2048, 4096)
    means = []
    for i, capacity in enumerate(capacities):
        group = sorted((r for r in rows if r[0] == capacity), key=lambda r: r[2])
        assert len(group) == 12
        means.append(statistics.mean(r[3] for r in group))
        for source, color, offset in (("code", RED, -0.11),
                                      ("conversation", TEAL, 0.11)):
            selected = [r for r in group if r[1] == source]
            assert len(selected) == 6
            xs = [i + offset + (j - 2.5) * 0.012 for j in range(6)]
            ax.scatter(xs, [r[3] for r in selected], s=17, c=color, marker="o",
                       edgecolors="white", linewidths=0.35, zorder=3,
                       label=source.title() if i == 0 else None)

    assert all(abs(a - b) < 0.02 for a, b in zip(means, (39.64, 11.46, -2.92)))
    ax.plot(range(3), means, color=BLUE, lw=1.35, zorder=4, label="Mean")
    ax.scatter(range(3), means, s=42, marker="D", c=BLUE, edgecolors="white",
               linewidths=0.55, zorder=5)

    ax.axhline(0, color=INK, lw=0.75, ls="--", zorder=1)
    ax.set_xlim(-0.30, 2.30)
    ax.set_ylim(-32, 67)
    ax.set_xticks(range(3), [f"{c:,}" for c in capacities])
    ax.set_yticks([-20, 0, 20, 40, 60])
    ax.set_xlabel("Configured KV blocks", fontsize=8.0, color=INK)
    ax.set_ylabel("Mean TTFT reduction vs SRP (%)", fontsize=8.0, color=INK)
    ax.tick_params(labelsize=7.5, width=0.8, length=3)
    ax.grid(axis="y", color="#E0E0E0", lw=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
              fontsize=7.1, frameon=False, handlelength=1.5, columnspacing=0.75)
    export(fig, "fig1b_gpu_capacity")


def main():
    plt.rcParams.update({
        "font.family": "Liberation Sans",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    decision_figure()
    gpu_figure(paired_improvements())


if __name__ == "__main__":
    main()
