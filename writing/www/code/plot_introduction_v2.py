"""Introduction figure: KV-growth decision and paired GPU TTFT results."""

from collections import defaultdict
from pathlib import Path
import csv
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "artifacts/www_evaluation_gpu_completion_20261003_r1/analysis/runs.csv"
OUT = ROOT / "writing/www/figures/introduction_v2"

BLUE = "#0F4D92"
TEAL = "#42949E"
RED = "#B64342"
GREEN = "#8BCF8B"
GRAY = "#767676"
LIGHT = "#CFCECE"
TEXT = "#272727"


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
        baseline_mean = statistics.mean(control)
        improvement = 100 * (baseline_mean - statistics.mean(spare)) / baseline_mean
        results.append((int(cell.rsplit("_c", 1)[1]), cell.split("_", 1)[0], cell, improvement))
    assert len(results) == 36
    return results


def plot_decision(ax):
    ax.set_xlim(-0.08, 1.04)
    ax.set_ylim(0, 2.38)
    ax.axis("off")
    ax.text(0, 2.32, "A  The decision within one scheduling round", weight="bold", size=10.5, color=TEXT)
    ax.text(0, 2.00, "Same order; decode and first-token completion remain protected", size=9, color=GRAY)

    bar_x, bar_w, bar_h = 0.18, 0.81, 0.42
    occupied = 0.24
    finish = 0.20
    partial = 0.31
    for y in (1.28, 0.53):
        ax.add_patch(Rectangle((bar_x, y), bar_w, bar_h, facecolor="white", edgecolor=GRAY, lw=1))
        ax.add_patch(Rectangle((bar_x, y), occupied * bar_w, bar_h, facecolor=LIGHT, edgecolor="white"))
        ax.add_patch(Rectangle((bar_x + occupied * bar_w, y), finish * bar_w, bar_h,
                               facecolor=GREEN, edgecolor="white"))

    ax.text(0, 1.49, "Base", va="center", weight="bold", size=10, color=TEXT)
    ax.add_patch(Rectangle((bar_x + (occupied + finish) * bar_w, 1.28), partial * bar_w,
                           bar_h, facecolor=RED, edgecolor="white"))
    ax.text(0, 0.74, "SPARE", va="center", weight="bold", size=10, color=BLUE)
    ax.text(bar_x + occupied * bar_w / 2, 1.49, "Existing KV", ha="center", va="center", size=8.2, color=TEXT)
    ax.text(bar_x + (occupied + finish / 2) * bar_w, 1.49, "Finish prefill",
            ha="center", va="center", size=8.2, color=TEXT)
    ax.text(bar_x + (occupied + finish + partial / 2) * bar_w, 1.49,
            "Partial prefill", ha="center", va="center", size=8.2, color="white")
    ax.text(bar_x + (occupied + finish + partial + (1 - occupied - finish - partial) / 2) * bar_w,
            1.49, "Free", ha="center", va="center", size=8.2, color=TEXT)
    ax.text(bar_x + occupied * bar_w / 2, 0.74, "Existing KV", ha="center", va="center", size=8.2, color=TEXT)
    ax.text(bar_x + (occupied + finish / 2) * bar_w, 0.74, "Finish prefill",
            ha="center", va="center", size=8.2, color=TEXT)
    ax.text(bar_x + (occupied + finish + (1 - occupied - finish) / 2) * bar_w,
            0.74, "Space retained for later work", ha="center", va="center",
            size=8.2, color=BLUE)
    ax.text(bar_x, 0.17, "Schematic: partial prefill adds KV before the request can emit its first token.",
            size=8.2, color=GRAY)


def plot_gpu(ax, rows):
    ax.set_title("B  Paired GPU results: benefit depends on KV capacity", loc="left",
                 weight="bold", fontsize=10.5, pad=13)
    capacities = (768, 2048, 4096)
    colors = {"code": RED, "conversation": TEAL}
    means = []
    for i, cap in enumerate(capacities):
        group = sorted((r for r in rows if r[0] == cap), key=lambda r: r[2])
        assert len(group) == 12
        means.append(statistics.mean(r[3] for r in group))
        for source in ("code", "conversation"):
            vals = sorted((r for r in group if r[1] == source), key=lambda r: r[2])
            assert len(vals) == 6
            x = [i + (-0.13 if source == "code" else 0.13) + (j - 2.5) * 0.014
                 for j in range(6)]
            ax.scatter(x, [r[3] for r in vals], s=42, color=colors[source], alpha=0.88,
                       edgecolor="white", linewidth=0.6, zorder=3,
                       label={"code": "Code", "conversation": "Conversation"}[source]
                       if i == 0 else None)

    assert all(abs(a - b) < 0.02 for a, b in zip(means, (39.64, 11.46, -2.92)))
    ax.plot(range(3), means, color=BLUE, lw=2.2, zorder=4, label="Mean across 12 scenarios")
    ax.scatter(range(3), means, s=95, marker="D", facecolor=BLUE, edgecolor="white",
               linewidth=0.8, zorder=5)
    for i, mean in enumerate(means):
        offset = (42, 0) if i < 2 else (0, -16)
        ax.annotate(f"{mean:+.1f}%", (i, mean), xytext=offset, textcoords="offset points",
                    ha="left" if i < 2 else "center", va="center" if i < 2 else "top", fontsize=9,
                    weight="bold", color=BLUE)

    ax.axhline(0, color=GRAY, lw=0.9, ls="--", zorder=1)
    ax.set_ylim(-36, 77)
    ax.set_xlim(-0.33, 2.33)
    ax.set_xticks(range(3), [f"{c:,}" for c in capacities])
    ax.set_xlabel("Configured KV blocks", fontsize=9.5)
    ax.set_ylabel("Mean TTFT reduction vs same-order control (%)", fontsize=9.5)
    ax.set_yticks([-20, 0, 20, 40, 60])
    ax.grid(axis="y", color="#e5e5e5", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=3, fontsize=8, frameon=False,
              columnspacing=1.2, handlelength=1.7)
    ax.text(0.01, -0.35, "Positive: SPARE is faster. Negative: the control is faster.",
            transform=ax.transAxes, fontsize=8, color=GRAY)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    rows = paired_improvements()
    fig = plt.figure(figsize=(7.2, 5.0), facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=(1.15, 1.85), hspace=0.32,
                            left=0.105, right=0.98, top=0.97, bottom=0.16)
    plot_decision(fig.add_subplot(grid[0, 0]))
    plot_gpu(fig.add_subplot(grid[1, 0]), rows)
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"kv_progress_and_ttft.{extension}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
