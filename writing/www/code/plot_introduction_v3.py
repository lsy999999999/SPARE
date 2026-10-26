"""Nine-capacity paired GPU TTFT figure for the WWW introduction."""

from collections import defaultdict
from pathlib import Path
import csv
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "artifacts/www_capacity_load_20261007_r1/analysis/runs.csv"
OUT = ROOT / "writing/www/figures/introduction_v3"
CAPACITIES = (512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144)
EXPECTED = (45.79, 39.54, 32.40, 18.04, 11.06, 2.20, -4.12, -3.66, -3.21)


def paired_values():
    repeats = defaultdict(list)
    with RUNS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            # The 2048-block sustained family uses 512 requests, not the
            # 128-request workloads shared by all nine capacity levels.
            if row["arm"] in {"arrival", "srp"} and "_sustained_" not in row["cell"]:
                repeats[row["cell"], row["arm"]].append(float(row["ttft_ms_mean"]))

    values = defaultdict(list)
    cells = {cell for cell, arm in repeats if arm == "arrival"}
    assert len(cells) == 108
    for cell in cells:
        spare, control = repeats[cell, "arrival"], repeats[cell, "srp"]
        assert len(spare) == len(control) == 4, cell
        capacity = int(cell.rsplit("_c", 1)[1])
        source = cell.split("_", 1)[0]
        assert capacity in CAPACITIES and source in {"code", "conversation"}
        improvement = 100 * (statistics.mean(control) - statistics.mean(spare)) / statistics.mean(control)
        values[capacity, source].append(improvement)

    for capacity in CAPACITIES:
        assert len(values[capacity, "code"]) == len(values[capacity, "conversation"]) == 6
        overall = statistics.mean(values[capacity, "code"] + values[capacity, "conversation"])
        assert abs(overall - EXPECTED[CAPACITIES.index(capacity)]) < 0.011
    return values


def main():
    plt.rcParams.update({
        "font.family": "Liberation Sans",
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    values = paired_values()
    xs = list(range(len(CAPACITIES)))
    code = [statistics.mean(values[c, "code"]) for c in CAPACITIES]
    conversation = [statistics.mean(values[c, "conversation"]) for c in CAPACITIES]
    overall = [(a + b) / 2 for a, b in zip(code, conversation)]

    fig, ax = plt.subplots(figsize=(3.35, 2.32), facecolor="white")
    fig.subplots_adjust(left=0.15, right=0.985, top=0.94, bottom=0.27)
    ax.axhline(0, color="#757575", lw=0.8, zorder=1)
    ax.grid(axis="y", color="#e4e7ea", lw=0.55, zorder=0)
    ax.plot(xs, code, color="#c65d32", marker="s", markersize=3.2,
            lw=1.15, label="Code (6)", zorder=3)
    ax.plot(xs, conversation, color="#147f82", marker="^", markersize=3.5,
            lw=1.15, label="Conversation (6)", zorder=3)
    ax.plot(xs, overall, color="#193b65", marker="o", markersize=4,
            lw=2, label="All (12)", zorder=4)
    ax.set_xlim(-0.18, 8.18)
    ax.set_ylim(-23, 62)
    ax.set_yticks((-20, 0, 20, 40, 60))
    ax.set_ylabel("Mean TTFT reduction vs SRP (%)", labelpad=3)
    ax.set_xticks(xs, [str(c) for c in CAPACITIES], rotation=45,
                  ha="right", rotation_mode="anchor")
    ax.set_xlabel("Configured KV blocks", labelpad=2)
    ax.tick_params(direction="out", length=2.5, width=0.7)
    ax.legend(loc="upper right", frameon=False, ncol=1, fontsize=6.5,
              handlelength=2.0, labelspacing=0.2, borderaxespad=0.2)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "gpu_capacity_nine.pdf")
    fig.savefig(OUT / "gpu_capacity_nine.svg")
    fig.savefig(OUT / "gpu_capacity_nine.png", dpi=600)
    fig.savefig(OUT / "gpu_capacity_nine.tiff", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
