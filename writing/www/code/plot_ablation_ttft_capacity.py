"""Plot measured mean-TTFT-limit versus input-capacity ablation curves."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
POINTS = ROOT / "artifacts/www_component_ablation_mean_20261019_r3/analysis/points.csv"
OUT = ROOT / "writing/www/figures/ablation_ttft_capacity"
THETAS = (1.0, 2.0, 4.0, 8.0)
SOURCES = ("code", "conversation")
KVS = (1024, 2048, 4096)
ARMS = {
    "srp": ("No deferral (SRP)", "#767676", "s", "--"),
    "chunk_family": ("Chunk candidates only", "#AA7329", "D", "-."),
    "drop_candidate1": ("Without candidate 1", "#B64342", "v", "--"),
    "constant1": ("Fixed n = 1", "#24845B", "^", ":"),
    "budget_only": ("Budget candidates only", "#42949E", "P", ":"),
    "spare": ("Full SPARE", "#0F4D92", "o", "-"),
}


def load_points():
    with POINTS.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    points = {}
    for row in rows:
        key = (row["source"], int(row["kv_blocks"]), row["arm"], float(row["theta_s"]))
        if key in points:
            raise ValueError(f"Duplicate capacity point: {key}")
        capacity = float(row["capacity_qps"])
        next_fail = float(row["next_fail_qps"])
        if not 0 < capacity < next_fail or not row["failure_causes"]:
            raise ValueError(f"Invalid capacity boundary: {key}")
        points[key] = capacity
    for source in SOURCES:
        for kv in KVS:
            for arm in ARMS:
                values = [points[(source, kv, arm, theta)] for theta in THETAS]
                if values != sorted(values):
                    raise ValueError(f"Nonmonotone capacity: {source}, {kv}, {arm}")
    return points


def plot_panel(ax, points, source, kv):
    for arm, (label, color, marker, linestyle) in ARMS.items():
        values = [points[(source, kv, arm, theta)] for theta in THETAS]
        ax.plot(THETAS, values, label=label, color=color, marker=marker,
                linestyle=linestyle, linewidth=1.45 if arm == "spare" else 1.05,
                markersize=4.3 if arm == "spare" else 3.3,
                markeredgecolor="white", markeredgewidth=.35,
                zorder=5 if arm == "spare" else 2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(THETAS, ["1", "2", "4", "8"])
    ax.set_xlim(.85, 9.2)
    ax.grid(axis="y", color="#D8DEE3", linewidth=.55)
    ax.set_axisbelow(True)
    ax.set_xlabel("Mean TTFT limit (s)")
    ax.set_title(f"{source.title()}  |  {kv:,} KV blocks", pad=6)


def save_figure(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        fig.savefig(OUT / f"{name}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def draw(points, kvs, name):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    width = 7.35 if len(kvs) == 3 else 7.1
    height = 5.35 if len(kvs) == 3 else 2.75
    fig, axes = plt.subplots(2 if len(kvs) == 3 else 1,
                             len(kvs) if len(kvs) == 3 else 2,
                             figsize=(width, height), squeeze=False)
    for row, source in enumerate(SOURCES):
        for col, kv in enumerate(kvs):
            ax = axes[row, col] if len(kvs) == 3 else axes[0, row]
            plot_panel(ax, points, source, kv)
            ax.set_ylabel("Maximum input QPS")
            vals = [points[(source, kv, arm, theta)]
                    for arm in ARMS for theta in THETAS]
            bottom, top = min(vals), max(vals)
            ax.set_ylim(max(0, bottom - .13 * (top - bottom) - .08),
                        top + .13 * (top - bottom) + .08)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(.5, .002), frameon=False,
               columnspacing=1.25, handlelength=2.1)
    fig.subplots_adjust(left=.075, right=.985, top=.945,
                        bottom=.17 if len(kvs) == 3 else .30,
                        wspace=.31, hspace=.43)
    save_figure(fig, name)


def main():
    points = load_points()
    draw(points, KVS, "ttft_capacity_all_kv")
    draw(points, (2048,), "ttft_capacity_kv2048")


if __name__ == "__main__":
    main()
