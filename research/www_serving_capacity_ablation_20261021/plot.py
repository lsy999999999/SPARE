"""Plot absolute serving capacity as the mean-TTFT limit changes."""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "plotting")
from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/www_serving_capacity_ablation_20261021/analysis"
THETAS = (1, 2, 4, 8)
SERIES = (
    ("spare", "Full SPARE", "#222b34", "o", "-", 2.0),
    ("srp", "No deferral (SRP)", "#888888", "s", "--", 1.2),
    ("drop_candidate1", "Without candidate 1", "#bd6345", "v", "-.", 1.3),
    ("no_finish_guard", "Without finish guard", "#29827b", "D", ":", 1.3),
    ("no_arrival", "Without arrival term", "#8a72aa", "^", "--", 1.3),
    ("fixed512", "Fixed budget 512", "#bd9b31", "P", "-.", 1.3),
)


def main():
    with (OUT / "capacity_by_limit.csv").open(newline="") as handle:
        records = list(csv.DictReader(handle))
    points = {(row["arm"], float(row["theta_s"])):
              float(row["variant_capacity_qps"])
              for row in records if row["source"] == "conversation"
              and int(row["kv_blocks"]) == 2048}
    assert len(points) == len(SERIES) * len(THETAS)
    assert all(theta > 0 for theta in THETAS)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": .8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(7.2, 3.55))
    for arm, label, color, marker, linestyle, width in SERIES:
        values = [points[arm, theta] for theta in THETAS]
        assert values == sorted(values), (arm, values)
        ax.plot(THETAS, values, label=label, color=color, marker=marker,
                linestyle=linestyle, linewidth=width, markersize=5,
                markeredgecolor="white", markeredgewidth=.6,
                zorder=5 if arm == "spare" else 2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(THETAS, ["1", "2", "4", "8"])
    ax.set_xlim(.86, 9.2)
    ax.set_ylim(2.25, 4.7)
    ax.set_yticks((2.5, 3.0, 3.5, 4.0, 4.5))
    ax.set_xlabel("Mean-TTFT limit (s)")
    ax.set_ylabel("Maximum input rate (QPS)")
    ax.grid(axis="y", color="#dfe2e5", linewidth=.55)
    ax.set_axisbelow(True)
    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(.5, .005),
               columnspacing=1.5, handlelength=2.2, frameon=False)
    fig.subplots_adjust(left=.11, right=.98, bottom=.30, top=.96)
    stem = OUT / "capacity_curve_conversation_2048"
    require_matplotlib_panel_alignment(
        fig, json_out=f"{stem}.alignment.json", strict=True)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
