"""Plot the KV-blind capacity control at a common KV configuration."""
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
OUT = ROOT / "artifacts/www_kv_blind_deferral_20261021_r3/analysis"
THETAS = (1.0, 2.0, 4.0, 8.0)


def main():
    with (OUT / "capacity_by_limit.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    points = {(row["source"], int(row["kv_blocks"]), float(row["theta_s"])): row
              for row in rows}
    assert all(theta > 0 for theta in THETAS)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True,
                             sharey=True)
    for ax, source in zip(axes, ("code", "conversation")):
        full = [float(points[source, 2048, theta]["full_capacity_qps"])
                for theta in THETAS]
        blind = [float(points[source, 2048, theta]["kv_blind_capacity_qps"])
                 for theta in THETAS]
        assert full == sorted(full) and blind == sorted(blind)
        ax.plot(THETAS, full, color="#25313b", linewidth=1.8, marker="o",
                markersize=5, label="Full SPARE", zorder=4)
        ax.plot(THETAS, blind, color="#b25a44", linewidth=1.5,
                linestyle="--", marker="s", markersize=4.8,
                label="KV-blind deferral", zorder=3)
        ax.set_title(source.capitalize(), pad=6)
        ax.set_xscale("log", base=2)
        ax.set_xticks(THETAS, ["1", "2", "4", "8"])
        ax.set_xlim(.85, 9.3)
        ax.set_ylim(2.15, 4.65)
        ax.set_yticks((2.5, 3.0, 3.5, 4.0, 4.5))
        ax.grid(axis="y", color="#e1e4e6", linewidth=.55)
        ax.set_axisbelow(True)
        ax.set_xlabel("Mean-TTFT limit (s)")
    axes[0].set_ylabel("Maximum input rate (QPS)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(.5, .005), frameon=False)
    fig.subplots_adjust(left=.09, right=.98, top=.89, bottom=.24, wspace=.12)
    stem = OUT / "kv_blind_capacity_kv2048"
    require_matplotlib_panel_alignment(
        fig, json_out=f"{stem}.alignment.json", strict=True)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
