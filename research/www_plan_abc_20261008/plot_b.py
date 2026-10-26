"""Plot all same-state branch effects and their time-segment components."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/www_plan_abc_20261008_r1/b_v2"


def main():
    with (OUT / "analysis/states.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads((OUT / "analysis/summary.json").read_text())
    windows = [f"{source}_new{i}" for source in ("code", "conversation") for i in range(3)]
    color = {768: "#16806f", 2048: "#bf5b3d"}
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), gridspec_kw={"width_ratios": [1.45, 1]})
    for row in rows:
        window = windows.index(row["window"])
        capacity = int(row["capacity"])
        offset = (-.18 if capacity == 768 else .18) + (-.055 if row["stage"] == "early" else .055)
        axes[0].scatter(float(row["mean_benefit_ms"]), window + offset,
                        marker="o" if row["stage"] == "early" else "^",
                        c=color[capacity], s=38, alpha=.85,
                        edgecolor="white", linewidth=.35)
    axes[0].axvline(0, color="#454545", linewidth=.8)
    axes[0].set_yticks(range(6), [w.replace("_", " ") for w in windows])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Base - SPARE remaining wait (ms / request)")
    axes[0].set_title("24 matched states, 4 repeats each", fontsize=10)
    for capacity in color:
        axes[0].scatter([], [], c=color[capacity], label=f"{capacity} blocks")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    values = [summary["first_round_benefit_ms"], summary["later_benefit_ms"],
              summary["mean_benefit_ms"]]
    axes[1].barh([0, 1, 2], values, color=["#16806f", "#bf5b3d", "#424a55"], height=.54)
    axes[1].axvline(0, color="#454545", linewidth=.8)
    axes[1].set_yticks([0, 1, 2], ["Action round", "Later rounds", "Net"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean wait difference (ms / request)")
    axes[1].set_title("Time segments, not a cost attribution", fontsize=10)
    low, high = summary["window_bootstrap_95ci_ms"]
    net = summary["mean_benefit_ms"]
    axes[1].errorbar(net, 2, xerr=[[net-low], [high-net]], fmt="none",
                     ecolor="#111111", capsize=3, linewidth=1)
    for ax in axes:
        ax.grid(axis="x", color="#e0e0e0", linewidth=.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "analysis/figure_b.png", dpi=200)
    fig.savefig(OUT / "analysis/figure_b.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
