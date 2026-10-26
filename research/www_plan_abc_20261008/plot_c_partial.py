"""Plot descriptive C effects while making the missing test cell explicit."""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/www_plan_abc_20261008_r1/c/analysis"


def main() -> None:
    summary = json.loads((OUT / "partial_summary.json").read_text())
    assert not summary["confirmatory"] and summary["accepted_cells"] == 23
    with (OUT / "partial_cell_means.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {(row["cell"], row["arm"]): row for row in rows}
    windows = sorted({row["window"] for row in rows})
    references = ("fixed_selected", "guarded_selected")
    colors = {768: "#16806f", 2048: "#bf5b3d"}
    marks = {"steady": "o", "burst_stop": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, arm, title in zip(axes, references, ("Calibrated native fixed 256", "Guarded fixed candidate 256")):
        by_window = defaultdict(list)
        for cell in sorted({row["cell"] for row in rows}):
            base = indexed[cell, arm]
            spare = indexed[cell, "spare"]
            effect = 100 * (float(base["ttft_ms_mean"]) - float(spare["ttft_ms_mean"])) / float(base["ttft_ms_mean"])
            window = base["window"]
            by_window[window].append(effect)
            y = windows.index(window) + (-.13 if base["shape"] == "steady" else .13)
            y += -.04 if int(base["capacity"]) == 768 else .04
            ax.scatter(effect, y, color=colors[int(base["capacity"])],
                       marker=marks[base["shape"]], s=38, alpha=.8,
                       edgecolor="white", linewidth=.4)
        for window, values in by_window.items():
            ax.scatter(st.fmean(values), windows.index(window), marker="D",
                       color="#303840", s=62, edgecolor="white", linewidth=.5)
        ax.axvline(0, color="#555555", linewidth=.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("SPARE mean TTFT improvement (%)")
        ax.grid(axis="x", color="#e0e0e0", linewidth=.5)
        ax.spines[["top", "right"]].set_visible(False)
    labels = [window.replace("_", " ") + (" (n=3)" if window == "conversation_test1" else "")
              for window in windows]
    axes[0].set_yticks(range(len(windows)), labels)
    axes[0].invert_yaxis()
    axes[0].scatter([], [], color=colors[768], label="768 blocks")
    axes[0].scatter([], [], color=colors[2048], label="2048 blocks")
    fig.suptitle("Exploratory C: 23/24 accepted cells; one A/A failure", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=2,
               loc="lower center", bbox_to_anchor=(.5, .01))
    fig.tight_layout(rect=(0, .07, 1, .94))
    fig.savefig(OUT / "figure_c_partial.png", dpi=200)
    fig.savefig(OUT / "figure_c_partial.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
