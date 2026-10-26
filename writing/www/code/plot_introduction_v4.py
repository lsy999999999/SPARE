"""WWW single-column G4 scan: paired TTFT benefits vary with KV capacity.

Style revision of V3; its complete 108-scenario data and pairing are reused.
The separate Conversation-only batch has a different arrival shape and is
reported separately. Cohort lines describe observed means, not uncertainty.
"""
import csv
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_introduction_v3 import CAPACITIES, ROOT, paired_values

OUT = ROOT / "writing/www/figures/introduction_v4"


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
    ax.axhline(0, color="#747980", lw=0.7, zorder=1)
    ax.grid(axis="y", color="#e9ebee", lw=0.5, zorder=0)
    ax.plot(xs, overall, color="#293241", marker="o", markersize=3.6,
            lw=1.7, label="All scenarios", zorder=4)
    ax.plot(xs, code, color="#547FA6", marker="s", markersize=3.4,
            markerfacecolor="white", markeredgewidth=0.9, linestyle="--",
            lw=1.15, label="Code", zorder=3)
    ax.plot(xs, conversation, color="#9982AC", marker="^", markersize=3.8,
            markerfacecolor="white", markeredgewidth=0.9, linestyle="-.",
            lw=1.15, label="Conversation", zorder=3)
    ax.set_xlim(-0.18, 8.18)
    ax.set_ylim(-23, 62)
    ax.set_yticks((-20, 0, 20, 40, 60))
    ax.set_ylabel("Mean TTFT reduction vs SRP (%)", labelpad=3)
    ax.set_xticks(xs, [str(c) for c in CAPACITIES], rotation=45,
                  ha="right", rotation_mode="anchor")
    ax.set_xlabel("Configured KV blocks", labelpad=2)
    ax.tick_params(direction="out", length=2.5, width=0.7)
    ax.legend(loc="upper right", frameon=False, fontsize=6.5,
              handlelength=2.4, labelspacing=0.3, borderaxespad=0.3)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "capacity_means.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("capacity_blocks", "all_12_mean_pct", "code_6_mean_pct", "conversation_6_mean_pct"))
        writer.writerows(zip(CAPACITIES, overall, code, conversation))
    # One axes: multi-panel alignment is not applicable. Preserve exact width.
    fig.savefig(OUT / "gpu_capacity_nine.pdf")
    fig.savefig(OUT / "gpu_capacity_nine.svg")
    fig.savefig(OUT / "gpu_capacity_nine.png", dpi=600)
    fig.savefig(OUT / "gpu_capacity_nine.tiff", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
