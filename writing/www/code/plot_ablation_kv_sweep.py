"""Plot paired component effects across the three measured KV capacities."""
from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "writing/www/figures/ablation_unified/unified_24windows.csv"
ARRIVAL_SOURCE = ROOT / "writing/www/manuscript/ablation_20261020/all_kv_24windows.csv"
OUT = ROOT / "writing/www/figures/ablation_kv_sweep"
sys.path.insert(0, str(Path.home() / ".codex/skills/nature-figure/scripts"))
from audit_panel_alignment import require_matplotlib_panel_alignment

KVS = (1024, 2048, 4096)
SOURCES = ("code", "conversation")
ARMS = ("srp", "drop_candidate1", "no_arrival", "no_finish_guard")
STYLES = {
    "srp": ("SRP (no candidates)", "#62666c", "s"),
    "drop_candidate1": ("Without candidate 1", "#bc513c", "o"),
    "no_arrival": ("Without arrival term", "#237d86", "D"),
    "no_finish_guard": ("Without completion protection", "#b1842f", "^"),
}


def load_rows():
    with SOURCE.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = {
        (row["source"], int(row["kv_blocks"]), row["arm"]): row
        for row in rows
        if row["source"] in SOURCES and int(row["kv_blocks"]) in KVS
        and row["arm"] in (*ARMS, "spare")
    }
    with ARRIVAL_SOURCE.open(newline="") as handle:
        arrival_rows = [row for row in csv.DictReader(handle)
                        if row["arm"] == "no_arrival"]
    for row in arrival_rows:
        source, kv = row["source"], int(row["kv_blocks"])
        full = selected[source, kv, "spare"]
        selected[source, kv, "no_arrival"] = {
            "windows": row["windows"],
            "capacity_delta_vs_full_qps": str(
                float(row["capacity_qps"]) - float(full["capacity_qps"])),
            "capacity_delta_low_qps": row["capacity_delta_95ci_low_qps"],
            "capacity_delta_high_qps": row["capacity_delta_95ci_high_qps"],
            "ttft_delta_vs_full_ms": str(
                float(row["mean_ttft_ms"]) - float(full["mean_ttft_at_2qps_ms"])),
            "ttft_delta_low_ms": row["ttft_delta_95ci_low_ms"],
            "ttft_delta_high_ms": row["ttft_delta_95ci_high_ms"],
            "finish_conflict_pct": row["finish_conflict_pct"],
        }
    assert len(selected) == len(SOURCES) * len(KVS) * (len(ARMS) + 1)
    assert all(int(row["windows"]) == 24 for row in selected.values())
    return selected


def paired_series(rows, source, arm, field, low, high):
    values = [float(rows[source, kv, arm][field]) for kv in KVS]
    lower = [float(rows[source, kv, arm][low]) for kv in KVS]
    upper = [float(rows[source, kv, arm][high]) for kv in KVS]
    values = [0.0 if abs(val) < 1e-10 else val for val in values]
    assert all(lo - 1e-10 <= val <= hi + 1e-10
               for lo, val, hi in zip(lower, values, upper))
    return values, [[max(0, val - lo) for val, lo in zip(values, lower)],
                    [max(0, hi - val) for val, hi in zip(values, upper)]]


def main():
    rows = load_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })
    fig = plt.figure(figsize=(7.2, 7.15), facecolor="white")
    lefts = (0.115, 0.565)
    bottoms = (0.66, 0.375, 0.09)
    axes = [[fig.add_axes((left, bottom, 0.365, 0.215))
             for left in lefts] for bottom in bottoms]
    fig.suptitle("Where KV capacity changes the ablation result", y=0.992,
                 fontsize=10, fontweight="bold")
    legend = [Line2D([0], [0], color=color, marker=marker, lw=1.2,
                     markersize=4.5, label=label)
              for label, color, marker in STYLES.values()]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.963),
               ncol=2, columnspacing=2.2, handlelength=2, fontsize=7)

    for col, source in enumerate(SOURCES):
        cap_ax, ttft_ax, conflict_ax = [axes[row][col] for row in range(3)]
        cap_ax.set_title("Code" if source == "code" else "Conversation",
                         fontsize=8.5, fontweight="bold", pad=8)
        for row_num, ax in enumerate((cap_ax, ttft_ax, conflict_ax)):
            ax.set_xlim(-0.31, 2.31)
            ax.set_xticks(range(3), [f"{kv:,}" for kv in KVS])
            ax.tick_params(axis="both", labelsize=7, length=3, width=0.6)
            ax.grid(axis="y", color="#e5e8e9", linewidth=0.55, zorder=0)
            ax.text(-0.05, 1.015, "abcdef"[row_num * 2 + col],
                    transform=ax.transAxes, va="bottom", ha="right",
                    fontsize=8, fontweight="bold")
            if row_num < 2:
                ax.tick_params(axis="x", labelbottom=False)
        cap_ax.axhline(0, color="#363a3c", lw=0.8, ls=(0, (3, 2)), zorder=1)
        for arm, offset in zip(ARMS, (-0.09, -0.03, 0.03, 0.09)):
            _, color, marker = STYLES[arm]
            y, err = paired_series(rows, source, arm,
                                   "capacity_delta_vs_full_qps",
                                   "capacity_delta_low_qps", "capacity_delta_high_qps")
            x = [i + offset for i in range(3)]
            cap_ax.errorbar(x, y, yerr=err, fmt=marker + "-", color=color,
                            markersize=4, lw=1.1, elinewidth=0.75,
                            capsize=2, capthick=0.75, zorder=3)
        cap_ax.set_ylim((-1.65, 0.82) if source == "code" else (-2.3, 1.12))

        ttft_ax.axhline(0, color="#363a3c", lw=0.8, ls=(0, (3, 2)), zorder=1)
        for arm, offset in zip(ARMS[1:], (-0.07, 0, 0.07)):
            _, color, marker = STYLES[arm]
            y, err = paired_series(rows, source, arm,
                                   "ttft_delta_vs_full_ms", "ttft_delta_low_ms",
                                   "ttft_delta_high_ms")
            x = [i + offset for i in range(3)]
            ttft_ax.errorbar(x, y, yerr=err, fmt=marker + "-", color=color,
                             markersize=4, lw=1.1, elinewidth=0.75,
                             capsize=2, capthick=0.75, zorder=3)
            if source == "conversation" and arm == "drop_candidate1":
                ttft_ax.plot(x[0], y[0], "x", color="#181a1b", markersize=8,
                             markeredgewidth=1.2, zorder=5)
        ttft_ax.set_ylim((-85, 375) if source == "code" else (-180, 1840))

        conflict_ax.axhline(0, color="#363a3c", lw=0.8, ls=(0, (3, 2)), zorder=1)
        conflict = [float(rows[source, kv, "no_finish_guard"]["finish_conflict_pct"])
                    for kv in KVS]
        _, color, marker = STYLES["no_finish_guard"]
        conflict_ax.plot(range(3), conflict, marker=marker, color=color,
                         lw=1.3, markersize=5, zorder=3)
        conflict_ax.set_ylim((-0.45, 8.8) if source == "code" else (-0.65, 13.4))
        for i, value in enumerate(conflict[:2]):
            conflict_ax.annotate(f"{value:.2f}%", (i, value),
                                 xytext=(0, 6 if i == 0 else 26),
                                 textcoords="offset points", ha="center",
                                 fontsize=6.5, color=color)
        conflict_ax.set_xlabel("KV blocks", fontsize=7)

    axes[0][0].set_ylabel("2-s capacity change vs Full (QPS)", fontsize=7)
    axes[1][0].set_ylabel("Mean TTFT change at 2 QPS (ms)", fontsize=7)
    axes[2][0].set_ylabel("Finish conflict among eligible rounds (%)", fontsize=7)
    fig.text(0.5, 0.022,
             "Points: 24 paired windows/workload; whiskers: paired 95% intervals. "
             "Black x: 2 QPS exceeds variant capacity.",
             ha="center", va="center", fontsize=6)
    fig.canvas.draw()
    base = OUT / "ablation_by_kv"
    require_matplotlib_panel_alignment(
        fig, json_out=str(base) + ".alignment.json",
        overlay_svg=str(base) + ".alignment.svg",
        panel_ids=list("abcdef"),
        row_groups=(("a", "b"), ("c", "d"), ("e", "f")),
        column_groups=(("a", "c", "e"), ("b", "d", "f")),
        strict=True)
    fig.savefig(str(base) + ".pdf")
    fig.savefig(str(base) + ".svg")
    fig.savefig(str(base) + ".png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
