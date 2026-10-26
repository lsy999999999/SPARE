"""Generate publication-ready plots for the shared native-V1 baseline matrix."""
from __future__ import annotations

import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "writing/www/experiments/native_simulator_20260929/summary.json"
OUTPUT = ROOT / "writing/www/figures/baselines_20260929"

LABELS = {
    "arrival": "Ours (arrival)",
    "apt_serve_kv_rule": "Apt-Serve KV rule",
    "mc_sf_oracle": "MC-SF oracle",
    "uniboost_static": "UniBoost static",
    "past_future": "Past-Future",
    "tie_oracle_proxy": "TIE oracle-proxy",
    "ssjf_oracle": "SSJF oracle",
    "trail_oracle": "TRAIL oracle",
    "sarathi": "Sarathi rule",
    "sola_rule": "SOLA rule",
}
ORDER = tuple(LABELS)
ORACLES = {"mc_sf_oracle", "tie_oracle_proxy", "ssjf_oracle", "trail_oracle"}

COLORS = {
    "arrival": "#b3261e",
    "apt_serve_kv_rule": "#2a6f97",
    "mc_sf_oracle": "#6a4c93",
    "uniboost_static": "#2d6a4f",
    "past_future": "#d97706",
    "tie_oracle_proxy": "#7b2cbf",
    "ssjf_oracle": "#5a189a",
    "trail_oracle": "#9d4edd",
    "sarathi": "#4d7c0f",
    "sola_rule": "#9f1239",
}


def load():
    rows = json.loads(INPUT.read_text())
    assert len(rows) == 416 and all(row["complete"] for row in rows)
    return rows


def aggregate(rows, arm, field):
    return st.fmean(row["metrics"][field] for row in rows if row["arm"] == arm)


def bar_style(arm):
    return dict(color=COLORS[arm], edgecolor="#202124", linewidth=.55,
                hatch="///" if arm in ORACLES else None)


def format_ms(value, _):
    return f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"


def latency_panel(ax, rows, mean_field, p99_field, title):
    y = list(range(len(ORDER)))
    means = [aggregate(rows, arm, mean_field) for arm in ORDER]
    p99s = [aggregate(rows, arm, p99_field) for arm in ORDER]
    ax.barh([v + .18 for v in y], p99s, height=.34, color="#d8dce2",
            edgecolor="#6b7280", linewidth=.5, label="P99")
    for index, arm in enumerate(ORDER):
        ax.barh(index - .18, means[index], height=.34, **bar_style(arm))
        ax.text(means[index] + max(p99s) * .012, index - .18,
                f"{means[index] / 1000:.2f}", va="center", fontsize=7.4)
    ax.set_yticks(y, [LABELS[arm] for arm in ORDER])
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Latency (ms); labels show mean in seconds")
    ax.xaxis.set_major_formatter(FuncFormatter(format_ms))
    ax.grid(axis="x", color="#d9dde3", linewidth=.6)


def overview(rows):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.facecolor": "#fbfbfc", "figure.facecolor": "white",
        "axes.titlepad": 9, "axes.labelpad": 6,
    })
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 11.2))
    latency_panel(axes[0, 0], rows, "ttft_ms_mean", "ttft_ms_p99",
                  "A  Time to first token")
    latency_panel(axes[0, 1], rows, "e2e_ms_mean", "e2e_ms_p99",
                  "B  End-to-end latency")

    ax = axes[1, 0]
    y = list(range(len(ORDER)))
    attainment = [100 * aggregate(rows, arm, "slo_ttft5000_maxitl200_attainment")
                  for arm in ORDER]
    throughput = [aggregate(rows, arm, "request_throughput_rps") for arm in ORDER]
    for index, arm in enumerate(ORDER):
        ax.barh(index, attainment[index], height=.58, **bar_style(arm))
        ax.text(attainment[index] + .8, index + .20, f"{attainment[index]:.1f}%",
                va="center", fontsize=7.4)
    ax.set_yticks(y, [LABELS[arm] for arm in ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, 103)
    ax.set_xlabel("SLO attainment (%)  [TTFT <= 5 s, max ITL <= 200 ms]")
    ax.set_title("C  SLO attainment and request throughput", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#d9dde3", linewidth=.6)
    twin = ax.twiny()
    twin.scatter(throughput, y, marker="D", s=26, facecolor="white",
                 edgecolor="#111827", linewidth=.9, zorder=5)
    twin.set_xlim(0, max(throughput) * 1.12)
    twin.set_xlabel("Request throughput (req/s); diamond markers")
    twin.spines["top"].set_visible(True)
    twin.spines["right"].set_visible(False)

    ax = axes[1, 1]
    recompute = [sum(row["metrics"]["recomputed_tokens"] for row in rows
                     if row["arm"] == arm) / 1e6 for arm in ORDER]
    preemptions = [sum(row["metrics"]["preemptions"] for row in rows
                       if row["arm"] == arm) for arm in ORDER]
    for index, arm in enumerate(ORDER):
        ax.barh(index, recompute[index], height=.58, **bar_style(arm))
        ax.text(recompute[index] + .045, index + .20, f"{recompute[index]:.2f}M",
                va="center", fontsize=7.4)
    ax.set_yticks(y, [LABELS[arm] for arm in ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, max(recompute) * 1.19)
    ax.set_xlabel("Recomputed tokens across 32 scenarios (millions)")
    ax.set_title("D  Recovery cost", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#d9dde3", linewidth=.6)
    twin = ax.twiny()
    twin.scatter(preemptions, y, marker="o", s=27, facecolor="white",
                 edgecolor="#111827", linewidth=.9, zorder=5)
    twin.set_xlim(0, max(preemptions) * 1.19)
    twin.set_xlabel("Preemptions; circle markers")
    twin.spines["top"].set_visible(True)
    twin.spines["right"].set_visible(False)

    legend = [
        Patch(facecolor=COLORS["arrival"], edgecolor="#202124", label="Current method"),
        Patch(facecolor="#6a4c93", edgecolor="#202124", hatch="///",
              label="Uses realized output length (oracle/proxy)"),
        Patch(facecolor="white", edgecolor="#202124", label="No realized-length input"),
        Patch(facecolor="#d8dce2", edgecolor="#6b7280", label="P99 latency"),
        Line2D([], [], marker="D", color="none", markeredgecolor="#111827",
               markerfacecolor="white", label="Throughput"),
        Line2D([], [], marker="o", color="none", markeredgecolor="#111827",
               markerfacecolor="white", label="Preemptions"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .042),
               ncol=3, frameon=False)
    fig.suptitle("Shared native-V1 simulator baseline comparison",
                 fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=.10, right=.985, top=.91, bottom=.16,
                        wspace=.22, hspace=.38)
    fig.text(.5, .012,
             "32 scenarios, 64 requests each. Scenario-level metrics are averaged equally. "
             "Synthetic, uncalibrated timing; not measured GPU performance.",
             ha="center", fontsize=8, color="#4b5563")
    save(fig, "baseline_overview")


def relative_distribution(rows):
    by_name = {row["name"]: row for row in rows}
    methods = ORDER[1:]
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 7.4), sharey=True)
    for ax, profile in zip(axes, ("resource", "resource_staircase")):
        data = []
        for arm in methods:
            values = []
            for row in rows:
                if row["arm"] != arm or row["profile"] != profile:
                    continue
                prefix = row["name"][:-len(arm)]
                baseline = by_name[prefix + "arrival"]["metrics"]["ttft_ms_mean"]
                values.append(100 * (row["metrics"]["ttft_ms_mean"] / baseline - 1))
            data.append(values)
        boxes = ax.boxplot(data, vert=False, patch_artist=True, widths=.62,
                           whis=(0, 100), showfliers=False,
                           medianprops=dict(color="#111827", linewidth=1.35),
                           whiskerprops=dict(color="#6b7280"),
                           capprops=dict(color="#6b7280"))
        for patch, arm in zip(boxes["boxes"], methods):
            patch.set_facecolor(COLORS[arm])
            patch.set_edgecolor("#202124")
            patch.set_alpha(.86)
            if arm in ORACLES:
                patch.set_hatch("///")
        ax.axvline(0, color="#111827", linewidth=1, linestyle="--")
        ax.set_xscale("symlog", linthresh=25, linscale=.8)
        ax.set_xticks([-50, -25, 0, 25, 50, 100, 200, 500, 1000])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
        ax.set_yticks(range(1, len(methods) + 1), [LABELS[arm] for arm in methods])
        ax.set_xlabel("Mean TTFT change vs. Ours (higher is worse)")
        ax.set_title(profile.replace("_", " ").title(), fontweight="bold")
        ax.grid(axis="x", color="#d9dde3", linewidth=.6)
        ax.set_facecolor("#fbfbfc")
    fig.suptitle("Per-scenario TTFT change relative to the current method",
                 fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=.12, right=.985, top=.87, bottom=.16, wspace=.06)
    fig.text(.5, .025,
             "Each box contains 16 matched scenarios; whiskers show the full range. "
             "Symmetric-log x-axis preserves improvements and large regressions.",
             ha="center", fontsize=8, color="#4b5563")
    save(fig, "baseline_relative_ttft")


def save(fig, stem):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    rows = load()
    overview(rows)
    relative_distribution(rows)
    print(OUTPUT)


if __name__ == "__main__":
    main()
