"""Plot the G2 paired result and a prespecified SPARE action trace."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/www_plan_abc_20261008_r1/a"


def rows(name):
    with (OUT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    paired = rows("paired.csv")
    trace = rows("case_trace.csv")
    summary = json.loads((OUT / "summary.json").read_text())
    action = json.loads((OUT / "case_action.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), gridspec_kw={"width_ratios": [1, 1.25]})
    colors = {"code": "#16806f", "conversation": "#bf5b3d"}
    marks = {"steady": "o", "burst_stop": "^"}
    for row in paired:
        capacity = int(row["capacity"])
        source = row["window"].split("_")[0]
        shape = row["shape"]
        offset = (-.14 if source == "code" else .14) + (-.045 if shape == "steady" else .045)
        axes[0].scatter([768, 2048, 4096].index(capacity)+offset,
                        float(row["improvement_pct"]), s=34,
                        c=colors[source], marker=marks[shape], alpha=.85,
                        edgecolor="white", linewidth=.35)
    axes[0].axhline(0, color="#454545", linewidth=.8)
    axes[0].set_xticks([0, 1, 2], ["768", "2048", "4096"])
    axes[0].set_xlabel("KV blocks")
    axes[0].set_ylabel("SPARE TTFT improvement vs SRP (%)")
    axes[0].set_title(f"Same-order pairs, n=36; mean {summary['paired_mean_improvement_pct']:.1f}%", fontsize=10)
    for source, color in colors.items():
        axes[0].scatter([], [], c=color, label=source)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")

    start = float(action["round"])
    action_time = float(trace[int(start)]["time_ms"])
    limit = min(float(trace[-1]["time_ms"]), action_time + 8000)
    segment = [row for row in trace if float(row["time_ms"]) <= limit]
    times = [float(row["time_ms"]) / 1000 for row in segment]
    kv = [int(row["unstarted_kv_blocks_before"]) for row in segment]
    firsts = [int(row["first_tokens_completed"]) for row in segment]
    axes[1].plot(times, kv, color="#16806f", linewidth=1.35, label="KV blocks held by unstarted requests")
    axes[1].axvline(action_time/1000, color="#bf5b3d", linewidth=1.15,
                    linestyle="--", label=f"Round {action['round']}: save {action['baseline_growth_blocks'] - action['actual_growth_blocks']} blocks")
    axes[1].set_xlabel("Episode time (s)")
    axes[1].set_ylabel("Unstarted-request KV blocks", color="#16806f")
    axes[1].tick_params(axis="y", labelcolor="#16806f")
    secondary = axes[1].twinx()
    secondary.plot(times, firsts, color="#424a55", linewidth=1.1,
                   label="Requests with first token")
    secondary.set_ylabel("Requests with first token", color="#424a55")
    secondary.tick_params(axis="y", labelcolor="#424a55")
    axes[1].set_title(action["cell"].replace("_", " "), fontsize=10)
    handles, labels = axes[1].get_legend_handles_labels()
    other_handles, other_labels = secondary.get_legend_handles_labels()
    axes[1].legend(handles+other_handles, labels+other_labels,
                   frameon=False, fontsize=7, loc="upper left")
    for ax in axes:
        ax.grid(axis="y", color="#dedede", linewidth=.5)
        ax.spines[["top", "right"]].set_visible(False)
    secondary.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figure_a.png", dpi=200)
    fig.savefig(OUT / "figure_a.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
