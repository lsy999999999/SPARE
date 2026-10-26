"""Measured TTFT-throughput positioning draft for a WWW introduction.

Seven main policies, all 36 G2 scenarios, four repetitions per policy/scenario.
Point estimates are descriptive; no claim of statistical dominance is made.
Style-only inspiration: Sarathi-Serve Figure 2; new coordinates and artwork.
"""

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/www_evaluation_gpu_completion_20261003_r1/analysis/runs.csv"
OUT = ROOT / "writing/www/figures/introduction_positioning"
POLICIES = ("arrival", "srp", "native", "sarathi", "past_future", "fixed256", "fixed512")
ABLATIONS = {"basic", "constant16", "constant32", "no_finish_guard",
             "budget_only", "chunk_only", "token_proxy"}
BLUE = "#17688A"
INK = "#252C34"
NEUTRAL = "#657380"
PURPLE = "#806288"


def load_means():
    rows = list(csv.DictReader(SOURCE.open()))
    assert len(rows) == 2016
    assert {r["arm"] for r in rows} == set(POLICIES) | ABLATIONS
    groups = defaultdict(list)
    for row in rows:
        groups[row["arm"], row["cell"]].append(row)
    cells = {r["cell"] for r in rows}
    assert len(cells) == 36
    scene_means = []
    for arm in sorted(set(POLICIES) | ABLATIONS):
        assert {cell for policy, cell in groups if policy == arm} == cells
        for cell in sorted(cells):
            group = groups[arm, cell]
            assert len(group) == 4 and {int(r["repeat"]) for r in group} == set(range(4))
            values = [(float(r["ttft_ms_mean"]) / 1000,
                       float(r["output_throughput_tps"])) for r in group]
            assert all(math.isfinite(v) and v > 0 for pair in values for v in pair)
            scene_means.append({"arm": arm, "cell": cell,
                                "mean_ttft_s": mean(v[0] for v in values),
                                "mean_output_tps": mean(v[1] for v in values)})
    summaries = {}
    for arm in sorted(set(POLICIES) | ABLATIONS):
        subset = [s for s in scene_means if s["arm"] == arm]
        summaries[arm] = {"arm": arm, "included_in_figure": arm in POLICIES,
                          "n_scenarios": len(subset), "n_runs": 4 * len(subset),
                          "mean_ttft_s": mean(s["mean_ttft_s"] for s in subset),
                          "mean_output_tps": mean(s["mean_output_tps"] for s in subset)}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, records in (("all_method_means.csv", list(summaries.values())),
                          ("scenario_means.csv", scene_means)):
        with (OUT / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    paired = {}
    spare = {s["cell"]: s for s in scene_means if s["arm"] == "arrival"}
    for arm in POLICIES[1:]:
        ref = {s["cell"]: s for s in scene_means if s["arm"] == arm}
        paired[arm] = {
            "mean_paired_ttft_reduction_pct": mean(
                100 * (ref[c]["mean_ttft_s"] - spare[c]["mean_ttft_s"]) /
                ref[c]["mean_ttft_s"] for c in cells),
            "mean_paired_throughput_increase_pct": mean(
                100 * (spare[c]["mean_output_tps"] - ref[c]["mean_output_tps"]) /
                ref[c]["mean_output_tps"] for c in cells),
        }
    (OUT / "data_contract.json").write_text(json.dumps({
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "input_rows": len(rows), "plotted_rows": len(POLICIES) * 36 * 4,
        "included_arms": POLICIES, "excluded_component_ablations": sorted(ABLATIONS),
        "excluded_scenarios": [], "excluded_repeats": [],
        "aggregation": "Mean of 4 repeats, then equal mean of all 36 scenarios",
        "uncertainty": "Descriptive point estimates; no significance claim",
        "paired_comparisons_not_used_as_coordinates": paired,
        "alignment": "Not applicable: one axes",
    }, indent=2) + "\n")
    return {arm: (s["mean_ttft_s"], s["mean_output_tps"]) for arm, s in summaries.items()}


def main():
    points = load_means()
    plt.rcParams.update({
        "font.family": "Liberation Sans", "font.size": 7,
        "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.75, "pdf.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(3.35, 2.65), facecolor="white")
    fig.subplots_adjust(left=0.175, right=0.98, bottom=0.19, top=0.97)
    ax.set_xlim(0, 29)
    ax.set_ylim(110, 190)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_yticks([120, 140, 160, 180])
    ax.set_xlabel("Mean TTFT (s)  ↓", labelpad=5)
    ax.set_ylabel("Output throughput (token/s)  ↑", labelpad=6)
    ax.tick_params(length=2.6, width=0.7, pad=3)

    # Labels are offset, but the data coordinates are never jittered.
    style = {
        "arrival": ("*", BLUE, 5.8, BLUE),
        "srp": ("o", NEUTRAL, 4.8, "white"),
        "native": ("o", NEUTRAL, 4.8, NEUTRAL),
        "sarathi": ("s", NEUTRAL, 4.5, NEUTRAL),
        "past_future": ("D", PURPLE, 4.8, PURPLE),
        "fixed256": ("^", NEUTRAL, 3.0, "white"),
        "fixed512": ("v", NEUTRAL, 4.7, "white"),
    }
    for arm in POLICIES:
        marker, color, size, face = style[arm]
        ax.plot(*points[arm], marker=marker, markersize=size, linestyle="none",
                markeredgecolor=color, markerfacecolor=face,
                markeredgewidth=0.9, zorder=6 if arm == "arrival" else 5)

    # Paired whole-policy comparison; not a causal trace of a single action.
    ax.add_patch(FancyArrowPatch(points["srp"], points["arrival"],
                 connectionstyle="arc3,rad=-0.25", arrowstyle="-|>",
                 mutation_scale=7, shrinkA=5, shrinkB=6,
                 linewidth=1.1, color=BLUE, zorder=3))
    ax.text(0.6, 140, "Selective\nprogress", fontsize=6.8,
            color=BLUE, ha="left", va="center", linespacing=1.05)
    ax.text(0.6, 156.5, "SPARE", fontsize=8.1, weight="bold", color=BLUE,
            ha="left", va="center")

    # Direct labels and short leaders keep nearby fixed-budget points visible.
    leader = {"arrowstyle": "-", "lw": 0.6, "color": NEUTRAL,
              "shrinkA": 3, "shrinkB": 4}
    ax.annotate("Fixed 256", xy=points["fixed256"], xytext=(0.8, 180),
                ha="left", va="center", fontsize=7,
                arrowprops={**leader, "connectionstyle": "angle3,angleA=0,angleB=90"})
    ax.annotate("Fixed 512", xy=points["fixed512"], xytext=(12, 151),
                ha="left", va="center", fontsize=7, arrowprops=leader)
    ax.annotate("Past-Future\n(rules)", xy=points["past_future"], xytext=(15, 180),
                ha="left", va="center", fontsize=7, color=INK,
                linespacing=1.1, arrowprops=leader)
    ax.text(8.4, 119, "SRP\n(same order)", ha="center", va="center",
            fontsize=7, linespacing=1.1)
    ax.annotate("vLLM V1", xy=points["native"], xytext=(23, 145),
                ha="center", va="center", fontsize=7, arrowprops=leader)
    ax.annotate("Sarathi\n(rules)", xy=points["sarathi"], xytext=(23, 117),
                ha="center", va="center", fontsize=7,
                linespacing=1.1, arrowprops=leader)

    # Single axes: multi-panel alignment is not applicable. Do not crop width.
    fig.savefig(OUT / "ttft_throughput_positioning.pdf")
    fig.savefig(OUT / "ttft_throughput_positioning.svg")
    fig.savefig(OUT / "ttft_throughput_positioning.png", dpi=600)
    fig.savefig(OUT / "ttft_throughput_positioning.tiff", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
