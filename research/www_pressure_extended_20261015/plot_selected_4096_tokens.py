"""Plot the four Introduction method representatives at 4096 KV blocks."""
from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from research.www_pressure_extended_20261015.plot_intro_comparison import (
    DATA, METHODS, OUT, RATES,
)

SOURCES = ("code", "conversation")
STEM = "arrival_pressure_selected_4096_output_tokens_from0"


def read_points(arms: tuple[str, ...]) -> dict[tuple[str, str, float], float]:
    path = DATA / "intro_output_tokens_0_120s.csv"
    points = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["kv_blocks"]) != 4096 or row["arm"] not in arms:
                continue
            if (row["source"] not in SOURCES or int(row["windows"]) != 3 or
                    (int(row["start_s"]), int(row["end_s"])) != (0, 120)):
                raise ValueError(f"unexpected experiment row: {row}")
            key = row["source"], row["arm"], float(row["arrival_qps"])
            if key in points:
                raise ValueError(f"duplicate point: {key}")
            points[key] = float(row["output_tokens_tps"])
    expected = {(source, arm, rate) for source in SOURCES
                for arm in arms for rate in RATES}
    if points.keys() != expected:
        raise ValueError(f"missing or extra points: {expected ^ points.keys()}")
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-method", choices=("sarathi", "sola_rule"),
                        default="sarathi")
    args = parser.parse_args()
    arms = ("arrival", args.batch_method, "uniboost_static", "past_future")
    stem = STEM if args.batch_method == "sarathi" else STEM.replace(
        "selected_4096", "selected_sola_4096")
    points = read_points(arms)
    styles = {arm: (label, color, linestyle, marker)
              for arm, label, color, linestyle, marker in METHODS if arm in arms}
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), sharex=True,
                             layout="constrained")
    for ax, source in zip(axes, SOURCES):
        for arm in arms:
            label, color, linestyle, marker = styles[arm]
            ax.plot(RATES, [points[source, arm, rate] for rate in RATES],
                    label=label, color=color, linestyle=linestyle, marker=marker,
                    markersize=5 if arm == "past_future" else 3.5,
                    linewidth=2.8 if arm == "past_future" else
                    (2 if arm == "arrival" else 1.5),
                    zorder={"arrival": 5, "sarathi": 4, "sola_rule": 4,
                            "uniboost_static": 3, "past_future": 2}[arm])
        ax.set_title(f"{source.capitalize()} | 4096 KV blocks")
        ax.set_xlim(0, 41)
        ax.set_ylim(bottom=0)
        ax.set_xticks((0, 8, 16, 24, 32, 40))
        ax.set_xlabel("Offered arrival rate (requests/s)")
        ax.grid(axis="y", color="#dddddd", linewidth=.5)
    axes[0].set_ylabel("Output tokens/s (0-120 s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4,
               frameon=False)
    OUT.mkdir(parents=True, exist_ok=True)
    for extension, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(OUT / f"{stem}.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
