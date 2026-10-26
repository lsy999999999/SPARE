"""Summarize the bounded overload scans without treating them as steady capacity."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.www_pareto_20261014.run import ROOT

OUT = ROOT / "artifacts/www_pressure_extended_20261015"
FIG = ROOT / "writing/www/figures/newidea"
EXPERIMENTS = (("coarse_120s", "120s SPARE", "arrival", "#195c9b", "-"),
               ("coarse_120s", "120s Apt-Serve", "apt_serve_kv_rule", "#a56a21", "--"),
               ("spare_180s", "180s SPARE", "arrival", "#238a83", ":"))


def load(tag):
    rows = [json.loads(path.read_text()) for path in (OUT / tag / "runs").glob("*.json")]
    design = json.loads((OUT / tag / "design.json").read_text())
    assert len(rows) == design["tasks"] and all(r["valid"] for r in rows)
    return rows


def group(rows):
    data = defaultdict(list)
    for row in rows:
        data[row["source"], row["capacity"], row["arm"], row["rate"]].append(row)
    return data


def summarize(tag, rows):
    result = []
    for (source, capacity, arm, rate), values in sorted(group(rows).items()):
        second = [r["second_half_qps"] for r in values]
        growth = [r["backlog_growth"]*2000/r["horizon_ms"] for r in values]
        result.append(dict(experiment=tag, source=source, capacity=capacity,
                           arm=arm, rate=rate, n=len(values),
                           completed_qps_second_half=round(float(np.mean(second)), 4),
                           completed_qps_sd=round(float(np.std(second, ddof=1)), 4),
                           backlog_growth_qps=round(float(np.mean(growth)), 4),
                           backlog_growth_sd=round(float(np.std(growth, ddof=1)), 4),
                           half_to_half_qps_change=round(float(np.mean([
                               r["second_half_qps"]-r["first_half_qps"] for r in values])), 4)))
    return result


def draw(series, metric, filename, ylabel, include_upper=False):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True,
                             layout="constrained")
    for row, source in enumerate(("code", "conversation")):
        for col, capacity in enumerate((768, 4096)):
            ax = axes[row, col]
            for tag, label, arm, color, linestyle in EXPERIMENTS:
                use_tag = ("conversation4096_refine_180s"
                           if tag == "spare_180s" and source == "conversation"
                           and capacity == 4096 else tag)
                points = sorted((r for r in series[use_tag]
                                 if r["source"] == source and r["capacity"] == capacity
                                 and r["arm"] == arm), key=lambda r: r["rate"])
                if include_upper and tag == "coarse_120s" and arm == "arrival":
                    points = sorted(points + [r for r in series["upper_120s"]
                                              if r["source"] == source and
                                              r["capacity"] == capacity],
                                    key=lambda r: r["rate"])
                assert points
                y = [r[metric] for r in points]
                ax.plot([r["rate"] for r in points], y, color=color,
                        linestyle=linestyle, linewidth=1.7, marker="o", markersize=2.8,
                        label=label + (" (n=5)" if use_tag.endswith("refine_180s") else " (n=3)"))
            ax.set_title(f"{source.capitalize()} | {capacity} KV blocks")
            ax.grid(axis="y", color="#dddddd", linewidth=.5)
            ax.set_xlim(7.5, 40.5 if include_upper else 24.5)
            ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xlabel("Offered arrival rate (requests/s)")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    labels[-1] = "180s SPARE (n=3; Conv/4096 n=5)"
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = {tag: load(tag) for tag in ("coarse_120s", "spare_180s",
                                       "conversation4096_refine_180s", "upper_120s")}
    series = {tag: summarize(tag, data) for tag, data in rows.items()}
    summary = [r for group_rows in series.values() for r in group_rows]
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    draw(series, "completed_qps_second_half", "arrival_pressure_extended_completed",
         "Completed QPS, latter half")
    draw(series, "backlog_growth_qps", "arrival_pressure_extended_backlog",
         "Backlog growth (requests/s)")
    draw(series, "completed_qps_second_half", "arrival_pressure_to40_completed",
         "Completed QPS, latter half", include_upper=True)
    draw(series, "backlog_growth_qps", "arrival_pressure_to40_backlog",
         "Backlog growth (requests/s)", include_upper=True)


if __name__ == "__main__":
    main()
