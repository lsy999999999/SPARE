"""Matched 120-second overload curves for the Introduction method classes."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.www_pareto_20261014.run import ROOT

DATA = ROOT / "artifacts/www_pressure_extended_20261015"
OUT = ROOT / "writing/www/figures/newidea"
RATES = (.5, 1., 2., 4., 6., 8., 12., 16., 20., 24., 32., 40.)
METHODS = (
    ("arrival", "SPARE", "#195c9b", "-", "o"),
    ("native", "vLLM V1", "#494949", "--", "s"),
    ("sarathi", "Sarathi rule", "#439bc2", "-.", "^"),
    ("sola_rule", "SOLA rule", "#d25a35", ":", "v"),
    ("uniboost_static", "UniBoost rule", "#9562a0", "--", "D"),
    ("past_future", "Past-Future rule", "#238a83", "-.", "P"),
    ("s3_proxy", "S3 proxy", "#b64342", ":", "X"),
    ("apt_serve_kv_rule", "Apt-Serve rule", "#a67b1e", "--", "*"),
)


def read_tag(tag):
    folder = DATA / tag
    design = json.loads((folder / "design.json").read_text())
    rows = [json.loads(path.read_text()) for path in (folder / "runs").glob("*.json")]
    if len(rows) != design["tasks"] or any(not row["valid"] for row in rows):
        raise RuntimeError(f"incomplete or invalid experiment: {tag}")
    if design["horizon_ms"] != 120000.:
        raise RuntimeError(f"incompatible window: {tag}")
    return design, rows


def assemble(start_s=60):
    if start_s not in (0, 60):
        raise ValueError("start_s must be 0 or 60")
    tags = ("intro_low_120s", "coarse_120s", "upper_120s", "apt_upper_120s",
            "intro_baselines_120s")
    experiments = [read_tag(tag) for tag in tags]
    selection_hashes = {d["source_selection_sha256"] for d, _ in experiments}
    timing_hashes = {d["timing_sha256"] for d, _ in experiments}
    simulator_hashes = {d["simulator_sha256"] for d, _ in experiments}
    if (len(selection_hashes) != 1 or len(timing_hashes) != 1 or
            len(simulator_hashes) != 1):
        raise RuntimeError("experiments used different frozen configuration or timing")
    inputs = {}
    for design, _ in experiments:
        for path, sha in design["input_sha256"].items():
            if path in inputs and inputs[path] != sha:
                raise RuntimeError("experiments used different request inputs")
            inputs[path] = sha
    data = defaultdict(list)
    paired_arrivals = {}
    for _, rows in experiments:
        for row in rows:
            mid, end = row["windows"]
            if (mid["at_ms"], end["at_ms"]) != (60000., 120000.):
                raise RuntimeError("incompatible measurement checkpoints")
            for checkpoint in row["windows"]:
                if checkpoint["arrivals"] - checkpoint["completed"] != checkpoint["outstanding"]:
                    raise RuntimeError("request accounting failed")
            key = row["source"], row["sample"], row["capacity"], row["rate"]
            arrivals = mid["arrivals"], end["arrivals"]
            if key in paired_arrivals and paired_arrivals[key] != arrivals:
                raise RuntimeError("methods used different arrival counts")
            paired_arrivals[key] = arrivals
            data[row["source"], row["capacity"], row["arm"], row["rate"]].append(row)
    summary = []
    for source in ("code", "conversation"):
        for capacity in (768, 4096):
            reference_samples = None
            for arm, label, *_ in METHODS:
                for rate in RATES:
                    rows = data[source, capacity, arm, rate]
                    names = {r["sample"] for r in rows}
                    if len(rows) != 3 or len(names) != 3:
                        raise RuntimeError(f"missing matched runs: {source} {capacity} {arm} {rate}")
                    if reference_samples is None:
                        reference_samples = names
                    if names != reference_samples:
                        raise RuntimeError("methods used different test windows")
                    seconds = 120 - start_s
                    counts = []
                    for r in rows:
                        mid, end = r["windows"]
                        start = mid if start_s == 60 else dict(
                            arrivals=0, completed=0, outstanding=0)
                        counts.append([end[key] - start[key] for key in
                                       ("arrivals", "completed", "outstanding")])
                    arrivals, completed, backlog = np.array(counts, dtype=float).T / seconds
                    if np.any(arrivals <= 0) or not np.allclose(arrivals, completed + backlog):
                        raise RuntimeError("invalid arrival rate or request conservation")
                    fractions = completed / arrivals
                    summary.append(dict(source=source, kv_blocks=capacity, method=label,
                                        arm=arm, arrival_qps=rate, windows=3,
                                        start_s=start_s, end_s=120,
                                        actual_arrival_qps=float(arrivals.mean()),
                                        completed_qps=float(completed.mean()),
                                        completed_qps_sd=float(completed.std(ddof=1)),
                                        backlog_growth_qps=float(backlog.mean()),
                                        backlog_growth_qps_sd=float(backlog.std(ddof=1)),
                                        completion_fraction=float(fractions.mean()),
                                        completion_fraction_sd=float(fractions.std(ddof=1))))
    return summary


def require_matplotlib_panel_alignment(fig, axes, output):
    fig.canvas.draw()
    scale = fig.get_size_inches() * 72
    boxes = [[tuple(ax.get_position().bounds[k] * scale[k % 2]
                    for k in range(4)) for ax in row] for row in axes]
    tolerance = 1.5
    for row in boxes:
        left, right = row
        if (abs(left[1] - right[1]) > tolerance or
                abs(left[3] - right[3]) > tolerance or
                abs(left[2] - right[2]) > tolerance):
            raise RuntimeError("misaligned panel row")
    for column in range(2):
        top, bottom = boxes[0][column], boxes[1][column]
        if (abs(top[0] - bottom[0]) > tolerance or
                abs(top[2] - bottom[2]) > tolerance or
                abs(top[3] - bottom[3]) > tolerance):
            raise RuntimeError("misaligned panel column")
    if abs((boxes[0][1][0] - boxes[0][0][0] - boxes[0][0][2]) -
           (boxes[1][1][0] - boxes[1][0][0] - boxes[1][0][2])) > tolerance:
        raise RuntimeError("unequal panel gutters")
    output.write_text(json.dumps({"tolerance_pt": tolerance, "boxes_pt": boxes}, indent=2))


def plot(rows, field, name, ylabel):
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
                         "font.size": 8,
                         "axes.titlesize": 9, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.7), sharex=True,
                             layout="constrained")
    for i, source in enumerate(("code", "conversation")):
        for j, capacity in enumerate((768, 4096)):
            ax = axes[i, j]
            for arm, label, color, linestyle, marker in METHODS:
                points = [r for r in rows if r["source"] == source
                          and r["kv_blocks"] == capacity and r["arm"] == arm]
                points.sort(key=lambda r: r["arrival_qps"])
                assert [r["arrival_qps"] for r in points] == list(RATES)
                ax.plot(RATES, [r[field] for r in points], label=label, color=color,
                        linestyle=linestyle, marker=marker, markersize=3.5,
                        linewidth=2.1 if arm == "arrival" else 1.3,
                        zorder=5 if arm == "arrival" else 2)
            ax.set_title(f"{source.capitalize()} | {capacity} KV blocks")
            ax.set_xticks((0, 4, 8, 12, 16, 20, 24, 32, 40))
            ax.set_xlim(0, 41)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", color="#dddddd", linewidth=.5)
    for ax in axes[1]:
        ax.set_xlabel("Offered arrival rate (requests/s)")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    OUT.mkdir(parents=True, exist_ok=True)
    require_matplotlib_panel_alignment(fig, axes, OUT / f"{name}.alignment.json")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-s", type=int, choices=(0, 60), default=0,
                        help="Beginning of the measurement window; arrivals end at 120 s")
    args = parser.parse_args()
    rows = assemble(args.start_s)
    suffix = "_from0" if args.start_s == 0 else ""
    csv_name = "intro_comparison_0_120s.csv" if args.start_s == 0 else "intro_comparison_120s.csv"
    with (DATA / csv_name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, "completed_qps", f"arrival_pressure_intro_throughput{suffix}",
         f"Completed requests/s ({args.start_s}-120 s)")
    plot(rows, "backlog_growth_qps", f"arrival_pressure_intro_backlog{suffix}",
         f"Backlog growth (requests/s, {args.start_s}-120 s)")


if __name__ == "__main__":
    main()
