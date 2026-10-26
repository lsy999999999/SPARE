"""Plot held-out arrival-pressure curves from the frozen CPU simulation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "artifacts/www_capacity_pareto_20261015_r1"
OUT = ROOT / "writing/www/figures/newidea"
ARMS = (
    ("arrival", "SPARE", "#195c9b", "-"),
    ("native", "vLLM V1", "#555555", "--"),
    ("past_future", "Past-Future", "#238a83", "-."),
    ("s3_proxy", "S3 proxy", "#b64342", ":"),
    ("apt_serve_kv_rule", "Apt-Serve rule", "#a56a21", (0, (5, 2))),
)


def read(path):
    return json.loads(path.read_text())


def main():
    design = read(DATA / "design.json")
    assert read(DATA / "test_status.json")["completed"] == 124800
    samples = [s for s in design["samples"] if s["split"] == "test"]
    selected = read(DATA / "selected_test.json")
    rates = design["rates"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True, sharey=True,
                             layout="constrained")
    for row, source in enumerate(("code", "conversation")):
        windows = [s for s in samples if s["source"] == source]
        assert len(windows) == 50
        for col, capacity in enumerate((768, 4096)):
            ax = axes[row, col]
            for arm, label, color, linestyle in ARMS:
                choices = [r for r in selected if r["source"] == source
                           and r["capacity"] == capacity and r["slo"] == "strict"
                           and r["weight"] == 0.5 and r["arm"] == arm]
                assert len(choices) == 1
                choice = choices[0]
                by_rate = {rate: [] for rate in rates}
                for sample in windows:
                    path = DATA / "test" / (f"{sample['name']}_c{capacity}_{arm}"
                                            f"_t{choice['budget']}.json")
                    for point in read(path):
                        assert point["valid"] and point["rate"] in by_rate
                        by_rate[point["rate"]].append(point["completed_by_end"] / 60.)
                assert all(len(by_rate[rate]) == 50 for rate in rates)
                mean = np.array([np.mean(by_rate[rate]) for rate in rates])
                se = np.array([np.std(by_rate[rate], ddof=1) / np.sqrt(50)
                               for rate in rates])
                ax.plot(rates, mean, label=label, color=color,
                        linestyle=linestyle, linewidth=1.6)
                if arm == "arrival":
                    ax.fill_between(rates, mean - 1.96 * se, mean + 1.96 * se,
                                    color=color, alpha=.13, linewidth=0)
                cap_rate = choice["test_qps"]
                if cap_rate in by_rate:
                    index = rates.index(cap_rate)
                    ax.plot(cap_rate, mean[index], marker="D", color=color,
                            markersize=4.5, markeredgecolor="white",
                            markeredgewidth=.5, linestyle="none", zorder=4)
            ax.plot([0, 10], [0, 10], color="#aaaaaa", linestyle="--",
                    linewidth=.8, zorder=0)
            ax.set_title(f"{source.capitalize()} | {capacity} KV blocks")
            ax.grid(axis="y", color="#dddddd", linewidth=.5)
            ax.set_xlim(0, 10.2)
            ax.set_ylim(0, 10.2)
    for ax in axes[1]:
        ax.set_xlabel("Offered arrival rate (requests/s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Completed requests/s (0-60 s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5,
               frameon=False)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "arrival_pressure_simulator.pdf", bbox_inches="tight")
    fig.savefig(OUT / "arrival_pressure_simulator.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
