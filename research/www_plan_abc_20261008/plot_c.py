"""Plot held-out paired TTFT effects and E2E tradeoffs for fixed controls."""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/www_plan_abc_20261008_r1/c/analysis"
REFERENCES = ("srp", "fixed_selected", "guarded_selected", "budget_only")
LABELS = {"srp": "SRP", "fixed_selected": "Fixed budget",
          "guarded_selected": "Guarded fixed", "budget_only": "Budget only"}
COLORS = {"srp": "#424a55", "fixed_selected": "#16806f",
          "guarded_selected": "#bf5b3d", "budget_only": "#8c6498"}


def load_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    summary = json.loads((OUT / "summary.json").read_text())
    assert summary["audited"] and summary["runs"] == 480
    rows = load_csv("cell_means.csv")
    paired = {(row["cell"], row["arm"]): row for row in rows}
    windows = sorted({row["window"] for row in rows})
    assert len(windows) == 6
    effects = defaultdict(list)
    tradeoffs = defaultdict(list)
    for cell in sorted({row["cell"] for row in rows}):
        spare = paired[cell, "spare"]
        for arm in REFERENCES:
            base = paired[cell, arm]
            ttft = 100 * (float(base["ttft_ms_mean"]) - float(spare["ttft_ms_mean"])) / float(base["ttft_ms_mean"])
            e2e = 100 * (float(base["e2e_ms_mean"]) - float(spare["e2e_ms_mean"])) / float(base["e2e_ms_mean"])
            effects[arm, base["window"]].append(ttft)
            if arm in ("fixed_selected", "guarded_selected"):
                tradeoffs[arm].append((ttft, e2e, int(base["capacity"])))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1, 1.05]})
    for i, arm in enumerate(REFERENCES):
        window_values = [st.fmean(effects[arm, window]) for window in windows]
        for j, value in enumerate(window_values):
            axes[0].scatter(value, i + (j - 2.5) * .065, s=35,
                            color=COLORS[arm], alpha=.7, edgecolor="white", linewidth=.4)
        axes[0].scatter(st.fmean(window_values), i, s=120, marker="D",
                        color=COLORS[arm], edgecolor="white", linewidth=.7)
    axes[0].axvline(0, color="#555555", linewidth=.8)
    axes[0].set_yticks(range(len(REFERENCES)), [LABELS[arm] for arm in REFERENCES])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("SPARE mean TTFT improvement (%)")
    axes[0].set_title("Six held-out windows; diamond = window mean", fontsize=10)

    for arm in ("fixed_selected", "guarded_selected"):
        for ttft, e2e, capacity in tradeoffs[arm]:
            axes[1].scatter(ttft, e2e, color=COLORS[arm],
                            marker="o" if capacity == 768 else "^", s=35,
                            alpha=.75, edgecolor="white", linewidth=.4)
        axes[1].scatter([], [], color=COLORS[arm], label=LABELS[arm])
    axes[1].axhline(0, color="#555555", linewidth=.8)
    axes[1].axvline(0, color="#555555", linewidth=.8)
    axes[1].set_xlabel("SPARE mean TTFT improvement (%)")
    axes[1].set_ylabel("SPARE mean E2E improvement (%)")
    axes[1].set_title("Paired cells; circles = 768, triangles = 2048 blocks", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(color="#e0e0e0", linewidth=.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figure_c.png", dpi=200)
    fig.savefig(OUT / "figure_c.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
