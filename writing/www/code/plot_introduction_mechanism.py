"""SPARE mechanism: a constructed round verified by the actual policy code.

Internal design draft, not measured performance. The only example numbers
represent token allocations and physical block counts, not timing predictions.
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research.v1_budget_20260922.policy import State
from research.v1_arrival_formula_20260924.policy import formula, select

sys.path.insert(0, "plotting")
from audit_panel_alignment import require_matplotlib_panel_alignment

OUT = ROOT / "writing/www/figures/introduction_mechanism"
BLUE = "#196B83"
PALE_BLUE = "#E8F2F5"
AMBER = "#B88245"
PALE_AMBER = "#F2E4D2"
INK = "#202C34"
RULE = "#B2BCC2"


def example():
    rows = [State("A", 16, 17, 16, 1, 1, True),
            State("B", 16, 32, 32, 0, 1, True),
            State("C", 32, 544, 544, 0, 2, True)]
    result = select(rows, 1024, 256, 18, 23, 16, 32, 8192,
                    threshold=formula(1.0), recent={"arrival_rate": 16384.0})
    base, chosen = result["base"], result["selected"]
    assert base.tokens == {"A": 1, "B": 16, "C": 256}
    assert chosen.tokens == {"A": 1, "B": 16, "C": 64}
    assert (base.growth, chosen.growth, result["threshold"]) == (18, 6, 11)
    assert (result["budget"], result["chunk"]) == (1024, 64)
    assert result["changed"] and base.growth - chosen.growth >= result["threshold"]
    assert chosen.tokens["C"] < base.tokens["C"] < rows[2].remaining
    for row in rows[:2]:
        assert chosen.tokens[row.rid] == base.tokens[row.rid]
    OUT.mkdir(parents=True, exist_ok=True)
    serial = {**result, "base": asdict(base), "selected": asdict(chosen),
              "example_kind": "constructed and policy-verified, not measured",
              "initial_requests": [asdict(r) for r in rows],
              "free_before": 18, "block_size_tokens": 16,
              "free_after_base": 0, "free_after_selected": 12,
              "work_segment_tokens": 64}
    (OUT / "verified_example.json").write_text(json.dumps(serial, indent=2) + "\n")
    return result


def main():
    result = example()
    plt.rcParams.update({
        "font.family": "Liberation Sans", "font.size": 7,
        "text.color": INK, "pdf.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(3.35, 3.75), facecolor="white")
    width, height = 3.35 * 72, 3.75 * 72

    def text(x, y, label, size=7, color=INK, weight="normal", ha="left"):
        return fig.text(x / width, 1 - y / height, label, fontsize=size,
                        color=color, weight=weight, ha=ha, va="center",
                        linespacing=1.12)

    text(8, 12, "Protect responses. Spare KV.", 10, weight="bold")
    text(8, 27, "One scheduling round · same state and request order", 7)

    columns = []
    for xpos, name, is_spare in ((78, "Default plan", False), (167, "SPARE", True)):
        ax = fig.add_axes([xpos / width, (height - 220) / height, 66 / width, 180 / height])
        ax.set_xlim(0, 66)
        ax.set_ylim(180, 0)
        ax.axis("off")
        columns.append(ax)
        ax.text(33, 3, name, ha="center", va="center", fontsize=8.5,
                weight="bold", color=BLUE if is_spare else INK)
        ax.text(33, 17, "Protected quotas" if is_spare else "Default quotas",
                ha="center", va="center", fontsize=6.8,
                color=BLUE if is_spare else INK)

        for y, label in ((33, "Next token"), (59, "First token")):
            ax.add_patch(Rectangle((3, y - 7), 60, 14, facecolor=PALE_BLUE,
                                   edgecolor=BLUE, linewidth=0.75))
            ax.text(33, y, label, ha="center", va="center", color=BLUE,
                    fontsize=7.2, weight="bold" if is_spare else "normal")

        # Four equal pieces of the baseline's partial-prefill work, not KV blocks.
        for i in range(4):
            deferred = is_spare and i > 0
            ax.add_patch(Rectangle((3 + 15.5 * i, 83), 13.5, 10,
                                   facecolor="white" if deferred else PALE_AMBER,
                                   edgecolor=AMBER, linewidth=0.85,
                                   linestyle=(0, (2, 1.5)) if deferred else "solid"))
        if is_spare:
            ax.text(43, 76, "Defer", color=AMBER, fontsize=7, ha="center", va="center")
        ax.text(33, 106, "No first token yet", ha="center", va="center", fontsize=6.8)

        # Each cell below is exactly one physical block from the same free pool.
        allocated = result["selected"].growth if is_spare else result["base"].growth
        for i in range(18):
            x = 5 + (i % 6) * 9.5
            y = 124 + (i // 6) * 9.5
            fill = BLUE if i < 2 else AMBER if i < allocated else "white"
            edge = BLUE if i < 2 or i >= allocated else AMBER
            ax.add_patch(Rectangle((x, y), 7.5, 7.5,
                                   facecolor=fill, edgecolor=edge, linewidth=0.65))
        ax.text(33, 161, f"{allocated} allocated", ha="center", va="center", fontsize=7)
        ax.text(33, 176, "12 kept free" if is_spare else "0 free",
                ha="center", va="center", fontsize=8.2,
                weight="bold" if is_spare else "normal",
                color=BLUE if is_spare else INK)

    text(8, 73, "Generating\nreply", 7.2)
    text(8, 99, "Finishing\ninput", 7.2)
    text(8, 128, "Unfinished\ninput", 7.2)
    text(8, 178, "New physical\nKV blocks", 7.2)

    text(width / 2, 236, "Defer only when enough KV blocks are saved", 7.5,
         color=BLUE, weight="bold", ha="center")
    text(width / 2, 250, "Threshold adapts to free KV and input pressure", 7, ha="center")
    text(width / 2, 262, "Otherwise, keep the default plan", 7, ha="center")

    require_matplotlib_panel_alignment(
        fig, axes=columns, panel_ids=["default", "spare"],
        row_groups=[["default", "spare"]], strict=True,
        json_out=OUT / "spare_mechanism.alignment.json",
        overlay_svg=OUT / "spare_mechanism.alignment.svg",
    )
    fig.savefig(OUT / "spare_mechanism.pdf")
    fig.savefig(OUT / "spare_mechanism.svg")
    fig.savefig(OUT / "spare_mechanism.png", dpi=600)
    fig.savefig(OUT / "spare_mechanism.tiff", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
