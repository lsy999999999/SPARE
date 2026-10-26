"""Constructed SPARE schedule, verified with the policy; not a measured trace.

Contract: illustrate how deferring unfinished prefill can preserve KV for D.
Two comparable panels show the same initial state and request ordering rule.
The C lane shows fractions of its default quota, not elapsed time. No timing claim
is inferred from these illustrative rounds. Original schematic, Python only.
"""
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/spare-timeline-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research.v1_budget_20260922.policy import State
from research.v1_arrival_formula_20260924.policy import ArrivalRate, formula, select

SKILL = Path("plotting")
sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

OUT = ROOT / "writing/www/figures/introduction_timeline"
BLUE = "#A9CCE5"
GRAY = "#F0F0F0"
PINK = "#EBC2C3"
GREEN = "#B8DDBB"
GREEN_INK = "#27753C"
RED = "#BD3439"
INK = "#252525"
EDGE = "#74787B"


def verified_example():
    histories = {}
    for method in ("baseline", "spare"):
        arrival_rate = ArrivalRate(1.0)
        arrival_rate.value = 32768.0  # Identical observed input-history state.
        rows = [State("A", 320, 321, 256, 65, 20, True),
                State("B", 16, 32, 32, 0, 1, True),
                State("C", 32, 4096, 4096, 0, 2, True)]
        history = []
        for k in range(3):
            if k == 1:
                rows.append(State("D", 0, 256, 256, 0, 0, False))
                arrival_rate.observe(k * .01, 256)
            rows.sort(key=lambda r: (not r.running, not r.decode, r.remaining, r.rid))
            free = 69 - sum(r.blocks for r in rows)
            result = select(rows, 1024, 256, free, 70, 16, 32, 8192,
                            threshold=formula(1) if method == "spare" else "off",
                            recent={"arrival_rate": arrival_rate.at(k * .01)})
            plan = result["selected"]
            assert plan is not None
            history.append(dict(round=k, before=[asdict(r) for r in rows],
                                free_before=free, base=asdict(result["base"]),
                                selected=asdict(plan), threshold=result["threshold"],
                                changed=result["changed"]))
            next_rows = []
            for r in rows:
                n = plan.tokens.get(r.rid, 0)
                if not n:
                    next_rows.append(r)
                    continue
                computed = r.computed + n
                blocks = max(r.blocks, (computed + 15) // 16)
                if k == 1 and r.rid == "A":
                    continue  # A's fixed remaining output length is two tokens.
                if computed >= r.prompt:
                    next_rows.append(State(r.rid, computed, computed + 1, r.prompt,
                                           max(1, r.outputs + 1), blocks, True))
                else:
                    next_rows.append(State(r.rid, computed, r.prompt, r.prompt,
                                           0, blocks, True))
            rows = next_rows
        histories[method] = history
    b, s = histories["baseline"], histories["spare"]
    assert b[0]["selected"]["tokens"] == {"A": 1, "B": 16, "C": 256}
    assert s[0]["selected"]["tokens"] == {"A": 1, "B": 16, "C": 64}
    assert "D" not in b[1]["selected"]["tokens"]
    assert s[1]["selected"]["tokens"]["D"] == 256
    assert b[2]["selected"]["tokens"]["D"] == 256
    assert s[2]["selected"]["tokens"]["D"] == 1
    # Free space after running requests advance, immediately before D admission.
    assert b[1]["free_before"] - b[1]["selected"]["growth"] == 11
    assert s[1]["free_before"] - s[1]["selected"]["growth"] + 16 == 23
    return histories


def box(ax, x, y, width, label, color, *, edge=EDGE, textcolor=INK, fs=9):
    ax.add_patch(Rectangle((x, y - .235), width, .47, facecolor=color,
                           edgecolor=edge, linewidth=.55))
    if label:
        ax.text(x + width / 2, y, label, ha="center", va="center",
                fontsize=fs, color=textcolor)


def panel(ax, method, history):
    spare = method == "spare"
    ax.set(xlim=(-.03, 3.04), ylim=(0, 3.35))
    ax.set_axis_off()
    ax.text(-.09, 1.91, "SPARE" if spare else "Default\nprogress",
            weight="bold", fontsize=9.5, va="center", ha="right", linespacing=1.35)
    # An incoming arrow for D and outgoing arrow for A replace event sentences.
    ax.annotate("D", xy=(1, .81), xytext=(1, .30),
                ha="center", va="center", fontsize=9,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=.8,
                                shrinkA=4, shrinkB=1))
    ax.annotate("", xy=(2.14, 3.21), xytext=(1.94, 2.97),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=.8))
    ax.text(2.19, 3.19, "A", fontsize=9, ha="left", va="center")
    ypos = dict(A=2.71, B=2.18, C=1.65, D=1.12)
    for k, record in enumerate(history):
        states = {r["rid"]: r for r in record["before"]}
        alloc = record["selected"]["tokens"]
        for rid in ("A", "B", "C", "D"):
            y = ypos[rid]
            if rid not in alloc:
                label = "KV wait" if rid == "D" and k == 1 else ""
                box(ax, k, y, 1, label, GRAY, textcolor=EDGE, fs=8)
                continue
            decode = states[rid]["outputs"] > 0
            n = alloc[rid]
            if rid == "C" and n < 256:
                box(ax, k, y, n / 256, r"$C_p$", BLUE)
                box(ax, k + n / 256, y, 1 - n / 256, "Defer", GREEN,
                    edge=GREEN_INK, textcolor=GREEN_INK, fs=8)
            else:
                box(ax, k, y, 1, rf"${rid}_{'d' if decode else 'p'}$", PINK if decode else BLUE)
    if spare:
        # Causal connector uses the empty pre-arrival D cell, not a KV scale.
        # The connector links the deferred quota to D's admission.
        ax.annotate("", xy=(1.09, 1.12), xytext=(.65, 1.39),
                    arrowprops=dict(arrowstyle="-|>", color=GREEN_INK, lw=.95,
                                    connectionstyle="angle,angleA=-90,angleB=180,rad=0"))
    # Red marks denote first-token events after the final prefill completes.
    first = 2 if spare else 3
    ax.plot(first, ypos["D"], marker="o", ms=4, color=RED, zorder=8, clip_on=False)
    ax.plot([first, first], [.48, .83], color=RED, lw=.8)
    if spare:
        ax.annotate("", xy=(2, .47), xytext=(3, .47),
                    arrowprops=dict(arrowstyle="<->", lw=.9, color=RED))
        ax.text(2.5, .10, "Earlier first token", color=RED, ha="center", fontsize=8.5,
                fontweight="bold")
    else:
        ax.text(2.99, .10, "First token", color=RED, ha="right", fontsize=8.5)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    histories = verified_example()
    (OUT / "illustrative_schedule.json").write_text(json.dumps(histories, indent=2) + "\n")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "mathtext.fontset": "dejavusans", "savefig.dpi": 600})
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 3.05))
    fig.subplots_adjust(left=.205, right=.975, bottom=.055, top=.85, hspace=.16)
    for ax, method in zip(axs, ("baseline", "spare")):
        panel(ax, method, histories[method])
    for k, label in enumerate((r"Round $t$", r"Round $t+1$", r"Round $t+2$")):
        x, _ = fig.transFigure.inverted().transform(axs[0].transData.transform((k+.5, 0)))
        fig.text(x, .93, label, fontsize=10, ha="center", va="center")
    fig.canvas.draw()
    stem = OUT / "spare_timeline"
    require_matplotlib_panel_alignment(fig, json_out=str(stem) + ".alignment.json", strict=True)
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
