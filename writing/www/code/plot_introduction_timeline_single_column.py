"""WWW single-column layout of the policy-checked SPARE timeline."""

from pathlib import Path

import matplotlib.pyplot as plt

from plot_introduction_timeline import panel, verified_example

from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "writing/www/figures/introduction_timeline"


def main():
    histories = verified_example()
    plt.rcParams.update({
        "font.family": "Liberation Sans", "font.size": 7,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans", "savefig.facecolor": "white",
    })
    fig, axs = plt.subplots(2, 1, figsize=(3.35, 3.35), facecolor="white")
    fig.subplots_adjust(left=.215, right=.975, bottom=.055, top=.84, hspace=.17)
    for ax, method in zip(axs, ("baseline", "spare")):
        panel(ax, method, histories[method])
    for item in axs[1].texts:
        if item.get_text() == "Earlier first token":
            item.set_x(2.38)
            item.set_fontsize(7.2)
    for k, label in enumerate((r"Round $t$", r"Round $t+1$", r"Round $t+2$")):
        x, _ = fig.transFigure.inverted().transform(
            axs[0].transData.transform((k + .5, 0)))
        fig.text(x, .926, label, fontsize=8, ha="center", va="center")
    stem = OUT / "spare_timeline_single_column"
    require_matplotlib_panel_alignment(
        fig, json_out=str(stem) + ".alignment.json", strict=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
