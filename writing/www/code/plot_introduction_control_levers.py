"""Original Figure-7-style comparison of scheduling decision levers.

The rows are illustrative decisions in one shared state, not measured traces
or faithful executions of the named systems. Methods may span several levers.
"""
from pathlib import Path
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/www_intro_control_levers_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'writing/www/figures/introduction_control_levers'
INK = '#27323A'
MUTED = '#5F6B73'
EDGE = '#9CA8AD'
DECODE = '#F3CEBF'
PREFILL = '#BBD9E4'
DEFER = '#CBE6D0'
WAIT = '#ECEFF0'
GATE = '#F4DFAD'
ACCENT = '#0F6D5D'


def segment(ax, x, y, width, label, color, *, dashed=False, ink=INK):
    ax.add_patch(Rectangle((x, y-.22), width, .44, facecolor=color,
                           edgecolor=EDGE if not dashed else ACCENT,
                           linewidth=.75, linestyle='--' if dashed else '-'))
    ax.text(x+width/2, y, label, ha='center', va='center', fontsize=7.3,
            color=ink)


def label(ax, x, y, first, second, *, accent=False):
    ax.text(x, y+.11, first, ha='left', va='center', fontsize=8.3,
            fontweight='bold', color=ACCENT if accent else INK)
    ax.text(x, y-.16, second, ha='left', va='center', fontsize=6.6,
            color=MUTED)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    fig.subplots_adjust(left=.025, right=.985, top=.97, bottom=.08)
    ax.set_xlim(0, 100)
    ax.set_ylim(.28, 5.18)
    ax.axis('off')

    ax.text(1, 5.00, 'Illustrative state: A decodes; B can finish prefill; C cannot; D arrives next round',
            fontsize=8.3, color=INK, va='center')
    ax.text(27, 4.58, 'Current round', fontsize=7.2, fontweight='bold', color=MUTED)
    ax.text(58, 4.58, 'Next round', fontsize=7.2, fontweight='bold', color=MUTED)
    ax.text(82, 4.58, 'Decision lever', fontsize=7.2, fontweight='bold', color=MUTED)
    ax.annotate('', xy=(79, 4.48), xytext=(27, 4.48),
                arrowprops=dict(arrowstyle='->', color=MUTED, lw=.8))

    y = (3.93, 3.03, 2.13, 1.23)
    for level in (3.48, 2.58, 1.68):
        ax.plot([1, 99], [level, level], color='#E2E6E8', lw=.6)

    label(ax, 1, y[0], 'Batch / chunk', 'e.g., Sarathi-Serve')
    for x, w, name, color in ((27, 7, r'$A_d$', DECODE),
                              (34, 8, r'$B_p$', PREFILL),
                              (42, 13, r'$C_p$', PREFILL)):
        segment(ax, x, y[0], w, name, color)
    segment(ax, 58, y[0], 10, r'$C_p$', PREFILL)
    segment(ax, 68, y[0], 11, 'D: KV-limited', WAIT)
    ax.text(82, y[0], 'Interleave / bound work\nUnfinished C holds KV',
            fontsize=7, va='center', color=INK)

    label(ax, 1, y[1], 'Request priority', 'e.g., UniBoost')
    for x, w, name, color in ((27, 7, r'$A_d$', DECODE),
                              (34, 8, r'$B_p$', PREFILL),
                              (42, 13, r'$C_p$', PREFILL)):
        segment(ax, x, y[1], w, name, color)
    segment(ax, 58, y[1], 11, r'$D_p$', PREFILL)
    segment(ax, 69, y[1], 10, r'$C_p$', PREFILL)
    ax.text(82, y[1], 'Choose who runs first\nC quota not set by rank',
            fontsize=7, va='center', color=INK)

    label(ax, 1, y[2], 'Cache / admission', 'e.g., Past-Future')
    for x, w, name, color in ((27, 7, r'$A_d$', DECODE),
                              (34, 8, r'$B_p$', PREFILL),
                              (42, 13, r'$C_p$', PREFILL)):
        segment(ax, x, y[2], w, name, color)
    segment(ax, 58, y[2], 21, 'D: admission decision', GATE)
    ax.text(82, y[2], 'Control who enters\nRunning C can still grow',
            fontsize=7, va='center', color=INK)

    label(ax, 1, y[3], 'SPARE', 'same order, selective progress', accent=True)
    segment(ax, 27, y[3], 7, r'$A_d$', DECODE)
    segment(ax, 34, y[3], 8, r'$B_p$', PREFILL)
    segment(ax, 42, y[3], 6, r'$C_p$', PREFILL)
    segment(ax, 48, y[3], 7, 'defer C', DEFER, dashed=True, ink=ACCENT)
    segment(ax, 58, y[3], 10, r'$C_p$', PREFILL)
    segment(ax, 68, y[3], 11, r'$D_p$', PREFILL)
    ax.text(82, y[3], 'Control C quota\nif blocks saved',
            fontsize=7, va='center', color=ACCENT, fontweight='bold')

    ax.text(1, .51, 'If C\'s extra KV blocks prevent D from entering, reducing C can free space. Rows show decision axes, not traces.',
            fontsize=6.6, color=MUTED)
    fig.savefig(OUT / 'control_levers.pdf', facecolor='white')
    fig.savefig(OUT / 'control_levers.svg', facecolor='white')
    fig.savefig(OUT / 'control_levers.png', dpi=600, facecolor='white')
    plt.close(fig)
    print(OUT / 'control_levers.pdf')


if __name__ == '__main__':
    main()
