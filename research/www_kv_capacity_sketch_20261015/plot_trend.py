"""KV-labelled TTFT-capacity scatter for paper SPARE and external baselines."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib as mpl

from research.www_kv_capacity_sketch_20261015.run import OUT, prior
from research.www_kv_capacity_paper_20261015.analyze import COLORS, DISPLAY

mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

sys.path.insert(0, str(Path.home() / '.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment


TARGET = OUT / 'analysis'
FIGURE = TARGET / 'kv_capacity_scatter_methods'
RED = '#BD3638'
MARKERS = dict(zip(prior.MAIN[1:], ('s', '^', 'v', 'D', 'P', 'X', '<', '>', 'h', 'p', '8')))

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': .7,
    'legend.frameon': False,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'savefig.facecolor': 'white',
})


def load():
    points = json.loads((TARGET / 'points.json').read_text())['means']
    assert len(points) == 14 * 12
    blocks = sorted({p['kv_blocks'] for p in points})
    rows = []
    for kv in blocks:
        ours = next(p for p in points if p['kv_blocks'] == kv and p['arm'] == 'arrival')
        baselines = [p for p in points if p['kv_blocks'] == kv and p['arm'] != 'arrival']
        assert len(baselines) == 11
        top_capacity = max(baselines, key=lambda p: (p['capacity_qps'], -p['mean_ttft_ms']))
        fastest = min(baselines, key=lambda p: (p['mean_ttft_ms'], -p['capacity_qps']))
        rows.append(dict(kv_blocks=kv, spare_capacity_qps=ours['capacity_qps'],
                         best_baseline_capacity_qps=top_capacity['capacity_qps'],
                         best_capacity_mean_ttft_ms=top_capacity['mean_ttft_ms'],
                         best_capacity_method=top_capacity['method'],
                         spare_mean_ttft_ms=ours['mean_ttft_ms'],
                         best_baseline_mean_ttft_ms=fastest['mean_ttft_ms'],
                         fastest_baseline_method=fastest['method']))
    return points, rows


def plot(points, rows):
    assert all(p['mean_ttft_ms'] > 0 and p['capacity_qps'] >= 0 for p in points)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    fig.subplots_adjust(left=.105, right=.97, top=.96, bottom=.31)
    for arm in prior.MAIN[1:]:
        baseline = [p for p in points if p['arm'] == arm]
        assert len(baseline) == 14
        ax.scatter([p['mean_ttft_ms'] for p in baseline],
                   [p['capacity_qps'] for p in baseline],
                   marker=MARKERS[arm], s=27, facecolor=COLORS[arm],
                   edgecolor='white', linewidth=.3, alpha=.6, zorder=2)
    groups = {}
    for r in rows:
        key = (round(r['spare_mean_ttft_ms'], 2), r['spare_capacity_qps'])
        groups.setdefault(key, []).append(r['kv_blocks'])
    for (ttft, capacity), kv in groups.items():
        ax.scatter(ttft, capacity, marker='o', s=88, facecolor='none',
                   edgecolor=RED, linewidth=1.6, zorder=5)
        label = ','.join(str(value) for value in kv)
        dx, dy, align = {
            768: (9, 7, 'left'), 896: (0, 11, 'center'),
            1024: (9, -10, 'left'),
        }.get(kv[0], (9, 0, 'left'))
        ax.annotate(label, (ttft, capacity), xytext=(dx, dy),
                    textcoords='offset points', color=RED, fontsize=7,
                    ha=align, va='center', zorder=6)
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator((300, 500, 1000, 2000, 5000, 10000)))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(265, 14000)
    ax.set_ylim(.25, 4.9)
    ax.set_xlabel('Mean TTFT')
    ax.set_ylabel('Mean serving capacity (QPS)')
    assert not ax.lines, 'Only isolated points may be used'
    handles = [Line2D([], [], linestyle='None', marker='o', markersize=7,
                      markerfacecolor='none', markeredgecolor=RED,
                      markeredgewidth=1.4, label='SPARE')]
    handles.extend(Line2D([], [], linestyle='None', marker=MARKERS[arm],
                          markersize=5, markerfacecolor=COLORS[arm],
                          markeredgecolor='white', alpha=.7, label=DISPLAY[arm])
                   for arm in prior.MAIN[1:])
    fig.legend(handles=handles, ncol=3, loc='lower center',
               bbox_to_anchor=(.52, .018), fontsize=7,
               columnspacing=1.3, labelspacing=.8, handletextpad=.5)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, axes=[ax], json_out=str(FIGURE) + '.alignment.json',
        overlay_svg=str(FIGURE) + '.alignment.svg', strict=True)
    fig.savefig(str(FIGURE) + '.png', dpi=350)
    fig.savefig(str(FIGURE) + '.pdf')
    fig.savefig(str(FIGURE) + '.svg')
    fig.savefig(str(FIGURE) + '.tiff', dpi=600)
    plt.close(fig)


def main():
    points, rows = load()
    with (TARGET / 'kv_capacity_trend_source.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    plot(points, rows)
    print(f'Plotted {len(points)} method-KV points from {len(rows)} KV capacities.')


if __name__ == '__main__':
    main()
