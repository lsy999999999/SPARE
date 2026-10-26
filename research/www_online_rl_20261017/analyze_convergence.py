"""Summarize and plot held-out learning curves without checkpoint selection."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/www_online_rl_mpl')
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.home() / '.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

from research.www_online_rl_20261017 import run

OUT = run.OUT / 'analysis'
SEEDS = (1017, 2027, 3037)
COLORS = ('#286b91', '#957343', '#89749b')

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8,
    'xtick.labelsize': 6, 'ytick.labelsize': 6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
})


def main():
    assert run.base.read(run.OUT / 'convergence_status.json')['complete']
    runs = [run.base.read(run.OUT / f'convergence_seed{s}.json') for s in SEEDS]
    assert runs[0]['matches_original_checkpoint']
    rows = []
    for item in runs:
        for point in item['points']:
            for score in point['summary']:
                rows.append(dict(seed=item['seed'], episodes=point['training_episodes'],
                                 td_updates=point['td_updates'], kv_blocks=score['kv_blocks'],
                                 rate=score['rate'], windows=score['windows'],
                                 mean_ttft_ms=score['mean_ttft_ms']))
    with (OUT / 'convergence.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), squeeze=False)
    fig.subplots_adjust(left=.12, right=.98, top=.93, bottom=.20,
                        wspace=.25, hspace=.31)
    for i, kv in enumerate((1024, 4096)):
        for j, rate in enumerate((.5, 4.)):
            ax = axes[j, i]
            values = [r for r in rows if r['kv_blocks'] == kv and r['rate'] == rate]
            baseline = next(r['mean_ttft_ms'] for r in values if r['episodes'] == 0)
            ax.axhline(baseline, color='#424f52', linestyle='--', linewidth=1,
                       label='SPARE')
            for seed, color in zip(SEEDS, COLORS):
                series = sorted((r for r in values if r['seed'] == seed),
                                key=lambda r: r['episodes'])
                ax.plot([r['episodes'] for r in series],
                        [r['mean_ttft_ms'] for r in series],
                        marker='o', markersize=3.2, linewidth=1.2,
                        color=color, label=f'RL seed {seed}')
            ax.set_title(f'KV {kv} | {rate:g} QPS', loc='left')
            ax.set_xlim(0, 512)
            ax.set_xticks((0, 128, 256, 384, 512))
            ax.grid(axis='y', color='#dce2e5', linewidth=.45)
            ax.set_axisbelow(True)
            if j == 1:
                ax.set_xlabel('Training episodes')
            if i == 0:
                ax.set_ylabel('Mean TTFT (ms)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               bbox_to_anchor=(.53, .05), fontsize=7)
    fig.canvas.draw()
    stem = OUT / 'online_rl_convergence'
    require_matplotlib_panel_alignment(fig, axes=list(axes.flat),
                                       json_out=str(stem)+'.alignment.json', strict=True)
    fig.savefig(str(stem)+'.png', dpi=300)
    fig.savefig(str(stem)+'.pdf')
    fig.savefig(str(stem)+'.svg')
    plt.close(fig)
    lines = ['# Online RL convergence', '',
             'Three independent seeds, 512 training episodes each. Evaluation',
             'uses the same eight held-out development windows at every checkpoint;',
             'the 50-window test set was not used. Episode 0 is the SPARE fallback.',
             'No checkpoint was selected from these curves.', '',
             '|KV|QPS|SPARE TTFT ms|RL seed 1017|RL seed 2027|RL seed 3037|',
             '|---:|---:|---:|---:|---:|---:|']
    for kv in (1024, 4096):
        for rate in (.5, 4.):
            subset = {(r['seed'], r['episodes']): r['mean_ttft_ms'] for r in rows
                      if r['kv_blocks'] == kv and r['rate'] == rate}
            lines.append(f'|{kv}|{rate:g}|{subset[1017, 0]:.2f}|'
                         + '|'.join(f'{subset[seed, 512]:.2f}' for seed in SEEDS) + '|')
    lines += ['', 'The learning curves are non-monotonic. At KV 4096 and 4 QPS,',
              'all three final RL policies have higher mean TTFT than SPARE.',
              'This run does not establish convergence or a general claim about',
              'other RL algorithms or reward designs.', '',
              'Seed 1017 exactly reproduced the original Q table. Each of the',
              'three remote runs took about 399 s wall time in parallel.']
    (OUT / 'CONVERGENCE_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
