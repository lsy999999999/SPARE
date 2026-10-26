"""Plot the fixed-KV workload comparison under the mean-TTFT capacity rule."""
from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/www_online_rl_kv2048_mpl')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'artifacts/www_online_rl_mean_capacity_20261017_r1/analysis/capacity_2s.csv'
OUT = DATA.parent
SPARE = '#b64342'
RL = '#0f4d92'


def main():
    with DATA.open(newline='') as stream:
        row = next(row for row in csv.DictReader(stream) if row['kv_blocks'] == '2048')

    sources = [('Code', 'code'), ('Conversation', 'conversation')]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    fig.subplots_adjust(left=.16, right=.97, top=.81, bottom=.30)
    for y, (label, key) in enumerate(sources):
        spare = float(row[f'spare_{key}_capacity_qps'])
        rl = float(row[f'rl_{key}_capacity_qps'])
        ax.plot((spare, rl), (y, y), color='#aeb7bb', linewidth=1.4, zorder=1)
        ax.scatter(spare, y, s=100, facecolors='white', edgecolors=SPARE,
                   linewidths=2, label='SPARE' if y == 0 else None, zorder=3)
        ax.scatter(rl, y, s=75, marker='D', color=RL,
                   label='Frozen Q-learning policy' if y == 0 else None, zorder=4)
        ax.annotate(f'{spare:.2f}', (spare, y), xytext=(0, -19),
                    textcoords='offset points', ha='center', fontsize=8, color=SPARE)
        ax.annotate(f'{rl:.2f}', (rl, y), xytext=(0, 11),
                    textcoords='offset points', ha='center', fontsize=8, color=RL)
        gain = 100 * (rl / spare - 1)
        ax.text(4.42, y, f'RL {gain:+.1f}%', ha='right', va='center',
                fontsize=9, fontweight='bold', color=RL if gain > 0 else SPARE)

    ax.set_yticks((0, 1), (label for label, _ in sources))
    ax.set_ylim(-.38, 1.38)
    ax.invert_yaxis()
    ax.set_xlim(2.95, 4.47)
    ax.set_xticks((3.0, 3.2, 3.4, 3.6, 3.8, 4.0))
    ax.set_xlabel('Maximum passing offered load (QPS)')
    ax.set_title(r'KV 2048 blocks | mean TTFT $\leq$ 2 s', loc='left', fontsize=11)
    ax.grid(axis='x', color='#e0e5e7', linewidth=.55)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.legend(loc='lower center', bbox_to_anchor=(.54, .02), ncol=2,
               frameon=False, fontsize=8)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'kv2048_code_conversation.{suffix}', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
