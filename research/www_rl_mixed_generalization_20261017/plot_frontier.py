"""Plot measured KV=2048 capacity points for SPARE and mixed-trained RL."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from statistics import mean

os.environ.setdefault('MPLCONFIGDIR', '/tmp/www_rl_frontier_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/www_rl_mixed_generalization_20261017_r1/frontier'
SEEDS = (1017, 2027, 3037)
SOURCES = ('code', 'conversation')
LIMITS = (250, 350, 500, 750, 1000, 1500, 2000, 3000, 4000,
          5000, 6000, 8000, 10000, 12000, 16000)
COLORS = {'spare': '#b64342', 'rl_mixed': '#176c82'}
LABELS = {'spare': 'SPARE', 'rl_mixed': 'RL-Mixed'}


def load_groups():
    groups = {}
    for source in SOURCES:
        for arm, seeds in (('spare', (0,)), ('rl_mixed', SEEDS)):
            for seed in seeds:
                path = OUT / 'groups' / f'original_{source}_c2048_{arm}_seed{seed}.json'
                with path.open() as stream:
                    group = json.load(stream)
                assert group['stage'] == 'complete'
                assert (group['source'], group['arm'], group['seed'], group['kv_blocks']) == (
                    source, arm, seed, 2048)
                assert tuple(p['mean_ttft_limit_ms'] for p in group['thresholds']) == LIMITS
                capacities = [p['capacity_qps'] for p in group['thresholds']]
                assert capacities == sorted(capacities)
                for point in group['thresholds']:
                    if point['capacity_qps']:
                        assert point['resolution_qps'] <= .020001
                        assert point['attained_mean_ttft_ms'] <= point['mean_ttft_limit_ms']
                        assert point['attained_p99_tbt_ms'] <= 200
                        assert point['attained_backlog_drift'] <= max(
                            2, point['capacity_qps'])
                    else:
                        assert point['below_minimum_measured_rate']
                groups[source, arm, seed] = group
    for arm, seeds in (('spare', (0,)), ('rl_mixed', SEEDS)):
        for source in SOURCES:
            assert len({groups[source, arm, seed]['input_sha256'] for seed in seeds}) == 1
    for source in SOURCES:
        assert groups[source, 'spare', 0]['input_sha256'] == (
            groups[source, 'rl_mixed', SEEDS[0]]['input_sha256'])
    return groups


def make_rows(groups):
    rows = []
    for source in (*SOURCES, 'mean'):
        for arm, seeds in (('spare', (0,)), ('rl_mixed', SEEDS)):
            for index, limit in enumerate(LIMITS):
                values = []
                below_floor = []
                for seed in seeds:
                    parts = (SOURCES if source == 'mean' else (source,))
                    values.append(mean(groups[part, arm, seed]['thresholds'][index]
                                       ['capacity_qps'] for part in parts))
                    below_floor.extend(groups[part, arm, seed]['thresholds'][index]
                                       ['below_minimum_measured_rate'] for part in parts)
                rows.append(dict(source=source, kv_blocks=2048, arm=arm,
                                 mean_ttft_limit_ms=limit,
                                 capacity_qps=mean(values),
                                 seed_min_qps=min(values), seed_max_qps=max(values),
                                 seed1017_qps=(values[0] if arm == 'rl_mixed' else ''),
                                 seed2027_qps=(values[1] if arm == 'rl_mixed' else ''),
                                 seed3037_qps=(values[2] if arm == 'rl_mixed' else ''),
                                 n_seeds=len(seeds), windows_per_source=50,
                                 below_minimum_rate_groups=sum(below_floor)))
    return rows


def plot(rows):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.titlesize': 11, 'axes.labelsize': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'legend.frameon': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.5), sharey=True)
    titles = {'code': 'Code', 'conversation': 'Conversation',
              'mean': 'Equal-weight mean'}
    for ax, source in zip(axes, (*SOURCES, 'mean')):
        for arm, marker in (('spare', 'o'), ('rl_mixed', 's')):
            points = [row for row in rows if row['source'] == source and row['arm'] == arm]
            xs = [row['mean_ttft_limit_ms'] / 1000 for row in points]
            ys = [row['capacity_qps'] for row in points]
            if arm == 'rl_mixed':
                lower = [row['capacity_qps'] - row['seed_min_qps'] for row in points]
                upper = [row['seed_max_qps'] - row['capacity_qps'] for row in points]
                ax.errorbar(xs, ys, yerr=[lower, upper], fmt='none',
                            color=COLORS[arm], alpha=.55, linewidth=.75,
                            capsize=1.6, zorder=2)
            ax.plot(xs, ys, color=COLORS[arm], linewidth=1.15, alpha=.7,
                    zorder=3)
            ax.scatter(xs, ys, marker=marker, s=30, color=COLORS[arm],
                       edgecolors='white', linewidths=.5, zorder=4,
                       label=LABELS[arm])
        ax.set_xscale('log')
        ax.set_xlim(.21, 19)
        ax.set_xticks((.25, .5, 1, 2, 4, 8, 16),
                      labels=('0.25', '0.5', '1', '2', '4', '8', '16'))
        ax.set_title(titles[source])
        ax.set_xlabel('Mean TTFT limit (s)')
        ax.grid(axis='y', color='#d9e0e3', linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Serving capacity (QPS)')
    axes[0].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2,
               bbox_to_anchor=(.5, .055))
    fig.text(.5, .015,
             'Zero denotes no feasible point at or above 0.25 QPS; '
             'RL whiskers show the range of three training seeds.',
             ha='center', fontsize=8, color='#4d5961')
    fig.suptitle('KV = 2048 blocks | P99 TBT <= 200 ms | '
                 'backlog <= max(2, offered QPS)', y=.99, fontsize=12)
    fig.tight_layout(rect=(0, .14, 1, .94), w_pad=2)
    analysis = OUT / 'analysis'
    analysis.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(analysis / f'kv2048_spare_rl_mixed_capacity.{suffix}', dpi=300)
    plt.close(fig)


def main():
    groups = load_groups()
    rows = make_rows(groups)
    analysis = OUT / 'analysis'
    analysis.mkdir(parents=True, exist_ok=True)
    with (analysis / 'kv2048_spare_rl_mixed_capacity.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    plot(rows)
    for source in (*SOURCES, 'mean'):
        chosen = [row for row in rows if row['source'] == source
                  and row['mean_ttft_limit_ms'] == 2000]
        print(source, {row['arm']: round(row['capacity_qps'], 3) for row in chosen})


if __name__ == '__main__':
    main()
