"""Export paired ablation outcomes without combining incompatible old runs."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from .run import OUT, ROOT, ARMS, LABELS, CAPACITIES, base, archive, prepare, group_path, rate_path

COLORS = dict(zip(ARMS, ('#0F4D92', '#767676', '#42949E', '#9A4D8E', '#B64342', '#bc8651')))
MARKERS = dict(zip(ARMS, ('o', 's', 'D', '^', 'v', 'X')))


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values, seed=20261016):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, (2.5, 97.5)))


def plot_points(points, folder):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/www_biobjective_ablation_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'font.size': 8, 'axes.titlesize': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': .7,
                         'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    folder.mkdir(parents=True, exist_ok=True)
    for source in ('code', 'conversation'):
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
        fig.subplots_adjust(left=.09, right=.985, bottom=.21, top=.91, hspace=.48, wspace=.32)
        fig.suptitle(source.title(), fontsize=11, y=.98)
        for ax, kv in zip(axes.flat, CAPACITIES):
            subset = [r for r in points if r['source'] == source and r['kv_blocks'] == kv]
            for row in sorted(subset, key=lambda r: r['arm'] == 'spare'):
                arm = row['arm']
                x, y = row['mean_ttft_ms'], row['capacity_qps']
                if x is None:
                    continue
                upper = row['next_failed_qps']
                if upper is not None:
                    ax.plot([x, x], [y, upper], linestyle=':', color=COLORS[arm], lw=.8, alpha=.75)
                ax.scatter(x, y, s=60 if arm == 'spare' else 28, marker=MARKERS[arm],
                           facecolors='none' if arm == 'spare' else COLORS[arm],
                           edgecolors=COLORS[arm], linewidths=1.3 if arm == 'spare' else .6,
                           zorder=5 if arm == 'spare' else 3)
            ax.set_title(f'KV = {kv} blocks', loc='left')
            ax.set_xscale('log')
            ax.grid(axis='y', color='#e5e7e9', linewidth=.5)
            ax.set_axisbelow(True)
            ax.set_ylim(bottom=0)
        for ax in axes[:, 0]:
            ax.set_ylabel('Serving capacity (QPS)')
        for ax in axes[-1]:
            ax.set_xlabel('Mean TTFT at 0.5 QPS (ms, log)')
        handles = [Line2D([], [], marker=MARKERS[a], color=COLORS[a], linestyle='none',
                          markerfacecolor='none' if a == 'spare' else COLORS[a],
                          label=LABELS[a]) for a in ARMS]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, .025),
                   ncol=3, fontsize=7, columnspacing=1.2)
        for ext in ('pdf', 'svg', 'png'):
            fig.savefig(folder / f'ablation_ttft_capacity_{source}.{ext}', dpi=300, facecolor='white')
        plt.close(fig)


def analyze():
    design = prepare()
    points, rates, paired = [], [], []
    references = {}
    for source in ('code', 'conversation'):
        for kv in CAPACITIES:
            for arm in ARMS:
                group = base.read(group_path(source, kv, arm))
                ref = base.read(rate_path(source, kv, arm, .5))
                assert ref['summary'] == archive.summary(ref['rows'], .5)
                references[source, kv, arm] = {r['name']: r for r in ref['rows']}
                valid = [r['mean_ttft_ms'] for r in ref['rows'] if r['valid']]
                ci = interval(valid) if len(valid) == 50 else (None, None)
                points.append(dict(source=source, kv_blocks=kv, arm=arm, label=LABELS[arm],
                                   mean_ttft_ms=ref['summary']['mean_ttft_ms'],
                                   ttft_ci_low_ms=ci[0], ttft_ci_high_ms=ci[1],
                                   capacity_qps=group['capacity_qps'],
                                   next_failed_qps=group['next_failed_qps'],
                                   capacity_is_lower_bound=group['capacity_is_lower_bound'],
                                   reference_feasible=group['reference_feasible'], windows=50))
                for rate in group['tested_rates']:
                    data = base.read(rate_path(source, kv, arm, rate))
                    assert data['summary'] == archive.summary(data['rows'], rate)
                    rates.append(dict(source=source, kv_blocks=kv, arm=arm, rate=rate,
                                      **{k: v for k, v in data['summary'].items() if k != 'failure_reasons'},
                                      failure_reasons='; '.join(data['summary']['failure_reasons'])))
            full = next(r for r in points if (r['source'], r['kv_blocks'], r['arm']) == (source, kv, 'spare'))
            ref = references[source, kv, 'spare']
            for arm in ARMS:
                if arm == 'spare':
                    continue
                ablated = references[source, kv, arm]
                assert set(ablated) == set(ref)
                diff = [ablated[n]['mean_ttft_ms'] - ref[n]['mean_ttft_ms']
                        for n in ref if ref[n]['valid'] and ablated[n]['valid']]
                ci = interval(diff) if len(diff) == 50 else (None, None)
                p = next(r for r in points if (r['source'], r['kv_blocks'], r['arm']) == (source, kv, arm))
                paired.append(dict(source=source, kv_blocks=kv, arm=arm,
                                   delta_ttft_ms=float(np.mean(diff)) if len(diff) == 50 else None,
                                   delta_ttft_ci_low_ms=ci[0], delta_ttft_ci_high_ms=ci[1],
                                   valid_pairs=len(diff),
                                   delta_capacity_qps=p['capacity_qps']-full['capacity_qps'],
                                   delta_definition='ablation minus SPARE'))
    analysis = OUT / 'analysis'
    csv_write(analysis / 'points.csv', points)
    csv_write(analysis / 'rate_metrics.csv', rates)
    csv_write(analysis / 'paired_deltas.csv', paired)
    plot_points(points, analysis)
    lines = ['# 同排序双目标消融', '',
             '所有结果为 CPU 拟合时钟模拟；SPARE 复用归档，其余五种消融补测。',
             '每负载50个配对窗口；T=1024、K=256、并发32；六档KV。',
             '横轴为共同0.5 QPS下mean TTFT；纵轴为满足冻结SLO和积压判据的最大连续通过网格点。',
             '点上方虚线表示到下一失败QPS的扫描区间，不是置信区间，也不连接不同方法。',
             'TTFT的窗口bootstrap区间及相对SPARE的配对差值见CSV；只表达固定日期内窗口差异。',
             'P95/P99先在各窗口内计算再取均值，沿用前沿图协议；未将其当作合并请求的总体分位数。',
             '重算和抢占在完整回放（含排空）统计，只作机制辅助；不同capacity点对应不同输入压力。',
             '12秒TTFT阈值沿用探索设置；有限测量窗和积压容差不能证明长期稳定。', '',
             '|负载|KV|方法|mean TTFT (ms)|capacity (QPS)|下一失败QPS|',
             '|---|---:|---|---:|---:|---:|']
    for p in points:
        latency = f"{p['mean_ttft_ms']:.3f}" if p['mean_ttft_ms'] is not None else '不可测'
        lines.append(f"|{p['source']}|{p['kv_blocks']}|{p['label']}|{latency}|{p['capacity_qps']:g}|{p['next_failed_qps']}|")
    (analysis / 'REPORT_CN.md').write_text('\n'.join(lines)+'\n')
    base.write(analysis / 'provenance.json', dict(design_sha256=base.digest(OUT / 'design.json'),
                                               analyzer_sha256=base.digest(Path(__file__)),
                                               source_windows=len(design['samples']), points=len(points)))


if __name__ == '__main__':
    analyze()
