"""Preliminary paired KV scan: all methods, exact coordinates, scatter only."""
from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D

from research.www_kv_capacity_sketch_20261015.run import (
    CAPACITIES, OUT, REFERENCE_QPS, SOURCES, prior, rate_path, read, summary, write,
)
from research.www_kv_capacity_paper_20261015.analyze import (
    COLORS, DISPLAY, csv_write, dominates, plt, mpl,
)

mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})
MARKERS = dict(zip(CAPACITIES, ('o', '^', 's', 'D', 'v', 'P', '*', 'X', '<', '>', 'h', 'p', '8', 'd')))


def analyze():
    target = OUT / 'analysis'
    target.mkdir(parents=True, exist_ok=True)
    scenes = []
    for source in SOURCES:
        for blocks in CAPACITIES:
            for arm in prior.MAIN:
                capacity, failed = 0., None
                for rate in prior.RATES:
                    data = read(rate_path(source, blocks, arm, rate))
                    assert data['summary'] == summary(data['rows'], rate)
                    if failed is None:
                        if data['summary']['passed']:
                            capacity = rate
                        else:
                            failed = rate
                    if rate == REFERENCE_QPS:
                        ttft = data['summary']['mean_ttft_ms']
                    if failed is not None and rate >= REFERENCE_QPS:
                        break
                scenes.append(dict(source=source, kv_blocks=blocks, arm=arm,
                                   capacity_qps=capacity, next_failed_qps=failed,
                                   capacity_is_lower_bound=failed is None,
                                   mean_ttft_ms=ttft, reference_feasible=capacity >= REFERENCE_QPS,
                                   n_windows=2))
    means, comparisons = [], []
    for blocks in CAPACITIES:
        for arm in prior.MAIN:
            pair = [p for p in scenes if p['kv_blocks'] == blocks and p['arm'] == arm]
            assert len(pair) == 2
            means.append(dict(kv_blocks=blocks, arm=arm, method=DISPLAY[arm],
                              capacity_qps=float(np.mean([p['capacity_qps'] for p in pair])),
                              mean_ttft_ms=float(np.mean([p['mean_ttft_ms'] for p in pair])),
                              reference_feasible=all(p['reference_feasible'] for p in pair),
                              n_windows_per_source=2))
        ours = next(p for p in means if p['kv_blocks'] == blocks and p['arm'] == 'arrival')
        for rival in (p for p in means if p['kv_blocks'] == blocks and p['arm'] != 'arrival'):
            comparisons.append(dict(kv_blocks=blocks, baseline=rival['arm'],
                                    baseline_dominates=dominates(rival, ours),
                                    spare_dominates=dominates(ours, rival)))
    csv_write(target / 'by_source.csv', scenes)
    csv_write(target / 'macro_averages.csv', means)
    csv_write(target / 'comparisons.csv', comparisons)
    write(target / 'points.json', dict(scenes=scenes, means=means, comparisons=comparisons))

    # Question: how do fixed paper policies compare as KV memory changes?
    # Single quantitative comparison panel; same paired windows for every point.
    # No CI: user requested a quick two-window sketch. Overlaps are not jittered.
    fig, ax = plt.subplots(figsize=(7.2, 6.5))
    fig.subplots_adjust(left=.12, right=.97, bottom=.35, top=.96)
    for arm in (*prior.MAIN[1:], 'arrival'):
        for p in (p for p in means if p['arm'] == arm):
            ours = arm == 'arrival'
            ax.scatter(p['mean_ttft_ms'], p['capacity_qps'],
                       marker=MARKERS[p['kv_blocks']], s=66 if ours else 32,
                       facecolors='none' if ours or not p['reference_feasible'] else COLORS[arm],
                       edgecolors=COLORS[arm] if ours or not p['reference_feasible'] else 'white',
                       linewidths=1.1 if ours else .4,
                       alpha=1. if ours else .8, zorder=5 if ours else 2)
    assert all(p['mean_ttft_ms'] > 0 for p in means)
    ax.set_xscale('log')
    ax.set_xticks([300, 500, 1000, 2000, 5000, 10000],
                  labels=['300', '500', '1000', '2000', '5000', '10000'])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel('Mean TTFT at 0.5 QPS (ms; log scale)')
    ax.set_ylabel('Mean serving capacity (QPS)')
    ax.grid(axis='y', color='#E7EBED', linewidth=.6)
    ax.set_axisbelow(True)
    ax.margins(x=.09, y=.12)
    methods = [Line2D([], [], linestyle='None', marker='o', color=COLORS[a],
                      markerfacecolor='none' if a == 'arrival' else COLORS[a],
                      markersize=5, label=DISPLAY[a]) for a in prior.MAIN]
    fig.legend(handles=methods, ncol=3, loc='upper center', bbox_to_anchor=(.53, .255),
               fontsize=7, columnspacing=2.1, handletextpad=.5)
    kv = [Line2D([], [], linestyle='None', marker=MARKERS[b], color='#58656C',
                 markersize=5, label=f'{b}') for b in CAPACITIES]
    fig.legend(handles=kv, ncol=7, loc='lower center', bbox_to_anchor=(.53, .015),
               title='KV cache (blocks)', title_fontsize=7,
               fontsize=6.5, columnspacing=1.2, handletextpad=.4)
    assert not ax.lines, 'Scatter plot must not connect data points'
    assert len(ax.collections) == len(CAPACITIES) * len(prior.MAIN)
    fig.savefig(target / 'kv_capacity_scatter.png', dpi=350)
    fig.savefig(target / 'kv_capacity_scatter.pdf')
    fig.savefig(target / 'kv_capacity_scatter.svg')
    plt.close(fig)

    lines = ['# KV 容量扫描：初步散点图', '',
             '14档KV，12种方法，共168个方法×KV点；只画点、不连线。颜色表示方法，形状表示KV；'
             'SPARE使用红色空心标记，尽量保留重合baseline的可见性。精确重合的点不偏移。'
             '横轴为对数刻度；baseline空心点表示至少一类负载在参考0.5 QPS下未通过容量判据。', '',
             '仅CPU simulator。固定论文原版SPARE，T=1024、K=256、并发32；不使用alpha。'
             '每类负载选两个日期各自的首个测试窗口，所有方法/KV共用；每类窗口内等权平均，'
             '再对Code和Conversation等权平均。旧结果只复用对应窗口原始记录，不混用50窗口均值。', '',
             'P95 TTFT≤12秒、P99 TBT≤200毫秒，最后20秒平均积压增长≤max(2, offered QPS)。'
             'SLO使用各窗口分位数的均值。capacity为从低到高连续通过的最大QPS网格点；'
             'mean TTFT统一在0.5 QPS测量。图为初步点估计，无置信区间。', '',
             '11个baseline为共享模拟后端上的规则/代理实现，范围见design.json。胜负只比较相同KV。', '',
             '|KV blocks|SPARE capacity QPS|SPARE mean TTFT ms|同时优于SPARE的baseline|',
             '|---:|---:|---:|---|']
    for blocks in CAPACITIES:
        p = next(p for p in means if p['kv_blocks'] == blocks and p['arm'] == 'arrival')
        rivals = [DISPLAY[c['baseline']] for c in comparisons if c['kv_blocks'] == blocks and c['baseline_dominates']]
        lines.append(f"|{blocks}|{p['capacity_qps']:g}|{p['mean_ttft_ms']:.2f}|{', '.join(rivals) or '无（含并列）'}|")
    (target / 'REPORT_CN.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[-16:]))


if __name__ == '__main__':
    analyze()
