"""Aggregate fixed-default KV scans and plot every external adapter."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

from research.www_kv_capacity_paper_20261015.run import (
    CAPACITIES, LABELS, OUT, REFERENCE_QPS, SOURCES,
    prior, rate_path, read, summary, write,
)

os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'matplotlib_cache'))
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path.home() / '.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

DISPLAY = dict(LABELS, arrival='SPARE', past_future='Past-Future rule',
               s3_proxy='S3 proxy', orca_max='Orca reservation',
               ltr_proxy='LTR proxy')
COLORS = dict(zip(prior.MAIN, (
    '#BD3638', '#606B73', '#809CA9', '#2877AD', '#7CA6C3', '#BA9657',
    '#1B8882', '#85805C', '#755694', '#9D7B9D', '#C37450', '#738F61')))
MARKERS = dict(zip(CAPACITIES, ('o', '^', 's', 'D', 'v', 'P')))

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 7, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'legend.frameon': False,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white',
})


def csv_write(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_group(source, blocks, arm):
    by_rate = {}
    qps, first_failed = 0., None
    for rate in prior.RATES:
        path = rate_path(source, blocks, arm, rate)
        if not path.exists():
            return None
        data = read(path)
        assert data['summary'] == summary(data['rows'], rate)
        by_rate[rate] = data
        if first_failed is None:
            if data['summary']['passed']:
                qps = rate
            else:
                first_failed = rate
        if first_failed is not None and rate >= REFERENCE_QPS:
            break
    reference = by_rate[REFERENCE_QPS]
    failure = by_rate[first_failed]['summary'] if first_failed is not None else None
    reasons = []
    if failure is not None:
        if failure['p95_ttft_ms'] > 12000:
            reasons.append('p95_ttft')
        if failure['p99_tbt_ms'] > 200:
            reasons.append('p99_tbt')
        if failure['backlog_drift'] > failure['drift_limit']:
            reasons.append('backlog')
    return dict(source=source, kv_blocks=blocks, arm=arm,
                capacity_qps=qps, next_failed_qps=first_failed,
                capacity_is_lower_bound=first_failed is None,
                first_failure_reasons=','.join(reasons),
                mean_ttft_ms=reference['summary']['mean_ttft_ms'],
                reference_qps=REFERENCE_QPS, reference_feasible=qps >= REFERENCE_QPS,
                n_windows=50), reference['rows']


def dominates(first, second):
    return (first['reference_feasible'] and second['reference_feasible']
            and first['capacity_qps'] >= second['capacity_qps']
            and first['mean_ttft_ms'] <= second['mean_ttft_ms'] + 1e-8
            and (first['capacity_qps'] > second['capacity_qps']
                 or first['mean_ttft_ms'] < second['mean_ttft_ms'] - 1e-8))


def equal_point(first, second):
    return (first['capacity_qps'] == second['capacity_qps']
            and abs(first['mean_ttft_ms']-second['mean_ttft_ms']) < 1e-8)


def bootstrap_indices(samples, rng):
    # Resample observed test windows within each date; no synthetic measurements.
    days = sorted({sample['day'] for sample in samples})
    return np.concatenate([
        rng.choice([i for i, sample in enumerate(samples) if sample['day'] == day],
                   (2000, sum(sample['day'] == day for sample in samples)), replace=True)
        for day in days], axis=1)


def analyze(partial=False):
    scenes, raw = [], {}
    capacities = []
    for blocks in CAPACITIES:
        groups = [load_group(source, blocks, arm)
                  for source in SOURCES for arm in prior.MAIN]
        if any(group is None for group in groups):
            if partial:
                continue
            raise RuntimeError(f'Incomplete KV {blocks}; use --partial for a labelled preview')
        capacities.append(blocks)
        for point, rows in groups:
            scenes.append(point)
            raw[point['source'], blocks, point['arm']] = rows
    assert capacities
    means = []
    for blocks in capacities:
        for arm in prior.MAIN:
            pair = [point for point in scenes
                    if point['kv_blocks'] == blocks and point['arm'] == arm]
            assert len(pair) == 2
            means.append(dict(kv_blocks=blocks, arm=arm, method=DISPLAY[arm],
                              capacity_qps=float(np.mean([p['capacity_qps'] for p in pair])),
                              mean_ttft_ms=float(np.mean([p['mean_ttft_ms'] for p in pair])),
                              reference_feasible=all(p['reference_feasible'] for p in pair),
                              reference_qps=REFERENCE_QPS, n_windows_per_source=50))

    rng = np.random.default_rng(20261015)
    indices = {source: bootstrap_indices(raw[source, capacities[0], 'arrival'], rng)
               for source in SOURCES}
    comparisons = []
    for blocks in capacities:
        ours = next(p for p in means if p['kv_blocks'] == blocks and p['arm'] == 'arrival')
        for rival in (p for p in means if p['kv_blocks'] == blocks and p['arm'] != 'arrival'):
            boot_deltas = []
            for source in SOURCES:
                a, b = raw[source, blocks, 'arrival'], raw[source, blocks, rival['arm']]
                assert [r['name'] for r in a] == [r['name'] for r in b]
                delta = np.array([y['mean_ttft_ms']-x['mean_ttft_ms'] for x, y in zip(a, b)])
                boot_deltas.append(delta[indices[source]].mean(axis=1))
            lo, hi = np.quantile(np.mean(boot_deltas, axis=0), [.025, .975])
            comparisons.append(dict(kv_blocks=blocks, baseline=rival['arm'],
                                    spare_capacity_qps=ours['capacity_qps'],
                                    baseline_capacity_qps=rival['capacity_qps'],
                                    spare_mean_ttft_ms=ours['mean_ttft_ms'],
                                    baseline_mean_ttft_ms=rival['mean_ttft_ms'],
                                    spare_capacity_gain_qps=ours['capacity_qps']-rival['capacity_qps'],
                                    spare_ttft_gain_ms=rival['mean_ttft_ms']-ours['mean_ttft_ms'],
                                    ttft_gain_ci95_low=float(lo), ttft_gain_ci95_high=float(hi),
                                    baseline_dominates=dominates(rival, ours),
                                    spare_dominates=dominates(ours, rival),
                                    equal_point=equal_point(ours, rival)))
    target = OUT / ('preview' if partial else 'analysis')
    target.mkdir(parents=True, exist_ok=True)
    csv_write(target / 'by_source.csv', scenes)
    csv_write(target / 'macro_averages.csv', means)
    csv_write(target / 'paired_comparisons.csv', comparisons)
    write(target / 'points.json', dict(scenes=scenes, means=means, comparisons=comparisons))
    report(target, scenes, means, comparisons, capacities, partial)
    figures(target, means, capacities)
    print(target, flush=True)
    for point in means:
        if point['arm'] == 'arrival':
            rivals = [DISPLAY[r['baseline']] for r in comparisons
                      if r['kv_blocks'] == point['kv_blocks'] and r['baseline_dominates']]
            print(f"KV {point['kv_blocks']}: SPARE {point['capacity_qps']:g} QPS / "
                  f"{point['mean_ttft_ms']:.2f} ms; dominated by {rivals}", flush=True)


def report(target, scenes, means, comparisons, capacities, partial):
    lines = ['# 论文默认 SPARE：KV 容量扫描', '',
             '**状态：仅已完成档位预览。**' if partial else '**状态：六档 KV 完整结果。**', '',
             '固定论文默认 T=1024、K=256、并发上限32。SPARE 保留 n0=8、门槛系数128、'
             'H=τ=1秒、四候选顺序、完成保护和恢复分支；不使用双目标 α 选预算。'
             '11个外部实现使用相同默认 T/K；这是共同默认配置比较。', '',
             'CPU simulator；P95 TTFT≤12秒、P99 TBT≤200毫秒；最后20秒平均积压增长≤max(2,λ)个请求。'
             '容量取自低到高连续通过的最大QPS档位，作为网格通过下界，并保留下一失败档位。'
             '各方法、各KV下的mean TTFT统一在0.5 QPS测量。每类负载50个相同测试窗口，'
             '预热20秒、测量40秒；统计窗口内到达的请求全部追踪至结束，排空阶段不抬高capacity。'
             'SLO判定沿用旧实验：先计算各窗口P95 TTFT和P99 TBT，再对50窗取均值；积压增长也取均值。', '',
             'mean TTFT先对50个窗口等权平均。主图每个点再对Code与Conversation等权平均，纵轴是两类负载capacity的宏平均，'
             '不是混合请求流的容量。颜色表示方法、形状表示KV，连线只连接资源扫描点。'
             '胜负只在同一KV下判断；同一方法跨KV的点不构成α权衡曲线。', '',
             '|KV blocks|SPARE平均capacity QPS|SPARE平均TTFT ms|同时优于SPARE的外部实现|',
             '|---:|---:|---:|---|']
    for blocks in capacities:
        ours = next(p for p in means if p['kv_blocks'] == blocks and p['arm'] == 'arrival')
        dom = [DISPLAY[r['baseline']] for r in comparisons
               if r['kv_blocks'] == blocks and r['baseline_dominates']]
        lines.append(f"|{blocks}|{ours['capacity_qps']:.3f}|{ours['mean_ttft_ms']:.2f}|"
                     f"{', '.join(dom) if dom else '无（点估计）'}|")
    non_dominated = sum(not any(r['baseline_dominates'] for r in comparisons
                               if r['kv_blocks'] == blocks) for blocks in capacities)
    lines += ['', f'同KV、共同默认配置下，SPARE在{non_dominated}/{len(capacities)}档的宏平均点估计上不被外部实现支配。', '',
              '完整对照（每格：平均capacity QPS / 平均TTFT ms）：', '',
              '|方法|' + '|'.join(f'KV {blocks}' for blocks in capacities) + '|',
              '|---|' + '|'.join('---:' for _ in capacities) + '|']
    for arm in prior.MAIN:
        points = [next(p for p in means if p['kv_blocks'] == blocks and p['arm'] == arm)
                  for blocks in capacities]
        lines.append('|' + DISPLAY[arm] + '|' + '|'.join(
            f"{p['capacity_qps']:.3f} / {p['mean_ttft_ms']:.2f}" for p in points) + '|')
    lines += ['', '各来源分别判断：', '',
              '|负载|KV blocks|SPARE capacity QPS|SPARE TTFT ms|支配SPARE的外部实现|',
              '|---|---:|---:|---:|---|']
    for source in SOURCES:
        for blocks in capacities:
            group = [p for p in scenes if p['source'] == source and p['kv_blocks'] == blocks]
            ours = next(p for p in group if p['arm'] == 'arrival')
            dom = [DISPLAY[p['arm']] for p in group
                   if p['arm'] != 'arrival' and dominates(p, ours)]
            lines.append(f"|{source}|{blocks}|{ours['capacity_qps']:g}|{ours['mean_ttft_ms']:.2f}|"
                         f"{', '.join(dom) if dom else '无（点估计）'}|")
    lines += ['', '主图：![平均性能随KV容量变化](kv_capacity_mean.png)', '',
              '辅助图：![KV容量与两项指标](kv_capacity_metrics.png)', '',
              '12秒阈值来自之前的探索，本轮固定采用；两个端点复用既有测试窗口，中间四档补跑同一组窗口。'
              '48条复现轨迹的请求指标、积压、抢占和执行轨迹哈希与缓存完全一致。', '',
              '图显示点估计，不宣称显著性。paired_comparisons.csv 提供同窗配对、日期内分层bootstrap的'
              'mean TTFT差值95%区间（2000次）；capacity采用通过/失败网格边界，不把网格步长称作置信区间。'
              '各baseline为共同V1后端上的规则/代理实现，具体范围见design.json。SOLA calibrated内部控制器'
              '仍沿用既有5秒/200毫秒目标；本轮统一更改的是外部评价SLO。', '',
              '完整72个方法×KV均值见macro_averages.csv，144个来源×方法×KV结果见by_source.csv（预览版仅含已完成档位）。']
    (target / 'REPORT_CN.md').write_text('\n'.join(lines)+'\n')


def export(fig, axes, base):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=axes,
                                       json_out=str(base)+'.alignment.json',
                                       overlay_svg=str(base)+'.alignment.svg', strict=True)
    fig.savefig(str(base)+'.png', dpi=350)
    fig.savefig(str(base)+'.pdf')
    fig.savefig(str(base)+'.svg')
    fig.savefig(str(base)+'.tiff', dpi=600)
    plt.close(fig)


def method_handles():
    return [Line2D([], [], color=COLORS[arm], linewidth=1.7 if arm == 'arrival' else .9,
                   marker='o', markersize=4, label=DISPLAY[arm]) for arm in prior.MAIN]


def figures(target, means, capacities):
    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    fig.subplots_adjust(left=.12, right=.97, bottom=.31, top=.96)
    # All methods and completed KV settings are retained; exact overlaps remain overlaps.
    for arm in (*prior.MAIN[1:], 'arrival'):
        points = sorted([p for p in means if p['arm'] == arm], key=lambda p: p['kv_blocks'])
        ours = arm == 'arrival'
        ax.plot([p['mean_ttft_ms'] for p in points], [p['capacity_qps'] for p in points],
                color=COLORS[arm], linewidth=1.8 if ours else .75,
                alpha=1. if ours else .65, zorder=5 if ours else 1)
        for p in points:
            ax.scatter(p['mean_ttft_ms'], p['capacity_qps'], marker=MARKERS[p['kv_blocks']],
                       s=66 if ours else 29, facecolors=COLORS[arm] if p['reference_feasible'] else 'none',
                       edgecolors='white' if p['reference_feasible'] else COLORS[arm],
                       linewidths=.5, alpha=1. if ours else .85, zorder=6 if ours else 2)
    ax.set_xlabel('Mean TTFT at 0.5 QPS (ms)')
    ax.set_ylabel('Mean serving capacity (QPS)')
    ax.grid(axis='y', color='#E7EBED', linewidth=.6)
    ax.set_axisbelow(True)
    ax.margins(x=.10, y=.12)
    fig.legend(handles=method_handles(), ncol=3, loc='upper center',
               bbox_to_anchor=(.54, .225), fontsize=7, columnspacing=2.1,
               handlelength=1.9, handletextpad=.6)
    kv_handles = [Line2D([], [], marker=MARKERS[blocks], color='#58656C', linestyle='',
                         markersize=5, label=f'KV {blocks}') for blocks in capacities]
    fig.legend(handles=kv_handles, ncol=6, loc='lower center', bbox_to_anchor=(.53, .01),
               fontsize=6.5, columnspacing=1.2, handletextpad=.4)
    export(fig, [ax], target / 'kv_capacity_mean')

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9))
    fig.subplots_adjust(left=.085, right=.98, bottom=.37, top=.96, wspace=.32)
    for ax, metric, ylabel in zip(axes,
                                 ('capacity_qps', 'mean_ttft_ms'),
                                 ('Mean serving capacity (QPS)', 'Mean TTFT at 0.5 QPS (ms)')):
        for arm in (*prior.MAIN[1:], 'arrival'):
            points = sorted([p for p in means if p['arm'] == arm], key=lambda p: p['kv_blocks'])
            ax.plot([p['kv_blocks'] for p in points], [p[metric] for p in points],
                    color=COLORS[arm], linewidth=1.9 if arm == 'arrival' else .8,
                    marker='o', markersize=4.5 if arm == 'arrival' else 2.5,
                    alpha=1. if arm == 'arrival' else .75, zorder=5 if arm == 'arrival' else 1)
        ax.set_xlabel('KV cache (blocks)')
        ax.set_ylabel(ylabel)
        ax.set_xticks((768, 2048, 4096))
        ax.grid(axis='y', color='#E7EBED', linewidth=.6)
        ax.set_axisbelow(True)
        ax.margins(y=.13)
    fig.legend(handles=method_handles(), ncol=3, loc='lower center',
               bbox_to_anchor=(.53, .015), fontsize=6.5, columnspacing=2.3)
    export(fig, list(axes), target / 'kv_capacity_metrics')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    analyze(parser.parse_args().partial)
