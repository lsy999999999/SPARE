"""Read archived 50-window capacity sweeps; compare TTFT at common offered load."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/www_ttft_load_existing_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from research.www_capacity_pareto_20261015 import run as base
from research.www_kv_capacity_50wide_20261016 import run as archive

ROOT = base.ROOT
SOURCE = ROOT / 'artifacts/www_kv_capacity_50wide_20261016_r1'
OUT = ROOT / 'artifacts/www_ttft_load_existing_20261016_r1'
CAPACITIES = (768, 4096)
METHODS = ('arrival', 'native', 'sarathi', 'sola_rule', 'uniboost_static',
           'past_future', 's3_proxy', 'apt_serve_kv_rule')
LABELS = ('SPARE', 'vLLM V1', 'Sarathi rule', 'SOLA rule', 'UniBoost rule',
          'Past-Future rule', 'S3 proxy', 'Apt-Serve rule')
COLORS = ('#0F4D92', '#767676', '#8da8b4', '#bc8651', '#8d78a0', '#218983', '#B64342', '#a19b5d')
MARKERS = ('o', 's', '^', 'v', 'D', 'P', 'X', '*')


def save_csv(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, include_failure=False):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'font.size': 9, 'axes.titlesize': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .7, 'legend.frameon': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.8), sharey=not include_failure)
    fig.subplots_adjust(left=.10, right=.98, top=.93, bottom=.23, hspace=.36, wspace=.22)
    for i, source in enumerate(('code', 'conversation')):
        for j, kv in enumerate(CAPACITIES):
            ax = axes[i, j]
            for index in (*range(1, len(METHODS)), 0):
                arm = METHODS[index]
                selected = sorted((r for r in rows if r['source'] == source and
                                   r['kv_blocks'] == kv and r['arm'] == arm), key=lambda r: r['rate'])
                passing = [r for r in selected if r['passed'] and r['within_capacity']]
                x = [r['rate'] for r in passing]
                y = [r['mean_ttft_ms']/1000 for r in passing]
                color, marker = COLORS[index], MARKERS[index]
                ax.plot(x, y, color=color, marker=marker, markersize=4,
                        linewidth=1.8 if index == 0 else 1.0,
                        zorder=6 if index == 0 else 3, alpha=1 if index == 0 else .85)
                if include_failure:
                    failed = [r for r in selected if not r['passed'] and r['mean_ttft_ms'] is not None]
                    for r in failed:
                        if passing:
                            last = passing[-1]
                            ax.plot([last['rate'], r['rate']],
                                    [last['mean_ttft_ms']/1000, r['mean_ttft_ms']/1000],
                                    color=color, linewidth=.9, linestyle='--')
                        ax.scatter(r['rate'], r['mean_ttft_ms']/1000, marker=marker,
                                   s=28, facecolors='none', edgecolors=color, zorder=4)
            ax.set_title(f'{source.title()} | KV = {kv}', loc='left')
            ax.grid(axis='y', color='#e5e7e9', linewidth=.5)
            ax.set_axisbelow(True)
            ax.set_xlim(left=0)
            if include_failure:
                ax.set_yscale('log')
            else:
                ax.set_ylim(bottom=0)
            if i == 1:
                ax.set_xlabel('Offered request rate (QPS)')
            if j == 0:
                ax.set_ylabel('Mean TTFT (s)' + ('; log scale' if include_failure else ''))
    if not include_failure:
        maximum = max(r['mean_ttft_ms']/1000 for r in rows
                      if r['passed'] and r['within_capacity'])
        axes[0, 0].set_ylim(0, maximum * 1.08)
        for ax in axes.flat:
            assert ax.get_ylim()[1] >= maximum
    handles = [Line2D([], [], color=c, marker=m, lw=1.4, markersize=4, label=l)
               for c, m, l in zip(COLORS, MARKERS, LABELS)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .045),
               ncol=4, columnspacing=1.1, fontsize=8)
    stem = 'ttft_vs_qps_with_failure_boundary' if include_failure else 'ttft_vs_qps_passing'
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=300, facecolor='white')
    plt.close(fig)


def analyze():
    OUT.mkdir(parents=True, exist_ok=True)
    design = base.read(SOURCE / 'design.json')
    assert design['windows_per_workload'] == 50
    assert design['budget'] == 1024 and design['chunk'] == 256
    hashes = {str((SOURCE / 'design.json').relative_to(ROOT)): base.digest(SOURCE / 'design.json')}
    rows, records = [], {}
    for source in ('code', 'conversation'):
        wanted = {s['name'] for s in design['samples'] if s['source'] == source}
        assert len(wanted) == 50
        for kv in CAPACITIES:
            for arm, label in zip(METHODS, LABELS):
                group_path = SOURCE / 'groups' / f'{source}_c{kv}_{arm}.json'
                group = base.read(group_path)
                hashes[str(group_path.relative_to(ROOT))] = base.digest(group_path)
                files = list((SOURCE / 'rates').glob(f'{source}_c{kv}_{arm}_r*.json'))
                for path in files:
                    data = base.read(path)
                    rr = data['rows']
                    rate = float(rr[0]['rate'])
                    assert len(rr) == 50 and {r['name'] for r in rr} == wanted
                    assert all(r['rate'] == rate and r['capacity'] == kv and r['arm'] == arm and
                               r['source'] == source and r['budget'] == 1024 for r in rr)
                    stats = archive.summary(rr, rate)
                    assert stats == data['summary'], str(path)
                    key = source, kv, arm, rate
                    assert key not in records
                    records[key] = {r['name']: r for r in rr}
                    hashes[str(path.relative_to(ROOT))] = base.digest(path)
                    rows.append(dict(source=source, kv_blocks=kv, arm=arm, label=label,
                                     rate=rate, mean_ttft_ms=stats['mean_ttft_ms'],
                                     p95_ttft_ms=stats['p95_ttft_ms'], p99_tbt_ms=stats['p99_tbt_ms'],
                                     backlog_drift=stats['backlog_drift'], passed=stats['passed'],
                                     failed_samples=stats['failed_samples'],
                                     within_capacity=rate <= group['capacity_qps'],
                                     capacity_qps=group['capacity_qps'], next_failed_qps=group['next_failed_qps'],
                                     windows=50))
    rows.sort(key=lambda r: (r['source'], r['kv_blocks'], r['arm'], r['rate']))
    lookup = {(r['source'], r['kv_blocks'], r['arm'], r['rate']): r for r in rows}
    paired = []
    rng = np.random.default_rng(20261016)
    boot = rng.integers(0, 50, (2000, 50))
    for row in rows:
        if row['arm'] == 'arrival':
            continue
        key = row['source'], row['kv_blocks'], 'arrival', row['rate']
        spare = lookup.get(key)
        if spare is None or not (spare['passed'] and row['passed'] and
                                 spare['within_capacity'] and row['within_capacity']):
            continue
        ours = records[key]
        other = records[row['source'], row['kv_blocks'], row['arm'], row['rate']]
        names = sorted(ours)
        delta = np.array([ours[n]['mean_ttft_ms']-other[n]['mean_ttft_ms'] for n in names])
        low, high = np.percentile(delta[boot].mean(axis=1), (2.5, 97.5))
        paired.append(dict(source=row['source'], kv_blocks=row['kv_blocks'], arm=row['arm'],
                           rate=row['rate'], spare_ttft_ms=spare['mean_ttft_ms'],
                           comparator_ttft_ms=row['mean_ttft_ms'],
                           reduction_pct=100*(1-spare['mean_ttft_ms']/row['mean_ttft_ms']),
                           paired_delta_ms=float(delta.mean()),
                           paired_delta_ci_low_ms=float(low), paired_delta_ci_high_ms=float(high),
                           spare_capacity_qps=spare['capacity_qps'],
                           comparator_capacity_qps=row['capacity_qps'], windows=50))
    save_csv('rate_metrics.csv', rows)
    save_csv('paired_common_feasible.csv', paired)
    plot(rows)
    plot(rows, include_failure=True)
    examples = [('code',4096,'s3_proxy',.5), ('code',4096,'s3_proxy',3.),
                ('code',768,'s3_proxy',1.), ('code',768,'s3_proxy',2.5),
                ('conversation',4096,'s3_proxy',4.), ('conversation',768,'s3_proxy',1.5)]
    lines = ['# 固定 KV 下的 mean TTFT—输入 QPS：已有数据复算', '',
             '本轮新增模拟次数：0。复用50窗口capacity扫描，固定T=1024、K=256、并发32。',
             f'共读取{len(rows)}个方法×负载×KV×QPS组，{len(rows)*50}条窗口记录；'
             f'得到{len(paired)}个双方均通过原判据的同QPS配对比较。', '',
             '## 图的读法', '',
             '- ttft_vs_qps_passing：只画通过原SLO/积压规则且位于连续通过区间内的点。'
             '曲线终点为各自的capacity通过点；不同终点不代表遗漏了相同负载下已知的可行结果。',
             '- ttft_vs_qps_with_failure_boundary：另以虚线及空心点展示首个失败网格点；'
             '失败点的mean TTFT是有限输入窗回放结果，不能视为稳定运行延迟。',
             '- 所有曲线点都是50窗口mean TTFT的等权均值。图不绘置信带；'
             'CSV给出同窗口配对差值的bootstrap区间，仅针对这些日期内的窗口。',
             '- 各方法按自身扫描记录停止，未对缺失高负载点外推。低负载曲线重合不做数据抖动。', '',
             '## 比capacity散点增加的信息', '',
             'capacity是按尾延迟与积压判据得到的边界，未编码边界内的平均起答表现。'
             '相同capacity和相同0.5 QPS延迟，可以对应不同的中高负载延迟。'
             '方法排序也可能随负载变化，因此不能只由单个参考负载推断全负载优势。', '',
             '|负载|KV|输入QPS|对照|SPARE TTFT ms|对照TTFT ms|相对降低|',
             '|---|---:|---:|---|---:|---:|---:|']
    for key in examples:
        r = next(r for r in paired if (r['source'],r['kv_blocks'],r['arm'],r['rate']) == key)
        lines.append(f"|{r['source']}|{r['kv_blocks']}|{r['rate']:g}|{r['arm']}|"
                     f"{r['spare_ttft_ms']:.2f}|{r['comparator_ttft_ms']:.2f}|{r['reduction_pct']:+.2f}%|")
    lines += ['', '负数表示SPARE更慢。例子用于解释信息增量，完整比较保留于CSV。', '',
              '## 与newidea第三点的关系', '',
              '第三点的120秒截断压力日志保存到达数、完成数、等待首token数量等检查点，'
              '未保存每个请求的首token时间，不能从这些计数唯一还原mean TTFT。'
              '其预算还是旧SLO下验证选出的配置，与本图固定预算不同，不能直接拼图当成同一次运行。',
              '本图取自60秒capacity扫描：20秒预热，统计20–60秒到达请求并追踪到完成，'
              '避免仅统计截止前已起答请求造成选择偏差。', '',
              '## 测量边界', '',
              '沿用窗口P95 TTFT均值≤12秒、窗口P99 TBT均值≤200毫秒、'
              '40–60秒未完成数增长均值≤max(2,输入QPS)，并要求50窗口全部有效。'
              '12秒为既有探索SLO；容量是离散网格和有限时间判据估计，不是严格稳定性证明。'
              '比较对象是共享V1后端的规则/代理实现，计时来自拟合模型。',
              '这张图补充性能随负载变化的形态，不单独证明省块或保护机制的因果贡献；'
              '该问题仍由正在补测的同排序消融回答。', '',
              '复现：`simulator/bin/python -B -m research.www_ttft_load_existing_20261016.analyze`。']
    (OUT/'REPORT_CN.md').write_text('\n'.join(lines)+'\n')
    base.write(OUT/'provenance.json',dict(new_simulations=0,rate_groups=len(rows),window_records=len(rows)*50,
                                       paired_comparisons=len(paired),source_sha256=hashes,
                                       analysis_sha256=base.digest(Path(__file__))))
    print(json.dumps(dict(new_simulations=0,rate_groups=len(rows),window_records=len(rows)*50,
                          paired_comparisons=len(paired),output=str(OUT))))


if __name__ == '__main__':
    analyze()
