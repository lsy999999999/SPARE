"""Plot equal-weight Code/Conversation TTFT at common feasible QPS and KV=4096."""
from __future__ import annotations

import csv
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/www_ttft_load_macro_4096_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from research.www_capacity_pareto_20261015 import run as base
from research.www_kv_capacity_50wide_20261016 import run as archive
from research.www_ttft_load_existing_20261016.analyze import (
    COLORS, LABELS, MARKERS, METHODS, OUT, SOURCE,
)

KV = 4096
SOURCES = ('code', 'conversation')


def build():
    design = base.read(SOURCE / 'design.json')
    assert design['budget'] == 1024 and design['chunk'] == 256
    assert design['windows_per_workload'] == 50
    rows = []
    for arm, label in zip(METHODS, LABELS):
        groups = {source: base.read(SOURCE / 'groups' / f'{source}_c{KV}_{arm}.json')
                  for source in SOURCES}
        rates = set()
        for source in SOURCES:
            rates.update(float(p.stem.rsplit('_r', 1)[1]) for p in
                         (SOURCE / 'rates').glob(f'{source}_c{KV}_{arm}_r*.json'))
        for rate in sorted(rates):
            halves = {}
            for source in SOURCES:
                path = SOURCE / 'rates' / f'{source}_c{KV}_{arm}_r{rate:g}.json'
                if not path.exists():
                    break
                data = base.read(path)
                assert len(data['rows']) == 50
                assert {r['source'] for r in data['rows']} == {source}
                assert {r['arm'] for r in data['rows']} == {arm}
                assert {r['capacity'] for r in data['rows']} == {KV}
                assert {r['rate'] for r in data['rows']} == {rate}
                assert {r['budget'] for r in data['rows']} == {1024}
                summary = archive.summary(data['rows'], rate)
                assert summary == data['summary'], str(path)
                halves[source] = summary
            if len(halves) != 2:
                continue
            feasible = all(halves[source]['passed'] and rate <= groups[source]['capacity_qps']
                           for source in SOURCES)
            if not feasible:
                continue
            assert all(halves[source]['mean_ttft_ms'] is not None for source in SOURCES)
            rows.append(dict(kv_blocks=KV, arm=arm, label=label, rate_qps=rate,
                             code_ttft_ms=halves['code']['mean_ttft_ms'],
                             conversation_ttft_ms=halves['conversation']['mean_ttft_ms'],
                             macro_ttft_ms=(halves['code']['mean_ttft_ms'] +
                                            halves['conversation']['mean_ttft_ms'])/2,
                             code_capacity_qps=groups['code']['capacity_qps'],
                             conversation_capacity_qps=groups['conversation']['capacity_qps'],
                             windows_per_workload=50))
    for arm in METHODS:
        selected = [r['rate_qps'] for r in rows if r['arm'] == arm]
        assert .5 in selected and selected == sorted(selected), arm
    return rows


def plot(rows):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'font.size': 9, 'axes.titlesize': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .7, 'legend.frameon': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    fig.subplots_adjust(left=.12, right=.98, top=.92, bottom=.27)
    for index in (*range(1, len(METHODS)), 0):
        arm = METHODS[index]
        chosen = [r for r in rows if r['arm'] == arm]
        ax.plot([r['rate_qps'] for r in chosen], [r['macro_ttft_ms']/1000 for r in chosen],
                color=COLORS[index], marker=MARKERS[index], markersize=4.4,
                linewidth=2.1 if index == 0 else 1.2,
                alpha=1 if index == 0 else .85, zorder=6 if index == 0 else 3)
    ax.set(xlabel='Offered request rate (QPS)',
           ylabel='Mean TTFT (s), Code / Conversation 50% each')
    ax.set_title(f'KV = {KV} blocks', loc='left')
    ax.set_xlim(0, max(r['rate_qps'] for r in rows) + .14)
    ax.set_ylim(0, max(r['macro_ttft_ms'] for r in rows)/1000 * 1.09)
    ax.grid(axis='y', color='#e5e7e9', linewidth=.5)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], color=c, marker=m, lw=1.4, markersize=4.4, label=l)
               for c, m, l in zip(COLORS, MARKERS, LABELS)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .035),
               ncol=4, columnspacing=1.0, fontsize=8)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'ttft_vs_qps_macro_4096.{suffix}', dpi=300,
                    facecolor='white')
    plt.close(fig)


def main():
    rows = build()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'ttft_vs_qps_macro_4096.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows)
    lookup = {(r['arm'], r['rate_qps']): r for r in rows}
    spare, s3 = lookup['arrival', 3.], lookup['s3_proxy', 3.]
    reduction = 100*(1-spare['macro_ttft_ms']/s3['macro_ttft_ms'])
    lines = [
        '# KV=4096, Code/Conversation 等权 mean TTFT—输入 QPS', '',
        '每个点是同一绝对输入QPS下，两类负载各50个窗口的mean TTFT先分别求均值，'
        '再以0.5/0.5加权。它代表两个工作负载的等权宏平均，不是混合请求流的实测延迟。',
        '只保留该方法在Code和Conversation上都通过既有SLO及积压判据的QPS点。'
        '未插值或外推，新增模拟次数为0；T=1024、K=256。', '',
        f'3 QPS时SPARE的等权mean TTFT为{spare["macro_ttft_ms"]:.2f} ms，'
        f'S3 proxy为{s3["macro_ttft_ms"]:.2f} ms；SPARE低{reduction:.2f}%。',
        f'SPARE在Code的capacity为{spare["code_capacity_qps"]:g} QPS，'
        f'在Conversation为{spare["conversation_capacity_qps"]:g} QPS。'
        '因此等权capacity为4 QPS，并不意味着两类负载都能稳定承载4 QPS；'
        '这条TTFT曲线在两类均通过的3 QPS处结束。', '',
        'CSV逐点保留Code、Conversation和等权TTFT以及两类capacity。'
        '纵轴是均值，capacity通过条件仍由尾延迟和积压同时决定。', '',
        '复现：`simulator/bin/python -B -m research.www_ttft_load_existing_20261016.plot_macro_4096`。'
    ]
    (OUT / 'MACRO_4096_CN.md').write_text('\n'.join(lines)+'\n')
    print(f'Created {len(rows)} common-feasible macro points; '
          f'SPARE/S3 at 3 QPS: {spare["macro_ttft_ms"]:.2f}/{s3["macro_ttft_ms"]:.2f} ms')


if __name__ == '__main__':
    main()
