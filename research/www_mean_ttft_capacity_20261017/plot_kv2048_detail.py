"""Plot the measured KV=2048 capacity frontier and the refined 2 s RL comparison."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

os.environ.setdefault('MPLCONFIGDIR', '/tmp/www_kv2048_detail_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from research.www_mean_ttft_capacity_20261017 import analyze as archive
from research.www_online_rl_20261017 import analyze_mean


KV = 2048
LIMIT_MS = 2000
OUT = archive.OUT / 'analysis'
RL_RESULTS = (archive.ROOT / 'artifacts/www_rl_mixed_generalization_20261017_r1'
              / 'analysis/capacity_three_seed.csv')
RL_COLOR = '#0f4d92'


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def baseline_curves():
    curves = defaultdict(list)
    for row in rows(archive.OUT / 'capacity_curves.csv'):
        if int(row['kv_blocks']) != KV or row['arm'] == 'arrival':
            continue
        curves[row['arm']].append(dict(
            mean_ttft_limit_ms=float(row['mean_ttft_limit_ms']),
            mean_capacity_qps=float(row['mean_capacity_qps']),
            code_capacity_qps=float(row['code_capacity_qps']),
            conversation_capacity_qps=float(row['conversation_capacity_qps']),
        ))
    assert set(curves) == set(archive.METHODS) - {'arrival'}
    return curves


def refined_spare_curve():
    paths = analyze_mean.records()
    eligible = {}
    coverage = {}
    for source in archive.SOURCES:
        points = []
        first_failure = None
        for rate, path in sorted(paths['spare', source, KV].items()):
            record = archive.read(path)
            assert len(record['rows']) == 50
            summary = record['summary']
            passed = (summary['failed_samples'] == 0
                      and summary['p99_tbt_ms'] <= 200
                      and summary['backlog_drift'] <= max(2, rate))
            if not passed:
                first_failure = rate
                break
            points.append((rate, summary['mean_ttft_ms']))
        assert first_failure is not None and points
        eligible[source] = points
        coverage[source] = dict(measured_rates=len(paths['spare', source, KV]),
                                first_resource_failure_qps=first_failure)
    curve = []
    last = 0.
    for limit in sorted({ttft for points in eligible.values() for _, ttft in points}):
        code = archive.capacity(eligible['code'], limit)
        conversation = archive.capacity(eligible['conversation'], limit)
        mean = (code + conversation) / 2
        if mean > last:
            curve.append(dict(mean_ttft_limit_ms=limit, mean_capacity_qps=mean,
                              code_capacity_qps=code,
                              conversation_capacity_qps=conversation))
            last = mean
    capacity_2s = {source: archive.capacity(points, LIMIT_MS)
                   for source, points in eligible.items()}
    assert capacity_2s == {'code': 3.32, 'conversation': 3.84}
    return curve, capacity_2s, coverage


def rl_at_2s():
    chosen = [row for row in rows(RL_RESULTS)
              if row['workload'] == 'original' and int(row['kv_blocks']) == KV]
    assert {row['source'] for row in chosen} == set(archive.SOURCES)
    return {row['source']: dict(mean=float(row['rl_mean_capacity_qps']),
                                seeds=[float(row[f'seed{seed}_qps'])
                                       for seed in (1017, 2027, 3037)])
            for row in chosen}


def write_sources(curves, spare, nondominated):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'kv2048_frontier_source.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            'arm', 'method', 'mean_ttft_limit_ms', 'mean_capacity_qps',
            'code_capacity_qps', 'conversation_capacity_qps', 'nondominated',
            'source'))
        writer.writeheader()
        for arm in archive.METHODS:
            points = spare if arm == 'arrival' else curves[arm]
            for index, point in enumerate(points):
                writer.writerow(dict(arm=arm, method=archive.LABELS[archive.METHODS.index(arm)],
                                     **point, nondominated=(nondominated[index]
                                                             if arm == 'arrival' else ''),
                                     source=('refined SPARE rate union' if arm == 'arrival'
                                             else 'archived 50-window curve')))


def plot(curves, spare, nondominated, spare_2s, rl):
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'axes.titlesize': 10,
                         'axes.labelsize': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'legend.frameon': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, (ax, detail) = plt.subplots(1, 2, figsize=(12.2, 6.1),
                                    gridspec_kw={'width_ratios': [3.1, 1]})
    for arm in archive.METHODS[1:]:
        index = archive.METHODS.index(arm)
        points = curves[arm]
        ax.step([p['mean_ttft_limit_ms'] / 1000 for p in points],
                [p['mean_capacity_qps'] for p in points], where='post',
                color=archive.COLORS[index], linewidth=1.35,
                linestyle='--' if arm in archive.EXTRA else '-',
                alpha=.84, zorder=2)
    ax.step([p['mean_ttft_limit_ms'] / 1000 for p in spare],
            [p['mean_capacity_qps'] for p in spare], where='post',
            color=archive.COLORS[0], linewidth=2.5, zorder=4)
    for status, face in ((True, 'white'), (False, archive.COLORS[0])):
        selected = [p for p, keep in zip(spare, nondominated) if keep == status]
        ax.scatter([p['mean_ttft_limit_ms'] / 1000 for p in selected],
                   [p['mean_capacity_qps'] for p in selected], s=33,
                   facecolors=face, edgecolors=archive.COLORS[0],
                   linewidths=1, zorder=5)
    ax.axvline(2, color='#8a9298', linewidth=.9, linestyle=':', zorder=1)
    ax.set_xscale('log')
    ax.set_xlim(.2, 5)
    ax.set_xticks((.25, .5, 1, 2, 3, 5), labels=('0.25', '0.5', '1', '2', '3', '5'))
    ax.set_ylim(0, 4.5)
    ax.set_xlabel('Mean TTFT limit (s; log scale)')
    ax.set_ylabel('Mean serving capacity (QPS)')
    ax.set_title('a  SPARE and 11 baselines', loc='left', fontweight='bold')
    ax.grid(axis='y', color='#dce2e5', linewidth=.55)
    ax.set_axisbelow(True)

    for source, y in (('code', 1), ('conversation', 0)):
        entry = rl[source]
        low, high = min(entry['seeds']), max(entry['seeds'])
        detail.errorbar(entry['mean'], y - .12,
                        xerr=[[entry['mean'] - low], [high - entry['mean']]],
                        fmt='D', markersize=6, color=RL_COLOR, capsize=3,
                        linewidth=1.5, zorder=5)
        detail.scatter(spare_2s[source], y + .12, s=62, facecolors='white',
                       edgecolors=archive.COLORS[0], linewidths=1.7, zorder=5)
        detail.text(spare_2s[source] + .025, y + .12,
                    f'{spare_2s[source]:.2f}', va='center', fontsize=8,
                    color=archive.COLORS[0])
        detail.text(entry['mean'] + .025, y - .12,
                    f'{entry["mean"]:.3f}', va='center', fontsize=8,
                    color=RL_COLOR)
    detail.set_xlim(3, 4.28)
    detail.set_ylim(-.5, 1.5)
    detail.set_yticks((0, 1), labels=('Conversation', 'Code'))
    detail.set_xticks((3, 3.4, 3.8, 4.2))
    detail.set_xlabel('Capacity at 2 s (QPS)')
    detail.set_title('b  Refined 2 s comparison', loc='left', fontweight='bold')
    detail.grid(axis='x', color='#dce2e5', linewidth=.55)
    detail.set_axisbelow(True)
    detail.text(.02, .03, 'RL whiskers: 3 training seeds', transform=detail.transAxes,
                fontsize=8, color='#4d5961')

    handles = [Line2D([], [], color=archive.COLORS[i],
                      linewidth=2.5 if i == 0 else 1.35,
                      linestyle='--' if arm in archive.EXTRA else '-',
                      label=archive.LABELS[i])
               for i, arm in enumerate(archive.METHODS)]
    handles.extend((Line2D([], [], marker='o', linestyle='None',
                           markerfacecolor='white', markeredgecolor=archive.COLORS[0],
                           label='SPARE nondominated'),
                    Line2D([], [], marker='o', linestyle='None',
                           markerfacecolor=archive.COLORS[0],
                           markeredgecolor=archive.COLORS[0],
                           label='SPARE dominated'),
                    Line2D([], [], marker='D', linestyle='None', color=RL_COLOR,
                           label='RL-Mixed at 2 s only')))
    fig.legend(handles=handles, loc='lower center', ncol=5,
               bbox_to_anchor=(.5, .015), fontsize=8.3, columnspacing=1.3,
               handlelength=2.1)
    fig.suptitle('KV = 2048 blocks | serving capacity under mean TTFT, P99 TBT and backlog limits',
                 y=.985, fontsize=12)
    fig.tight_layout(rect=(0, .22, 1, .935), w_pad=2.1)
    for suffix in ('png', 'pdf'):
        fig.savefig(OUT / f'kv2048_mean_ttft_capacity.{suffix}', dpi=300)
    plt.close(fig)


def main():
    design = archive.read(archive.OUT / 'design.json')
    from research.www_online_rl_20261017 import run as old
    old_design = old.base.read(old.base.OUT / 'design.json')
    assert {(s['name'], s['input']) for s in design['samples']} == {
        (s['name'], s['input']) for s in old_design['samples'] if s['split'] == 'test'}
    assert design['timing_sha256'] == old.base.digest(old.base.PREVIOUS / 'timing_profile.json')
    curves = baseline_curves()
    spare, spare_2s, coverage = refined_spare_curve()
    rl = rl_at_2s()
    nondominated = [not any(archive.dominates(other, point)
                            for baseline in curves.values() for other in baseline)
                    for point in spare]
    write_sources(curves, spare, nondominated)
    summary = dict(kv_blocks=KV, baseline_methods=len(curves),
                   archived_curve_points=sum(map(len, curves.values())),
                   refined_spare_curve_points=len(spare),
                   spare_dominated_points=nondominated.count(False),
                   spare_nondominated_points=nondominated.count(True),
                   spare_capacity_at_2s=spare_2s, rl_mixed_at_2s=rl,
                   spare_rate_coverage=coverage,
                   note='Baseline frontiers use archived measured rates; '
                        'SPARE adds dedicated 2 s rates. RL-Mixed has only '
                        'a 2 s operating point, not a full frontier.')
    (OUT / 'kv2048_frontier_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    plot(curves, spare, nondominated, spare_2s, rl)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
