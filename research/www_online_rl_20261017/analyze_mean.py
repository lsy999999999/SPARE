"""Compare frozen RL with SPARE using the current mean-TTFT capacity rule."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/www_online_rl_mean_mpl')
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/www_mean_ttft_capacity_20261017_r2'
RL = ROOT / 'artifacts/www_online_rl_20261017_r1'
EXTRA = ROOT / 'artifacts/www_online_rl_mean_capacity_20261017_r1'
FIG = EXTRA / 'analysis'
TRAIN_KV = {1024, 2048, 4096, 8192}
LIMIT_MS = 2000
SPARE_COLOR = '#b64342'
RL_COLOR = '#0f4d92'


def read(path):
    with path.open() as stream:
        return json.load(stream)


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def records():
    paths = defaultdict(dict)
    with (BASE / 'rate_audit.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['arm'] == 'arrival':
                paths['spare', row['source'], int(row['kv_blocks'])][
                    float(row['rate_qps'])] = ROOT / row['rate_record_path']
    for folder in (RL / 'rates', EXTRA / 'rates'):
        for path in folder.glob('*_online_rl_r*.json'):
            stem, rate_text = path.stem.rsplit('_r', 1)
            source, kv_text, arm = stem.split('_', 2)
            assert arm == 'online_rl'
            key = ('rl', source, int(kv_text[1:]))
            rate = float(rate_text)
            assert rate not in paths[key]
            paths[key][rate] = path
    for path in (EXTRA / 'rates').glob('*_arrival_r*.json'):
        stem, rate_text = path.stem.rsplit('_r', 1)
        source, kv_text, arm = stem.split('_', 2)
        assert arm == 'arrival'
        key = ('spare', source, int(kv_text[1:]))
        rate = float(rate_text)
        assert rate not in paths[key]
        paths[key][rate] = path
    return paths


def passed(summary, rate, limit):
    return (summary['failed_samples'] == 0
            and summary['mean_ttft_ms'] <= limit
            and summary['p99_tbt_ms'] <= 200
            and summary['backlog_drift'] <= max(2, rate))


def capacity(paths, limit):
    last = 0.
    for rate, path in sorted(paths.items()):
        if not passed(read(path)['summary'], rate, limit):
            return last
        last = rate
    raise ValueError(f'No measured failure at mean TTFT limit {limit} ms')


def comparison(paths):
    kvs = sorted({kv for arm, source, kv in paths if arm == 'rl' and source == 'code'}
                 & {kv for arm, source, kv in paths if arm == 'spare' and source == 'code'})
    rows = []
    for kv in kvs:
        points = {}
        for arm in ('spare', 'rl'):
            values = []
            for source in ('code', 'conversation'):
                samples = paths[arm, source, kv]
                assert len(read(samples[.5])['rows']) == 50
                values.append((capacity(samples, LIMIT_MS),
                               read(samples[.5])['summary']['mean_ttft_ms']))
            points[arm] = values
        spare_capacity = sum(value[0] for value in points['spare']) / 2
        rl_capacity = sum(value[0] for value in points['rl']) / 2
        spare_ttft = sum(value[1] for value in points['spare']) / 2
        rl_ttft = sum(value[1] for value in points['rl']) / 2
        rows.append(dict(kv_blocks=kv, training_kv=kv in TRAIN_KV,
                         spare_code_capacity_qps=points['spare'][0][0],
                         rl_code_capacity_qps=points['rl'][0][0],
                         spare_conversation_capacity_qps=points['spare'][1][0],
                         rl_conversation_capacity_qps=points['rl'][1][0],
                         spare_macro_capacity_qps=spare_capacity,
                         rl_macro_capacity_qps=rl_capacity,
                         capacity_gain_pct=100 * (rl_capacity / spare_capacity - 1),
                         spare_reference_ttft_ms=spare_ttft,
                         rl_reference_ttft_ms=rl_ttft,
                         rl_dominates_spare_point=(rl_capacity >= spare_capacity
                             and rl_ttft <= spare_ttft
                             and (rl_capacity > spare_capacity or rl_ttft < spare_ttft))))
    return rows


def paired_latency(paths, kv, rate, *, draws=10000):
    source = 'conversation'
    left = read(paths['spare', source, kv][rate])
    right = read(paths['rl', source, kv][rate])
    lrows = {row['name']: row for row in left['rows']}
    rrows = {row['name']: row for row in right['rows']}
    assert len(lrows) == len(rrows) == 50 and set(lrows) == set(rrows)
    assert all(lrows[name]['valid'] and rrows[name]['valid'] for name in lrows)
    names = sorted(lrows)
    spare = np.array([lrows[name]['mean_ttft_ms'] for name in names])
    rl = np.array([rrows[name]['mean_ttft_ms'] for name in names])
    rng = np.random.default_rng(10172)
    index = rng.integers(0, 50, (draws, 50))
    gains = 100 * (1 - rl[index].mean(axis=1) / spare[index].mean(axis=1))
    return dict(kv_blocks=kv, source=source, offered_qps=rate,
                spare_mean_ttft_ms=float(spare.mean()), rl_mean_ttft_ms=float(rl.mean()),
                rl_gain_pct=float(100 * (1 - rl.mean() / spare.mean())),
                paired_bootstrap_pct_low=float(np.quantile(gains, .025)),
                paired_bootstrap_pct_high=float(np.quantile(gains, .975)),
                rl_faster_windows=int(np.sum(rl < spare)),
                rl_slower_windows=int(np.sum(rl > spare)),
                spare_p99_tbt_ms=left['summary']['p99_tbt_ms'],
                rl_p99_tbt_ms=right['summary']['p99_tbt_ms'],
                spare_backlog_drift=left['summary']['backlog_drift'],
                rl_backlog_drift=right['summary']['backlog_drift'],
                spare_preemptions=left['summary']['preemptions'],
                rl_preemptions=right['summary']['preemptions'])


def plot_operating_points(paths, shift):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    fig.subplots_adjust(left=.085, right=.985, top=.76, bottom=.25, wspace=.27)
    for ax, row in zip(axes, shift):
        kv = row['kv_blocks']
        common = sorted(set(paths['spare', 'conversation', kv])
                        & set(paths['rl', 'conversation', kv]))
        rates = []
        for rate in common:
            if not 2.5 <= rate <= 4.0:
                continue
            spare = read(paths['spare', 'conversation', kv][rate])['summary']
            rl = read(paths['rl', 'conversation', kv][rate])['summary']
            if (spare['failed_samples'] == rl['failed_samples'] == 0
                    and spare['p99_tbt_ms'] <= 200 and rl['p99_tbt_ms'] <= 200
                    and spare['backlog_drift'] <= max(2, rate)
                    and rl['backlog_drift'] <= max(2, rate)):
                rates.append((rate, spare['mean_ttft_ms'], rl['mean_ttft_ms']))
        assert any(abs(rate - 3.5) < 1e-9 for rate, _, _ in rates)
        for arm, column, color, marker in (('SPARE', 1, SPARE_COLOR, 'o'),
                                           ('Online RL', 2, RL_COLOR, 'D')):
            ordinary = [item for item in rates if item[0] != 3.5]
            if ordinary:
                ax.scatter([item[column] for item in ordinary],
                           [item[0] for item in ordinary], marker=marker,
                           facecolors='white' if arm == 'SPARE' else color,
                           edgecolors=color, s=25, linewidths=1, alpha=.7,
                           label=arm, zorder=2)
            target = next(item for item in rates if item[0] == 3.5)
            ax.scatter(target[column], 3.5, marker=marker,
                       facecolors='white' if arm == 'SPARE' else color,
                       edgecolors=color, s=85, linewidths=1.7, zorder=4)
        ax.set_title(f'KV {kv}  ' + ('trained size' if kv in TRAIN_KV else 'unseen size')
                     + f'\nAt 3.5 QPS, RL {abs(row["rl_gain_pct"]):.1f}% '
                     + ('faster' if row['rl_gain_pct'] > 0 else 'slower'), loc='left')
        ax.set_xlim((300, 2350) if kv == 2048 else (300, 1150))
        ax.set_ylim(2.4, 4.1)
        ax.set_xlabel('Mean TTFT (ms)')
        ax.grid(axis='both', color='#dce2e5', linewidth=.45)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Offered load (QPS), resource-feasible')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(.5, .015))
    for suffix in ('pdf', 'png', 'svg'):
        fig.savefig(FIG / f'matched_load_latency.{suffix}', dpi=300)
    plt.close(fig)


def plot_scatter(rows):
    import matplotlib.pyplot as plt

    chosen = {row['kv_blocks']: row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    fig.subplots_adjust(left=.085, right=.985, top=.77, bottom=.26, wspace=.26)
    for ax, kv in zip(axes, (2048, 12032)):
        row = chosen[kv]
        ax.scatter(row['spare_reference_ttft_ms'], row['spare_macro_capacity_qps'],
                   facecolors='white', edgecolors=SPARE_COLOR, linewidths=1.7,
                   s=90, label='SPARE', zorder=3)
        ax.scatter(row['rl_reference_ttft_ms'], row['rl_macro_capacity_qps'],
                   color=RL_COLOR, marker='D', s=60, label='Online RL', zorder=4)
        ax.set_xlim(304.8, 306.25)
        center = (row['spare_macro_capacity_qps'] + row['rl_macro_capacity_qps']) / 2
        ax.set_ylim(center - .21, center + .21)
        ax.set_title(f'KV {kv}  ' + ('trained size' if kv in TRAIN_KV else 'unseen size')
                     + f'\nRL capacity {row["capacity_gain_pct"]:+.1f}%', loc='left')
        ax.set_xlabel('Mean TTFT at 0.5 QPS (ms)')
        ax.grid(axis='both', color='#dce2e5', linewidth=.45)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Capacity at 2 s mean TTFT (QPS)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(.5, .025))
    for suffix in ('pdf', 'png', 'svg'):
        fig.savefig(FIG / f'fixed_kv_pareto_mean2s.{suffix}', dpi=300)
    plt.close(fig)


def plot_shift(shift):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.9, 3.0))
    fig.subplots_adjust(left=.18, right=.96, top=.88, bottom=.20)
    for x, row in enumerate(shift):
        center = row['rl_gain_pct']
        low = center - row['paired_bootstrap_pct_low']
        high = row['paired_bootstrap_pct_high'] - center
        ax.errorbar(x, center, yerr=[[low], [high]], fmt='D', markersize=6,
                    capsize=4, linewidth=1.3,
                    color=RL_COLOR if center > 0 else SPARE_COLOR, zorder=3)
        ax.annotate(f'{center:+.1f}%', (x, center),
                    xytext=(10, -12 if center < 0 else 0),
                    textcoords='offset points', va='center', fontsize=8)
    ax.axhline(0, color='#606b73', linewidth=.8)
    ax.set_xlim(-.45, 1.45)
    ax.set_ylim(-29, 19)
    ax.set_xticks((0, 1), ('KV 2048\ntrained size', 'KV 12032\nunseen size'))
    ax.set_ylabel('RL mean TTFT improvement (%)')
    ax.set_title('Conversation at 3.5 QPS', loc='left')
    ax.grid(axis='y', color='#dce2e5', linewidth=.45)
    ax.set_axisbelow(True)
    for suffix in ('pdf', 'png', 'svg'):
        fig.savefig(FIG / f'matched_qps_kv_shift.{suffix}', dpi=300)
    plt.close(fig)


def report(rows, shift):
    apparent_winners = (1792, 2048, 2816, 3072, 3840, 4096)
    chosen = {r['kv_blocks']: r for r in rows}
    assert all(chosen[kv]['capacity_gain_pct'] < 0 for kv in apparent_winners)
    lines = [
        '# Frozen online RL under the mean-TTFT capacity definition', '',
        'The capacity rule uses a variable mean-TTFT ceiling, P99 TBT <= 200 ms,',
        'backlog drift <= max(2, offered QPS), all 50 windows valid, and first',
        'failure in the measured ascending QPS grid. P95 TTFT is not a constraint.',
        'Code and Conversation were both present in RL training; KV 12032 was not.', '',
        'At a 2 s mean-TTFT ceiling, six KV sizes appeared to favor RL on the',
        'archived coarse rate grid (1792, 2048, 2816, 3072, 3840, 4096).',
        'After refining both methods and both sources to 0.01-0.02 QPS at these',
        'sizes, SPARE has higher equal-mean capacity at all six. KV 12032 was',
        'also refined and favors SPARE. The other 20 original sizes retain the',
        'archived mean-ceiling grid; they were not fully refined for this comparison.',
        f'In the assembled 26-size table RL is higher at '
        f'{sum(r["capacity_gain_pct"] > 1e-9 for r in rows if r["kv_blocks"] <= 8192)}, '
        f'lower at {sum(r["capacity_gain_pct"] < -1e-9 for r in rows if r["kv_blocks"] <= 8192)}, '
        f'and tied at {sum(abs(r["capacity_gain_pct"]) <= 1e-9 for r in rows if r["kv_blocks"] <= 8192)}.', '',
        '| KV | SPARE capacity | RL capacity | RL relative gain | SPARE 0.5-QPS TTFT | RL 0.5-QPS TTFT |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for kv in (2048, 4096, 12032):
        row = chosen[kv]
        lines.append(f'| {kv} | {row["spare_macro_capacity_qps"]:.3f} | '
                     f'{row["rl_macro_capacity_qps"]:.3f} | '
                     f'{row["capacity_gain_pct"]:+.2f}% | '
                     f'{row["spare_reference_ttft_ms"]:.2f} ms | '
                     f'{row["rl_reference_ttft_ms"]:.2f} ms |')
    lines.extend(['', 'Source-specific 2 s capacities at the same KV:',
                  '| KV | Code SPARE / RL | Conversation SPARE / RL |',
                  '|---:|---:|---:|'])
    for kv in (2048, 4096, 12032):
        row = chosen[kv]
        lines.append(f'| {kv} | {row["spare_code_capacity_qps"]:.2f} / '
                     f'{row["rl_code_capacity_qps"]:.2f} | '
                     f'{row["spare_conversation_capacity_qps"]:.2f} / '
                     f'{row["rl_conversation_capacity_qps"]:.2f} |')
    lines.extend(['', 'The matched-load plot uses measured offered QPS on y, not maximum '
                  'serving capacity. Both methods satisfy P99 TBT and backlog limits '
                  'at every plotted point. The highlighted 3.5-QPS RL point has '
                  'lower mean TTFT at KV 2048 and higher mean TTFT at KV 12032.',
                  'The fixed-KV scatter uses reference TTFT at 0.5 QPS on x '
                  'and 2 s mean-TTFT-limited capacity on y. These are separate runs.',
                  'The capacity differences are discrete point estimates, not confidence '
                  'intervals. In a paired-window bootstrap, some resamples did not fail '
                  'at the highest measured QPS, so a two-sided capacity interval is '
                  'not identified by the existing rate scans.', '',
                  'Matched Conversation, 3.5 QPS, 50 paired test windows:',
                  '| KV | SPARE mean TTFT | RL mean TTFT | RL improvement | '
                  '95% paired bootstrap interval | RL faster windows |',
                  '|---:|---:|---:|---:|---:|---:|'])
    for row in shift:
        lines.append(f'| {row["kv_blocks"]} | {row["spare_mean_ttft_ms"]:.1f} ms | '
                     f'{row["rl_mean_ttft_ms"]:.1f} ms | {row["rl_gain_pct"]:+.2f}% | '
                     f'[{row["paired_bootstrap_pct_low"]:+.2f}%, '
                     f'{row["paired_bootstrap_pct_high"]:+.2f}%] | '
                     f'{row["rl_faster_windows"]}/50 |')
    lines.extend(['', 'Both policies satisfy the current P99 TBT and backlog limits '
                  'at the two matched 3.5-QPS points. At KV 12032 neither policy '
                  'preempts; RL nevertheless raises mean TTFT. This establishes '
                  'a workload/KV-specific reversal for this frozen controller, '
                  'not a general theorem about RL or a causal attribution.',
                  'Code and Conversation are different workload types, but both '
                  'were included in training. KV 12032 is a genuine KV extrapolation; '
                  'the two selected KV points should not be read as a generalization '
                  'average over all possible KV sizes.', '',
                  '[Matched load-latency plot](matched_load_latency.pdf) · '
                  '[Fixed-KV capacity scatter](fixed_kv_pareto_mean2s.pdf) · '
                  '[matched-QPS shift](matched_qps_kv_shift.pdf) · '
                  '[all 2-s KV results](capacity_2s.csv) · '
                  '[paired shift numbers](paired_shift.json)'])
    (FIG / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    paths = records()
    assert read(EXTRA / 'extreme_status.json')['complete']
    rows = comparison(paths)
    assert len(rows) == 27 and rows[-1]['kv_blocks'] == 12032
    shift = [paired_latency(paths, kv, 3.5) for kv in (2048, 12032)]
    write_csv(FIG / 'capacity_2s.csv', rows)
    (FIG / 'paired_shift.json').write_text(json.dumps(shift, indent=2) + '\n')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8,
                         'axes.titlesize': 9, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    plot_operating_points(paths, shift)
    plot_scatter(rows)
    plot_shift(shift)
    report(rows, shift)
    print((FIG / 'REPORT.md').read_text())


if __name__ == '__main__':
    main()
