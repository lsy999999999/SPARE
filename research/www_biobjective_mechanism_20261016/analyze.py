"""Show component crossovers and the SLO mechanism behind capacity."""
from __future__ import annotations

import csv
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/www_biobjective_mechanism_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import run

OLD = run.prior.OUT
ARMS = ('spare', 'srp', 'constant16', 'constant32',
        'budget_only', 'chunk_only', 'token_proxy', 'no_decode_guard')
PLOT_ARMS = ('spare', 'srp', 'constant16', 'constant32', 'chunk_only')
LABELS = {'spare': 'SPARE', 'srp': 'No selective deferral',
          'constant16': 'Fixed threshold 16', 'constant32': 'Fixed threshold 32',
          'budget_only': 'Budget candidates only',
          'chunk_only': 'Chunk candidate only',
          'token_proxy': 'Token proxy', 'no_decode_guard': 'No decode guard'}
COLORS = {'spare': '#0F4D92', 'srp': '#767676',
          'constant16': '#B64342', 'constant32': '#9A4D8E',
          'budget_only': '#2B8B69', 'chunk_only': '#D08028',
          'token_proxy': '#42949E', 'no_decode_guard': '#7B658A'}
MARKERS = {'spare': 'o', 'srp': 's', 'constant16': '^', 'constant32': 'v',
           'budget_only': 'D', 'chunk_only': 'P',
           'token_proxy': 'X', 'no_decode_guard': 'h'}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect():
    groups, pressure = [], []
    for source in ('code', 'conversation'):
        for kv in (768, *run.CAPACITIES):
            for arm in ARMS:
                if arm not in run.prior.ARMS and kv == 768:
                    continue
                root = OLD if arm in run.prior.ARMS else run.OUT
                path = root / 'groups' / f'{source}_c{kv}_{arm}.json'
                if not path.exists():
                    continue
                group = run.prior.base.read(path)
                groups.append(dict(source=source, kv_blocks=kv, arm=arm,
                                   capacity_qps=group['capacity_qps'],
                                   next_failed_qps=group['next_failed_qps']))
                rate_path = root / 'rates' / f'{source}_c{kv}_{arm}_r3.json'
                if rate_path.exists():
                    rate = run.prior.base.read(rate_path)
                    summary = rate['summary']
                    assert summary == run.prior.archive.summary(rate['rows'], 3.)
                    failures = []
                    if summary['p95_ttft_ms'] > 12000:
                        failures.append('p95_ttft')
                    if summary['p99_tbt_ms'] > 200:
                        failures.append('p99_tbt')
                    if summary['backlog_drift'] > summary['drift_limit']:
                        failures.append('backlog_drift')
                    if summary['failed_samples']:
                        failures.append('invalid_replay')
                    valid = [r for r in rate['rows'] if r['valid']]
                    pressure.append(dict(source=source, kv_blocks=kv, arm=arm,
                                         rate_qps=3., passed=summary['passed'],
                                         failed_criteria=';'.join(failures),
                                         mean_ttft_ms=summary['mean_ttft_ms'],
                                         p95_ttft_ms=summary['p95_ttft_ms'],
                                         p99_tbt_ms=summary['p99_tbt_ms'],
                                         backlog_drift=summary['backlog_drift'],
                                         drift_limit=summary['drift_limit'],
                                         preemptions=summary['preemptions'],
                                         recomputed_tokens=summary['recomputed_tokens'],
                                         mean_interventions=sum(r['interventions'] for r in valid)/len(valid)
                                         if valid and all('interventions' in r for r in valid) else None,
                                         mean_rounds=sum(r['rounds'] for r in valid)/len(valid)
                                         if valid and all('rounds' in r for r in valid) else None))
    return groups, pressure


def paired_changes():
    rows = []
    for source in ('code', 'conversation'):
        for kv in run.CAPACITIES:
            for arm in run.OPTIONS:
                group_path = run.OUT / 'groups' / f'{source}_c{kv}_{arm}.json'
                if not group_path.exists():
                    continue
                group = run.prior.base.read(group_path)
                for rate in group['tested_rates']:
                    old_path = OLD / 'rates' / f'{source}_c{kv}_spare_r{rate:g}.json'
                    new_path = run.OUT / 'rates' / f'{source}_c{kv}_{arm}_r{rate:g}.json'
                    if not old_path.exists() or not new_path.exists():
                        continue
                    old = run.prior.base.read(old_path)
                    new = run.prior.base.read(new_path)
                    left = {r['name']: r for r in old['rows']}
                    right = {r['name']: r for r in new['rows']}
                    assert left.keys() == right.keys() and len(left) == 50
                    pairs = [(left[name], right[name]) for name in left]
                    valid = [(a, b) for a, b in pairs if a['valid'] and b['valid']]
                    rows.append(dict(source=source, kv_blocks=kv, arm=arm,
                                     rate_qps=rate,
                                     spare_passed=old['summary']['passed'],
                                     variant_passed=new['summary']['passed'],
                                     valid_pairs=len(valid),
                                     different_trajectories=sum(
                                         a['trajectory_sha256'] != b['trajectory_sha256']
                                         for a, b in valid),
                                     mean_delta_ttft_ms=sum(
                                         b['mean_ttft_ms']-a['mean_ttft_ms']
                                         for a, b in valid)/len(valid) if valid else None))
    return rows


def plot(groups, pressure, prefix):
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'font.size': 8, 'axes.titlesize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .75, 'legend.frameon': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.0), sharex='col')
    fig.subplots_adjust(left=.085, right=.985, top=.93, bottom=.19,
                        hspace=.29, wspace=.2)
    for col, source in enumerate(('code', 'conversation')):
        for arm in PLOT_ARMS:
            points = sorted((r for r in groups if r['source'] == source and
                             r['arm'] == arm), key=lambda r: r['kv_blocks'])
            if points:
                axes[0, col].plot([r['kv_blocks'] for r in points],
                                  [r['capacity_qps'] for r in points],
                                  marker=MARKERS[arm], markersize=4,
                                  lw=2.1 if arm == 'spare' else 1.1,
                                  color=COLORS[arm], label=LABELS[arm],
                                  zorder=5 if arm == 'spare' else 2)
            p = sorted((r for r in pressure if r['source'] == source and
                        r['arm'] == arm), key=lambda r: r['kv_blocks'])
            if p:
                axes[1, col].plot([r['kv_blocks'] for r in p],
                                  [r['p95_ttft_ms']/1000 for r in p],
                                  marker=MARKERS[arm], markersize=3.7,
                                  lw=1.8 if arm == 'spare' else 1.,
                                  color=COLORS[arm], zorder=5 if arm == 'spare' else 2)
                for r in p:
                    if not r['passed']:
                        axes[1, col].scatter(r['kv_blocks'], r['p95_ttft_ms']/1000,
                                             marker=MARKERS[arm], s=27,
                                             facecolors='white', edgecolors=COLORS[arm],
                                             linewidths=1.2, zorder=6)
        axes[0, col].set_title(source.title(), loc='left')
        axes[0, col].set_ylim(bottom=0)
        axes[1, col].axhline(12, color='#333333', lw=.7, ls='--')
        axes[1, col].set_xlabel('Configured KV blocks')
        axes[1, col].set_xticks((768, 1024, 1536, 2048, 3072, 4096))
        axes[1, col].tick_params(axis='x', labelrotation=60)
        for ax in axes[:, col]:
            ax.grid(axis='y', color='#e1e4e6', lw=.5)
            ax.set_axisbelow(True)
    axes[0, 0].set_ylabel('Serving capacity (QPS)')
    axes[1, 0].set_ylabel('P95 TTFT at 3 QPS (s)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', ncol=5,
               bbox_to_anchor=(.52, .54), fontsize=7, columnspacing=.75)
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(prefix.with_suffix('.' + suffix), dpi=300,
                    facecolor='white')
    plt.close(fig)


def plot_boundaries(folder):
    cases = (('conversation', 2048, 4., ('spare', 'constant16')),
             ('conversation', 4096, 5., ('spare', 'constant16', 'srp')),
             ('code', 4096, 4., ('spare', 'srp')))
    records = []
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.95))
    fig.subplots_adjust(left=.075, right=.99, top=.81, bottom=.27, wspace=.38)
    for index, (source, kv, rate, arms) in enumerate(cases):
        ax = axes[index]
        values = []
        for arm in arms:
            path = OLD / 'rates' / f'{source}_c{kv}_{arm}_r{rate:g}.json'
            datum = run.prior.base.read(path)
            summary = datum['summary']
            assert summary == run.prior.archive.summary(datum['rows'], rate)
            if arm == 'spare':
                proxy = run.prior.base.read(
                    run.OUT / 'rates' /
                    f'{source}_c{kv}_no_decode_guard_r{rate:g}.json')
                original = {r['name']: r for r in datum['rows']}
                assert all(r['trajectory_sha256'] == original[r['name']]['trajectory_sha256']
                           for r in proxy['rows'])
                interventions = sum(r['interventions'] for r in proxy['rows'])/50
            else:
                interventions = sum(r.get('interventions', 0) for r in datum['rows'])/50
            group = run.prior.base.read(OLD / 'groups' /
                                        f'{source}_c{kv}_{arm}.json')
            records.append(dict(source=source, kv_blocks=kv, offered_qps=rate,
                                arm=arm, capacity_qps=group['capacity_qps'],
                                passed=summary['passed'],
                                mean_interventions_per_run=interventions,
                                p95_ttft_s=summary['p95_ttft_ms']/1000,
                                p99_tbt_ms=summary['p99_tbt_ms'],
                                backlog_drift=summary['backlog_drift'],
                                backlog_limit=summary['drift_limit']))
            values.append(summary['p95_ttft_ms']/1000)
        x = list(range(len(arms)))
        ax.bar(x, values, width=.58, color=[COLORS[a] for a in arms],
               edgecolor='#333333', linewidth=.6)
        ax.axhline(12, color='#333333', ls='--', lw=.9)
        ax.set_xticks(x, [LABELS[a].replace('No selective deferral', 'SRP')
                          .replace('Fixed threshold 16', 'Fixed 16') for a in arms])
        ax.set_ylim(0, max(15, max(values)*1.15))
        ax.set_title(f'{source.title()}  /  {kv} KV\n{rate:g} offered QPS', loc='left')
        for j, value in enumerate(values):
            row = records[-len(arms)+j]
            printed = f'{value:.3f}' if abs(value-12) < .05 else f'{value:.2f}'
            ax.text(j, max(value+.25, 12.75), f'{printed}s',
                    ha='center', va='bottom', fontsize=7)
            ax.text(j, 3.75, f'C={row["capacity_qps"]:g}',
                    ha='center', va='center', fontsize=7, color='white')
        ax.grid(axis='y', color='#e1e4e6', lw=.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('P95 TTFT at offered load (s)')
    fig.text(.53, .08, 'Dashed line: 12 s TTFT SLO; C: measured serving capacity (QPS)',
             ha='center', fontsize=7)
    write_csv(folder / 'capacity_boundary_witnesses.csv', records)
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(folder / f'capacity_boundary_crossovers.{suffix}', dpi=300,
                    facecolor='white')
    plt.close(fig)


def main():
    groups, pressure = collect()
    assert groups and pressure
    folder = run.OUT / 'analysis'
    write_csv(folder / 'capacity_components.csv', groups)
    write_csv(folder / 'p95_at_3qps.csv', pressure)
    changes = paired_changes()
    if changes:
        write_csv(folder / 'paired_trajectory_changes.csv', changes)
    plot(groups, pressure, folder / 'capacity_and_slo_by_kv')
    plot_boundaries(folder)
    print(f'{len(groups)} capacity points, {len(pressure)} 3-QPS diagnostics, '
          f'{len(changes)} paired rates')


if __name__ == '__main__':
    main()
