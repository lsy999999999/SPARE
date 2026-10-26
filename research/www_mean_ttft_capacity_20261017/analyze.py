"""Re-evaluate archived KV sweeps under a mean-TTFT capacity constraint."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/www_mean_ttft_capacity_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'artifacts/www_kv_capacity_dense_20261016_r1'
FINE = ROOT / 'artifacts/www_kv_capacity_fine50_20261016_r1'
PAPER = ROOT / 'artifacts/www_kv_capacity_paper_20261015_r1'
OUT = ROOT / 'artifacts/www_mean_ttft_capacity_20261017_r2'
SOURCES = ('code', 'conversation')
CORE = ('arrival', 'native', 'sarathi', 'sola_rule', 'uniboost_static',
        'past_future', 's3_proxy', 'apt_serve_kv_rule')
EXTRA = ('past_future_multi', 'orca_max', 'ltr_proxy', 'sola_calibrated')
METHODS = CORE + EXTRA
LABELS = ('SPARE', 'vLLM V1', 'Sarathi rule', 'SOLA rule', 'UniBoost rule',
          'Past-Future rule', 'S3 proxy', 'Apt-Serve rule',
          'Past-Future multi', 'Orca reservation', 'LTR proxy', 'SOLA calibrated')
COLORS = ('#b83b3b', '#606b73', '#8da8b4', '#bc8651', '#8d78a0',
          '#218983', '#c37757', '#a19b5d',
          '#125f59', '#426d9e', '#565198', '#9a5b7d')
CHECK_LIMITS_MS = (250, 350, 500, 750, 1000, 1500, 2000, 3000,
                   4000, 5000, 6000, 8000, 10000, 12000, 16000)
PLOT_MAX_MS = 16000


def read(path):
    with path.open() as stream:
        return json.load(stream)


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def load_records():
    from research.www_mean_ttft_capacity_20261017 import run as scan

    design = read(ARCHIVE / 'design.json')
    kvs = scan.CAPACITIES
    assert tuple(design['methods']) == CORE
    assert read(FINE / 'design.json')['source_design_sha256']
    paper_design = read(PAPER / 'design.json')
    assert set(paper_design['main_arms']) == set(METHODS)
    dense_samples = {(s['name'], s['input']) for s in design['samples']}
    assert dense_samples == {(s['name'], s['input']) for s in paper_design['samples']}
    status = read(OUT / 'status.json')
    assert status['stage'] == 'complete' and not status['errors']
    assert status['groups_complete'] == status['groups_total'] == len(kvs) * len(METHODS) * 2
    records = defaultdict(dict)
    provenance = {}
    for root in (ARCHIVE, FINE, PAPER, OUT):
        for path in (root / 'rates').glob('*.json'):
            stem, rate_string = path.stem.rsplit('_r', 1)
            source, kv_string, arm = stem.split('_', 2)
            kv = int(kv_string[1:])
            if kv not in kvs:
                continue
            if root == PAPER and arm not in EXTRA:
                continue
            rate = float(rate_string)
            data = read(path)
            assert len(data['rows']) == 50
            assert len({row['name'] for row in data['rows']}) == 50
            key = (source, kv, arm)
            if rate in records[key]:
                assert records[key][rate] == data['summary'], path
            else:
                provenance[key, rate] = str(path.relative_to(ROOT))
                records[key][rate] = data['summary']
    expected = len(kvs) * len(METHODS) * len(SOURCES)
    assert len(records) == expected
    for kv in kvs:
        for arm in METHODS:
            for source in SOURCES:
                group = read(OUT / 'groups' / f'{source}_c{kv}_{arm}.json')
                assert group['stage'] == 'complete' and group['resolution_qps'] <= .020001
    return kvs, records, provenance


def source_points(records, provenance):
    rows = []
    by_source = {}
    for key, scans in sorted(records.items()):
        source, kv, arm = key
        ordered = sorted(scans.items())
        eligible = []
        first_failure = None
        for rate, s in ordered:
            passed = (s.get('failed_samples', 0) == 0 and s['p99_tbt_ms'] <= 200
                      and s['backlog_drift'] <= max(2, rate))
            rows.append(dict(source=source, kv_blocks=kv, arm=arm, rate_qps=rate,
                             rate_record_path=provenance[key, rate], windows=50,
                             mean_ttft_ms=s['mean_ttft_ms'], p99_tbt_ms=s['p99_tbt_ms'],
                             backlog_drift=s['backlog_drift'], drift_limit=max(2, rate),
                             failed_samples=s.get('failed_samples', 0), eligible=passed,
                             old_p95_ttft_ms=s['p95_ttft_ms']))
            if first_failure is None:
                if passed:
                    eligible.append((rate, s['mean_ttft_ms']))
                else:
                    first_failure = rate
        by_source[key] = (eligible, first_failure, ordered[-1][0])
    return rows, by_source


def capacity(points, limit):
    passed = 0.0
    for rate, mean in points:
        if mean > limit:
            break
        passed = rate
    return passed


def curves(kvs, by_source):
    curves_by_key = {}
    audits = []
    for kv in kvs:
        for arm in METHODS:
            code, code_fail, code_max = by_source['code', kv, arm]
            conv, conv_fail, conv_max = by_source['conversation', kv, arm]
            assert code_fail is not None and conv_fail is not None, (kv, arm)
            limits = sorted({mean for _, mean in code + conv if mean is not None})
            curve = []
            last = 0.0
            for limit in limits:
                c = capacity(code, limit)
                v = capacity(conv, limit)
                value = (c + v) / 2
                if value > last:
                    curve.append(dict(kv_blocks=kv, arm=arm, method=LABELS[METHODS.index(arm)],
                                      mean_ttft_limit_ms=limit, mean_capacity_qps=value,
                                      code_capacity_qps=c, conversation_capacity_qps=v))
                    last = value
            curves_by_key[kv, arm] = curve
            audits.append(dict(kv_blocks=kv, arm=arm, code_first_resource_failure=code_fail,
                               conversation_first_resource_failure=conv_fail,
                               code_max_measured_qps=code_max,
                               conversation_max_measured_qps=conv_max,
                               code_last_eligible_qps=code[-1][0] if code else 0,
                               conversation_last_eligible_qps=conv[-1][0] if conv else 0,
                               code_censored=code_fail is None,
                               conversation_censored=conv_fail is None))
    return curves_by_key, audits


def dominates(a, b):
    return (a['mean_ttft_limit_ms'] <= b['mean_ttft_limit_ms'] + 1e-9
            and a['mean_capacity_qps'] >= b['mean_capacity_qps'] - 1e-9
            and (a['mean_ttft_limit_ms'] < b['mean_ttft_limit_ms'] - 1e-9
                 or a['mean_capacity_qps'] > b['mean_capacity_qps'] + 1e-9))


def available(kv):
    return METHODS


def comparisons(kvs, curves_by_key):
    checks = []
    pareto = []
    for kv in kvs:
        points = [p for arm in available(kv) for p in curves_by_key[kv, arm]]
        for p in points:
            rivals = [q for q in points if q['arm'] != p['arm'] and dominates(q, p)]
            pareto.append(dict(p, dominated_by='; '.join(sorted({q['method'] for q in rivals})),
                               nondominated=not rivals))
        for limit in CHECK_LIMITS_MS:
            values = {arm: max((p['mean_capacity_qps'] for p in curves_by_key[kv, arm]
                                if p['mean_ttft_limit_ms'] <= limit), default=0)
                      for arm in available(kv)}
            ours = values['arrival']
            best = max(values[arm] for arm in available(kv)[1:])
            selected = max((p for p in curves_by_key[kv, 'arrival']
                            if p['mean_ttft_limit_ms'] <= limit),
                           key=lambda p: p['mean_ttft_limit_ms'], default=None)
            selected_dominators = ([q for q in points if q['arm'] != 'arrival'
                                    and dominates(q, selected)] if selected else [])
            checks.append(dict(kv_blocks=kv, mean_ttft_limit_ms=limit,
                               spare_capacity_qps=ours, best_baseline_capacity_qps=best,
                               comparators=len(available(kv)) - 1,
                               spare_is_best=ours >= best - 1e-9,
                               spare_attained_ttft_ms=(selected['mean_ttft_limit_ms']
                                                       if selected else None),
                               spare_nondominated=not selected_dominators,
                               spare_dominated_by='; '.join(sorted({q['method']
                                                                   for q in selected_dominators})),
                               higher_baselines='; '.join(LABELS[METHODS.index(arm)]
                                                          for arm in available(kv)[1:]
                                                          if values[arm] > ours + 1e-9)))
    return pareto, checks


def plot_rows(pareto, audits, rate_rows):
    coverage = {(a['kv_blocks'], a['arm']): a for a in audits}
    measured = {(r['source'], r['kv_blocks'], r['arm'], r['rate_qps']): r
                for r in rate_rows}
    by_method = defaultdict(list)
    for p in pareto:
        by_method[p['kv_blocks'], p['arm']].append(p)
    rows = []
    for key, points in by_method.items():
        points.sort(key=lambda p: p['mean_ttft_limit_ms'])
        assert points[-1]['mean_ttft_limit_ms'] < PLOT_MAX_MS, key
        audit = coverage[key]
        for p in points:
            fields = dict(windows_per_workload=50,
                          code_first_failed_qps=audit['code_first_resource_failure'],
                          conversation_first_failed_qps=audit['conversation_first_resource_failure'],
                          scan_complete=True)
            for source in SOURCES:
                rate = p[f'{source}_capacity_qps']
                record = measured.get((source, p['kv_blocks'], p['arm'], rate)) if rate else None
                if record is not None:
                    assert record['eligible'] and record['mean_ttft_ms'] <= p['mean_ttft_limit_ms']
                    assert record['failed_samples'] == 0
                fields.update({f'{source}_{name}': record[name] if record else None
                               for name in ('mean_ttft_ms', 'p99_tbt_ms', 'backlog_drift',
                                            'failed_samples', 'rate_record_path')})
            rows.append(dict(p, **fields, is_plot_extension=False))
        rows.append(dict(rows[-1], mean_ttft_limit_ms=PLOT_MAX_MS,
                         dominated_by='', nondominated='', is_plot_extension=True))
    return rows


def plot(kvs, all_rows):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    by_method = defaultdict(list)
    for p in all_rows:
        by_method[p['kv_blocks'], p['arm']].append(p)
    max_y = max(p['mean_capacity_qps'] for p in all_rows)
    for page in range((len(kvs) + 7) // 8):
        shown = kvs[page * 8:(page + 1) * 8]
        nrows = 1 if len(shown) <= 4 else 2
        fig, axes = plt.subplots(nrows, 4, figsize=(14, 4.2 if nrows == 1 else 7.5),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, kv in zip(axes.flat, shown):
            for arm in METHODS:
                i = METHODS.index(arm)
                points = by_method[kv, arm]
                if not points:
                    continue
                ax.step([p['mean_ttft_limit_ms'] / 1000 for p in points],
                        [p['mean_capacity_qps'] for p in points], where='post',
                        color=COLORS[i], linewidth=2 if arm == 'arrival' else 1,
                        linestyle='--' if arm in EXTRA else '-',
                        alpha=1 if arm == 'arrival' else .8, zorder=4 if arm == 'arrival' else 2)
                if arm == 'arrival':
                    observed = [p for p in points if not p['is_plot_extension']]
                    ax.scatter([p['mean_ttft_limit_ms'] / 1000 for p in observed],
                               [p['mean_capacity_qps'] for p in observed], s=19,
                               facecolors=['white' if p['nondominated'] else COLORS[0]
                                           for p in observed],
                               edgecolors=COLORS[0], linewidths=.8, zorder=5)
            ax.set_title(f'KV {kv}', loc='left')
            ax.grid(axis='y', linewidth=.4, color='#dce2e5')
            ax.set_axisbelow(True)
            ax.set_xscale('log')
            ax.set_xlim(.2, PLOT_MAX_MS / 1000)
            ax.set_xticks((.25, .5, 1, 2, 4, 8, 16),
                          labels=('0.25', '0.5', '1', '2', '4', '8', '16'))
            ax.set_ylim(0, max_y * 1.07)
        for ax in axes.flat[len(shown):]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel('Mean TTFT limit (s; log scale)')
        for ax in axes[:, 0]:
            ax.set_ylabel('Mean capacity (QPS)')
        shown_methods = METHODS
        handles = [Line2D([], [], color=COLORS[i], linewidth=2 if i == 0 else 1.3,
                          linestyle='--' if i >= len(CORE) else '-', label=LABELS[i])
                   for i, arm in enumerate(METHODS) if arm in shown_methods]
        handles += [Line2D([], [], marker='o', linestyle='None', color=COLORS[0],
                           markerfacecolor='white', markersize=5,
                           label='SPARE nondominated point'),
                    Line2D([], [], marker='o', linestyle='None', color=COLORS[0],
                           markerfacecolor=COLORS[0], markersize=5,
                           label='SPARE dominated point')]
        fig.legend(handles=handles, ncol=7 if len(shown_methods) > 8 else 5,
                   loc='lower center', bbox_to_anchor=(.5, .01),
                   fontsize=7.5)
        fig.suptitle('Capacity under mean TTFT, P99 TBT and backlog limits', y=.99, fontsize=12)
        fig.tight_layout(rect=(0, .13 if nrows == 1 else .075, 1, .94 if nrows == 1 else .965),
                         h_pad=2.2, w_pad=1.5)
        stem = OUT / f'mean_ttft_capacity_page{page + 1}'
        fig.savefig(stem.with_suffix('.png'), dpi=300)
        fig.savefig(stem.with_suffix('.pdf'))
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kvs, records, provenance = load_records()
    rate_rows, by_source = source_points(records, provenance)
    curves_by_key, audits = curves(kvs, by_source)
    pareto, checks = comparisons(kvs, curves_by_key)
    all_rows = plot_rows(pareto, audits, rate_rows)
    write_csv('rate_audit.csv', rate_rows)
    write_csv('coverage.csv', audits)
    write_csv('capacity_curves.csv', [p for curve in curves_by_key.values() for p in curve])
    write_csv('pareto_points.csv', pareto)
    write_csv('threshold_checks.csv', checks)
    write_csv('plot_data_all.csv', all_rows)
    plot(kvs, all_rows)
    print(f'{len(kvs)} KV levels; {len(rate_rows)} measured rates; '
          f'{sum(a["code_censored"] or a["conversation_censored"] for a in audits)} censored methods')
    for limit in CHECK_LIMITS_MS:
        subset = [c for c in checks if c['mean_ttft_limit_ms'] == limit]
        print(f'{limit} ms: SPARE best at {sum(c["spare_is_best"] for c in subset)}/{len(subset)} KV')


if __name__ == '__main__':
    main()
