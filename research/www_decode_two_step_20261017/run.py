"""Paired two-step decode-headroom ablation on the frozen capacity workload."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import statistics
from unittest.mock import patch

from research.www_biobjective_ablation_20261016 import run as comparison
from research.www_capacity_pareto_20261015 import run as base
from research.www_evaluation_completion_20261003 import scheduler as evaluation
from research.www_kv_capacity_50wide_20261016 import run as archive
from .policy import select

ROOT = base.ROOT
OUT = ROOT / 'artifacts/www_decode_two_step_20261017_r1'
SOURCES = ('code', 'conversation')
CAPACITIES = comparison.CAPACITIES


def initialize():
    comparison.initialize()


def one(sample, kv, rate):
    counts = {'action_changes': 0, 'candidate_vetoes': 0}

    def guarded(*args, **kwargs):
        result = select(*args, **kwargs)
        counts['action_changes'] += int(result['two_step_action_changed'])
        counts['candidate_vetoes'] += result['two_step_vetoes']
        return result

    with patch.object(evaluation, 'select', guarded):
        result = comparison.simulate_one(sample, kv, 'spare', rate)
    result['arm'] = 'two_step_decode'
    result.update(counts)
    return result


def prepare():
    frozen = base.read(comparison.OUT / 'design.json')
    assert base.read(comparison.OUT / 'status.json')['stage'] == 'complete'
    design = dict(parent_design_sha256=base.digest(comparison.OUT / 'design.json'),
                  samples=frozen['samples'], capacities=CAPACITIES,
                  rates=comparison.RATES, budget=1024, chunk=256,
                  decode_protection='current decode token plus physical KV headroom '
                                    'for one more decode token per protected request',
                  source_sha256={p: base.digest(ROOT / p) for p in (
                      'research/www_decode_two_step_20261017/policy.py',
                      'research/www_decode_two_step_20261017/run.py')},
                  timing_sha256=frozen['timing_sha256'],
                  cpu_fitted_simulator=True, gpu_execution=False)
    path = OUT / 'design.json'
    normalized = json.loads(json.dumps(design))
    if path.exists():
        assert base.read(path) == normalized
    else:
        base.write(path, normalized)
    return normalized


def run_group(source, kv, samples):
    group_path = OUT / 'groups' / f'{source}_c{kv}.json'
    if group_path.exists():
        return base.read(group_path)
    capacity, failed, tested = 0., None, []
    for rate in comparison.RATES:
        path = OUT / 'rates' / f'{source}_c{kv}_r{rate:g}.json'
        if path.exists():
            record = base.read(path)
        else:
            checkpoint = OUT / 'checkpoints' / path.name
            rows = base.read(checkpoint)['rows'] if checkpoint.exists() else []
            done = {row['name'] for row in rows}
            for sample in samples:
                if sample['name'] in done:
                    continue
                rows.append(one(sample, kv, rate))
                base.write(checkpoint, {'rows': rows})
            rows.sort(key=lambda row: row['name'])
            record = {'rows': rows, 'summary': archive.summary(rows, rate)}
            base.write(path, record)
        assert len(record['rows']) == 50
        assert record['summary'] == archive.summary(record['rows'], rate)
        tested.append(rate)
        if failed is None:
            if record['summary']['passed']:
                capacity = rate
            else:
                failed = rate
        if failed is not None and rate >= .5:
            break
    result = dict(source=source, kv_blocks=kv, capacity_qps=capacity,
                  next_failed_qps=failed, tested_rates=tested, windows=50)
    base.write(group_path, result)
    return result


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze():
    points, pairs = [], []
    for source in SOURCES:
        for kv in CAPACITIES:
            group = base.read(OUT / 'groups' / f'{source}_c{kv}.json')
            original = base.read(comparison.group_path(source, kv, 'spare'))
            new = base.read(OUT / 'rates' / f'{source}_c{kv}_r0.5.json')
            old = base.read(comparison.rate_path(source, kv, 'spare', .5))
            by_name = {row['name']: row for row in old['rows']}
            assert len(by_name) == len(new['rows']) == 50
            assert all(row['valid'] and by_name[row['name']]['valid'] for row in new['rows'])
            deltas = [row['mean_ttft_ms'] - by_name[row['name']]['mean_ttft_ms']
                      for row in new['rows']]
            changed = sum(row['trajectory_sha256'] != by_name[row['name']]['trajectory_sha256']
                          for row in new['rows'])
            vetoes = sum(row['candidate_vetoes'] for row in new['rows'])
            actions = sum(row['action_changes'] for row in new['rows'])
            points.append(dict(source=source, kv_blocks=kv,
                               spare_capacity_qps=original['capacity_qps'],
                               two_step_capacity_qps=group['capacity_qps'],
                               capacity_delta_qps=group['capacity_qps']-original['capacity_qps'],
                               spare_mean_ttft_ms=old['summary']['mean_ttft_ms'],
                               two_step_mean_ttft_ms=new['summary']['mean_ttft_ms'],
                               paired_mean_ttft_delta_ms=statistics.fmean(deltas),
                               changed_windows_at_0_5=changed,
                               changed_actions_at_0_5=actions,
                               vetoed_candidates_at_0_5=vetoes,
                               spare_next_failed_qps=original['next_failed_qps'],
                               two_step_next_failed_qps=group['next_failed_qps']))
            for row in new['rows']:
                prior = by_name[row['name']]
                pairs.append(dict(source=source, kv_blocks=kv, name=row['name'],
                                  ttft_delta_ms=row['mean_ttft_ms']-prior['mean_ttft_ms'],
                                  tbt_delta_ms=row['p99_tbt_ms']-prior['p99_tbt_ms'],
                                  trajectory_changed=row['trajectory_sha256'] != prior['trajectory_sha256'],
                                  changed_actions=row['action_changes'],
                                  vetoed_candidates=row['candidate_vetoes']))
    write_csv(OUT / 'analysis' / 'points.csv', points)
    write_csv(OUT / 'analysis' / 'paired_windows.csv', pairs)
    lines = ['# Two-step decode protection ablation', '',
             'The frozen SPARE policy protects the current one-token decode allocation. '
             'This variant additionally requires enough free physical KV blocks after '
             'each accepted candidate for one more decode token per protected request. '
             'It does not schedule two decode tokens in one iteration or reserve blocks '
             'across later iterations.', '',
             'CPU simulator with the same 50 paired windows per source, fitted timing, '
             'six KV capacities, budget, chunk, load grid, and pass criteria as the '
             'completed bi-objective ablation. SPARE results are reused from that archive.', '',
             '|Source|KV blocks|SPARE capacity|Two-step capacity|Capacity delta|'
             'Paired mean TTFT delta at 0.5 QPS (ms)|Changed windows / 50|Changed actions|Vetoed candidates|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in points:
        lines.append(f"|{row['source']}|{row['kv_blocks']}|{row['spare_capacity_qps']:g}|"
                     f"{row['two_step_capacity_qps']:g}|{row['capacity_delta_qps']:+g}|"
                     f"{row['paired_mean_ttft_delta_ms']:+.3f}|{row['changed_windows_at_0_5']}|"
                     f"{row['changed_actions_at_0_5']}|{row['vetoed_candidates_at_0_5']}|")
    (OUT / 'analysis' / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run', 'analyze', 'status'))
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps({'samples': len(prepare()['samples']), 'groups': 12}))
    elif args.command == 'run':
        design = prepare()
        tasks = [(source, kv, [s for s in design['samples'] if s['source'] == source])
                 for source in SOURCES for kv in CAPACITIES]
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
            future_map = {pool.submit(run_group, *task): task[:2] for task in tasks}
            for future in as_completed(future_map):
                print(json.dumps({'group': future_map[future], 'result': future.result()}), flush=True)
        print(json.dumps({'points': len(analyze())}), flush=True)
    elif args.command == 'analyze':
        print(json.dumps({'points': len(analyze())}))
    else:
        print(json.dumps({'groups_complete': len(list((OUT / 'groups').glob('*.json'))),
                          'rates_complete': len(list((OUT / 'rates').glob('*.json'))),
                          'partial_rates': len(list((OUT / 'checkpoints').glob('*.json')))},
                         indent=2))


if __name__ == '__main__':
    main()
