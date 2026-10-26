"""Small paired-window, fine-QPS follow-up to the 32-KV scan."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np

from research.www_capacity_pareto_20261015 import run as base
from research.www_kv_capacity_50wide_20261016 import run as old
from research.www_kv_extreme_stress_20261016 import run as sim

ROOT = base.ROOT
OUT = ROOT / 'artifacts/www_kv_capacity_fine_20261016_r1'
ABS_OUT = OUT
ARCHIVE = ROOT / 'artifacts/www_kv_capacity_dense_20261016_r1'
CAPACITIES = old.DENSE_CAPACITIES
METHODS = old.METHODS
SHARDS = {'local': CAPACITIES[::2], 'remote': CAPACITIES[1::2]}
PASS_KEY = 'passed_absolute'


def configure(rule):
    global OUT, PASS_KEY
    if rule == 'absolute':
        OUT, PASS_KEY = ABS_OUT, 'passed_absolute'
    else:
        assert rule == 'drift'
        OUT = ROOT / 'artifacts/www_kv_capacity_fine_drift_20261016_r1'
        PASS_KEY = 'passed_drift'


def prepare():
    archive = base.read(ARCHIVE / 'design.json')
    samples = [s for s in archive['samples'] if s['window'] == 0]
    assert len(samples) == 4
    assert all(sum(s['source'] == source for s in samples) == 2
               for source in ('code', 'conversation'))
    for sample in samples:
        assert base.digest(ROOT / sample['input']) == archive['input_sha256'][sample['input']]
    design = dict(samples=samples, capacities=CAPACITIES, methods=METHODS,
                  shards=SHARDS, windows_per_workload=2,
                  window_selection='window 0 from each of the two frozen test dates',
                  archived_rates=old.RATES, fine_target_qps=.1,
                  capacity_rule='all observed rates up to capacity pass; first failure bounds capacity',
                  pass_rule='no invalid run; mean P95 TTFT <= 12000 ms; mean P99 TBT <= 200 ms; mean pending_end < max(2, offered QPS)',
                  old_rule='mean backlog_drift <= max(2, offered QPS)',
                  horizon_ms=60000, warm_ms=20000, pending_probe_ms=60000,
                  reference_qps=.5, budget=1024, chunk=256, concurrency=32,
                  gpu_execution=False, source_design_sha256=base.digest(ARCHIVE / 'design.json'),
                  input_sha256={s['input']: archive['input_sha256'][s['input']] for s in samples},
                  timing_sha256=archive['timing_sha256'],
                  simulator_sha256=base.digest(ROOT / 'research/www_evaluation_completion_20261003/simulate.py'))
    if PASS_KEY == 'passed_drift':
        design['capacity_metric'] = '20-second growth in unfinished requests'
        design['pass_rule'] = ('no invalid run; mean P95 TTFT <= 12000 ms; '
                               'mean P99 TBT <= 200 ms; mean backlog_drift <= max(2, offered QPS)')
    target = OUT / 'design.json'
    if target.exists():
        assert base.read(target) == __import__('json').loads(__import__('json').dumps(design))
    else:
        base.write(target, design)
    return design


def summary(rows, rate):
    assert len(rows) == 2 and len({r['name'] for r in rows}) == 2
    valid = [r for r in rows if r['valid']]
    stats = {key: float(np.mean([r[key] for r in valid])) if valid else None
             for key in ('mean_ttft_ms', 'p95_ttft_ms', 'p99_tbt_ms',
                         'pending_mid', 'pending_end', 'backlog_drift')}
    stats['invalid_windows'] = len(rows) - len(valid)
    stats['limit'] = max(2., rate)
    latency = (len(valid) == len(rows) and stats['p95_ttft_ms'] <= 12000.
               and stats['p99_tbt_ms'] <= 200.)
    stats['passed_absolute'] = bool(latency and stats['pending_end'] < stats['limit'])
    stats['passed_drift'] = bool(latency and stats['backlog_drift'] <= stats['limit'])
    return stats


def rate_path(source, kv, arm, rate):
    return OUT / 'rates' / f'{source}_c{kv}_{arm}_r{rate:g}.json'


def get_rate(source, kv, arm, rate, samples):
    path = rate_path(source, kv, arm, rate)
    if path.exists():
        record = base.read(path)
        assert {r['name'] for r in record['rows']} == {s['name'] for s in samples}
        assert record['summary'] == summary(record['rows'], rate)
        return record
    prior = ARCHIVE / 'rates' / path.name
    if not prior.exists() and PASS_KEY == 'passed_drift':
        prior = ABS_OUT / 'rates' / path.name
    rows = []
    if prior.exists():
        wanted = {s['name'] for s in samples}
        rows = [r for r in base.read(prior)['rows'] if r['name'] in wanted]
    found = {r['name'] for r in rows}
    assert len(found) == len(rows)
    reused = len(rows)
    for sample in samples:
        if sample['name'] not in found:
            rows.append(sim.simulate_sample(sample, kv, arm, rate))
    rows.sort(key=lambda r: r['name'])
    record = dict(rows=rows, summary=summary(rows, rate),
                  reused_rows=reused, newly_simulated=len(rows)-reused)
    base.write(path, record)
    return record


def run_group(task):
    source, kv, arm, samples = task
    known = {}
    new_rows = reused_rows = 0
    for rate in old.RATES:
        prior = ARCHIVE / 'rates' / rate_path(source, kv, arm, rate).name
        first_failure = next((r for r in sorted(known) if not known[r]), None)
        if prior.exists() or rate in (.25, .5) or first_failure is None:
            record = get_rate(source, kv, arm, rate, samples)
            known[rate] = record['summary'][PASS_KEY]
            new_rows += record['newly_simulated']
            reused_rows += record['reused_rows']
    assert .25 in known and .5 in known
    ordered = sorted(known)
    failed = next((rate for rate in ordered if not known[rate]), None)
    passed = max((rate for rate in ordered if rate < failed and known[rate]), default=0.) if failed else max(ordered)
    if failed is not None and passed:
        while failed - passed > .1000001:
            candidate = round((failed + passed) / .2) / 10
            if not passed < candidate < failed:
                break
            record = get_rate(source, kv, arm, candidate, samples)
            new_rows += record['newly_simulated']
            reused_rows += record['reused_rows']
            known[candidate] = record['summary'][PASS_KEY]
            if known[candidate]:
                passed = candidate
            else:
                failed = candidate
    result = dict(source=source, kv_blocks=kv, arm=arm, capacity_qps=passed,
                  next_failed_qps=failed, capacity_is_lower_bound=failed is None,
                  resolution_qps=round(failed-passed, 4) if failed is not None and passed else None,
                  reference_feasible=known[.5], rates_complete=len(known),
                  new_rows=new_rows, reused_rows=reused_rows,
                  nonmonotone_archived_rates=sum(known[r] for r in ordered if failed is not None and r > failed))
    base.write(OUT / 'groups' / f'{source}_c{kv}_{arm}.json', result)
    return result


def run(shard, workers):
    design = prepare()
    tasks = [(source, kv, arm, [s for s in design['samples'] if s['source'] == source])
             for source in ('code', 'conversation') for kv in SHARDS[shard] for arm in METHODS]
    status = dict(stage='running', shard=shard, groups_total=len(tasks), groups_complete=0,
                  new_rows=0, reused_rows=0, errors=[], updated_at=time.time())
    target = OUT / shard / 'status.json'
    base.write(target, status)
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=sim.initialize) as pool:
            for future in as_completed([pool.submit(run_group, task) for task in tasks]):
                result = future.result()
                status['groups_complete'] += 1
                status['new_rows'] += result['new_rows']
                status['reused_rows'] += result['reused_rows']
                status['updated_at'] = time.time()
                base.write(target, status)
                if status['groups_complete'] % 16 == 0:
                    print(status, flush=True)
    except Exception as exc:
        status['stage'] = 'failed'
        status['errors'].append(repr(exc))
        base.write(target, status)
        raise
    status['stage'] = 'complete'
    status['updated_at'] = time.time()
    base.write(target, status)
    print(status, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--shard', choices=SHARDS)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--rule', choices=('absolute', 'drift'), default='absolute')
    args = parser.parse_args()
    configure(args.rule)
    if args.prepare:
        print(prepare())
    else:
        if not args.shard:
            parser.error('--shard is required')
        run(args.shard, args.workers)
