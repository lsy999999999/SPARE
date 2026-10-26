"""Refine all 32 KV capacity brackets with 50 paired windows per workload."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

from research.www_capacity_pareto_20261015 import run as base
from research.www_kv_capacity_50wide_20261016 import run as old
from research.www_kv_extreme_stress_20261016 import run as sim

old.configure('dense')
ROOT = base.ROOT
ARCHIVE = old.OUT
OUT = ROOT / 'artifacts/www_kv_capacity_fine50_20261016_r1'
CAPACITIES = old.CAPACITIES
METHODS = old.METHODS
SHARDS = old.SHARDS


def prepare():
    archive = base.read(ARCHIVE / 'design.json')
    assert tuple(archive['capacities']) == CAPACITIES
    assert tuple(archive['methods']) == METHODS
    assert archive['windows_per_workload'] == 50
    for sample in archive['samples']:
        assert base.digest(ROOT / sample['input']) == archive['input_sha256'][sample['input']]
    design = dict(samples=archive['samples'], capacities=CAPACITIES, methods=METHODS,
                  shards=SHARDS, archived_rates=old.RATES, windows_per_workload=50,
                  reference_qps=.5, fine_target_qps=.02,
                  capacity_rule='first failed offered QPS bounds the highest prior passing QPS',
                  pass_rule='all 50 valid; mean P95 TTFT <= 12000 ms; mean P99 TBT <= 200 ms; mean 40-60s backlog drift <= max(2, offered QPS)',
                  aggregation='equal mean of Code and Conversation capacities',
                  gpu_execution=False, source_design_sha256=base.digest(ARCHIVE / 'design.json'),
                  input_sha256=archive['input_sha256'],
                  timing_sha256=archive['timing_sha256'],
                  simulator_sha256=base.digest(ROOT / 'research/www_evaluation_completion_20261003/simulate.py'))
    target = OUT / 'design.json'
    if target.exists():
        assert base.read(target) == __import__('json').loads(__import__('json').dumps(design))
    else:
        base.write(target, design)
    return design


def rate_path(source, kv, arm, rate):
    return OUT / 'rates' / f'{source}_c{kv}_{arm}_r{rate:g}.json'


def get_rate(source, kv, arm, rate, samples):
    path = rate_path(source, kv, arm, rate)
    prior = ARCHIVE / 'rates' / path.name
    record = base.read(path if path.exists() else prior) if (path.exists() or prior.exists()) else None
    if record is None:
        rows = [sim.simulate_sample(sample, kv, arm, rate) for sample in samples]
        rows.sort(key=lambda row: row['name'])
        record = dict(rows=rows, summary=old.summary(rows, rate),
                      reused_rows=0, newly_simulated=50)
        base.write(path, record)
    assert {row['name'] for row in record['rows']} == {s['name'] for s in samples}
    assert record['summary'] == old.summary(record['rows'], rate)
    return record, path.exists() and not prior.exists()


def run_group(task):
    source, kv, arm, samples = task
    archived = base.read(ARCHIVE / 'groups' / f'{source}_c{kv}_{arm}.json')
    passed = float(archived['capacity_qps'])
    failed = archived['next_failed_qps']
    assert failed is not None and failed > passed
    assert base.read(ARCHIVE / 'rates' / rate_path(source, kv, arm, .5).name)['summary'] == old.summary(
        base.read(ARCHIVE / 'rates' / rate_path(source, kv, arm, .5).name)['rows'], .5)
    tested = {}
    while failed - passed > .0200001:
        candidate = round((failed + passed) * 50) / 100
        if not passed < candidate < failed:
            break
        record, is_new = get_rate(source, kv, arm, candidate, samples)
        tested[candidate] = dict(passed=record['summary']['passed'], newly_simulated=is_new)
        if record['summary']['passed']:
            passed = candidate
        else:
            failed = candidate
    result = dict(source=source, kv_blocks=kv, arm=arm, capacity_qps=passed,
                  next_failed_qps=failed, resolution_qps=round(failed-passed, 4),
                  reference_qps=.5, windows_per_workload=50, rates_tested=tested,
                  archived_capacity_qps=archived['capacity_qps'],
                  archived_next_failed_qps=archived['next_failed_qps'])
    base.write(OUT / 'groups' / f'{source}_c{kv}_{arm}.json', result)
    return result


def run(shard, workers):
    design = prepare()
    tasks = [(source, kv, arm, [s for s in design['samples'] if s['source'] == source])
             for source in ('code', 'conversation') for kv in SHARDS[shard] for arm in METHODS]
    assert all(len(task[3]) == 50 for task in tasks)
    target = OUT / shard / 'status.json'
    status = dict(stage='running', shard=shard, host=os.uname().nodename,
                  pid=os.getpid(), workers=workers, groups_total=len(tasks),
                  groups_complete=0, new_rates=0, errors=[], updated_at=time.time())
    base.write(target, status)
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=sim.initialize) as pool:
            for future in as_completed([pool.submit(run_group, task) for task in tasks]):
                result = future.result()
                status['groups_complete'] += 1
                status['new_rates'] += sum(v['newly_simulated'] for v in result['rates_tested'].values())
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
    args = parser.parse_args()
    if args.prepare:
        print(f'Prepared {len(prepare()["samples"])} windows across {len(CAPACITIES)} KV levels')
    else:
        if args.shard is None:
            parser.error('--shard is required')
        run(args.shard, args.workers)
