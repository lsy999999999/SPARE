"""Resume the fixed-policy 21-KV scan with 50 paired windows per workload."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import shutil
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np

from research.www_capacity_pareto_20261015 import run as base
from research.www_kv_extreme_stress_20261016 import run as previous

ROOT = base.ROOT
OUT = ROOT / 'artifacts/www_kv_capacity_50wide_20261016_r1'
WIDE_OUT = OUT
DENSE_OUT = ROOT / 'artifacts/www_kv_capacity_dense_20261016_r1'
OLD_50 = ROOT / 'artifacts/www_kv_capacity_paper_20261015_r1'
OLD_10 = ROOT / 'artifacts/www_kv_extreme_stress_20261016_r1/full'
CAPACITIES = tuple(previous.CAPACITIES['full'])
DENSE_CAPACITIES = (672, 768, 896) + tuple(range(1024, 8193, 256))
METHODS = tuple(previous.METHODS)
RATES = tuple(previous.RATES['full'])
SHARDS = {
    'local': tuple(kv for kv in CAPACITIES if kv <= 768),
    'remote': tuple(kv for kv in CAPACITIES if kv > 768),
}


def configure(profile):
    global OUT, CAPACITIES, SHARDS
    if profile == 'wide':
        return
    assert profile == 'dense'
    OUT = DENSE_OUT
    CAPACITIES = DENSE_CAPACITIES
    old = set(previous.CAPACITIES['full'])
    new = [kv for kv in CAPACITIES if kv not in old]
    reused = [kv for kv in CAPACITIES if kv in old]
    SHARDS = {
        'local': tuple(sorted(reused[:7] + new[::2])),
        'remote': tuple(sorted(reused[7:] + new[1::2])),
    }


def design():
    old = base.read(OLD_50 / 'design.json')
    samples = sorted((s for s in old['samples'] if s['split'] == 'test'),
                     key=lambda s: s['name'])
    assert len(samples) == 100
    assert all(sum(s['source'] == source for s in samples) == 50
               for source in ('code', 'conversation'))
    assert tuple(old['rates']) == RATES
    assert set(METHODS) <= set(old['main_arms'])
    assert old['default_budget'] == previous.BUDGET == 1024
    assert old['chunk'] == 256 and old['concurrency'] == 32
    assert old['slos'] == previous.SLO
    assert old['reference_qps'] == previous.REFERENCE_QPS == .5
    assert set(SHARDS['local']).isdisjoint(SHARDS['remote'])
    assert set(SHARDS['local']) | set(SHARDS['remote']) == set(CAPACITIES)
    for sample in samples:
        assert base.digest(ROOT / sample['input']) == old['input_sha256'][sample['input']]
    prior_design = base.read(OLD_10 / 'design.json')
    for path, digest in prior_design['source_sha256'].items():
        assert base.digest(ROOT / path) == digest, path
    assert base.digest(base.PREVIOUS / 'timing_profile.json') == old['timing_sha256']
    return dict(samples=samples, capacities=CAPACITIES, shards=SHARDS,
                methods=METHODS, rates=RATES, windows_per_workload=50,
                reference_qps=.5, slo=previous.SLO, budget=1024, chunk=256,
                concurrency=32, cohort='original requests',
                aggregation='Equal mean of Code and Conversation, not mixed-stream capacity',
                old_50_design_sha256=base.digest(OLD_50 / 'design.json'),
                input_sha256={s['input']: old['input_sha256'][s['input']] for s in samples},
                source_sha256=prior_design['source_sha256'],
                timing_sha256=old['timing_sha256'])


def prepare():
    expected = design()
    target = OUT / 'design.json'
    if target.exists():
        assert base.read(target) == __import__('json').loads(__import__('json').dumps(expected))
    else:
        base.write(target, expected)
    return expected


def summary(rows, rate):
    assert len(rows) == 50 and len({r['name'] for r in rows}) == 50
    valid = [r for r in rows if r['valid']]
    failures = [r for r in rows if not r['valid']]
    keys = ('mean_ttft_ms', 'p95_ttft_ms', 'p99_tbt_ms', 'backlog_drift',
            'preemptions', 'recomputed_tokens')
    stats = {key: float(np.mean([r[key] for r in valid])) if valid else None
             for key in keys}
    stats['drift_limit'] = max(2., rate)
    stats['failed_samples'] = len(failures)
    stats['failure_reasons'] = sorted({r['failure'] for r in failures})
    stats['passed'] = (not failures and stats['p95_ttft_ms'] <= 12000.
                       and stats['p99_tbt_ms'] <= 200.
                       and stats['backlog_drift'] <= stats['drift_limit'])
    return stats


def prior_rows(source, kv, arm, rate, wanted):
    filename = f'{source}_c{kv}_{arm}_r{rate:g}.json'
    for root in (WIDE_OUT, OLD_50, OLD_10):
        path = root / 'rates' / filename
        if path.exists():
            rows = [r for r in base.read(path)['rows'] if r['name'] in wanted]
            assert len({r['name'] for r in rows}) == len(rows)
            return rows
    return []


def run_group(task):
    source, kv, arm, samples = task
    wanted = {s['name'] for s in samples}
    assert len(wanted) == 50
    old_group = WIDE_OUT / 'groups' / f'{source}_c{kv}_{arm}.json'
    if OUT != WIDE_OUT and old_group.exists():
        result = base.read(old_group)
        assert result['source'] == source and result['kv_blocks'] == kv and result['arm'] == arm
        rate_files = list((WIDE_OUT / 'rates').glob(f'{source}_c{kv}_{arm}_r*.json'))
        assert rate_files
        for old_rate in rate_files:
            target = OUT / 'rates' / old_rate.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_rate, target)
        target = OUT / 'groups' / old_group.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_group, target)
        return dict(result, rates_complete=len(rate_files), new_rows=0,
                    reused_rows=50 * len(rate_files))
    capacity, failure = 0., None
    new_rows = reused_rows = rates_complete = 0
    for rate in RATES:
        path = OUT / 'rates' / f'{source}_c{kv}_{arm}_r{rate:g}.json'
        if path.exists():
            record = base.read(path)
            assert {r['name'] for r in record['rows']} == wanted
            assert record['summary'] == summary(record['rows'], rate)
        else:
            rows = prior_rows(source, kv, arm, rate, wanted)
            found = {r['name'] for r in rows}
            assert found <= wanted
            reused_rows += len(rows)
            for sample in samples:
                if sample['name'] not in found:
                    rows.append(previous.simulate_sample(sample, kv, arm, rate))
                    new_rows += 1
            rows.sort(key=lambda r: r['name'])
            record = dict(rows=rows, summary=summary(rows, rate),
                          reused_rows=len(found), newly_simulated=len(rows)-len(found))
            base.write(path, record)
        rates_complete += 1
        if failure is None:
            if record['summary']['passed']:
                capacity = rate
            else:
                failure = rate
        if failure is not None and rate >= .5:
            break
    result = dict(source=source, kv_blocks=kv, arm=arm,
                  capacity_qps=capacity, next_failed_qps=failure,
                  capacity_is_lower_bound=failure is None,
                  reference_feasible=capacity >= .5,
                  rates_complete=rates_complete, new_rows=new_rows,
                  reused_rows=reused_rows)
    base.write(OUT / 'groups' / f'{source}_c{kv}_{arm}.json', result)
    return result


def run(shard, workers):
    frozen = prepare()
    tasks = [(source, kv, arm,
              [s for s in frozen['samples'] if s['source'] == source])
             for source in ('code', 'conversation')
             for kv in SHARDS[shard] for arm in METHODS]
    assert all(len(t[3]) == 50 for t in tasks)
    started = time.monotonic()
    status = dict(stage='starting', shard=shard, host=os.uname().nodename,
                  pid=os.getpid(), workers=workers, groups_total=len(tasks),
                  groups_complete=0, rates_complete=0, new_rows=0,
                  reused_rows=0, errors=[], updated_at=time.time())
    target = OUT / shard / 'status.json'
    base.write(target, status)
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=previous.initialize) as pool:
            status['stage'] = 'running'
            base.write(target, status)
            futures = {pool.submit(run_group, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                status['groups_complete'] += 1
                for field in ('rates_complete', 'new_rows', 'reused_rows'):
                    status[field] += result[field]
                status['elapsed_s'] = round(time.monotonic()-started, 1)
                status['updated_at'] = time.time()
                base.write(target, status)
                print(status, flush=True)
    except Exception as exc:
        status['stage'] = 'failed'
        status['errors'].append(repr(exc))
        status['updated_at'] = time.time()
        base.write(target, status)
        raise
    status['stage'] = 'complete'
    status['updated_at'] = time.time()
    base.write(target, status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=('wide', 'dense'), default='wide')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--shard', choices=SHARDS)
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()
    configure(args.profile)
    if args.prepare:
        print(f'Prepared {len(prepare()["samples"])} test windows across {len(CAPACITIES)} KV levels')
    else:
        if args.shard is None:
            parser.error('--shard is required unless --prepare is used')
        run(args.shard, args.workers)
