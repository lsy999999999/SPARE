"""Complete 50-window QPS scans for the mean-TTFT capacity plots."""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

from research.www_capacity_pareto_20261015 import run as base
from research.www_kv_capacity_50wide_20261016 import run as old
from research.www_kv_extreme_stress_20261016 import run as sim

ROOT = base.ROOT
ARCHIVE = ROOT / 'artifacts/www_kv_capacity_dense_20261016_r1'
FINE = ROOT / 'artifacts/www_kv_capacity_fine50_20261016_r1'
PAPER = ROOT / 'artifacts/www_kv_capacity_paper_20261015_r1'
CORE = tuple(old.METHODS)
EXTRA = ('past_future_multi', 'orca_max', 'ltr_proxy', 'sola_calibrated')
METHODS = CORE + EXTRA
OUT = ROOT / 'artifacts/www_mean_ttft_capacity_20261017_r2'
CAPACITIES = tuple(kv for kv in sorted(set(old.DENSE_CAPACITIES) | {12032})
                   if kv >= 1024 and kv not in (1280, 1536, 2304))
COARSE_RATES = base.RATES + (12., 16., 20.)
SOURCES = ('code', 'conversation')


def design():
    archive = base.read(ARCHIVE / 'design.json')
    paper = base.read(PAPER / 'design.json')
    assert {(s['name'], s['input']) for s in archive['samples']} == {
        (s['name'], s['input']) for s in paper['samples']}
    assert len(archive['samples']) == 100
    assert all(sum(s['source'] == source for s in archive['samples']) == 50
               for source in SOURCES)
    assert set(paper['main_arms']) == set(METHODS)
    for sample in archive['samples']:
        assert base.digest(ROOT / sample['input']) == archive['input_sha256'][sample['input']]
    assert base.digest(base.PREVIOUS / 'timing_profile.json') == archive['timing_sha256']
    return dict(samples=archive['samples'], capacities=CAPACITIES, methods=METHODS,
                sources=SOURCES, coarse_rates=COARSE_RATES, windows_per_workload=50,
                chunk=256, budget=1024, concurrency=32,
                horizon_ms=60000, warm_ms=20000, backlog_interval_ms=(40000, 60000),
                rule='mean TTFT threshold varies; all 50 valid; mean P99 TBT <= 200 ms; '
                     'mean backlog drift <= max(2, offered QPS); first failure bounds capacity',
                refinement_qps=.02, gpu_execution=False,
                timing_sha256=archive['timing_sha256'],
                input_sha256=archive['input_sha256'],
                source_sha256={p: base.digest(ROOT / p) for p in (
                    'research/www_capacity_pareto_20261015/run.py',
                    'research/www_kv_extreme_stress_20261016/run.py',
                    'research/www_evaluation_completion_20261003/simulate.py',
                    'research/v1_paper_comparison_20260922/scheduler.py',
                    'research/www_baseline_fidelity_20261013/scheduler.py')})


def prepare():
    expected = json.loads(json.dumps(design()))
    path = OUT / 'design.json'
    if path.exists():
        previous = base.read(path)
        assert {k: v for k, v in previous.items() if k != 'source_sha256'} == {
            k: v for k, v in expected.items() if k != 'source_sha256'}
        assert previous['source_sha256'] == expected['source_sha256']
    else:
        base.write(path, expected)
    return expected


def name(source, kv, arm, rate):
    return f'{source}_c{kv}_{arm}_r{rate:g}.json'


def cached_path(source, kv, arm, rate):
    filename = name(source, kv, arm, rate)
    roots = (OUT, FINE, ARCHIVE, PAPER)
    for root in roots:
        if root == PAPER and arm not in EXTRA:
            continue
        path = root / 'rates' / filename
        if path.exists():
            return path
    return None


def cached_rates(source, kv, arm):
    result = set()
    for root in (OUT, FINE, ARCHIVE, PAPER):
        if root == PAPER and arm not in EXTRA:
            continue
        prefix = f'{source}_c{kv}_{arm}_r'
        for path in (root / 'rates').glob(prefix + '*.json'):
            result.add(float(path.stem[len(prefix):]))
    return result


def get_rate(source, kv, arm, rate, samples):
    path = cached_path(source, kv, arm, rate)
    if path is None:
        rows = [sim.simulate_sample(sample, kv, arm, rate) for sample in samples]
        rows.sort(key=lambda row: row['name'])
        path = OUT / 'rates' / name(source, kv, arm, rate)
        base.write(path, dict(rows=rows, summary=old.summary(rows, rate),
                              newly_simulated=50))
        fresh = True
    else:
        fresh = False
    data = base.read(path)
    assert len(data['rows']) == 50
    assert {row['name'] for row in data['rows']} == {s['name'] for s in samples}
    return old.summary(data['rows'], rate), fresh


def resource_passed(summary, rate):
    return (summary['failed_samples'] == 0 and summary['p99_tbt_ms'] <= 200
            and summary['backlog_drift'] <= max(2, rate))


def run_group(task, force=False):
    source, kv, arm, samples = task
    target = OUT / 'groups' / f'{source}_c{kv}_{arm}.json'
    if target.exists() and not force:
        result = base.read(target)
        assert result['stage'] == 'complete'
        return result
    known = cached_rates(source, kv, arm)
    coarse = sorted(set(COARSE_RATES) | known)
    last_pass = 0.
    first_fail = None
    new_rates = 0
    checked = []
    for rate in coarse:
        summary, fresh = get_rate(source, kv, arm, rate, samples)
        new_rates += int(fresh)
        checked.append(rate)
        if resource_passed(summary, rate):
            last_pass = rate
        else:
            first_fail = rate
            break
    while first_fail is None:
        rate = 2 * checked[-1]
        summary, fresh = get_rate(source, kv, arm, rate, samples)
        new_rates += int(fresh)
        checked.append(rate)
        if resource_passed(summary, rate):
            last_pass = rate
        else:
            first_fail = rate
        if rate >= 64 and first_fail is None:
            raise RuntimeError(f'no resource failure through {rate:g} QPS: {source}/{kv}/{arm}')
    while first_fail - last_pass > .0200001:
        candidate = round((first_fail + last_pass) * 50) / 100
        if not last_pass < candidate < first_fail:
            break
        summary, fresh = get_rate(source, kv, arm, candidate, samples)
        new_rates += int(fresh)
        checked.append(candidate)
        if resource_passed(summary, candidate):
            last_pass = candidate
        else:
            first_fail = candidate
    result = dict(stage='complete', source=source, kv_blocks=kv, arm=arm,
                  capacity_qps=last_pass, next_failed_qps=first_fail,
                  resolution_qps=round(first_fail - last_pass, 4),
                  rates_checked=len(checked), new_rates=new_rates,
                  windows_per_workload=50)
    base.write(target, result)
    return result


def run(workers, kv_min, kv_max, status_name, reconcile=False):
    frozen = prepare()
    status_path = OUT / status_name
    tasks = [(source, kv, arm,
              [s for s in frozen['samples'] if s['source'] == source])
             for kv in CAPACITIES if kv_min <= kv <= kv_max
             for arm in METHODS for source in SOURCES]
    started = time.monotonic()
    complete = new = 0
    failures = []
    with ProcessPoolExecutor(max_workers=workers, initializer=sim.initialize) as pool:
        futures = {pool.submit(run_group, task, reconcile): task[:3] for task in tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
                complete += 1
                new += result['new_rates']
            except Exception as exc:
                failures.append(dict(task=futures[future], error=repr(exc)))
            if complete % 8 == 0 or failures:
                status = dict(stage='running', groups_complete=complete,
                              groups_total=len(tasks), new_rates=new,
                              errors=failures, elapsed_s=round(time.monotonic()-started, 1))
                base.write(status_path, status)
                print(status, flush=True)
    status = dict(stage='complete' if not failures and complete == len(tasks) else 'failed',
                  groups_complete=complete, groups_total=len(tasks), new_rates=new,
                  errors=failures, elapsed_s=round(time.monotonic()-started, 1))
    base.write(status_path, status)
    if status['stage'] != 'complete':
        raise RuntimeError(status)
    print(status, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--kv-min', type=int, default=0)
    parser.add_argument('--kv-max', type=int, default=12032)
    parser.add_argument('--status-name', default='status.json')
    parser.add_argument('--reconcile', action='store_true',
                        help='recompute first-failure boundaries from all cached rates')
    args = parser.parse_args()
    if args.prepare:
        print(f'prepared {len(prepare()["samples"])} windows, {len(CAPACITIES)} KV levels, '
              f'{len(METHODS)} methods')
    else:
        run(args.workers, args.kv_min, args.kv_max, args.status_name, args.reconcile)
