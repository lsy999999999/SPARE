"""Resumable 50-window mean-TTFT capacity scans for mixed RL and SPARE."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import time

from research.www_kv_extreme_stress_20261016 import run as sim
from research.www_rl_mixed_generalization_20261017 import inputs, run


OUT = run.OUT
KVS = (2048, 3072, 4096, 6144, 12032)
COARSE = (.25, .5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10., 12.)
LIMIT_MS = 2000


def passed(summary, rate):
    return (summary['failed_samples'] == 0
            and summary['mean_ttft_ms'] <= LIMIT_MS
            and summary['p99_tbt_ms'] <= 200
            and summary['backlog_drift'] <= max(2, rate))


def cohort_for(workload, source):
    cohort = [s for s in inputs.samples_for(workload)
              if s['source'] == source]
    assert len(cohort) == 50
    return cohort


def source_hash(cohort):
    hashes = [run.old.base.digest(run.old.ROOT / s['input']) for s in cohort]
    return hashlib.sha256(''.join(hashes).encode()).hexdigest()


@lru_cache(maxsize=1)
def original_records():
    from research.www_online_rl_20261017 import analyze_mean
    return analyze_mean.records()


def rate_path(workload, source, kv, arm, seed, rate):
    return (OUT / 'rates' /
            f'{workload}_{source}_c{kv}_{arm}_seed{seed}_r{rate:g}.json')


def get_rate(workload, source, kv, arm, seed, rate, cohort, signature, weights=None):
    if workload == 'original' and arm == 'spare':
        original = original_records()['spare', source, kv].get(rate)
        if original is not None:
            record = run.old.base.read(original)
            assert len(record['rows']) == 50
            return record['summary'], False
    path = rate_path(workload, source, kv, arm, seed, rate)
    checkpoint_sha = (None if arm == 'spare' else
                      run.old.base.digest(OUT / run.old.base.read(
                          OUT / 'checkpoints' / f'seed{seed}_status.json')['selected']['path']))
    if path.exists():
        record = run.old.base.read(path)
        assert record['input_sha256'] == signature
        assert record['checkpoint_sha256'] == checkpoint_sha
        fresh = False
    else:
        if arm == 'spare':
            if not hasattr(run.old.base, 'SIM'):
                sim.initialize()
            rows = [sim.simulate_sample(sample, kv, 'arrival', rate)
                    for sample in cohort]
        else:
            assert weights is not None
            rows = [run.eval_one(sample, kv, rate, weights) for sample in cohort]
        rows.sort(key=lambda row: row['name'])
        record = dict(rows=rows, summary=run.old.prior.old.summary(rows, rate),
                      input_sha256=signature, checkpoint_sha256=checkpoint_sha,
                      windows=50)
        run.old.base.write(path, record)
        fresh = True
    assert len(record['rows']) == 50
    assert {row['name'] for row in record['rows']} == {s['name'] for s in cohort}
    assert record['summary'] == run.old.prior.old.summary(record['rows'], rate)
    return record['summary'], fresh


def group(task):
    workload, source, kv, arm, seed = task
    target = OUT / 'groups' / f'{workload}_{source}_c{kv}_{arm}_seed{seed}.json'
    if target.exists():
        result = run.old.base.read(target)
        assert result['stage'] == 'complete'
        return result
    cohort = cohort_for(workload, source)
    signature = source_hash(cohort)
    weights = run.selected_weights(seed) if arm == 'rl_mixed' else None
    passed_rate = 0.
    failed_rate = None
    checked = []
    for rate in COARSE:
        summary, _ = get_rate(workload, source, kv, arm, seed, rate,
                              cohort, signature, weights)
        checked.append(rate)
        if passed(summary, rate):
            passed_rate = rate
        else:
            failed_rate = rate
            break
    if failed_rate is None:
        raise RuntimeError(f'no failed rate through {COARSE[-1]} for {task}')
    while failed_rate - passed_rate > .0200001:
        rate = round((failed_rate + passed_rate) * 50) / 100
        if not passed_rate < rate < failed_rate:
            break
        summary, _ = get_rate(workload, source, kv, arm, seed, rate,
                              cohort, signature, weights)
        checked.append(rate)
        if passed(summary, rate):
            passed_rate = rate
        else:
            failed_rate = rate
    result = dict(stage='complete', workload=workload, source=source,
                  kv_blocks=kv, arm=arm, seed=seed,
                  capacity_qps=passed_rate, next_failed_qps=failed_rate,
                  resolution_qps=round(failed_rate - passed_rate, 4),
                  rates_checked=len(checked), windows=50,
                  input_sha256=signature)
    run.old.base.write(target, result)
    return result


def tasks(seeds, workloads, arms, kvs):
    result = []
    for workload in workloads:
        work_kvs = kvs if workload == 'original' else (2048,)
        sources = ('code', 'conversation') if workload == 'original' else ('conversation',)
        for kv in work_kvs:
            for source in sources:
                for arm in arms:
                    for seed in ((0,) if arm == 'spare' else seeds):
                        result.append((workload, source, kv, arm, seed))
    return result


def main(seeds, workloads, arms, kvs, workers):
    if any(workload != 'original' for workload in workloads):
        inputs.prepare()
    pending = tasks(seeds, workloads, arms, kvs)
    started = time.monotonic()
    failures = []
    done = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(group, task): task for task in pending}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                done.append(result)
                print(f'{len(done)}/{len(pending)} {task} '
                      f'capacity={result["capacity_qps"]:.3f}', flush=True)
            except Exception as exc:
                failures.append(dict(task=task, error=repr(exc)))
                print(f'FAILED {task}: {exc!r}', flush=True)
    status = dict(complete=not failures and len(done) == len(pending),
                  groups=len(done), expected=len(pending), failures=failures,
                  wall_s=time.monotonic()-started)
    run.old.base.write(OUT / 'sweep_status.json', status)
    if not status['complete']:
        raise RuntimeError(status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='1017,2027,3037')
    parser.add_argument('--workloads', default='original,A_short,B_balanced,C_long')
    parser.add_argument('--arms', default='spare,rl_mixed')
    parser.add_argument('--kvs', default=','.join(map(str, KVS)))
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    main(tuple(map(int, args.seeds.split(','))), tuple(args.workloads.split(',')),
         tuple(args.arms.split(',')), tuple(map(int, args.kvs.split(','))), args.workers)
