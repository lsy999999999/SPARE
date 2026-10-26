"""Frozen seed-7047 capacity curves, reusing audited paired-window records."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time

from research.www_rl_standalone_v2_20261019 import sweep, train

ROOT = train.run.ROOT
OUT = ROOT / 'artifacts/www_rl7047_frontier_20261020_r1'
ORIGINAL = train.OUT
ABLATION = ROOT / 'artifacts/www_biobjective_ablation_mean_20261019_r2'
SEED = 7047
LIMITS_MS = (250, 300, 350, 400, 500, 600, 750, 1000, 1250, 1500,
             1750, 2000, 2500, 3000, 4000, 6000, 8000)
CELLS = tuple((source, kv, arm) for source in ('conversation', 'code')
              for kv in (2048, 4096) for arm in ('spare', 'rl_standalone'))
METRICS = ('valid', 'mean_ttft_ms', 'p99_tbt_ms', 'backlog_drift',
           'measured_requests', 'completed_by_end', 'preemptions', 'recomputed_tokens')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_rows(left, right):
    a = {r['name']: r for r in left}
    b = {r['name']: r for r in right}
    assert a.keys() == b.keys()
    for name in a:
        for key in METRICS:
            if key in a[name] or key in b[name]:
                x, y = a[name].get(key), b[name].get(key)
                assert x == y or (isinstance(x, (int, float)) and
                                 isinstance(y, (int, float)) and
                                 math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-7)), (name, key, x, y)


def resources(summary, rate):
    return (summary['failed_samples'] == 0 and summary['p99_tbt_ms'] <= 200
            and summary['backlog_drift'] <= max(2, rate))


def passes(summary, rate, limit):
    return resources(summary, rate) and summary['mean_ttft_ms'] <= limit


def reasons(summary, rate, limit):
    return [name for name, failed in (
        ('mean_ttft', summary['mean_ttft_ms'] > limit),
        ('p99_tbt', summary['p99_tbt_ms'] > 200),
        ('backlog_drift', summary['backlog_drift'] > max(2, rate)),
        ('invalid_window', summary['failed_samples'] > 0)) if failed]


def window_job(sample, kv, arm, rate):
    if arm == 'spare':
        if not hasattr(train.run.old.base, 'SIM'):
            sweep.sim.initialize()
        return sweep.sim.simulate_sample(sample, kv, 'arrival', rate)
    weights = sweep.selected_weights(SEED)
    outcome = train.eval_one(sample, kv, rate, weights)
    return dict(source=sample['source'], name=sample['name'], day=sample['day'],
                capacity=kv, arm=arm, budget=1024, rate=rate, **outcome)


def prepare():
    selected = read(ORIGINAL / 'checkpoints/seed7047_status.json')['selected']
    assert selected['episode'] == 256
    assert digest(ORIGINAL / selected['path']) == selected['sha256']
    design = read(ABLATION / 'design.json')
    assert design['timing_sha256'] == digest(train.run.old.base.PREVIOUS / 'timing_profile.json')
    banks, cohorts, signatures, audit = {}, {}, {}, []
    for source in ('conversation', 'code'):
        cohorts[source] = sweep.cohort_for(source)
        signatures[source] = hashlib.sha256(''.join(
            digest(ROOT / s['input']) for s in cohorts[source]).encode()).hexdigest()
        assert {(s['name'], s['input']) for s in cohorts[source]} == {
            (s['name'], s['input']) for s in design['samples'] if s['source'] == source}
    for source, kv, arm in CELLS:
        bank = {}
        pattern = f'{source}_c{kv}_{arm}_seed{SEED}_r*.json'
        for path in (ORIGINAL / 'rates').glob(pattern):
            rate = float(path.stem.split('_r')[-1])
            data = read(path)
            assert data['input_sha256'] == signatures[source]
            assert data['checkpoint_sha256'] == (selected['sha256'] if arm != 'spare' else None)
            bank[rate] = path
        overlaps = 0
        if arm == 'spare':
            for path in (ABLATION / 'rates').glob(f'{source}_c{kv}_spare_r*.json'):
                rate = float(path.stem.split('_r')[-1])
                if rate in bank:
                    same_rows(read(bank[rate])['rows'], read(path)['rows'])
                    overlaps += 1
                else:
                    bank[rate] = path
            assert overlaps >= 5
        banks[source, kv, arm] = bank
        audit.append(dict(source=source, kv_blocks=kv, arm=arm,
                          available_rates=len(bank), ablation_overlap_rates=overlaps))
    write(OUT / 'design.json', dict(seed=SEED, selected=selected, limits_ms=LIMITS_MS,
          minimum_qps=.25, resolution_qps=.02, cells=CELLS, windows_per_source=50,
          input_sha256=signatures, timing_sha256=design['timing_sha256'],
          capacity_rule='ascending measured QPS through first failure; mean TTFT <= limit; '
                        'mean window P99 TBT <=200 ms; mean backlog drift <=max(2,QPS); all valid',
          seed_selection='highest Conversation/KV2048 test capacity among three runs; '
                         'checkpoint selected on validation windows',
          reuse_audit=audit, simulator_only=True))
    return banks, cohorts, signatures, selected


def scan(cell, pool, bank, cohort, signature, selected):
    source, kv, arm = cell
    name = f'{source}_c{kv}_{arm}_seed{SEED}'
    cache, files, reused, fresh = {}, {}, 0, 0

    def evaluate(rate):
        nonlocal reused, fresh
        rate = round(rate, 2)
        if rate in cache:
            return cache[rate]
        target = OUT / 'rates' / f'{name}_r{rate:g}.json'
        if target.exists():
            data = read(target)
        elif rate in bank:
            origin = bank[rate]
            data = dict(rows=read(origin)['rows'], reused_from=str(origin.relative_to(ROOT)),
                        reused_sha256=digest(origin), input_sha256=signature,
                        checkpoint_sha256=selected['sha256'] if arm != 'spare' else None)
            data['summary'] = train.run.old.prior.old.summary(data['rows'], rate)
            write(target, data)
        else:
            jobs = [pool.submit(window_job, s, kv, arm, rate) for s in cohort]
            rows = sorted((j.result() for j in jobs), key=lambda row: row['name'])
            data = dict(rows=rows, summary=train.run.old.prior.old.summary(rows, rate),
                        input_sha256=signature,
                        checkpoint_sha256=selected['sha256'] if arm != 'spare' else None,
                        newly_simulated=50)
            write(target, data)
            print(f'NEW {name} QPS={rate:g} TTFT={data["summary"]["mean_ttft_ms"]:.1f}', flush=True)
        assert data['input_sha256'] == signature
        assert data['checkpoint_sha256'] == (selected['sha256'] if arm != 'spare' else None)
        assert len(data['rows']) == 50
        assert {r['name'] for r in data['rows']} == {s['name'] for s in cohort}
        assert all(r['capacity'] == kv and r['rate'] == rate for r in data['rows'])
        assert data['summary'] == train.run.old.prior.old.summary(data['rows'], rate)
        reused += int('reused_from' in data)
        fresh += int('newly_simulated' in data)
        cache[rate], files[rate] = data['summary'], str(target.relative_to(ROOT))
        return cache[rate]

    # Load all available observations so local nonmonotonicities are not hidden.
    for rate in sorted(bank):
        evaluate(rate)
    for path in (OUT / 'rates').glob(f'{name}_r*.json'):
        evaluate(float(path.stem.split('_r')[-1]))

    def bracket(predicate):
        passed = None
        for rate in sorted(cache):
            if predicate(cache[rate], rate):
                passed = rate
            else:
                return passed, rate
        return passed, None

    def refine(low, high, predicate):
        if low is None:
            return low, high
        while high - low > .0200001:
            mid = round((low + high) * 50) / 100
            if predicate(evaluate(mid), mid):
                low = mid
            else:
                high = mid
        return low, high

    # Add coarse observations until a non-TTFT constraint fails.
    low, high = bracket(resources)
    while high is None:
        rate = round(max(cache) + .5, 2)
        assert rate <= 16, 'No resource failure by 16 QPS'
        evaluate(rate)
        low, high = bracket(resources)
    resource_low, resource_high = refine(low, high, resources)
    # Revisit earlier limits after all refinements; enforce first-failure semantics
    # against the final union of observed QPS, including nonmonotonic observations.
    while True:
        before = len(cache)
        points = []
        for limit in LIMITS_MS:
            predicate = lambda s, r: passes(s, r, limit)
            low, high = bracket(predicate)
            assert high is not None
            low, high = refine(low, high, predicate)
            at = cache[low] if low is not None else None
            points.append(dict(mean_ttft_limit_ms=limit, capacity_qps=low,
                next_failed_qps=high, resolution_qps=round(high-low, 4) if low is not None else None,
                below_minimum_measured_rate=low is None,
                attained_mean_ttft_ms=at['mean_ttft_ms'] if at else None,
                first_failure_reasons=reasons(cache[high], high, limit)))
        if len(cache) == before:
            break
    resource_low, resource_high = bracket(resources)
    result = dict(stage='complete', source=source, kv_blocks=kv, arm=arm, seed=SEED,
        input_sha256=signature, checkpoint_sha256=selected['sha256'] if arm != 'spare' else None,
        windows=50, minimum_qps=.25, thresholds=points,
        resource_last_pass_qps=resource_low, resource_first_fail_qps=resource_high,
        reused_rates=reused, new_rates=fresh, rate_records=len(cache),
        records=[dict(rate_qps=r, path=files[r], sha256=digest(ROOT/files[r]), summary=cache[r])
                 for r in sorted(cache)])
    write(OUT / 'groups' / f'{name}.json', result)
    print(f'DONE {name}: reused={reused} new={fresh} resource={resource_low}', flush=True)
    return result


def main(workers):
    started = time.monotonic()
    banks, cohorts, signatures, selected = prepare()
    # A current-code replay ensures archived trajectories still match execution.
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        parity = []
        jobs = [(cell, pool.submit(window_job, cohorts[cell[0]][0], cell[1], cell[2], 2.))
                for cell in CELLS]
        for cell, job in jobs:
            row = job.result()
            old = next(r for r in read(banks[cell][2.])['rows'] if r['name'] == row['name'])
            same_rows([row], [old])
            parity.append(dict(cell=cell, sample=row['name'], rate_qps=2., passed=True))
        write(OUT / 'current_code_parity.json', parity)
        print('Current-code parity: 8/8 passed', flush=True)
        done = []
        with ThreadPoolExecutor(max_workers=8) as threads:
            futures = {threads.submit(scan, cell, pool, banks[cell], cohorts[cell[0]],
                                      signatures[cell[0]], selected): cell for cell in CELLS}
            for future in as_completed(futures):
                result = future.result()
                done.append(result)
                write(OUT / 'status.json', dict(complete=len(done)==len(CELLS),
                    groups=len(done), expected=len(CELLS), wall_s=time.monotonic()-started,
                    new_rates=sum(g['new_rates'] for g in done),
                    reused_rates=sum(g['reused_rates'] for g in done)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    main(args.workers)
