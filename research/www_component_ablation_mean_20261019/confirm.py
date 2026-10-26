"""Confirm decisive component contrasts on 24 paired windows per source."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

from research.www_component_ablation_mean_20261019 import run as pilot

ROOT = pilot.ROOT
OUT = ROOT / 'artifacts/www_component_ablation_mean_20261019_r3'
WINDOWS = tuple(range(0, 24, 2))
ARMS = ('spare', 'srp', 'constant1', 'constant8', 'constant16',
        'no_arrival', 'budget_only', 'chunk_family', 'drop_candidate1',
        'no_finish_guard', 'no_decode_guard')


def prepare():
    frozen = pilot.old.prepare()
    samples = [s for s in frozen['samples'] if s['window'] in WINDOWS]
    assert len(samples) == 48
    assert all(sum(s['source'] == source for s in samples) == 24
               for source in pilot.SOURCES)
    design = dict(samples=samples, windows=WINDOWS, arms=ARMS,
                  sources=pilot.SOURCES, kv_blocks=pilot.KVS,
                  theta_ms=pilot.THETA_MS, windows_per_source=24,
                  capacity_rule='all 24 valid; mean TTFT <= theta; '
                                'mean P99 TBT <= 200 ms; mean backlog drift '
                                '<= max(2,QPS); continuous pass through first failure',
                  refinement_qps=.02, gpu_execution=False,
                  pilot_design_sha256=pilot.old.prior.base.digest(pilot.OUT / 'design.json'),
                  policy_sha256=pilot.old.prior.base.digest(Path(pilot.policy.__file__)),
                  runner_sha256=pilot.old.prior.base.digest(Path(__file__)))
    path = OUT / 'design.json'
    if path.exists():
        assert pilot.old.prior.base.read(path) == json.loads(json.dumps(design))
    else:
        pilot.old.prior.base.write(path, design)
    return samples


def rate_path(source, kv, arm, rate):
    return OUT / 'rates' / pilot.old.rate_name(source, kv, arm, rate)


def summary(rows, rate):
    assert len(rows) == 24 and len({r['name'] for r in rows}) == 24
    valid = [r for r in rows if r['valid']]
    result = {key: sum(r[key] for r in valid)/len(valid) if valid else None
              for key in ('mean_ttft_ms', 'p99_tbt_ms', 'backlog_drift',
                          'preemptions', 'recomputed_tokens')}
    result['failed_samples'] = len(rows)-len(valid)
    result['drift_limit'] = max(2, rate)
    return result


def archived_path(source, kv, arm, rate):
    name = pilot.old.rate_name(source, kv, arm, rate)
    if arm in pilot.EXISTING:
        path = pilot.old.rate_path(source, kv, arm, rate)
        if path.exists():
            return path
        path = pilot.old.cached_record(source, kv, arm, rate)
        if path is not None:
            return path
    path = pilot.OUT / 'rates' / name
    return path if path.exists() else None


def get_rate(source, kv, arm, rate, samples):
    target = rate_path(source, kv, arm, rate)
    wanted = {s['name'] for s in samples}
    if target.exists():
        data = pilot.old.prior.base.read(target)
    else:
        source_path = archived_path(source, kv, arm, rate)
        rows = [] if source_path is None else [r for r in
               pilot.old.prior.base.read(source_path)['rows'] if r['name'] in wanted]
        found = {r['name'] for r in rows}
        assert len(rows) == len(found) and found <= wanted
        for sample in samples:
            if sample['name'] not in found:
                rows.append(pilot.old.prior.simulate_one(sample, kv, arm, rate))
        rows.sort(key=lambda row: row['name'])
        data = dict(rows=rows, reused_from=(str(source_path.relative_to(ROOT))
                    if source_path is not None else None),
                    newly_simulated=len(rows)-len(found))
        data['summary'] = summary(rows, rate)
        pilot.old.prior.base.write(target, data)
    assert {r['name'] for r in data['rows']} == wanted
    assert data['summary'] == summary(data['rows'], rate)
    return data['summary']


def run_group(task):
    source, kv, arm, samples = task
    target = OUT / 'groups' / f'{source}_c{kv}_{arm}.json'
    if target.exists():
        return pilot.old.prior.base.read(target)
    scans = {}
    for rate in sorted(set(pilot.old.main.COARSE_RATES) | {.5}):
        scans[rate] = get_rate(source, kv, arm, rate, samples)
        if not pilot.old.resource_ok(scans[rate], rate):
            break
    while pilot.old.resource_ok(scans[max(scans)], max(scans)):
        rate = 2 * max(scans)
        if rate > 64:
            raise RuntimeError(f'No resource failure: {source}/{kv}/{arm}')
        scans[rate] = get_rate(source, kv, arm, rate, samples)
    for theta in pilot.THETA_MS:
        low, high, _ = pilot.old.bounds(scans, theta)
        assert high is not None
        while high-low > .0200001:
            rate = round((low+high)*50)/100
            if not low < rate < high:
                break
            scans[rate] = get_rate(source, kv, arm, rate, samples)
            if (pilot.old.resource_ok(scans[rate], rate) and
                    scans[rate]['mean_ttft_ms'] <= theta):
                low = rate
            else:
                high = rate
    capacities = {}
    for theta in pilot.THETA_MS:
        low, high, cause = pilot.old.bounds(scans, theta)
        capacities[f'{theta:g}'] = dict(last_pass_qps=low, first_fail_qps=high,
                                        resolution_qps=round(high-low, 4),
                                        failure_causes=cause)
    result = dict(source=source, kv_blocks=kv, arm=arm, capacities=capacities,
                  rates=sorted(scans), completed_at=time.time())
    pilot.old.prior.base.write(target, result)
    return result


def smoke():
    samples = prepare()
    pilot.initialize()
    for source, kv, rate in (('code', 1024, 2.), ('conversation', 2048, 3.)):
        sample = next(s for s in samples if s['source'] == source and
                      s['window'] in pilot.WINDOWS)
        old = pilot.old.prior.base.read(pilot.rate_path(source, kv, 'spare', rate))
        archived = next(r for r in old['rows'] if r['name'] == sample['name'])
        current = pilot.old.prior.simulate_one(sample, kv, 'spare', rate)
        assert archived['trajectory_sha256'] == current['trajectory_sha256']
    pilot.old.prior.base.write(OUT / 'smoke.json', dict(parity=True))
    print('24-window parity passed')


def run(workers):
    samples = prepare()
    assert pilot.old.prior.base.read(OUT / 'smoke.json')['parity']
    tasks = [(source, kv, arm, [s for s in samples if s['source'] == source])
             for source in pilot.SOURCES for kv in pilot.KVS for arm in ARMS]
    pending = [task for task in tasks if not (OUT / 'groups' /
               f'{task[0]}_c{task[1]}_{task[2]}.json').exists()]
    done, errors = len(tasks)-len(pending), []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=pilot.initialize) as pool:
        futures = {pool.submit(run_group, task): task[:3] for task in pending}
        for future in as_completed(futures):
            try:
                future.result()
                done += 1
            except Exception as exc:
                errors.append(dict(task=futures[future], error=repr(exc)))
            if done % 6 == 0 or errors:
                status = dict(completed=done, total=len(tasks), errors=errors,
                              elapsed_s=round(time.monotonic()-started, 1))
                pilot.old.prior.base.write(OUT / 'status.json', status)
                print(status, flush=True)
    status = dict(completed=done, total=len(tasks), errors=errors,
                  elapsed_s=round(time.monotonic()-started, 1))
    pilot.old.prior.base.write(OUT / 'status.json', status)
    if errors:
        raise RuntimeError(status)
    print(status, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()
    smoke() if args.smoke else run(args.workers)
