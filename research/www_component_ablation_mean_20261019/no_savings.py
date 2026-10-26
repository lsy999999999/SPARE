"""Matched 24-window scans removing the entire candidate savings condition."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

from research.www_component_ablation_mean_20261019 import complete, confirm, behavior, run as pilot
from research.www_component_ablation_mean_20261019 import no_savings_policy, no_savings_observer

ROOT = complete.ROOT
OUT = ROOT / 'artifacts/www_no_savings_condition_20261020_r1'
BEHAVIOR_OUT = OUT / 'behavior'
ARMS = ('spare', 'no_savings_condition', 'constant1')
OLD_BEHAVIOR = behavior.OUT
READ, WRITE, DIGEST = complete.READ, complete.WRITE, complete.DIGEST


def prepare():
    samples = complete.prepare()
    design = dict(samples=samples, arms=ARMS, sources=pilot.SOURCES, kv_blocks=pilot.KVS,
                  theta_ms=pilot.THETA_MS, windows_per_source=24,
                  parent_design_sha256=DIGEST(complete.OUT / 'design.json'),
                  policy_sha256=DIGEST(Path(no_savings_policy.__file__)),
                  original_policy_sha256=DIGEST(Path(pilot.policy.__file__)),
                  observer_sha256=DIGEST(Path(no_savings_observer.__file__)),
                  runner_sha256=DIGEST(Path(__file__)),
                  intervention='Delete the entire acceptance_saving >= n_t condition; '
                  'zero and negative savings cannot reject a candidate',
                  unchanged='Ordering, candidate set/order, feasibility, both output guards, '
                  'base.growth > 0 and len(rows) >= 2 triggers, engine and timing',
                  capacity_rule='all 24 valid; mean TTFT <= theta; mean window P99 TBT <=200 ms; '
                  'mean backlog drift <=max(2,QPS); continuous pass through first failure',
                  refinement_qps=.02, behavior_rates=behavior.RATES, gpu_execution=False)
    normalized = json.loads(json.dumps(design))
    for directory in (OUT, BEHAVIOR_OUT):
        path = directory / 'design.json'
        if path.exists():
            assert READ(path) == normalized, 'Frozen no-savings design changed'
        else:
            WRITE(path, normalized)
    return samples


def configure():
    confirm.OUT = OUT
    confirm.ARMS = ARMS
    confirm.get_rate = get_rate


def initialize():
    configure()
    pilot.initialize()
    pilot.old.prior.ARMS['no_savings_condition'] = ('arrival', {'disable_saving_condition': True})
    from research.www_evaluation_completion_20261003 import scheduler
    scheduler.select = no_savings_policy.select
    behavior.Observer = no_savings_observer.Observer


def get_rate(source, kv, arm, rate, samples):
    path = OUT / 'rates' / pilot.old.rate_name(source, kv, arm, rate)
    if path.exists():
        data = READ(path)
    else:
        old = complete.OUT / 'rates' / path.name
        if old.exists():
            raw = READ(old)
            data = dict(rows=raw['rows'], reused_from=str(old.relative_to(ROOT)),
                        reused_sha256=DIGEST(old), new_simulations=0,
                        summary=confirm.summary(raw['rows'], rate))
        else:
            checkpoint = OUT / 'checkpoints' / path.name
            rows = READ(checkpoint)['rows'] if checkpoint.exists() else []
            found = {r['name'] for r in rows}
            assert len(found) == len(rows)
            for sample in samples:
                if sample['name'] not in found:
                    rows.append(pilot.old.prior.simulate_one(sample, kv, arm, rate))
                    WRITE(checkpoint, dict(rows=rows))
            rows.sort(key=lambda r: r['name'])
            data = dict(rows=rows, new_simulations=len(rows), reused_from=None,
                        summary=confirm.summary(rows, rate))
        WRITE(path, data)
    assert {r['name'] for r in data['rows']} == {s['name'] for s in samples}
    assert data['summary'] == confirm.summary(data['rows'], rate)
    return data['summary']


def capacity_group(task):
    source, kv, arm, samples = task
    prior = complete.OUT / 'groups' / f'{source}_c{kv}_{arm}.json'
    if prior.exists():
        result = READ(prior)
        for rate in result['rates']:
            get_rate(source, kv, arm, rate, samples)
        result = dict(result, reused_from=str(prior.relative_to(ROOT)), reused_sha256=DIGEST(prior))
        WRITE(OUT / 'groups' / prior.name, result)
        return result
    return confirm.run_group(task)


def behavior_group(task):
    source, kv, arm, rate, samples = task
    path = BEHAVIOR_OUT / 'rates' / pilot.old.rate_name(source, kv, arm, rate)
    if path.exists():
        return READ(path)
    old = OLD_BEHAVIOR / 'rates' / path.name
    if old.exists():
        data = dict(READ(old), reused_from=str(old.relative_to(ROOT)), reused_sha256=DIGEST(old))
    else:
        checkpoint = BEHAVIOR_OUT / 'checkpoints' / path.name
        rows = READ(checkpoint)['rows'] if checkpoint.exists() else []
        found = {r['name'] for r in rows}
        for sample in samples:
            if sample['name'] not in found:
                rows.append(behavior.replay(sample, kv, arm, rate))
                WRITE(checkpoint, dict(rows=rows))
        rows.sort(key=lambda r: r['name'])
        data = dict(rows=rows, summary=confirm.summary(rows, rate))
    WRITE(path, data)
    return data


def smoke():
    samples = prepare()
    initialize()
    checks = []
    for source, kv in (('code', 1024), ('conversation', 2048)):
        sample = next(s for s in samples if s['source'] == source and s['window'] == 2)
        for arm in ARMS:
            audited = behavior.replay(sample, kv, arm, 2.)
            plain = pilot.old.prior.simulate_one(sample, kv, arm, 2.)
            assert plain['trajectory_sha256'] == audited['trajectory_sha256']
            assert plain['mean_ttft_ms'] == audited['mean_ttft_ms']
            if arm != 'no_savings_condition':
                assert audited['parity_sources']
            checks.append(audited)
    assert sum(r['counters'].get('selected_zero_saving_rounds', 0) for r in checks
               if r['arm'] == 'no_savings_condition') > 0
    WRITE(OUT / 'smoke.json', dict(passed=True, checks=checks))
    print('No-savings smoke: old arms reproduce, passive audit agrees, zero-savings actions accepted', flush=True)


def run(workers):
    samples = prepare()
    assert READ(OUT / 'smoke.json')['passed']
    capacity_tasks = [(source, kv, arm, [s for s in samples if s['source'] == source])
                      for source in pilot.SOURCES for kv in pilot.KVS for arm in ARMS]
    behavior_tasks = [(source, kv, arm, rate, [s for s in samples if s['source'] == source])
                     for rate in behavior.RATES for source in pilot.SOURCES for kv in pilot.KVS for arm in ARMS]
    tasks = []
    completed = {'capacity': 0, 'behavior': 0}
    for task in capacity_tasks:
        if (OUT / 'groups' / f'{task[0]}_c{task[1]}_{task[2]}.json').exists():
            completed['capacity'] += 1
        else:
            tasks.append(('capacity', task))
    for task in behavior_tasks:
        if (BEHAVIOR_OUT / 'rates' / pilot.old.rate_name(*task[:4])).exists():
            completed['behavior'] += 1
        else:
            tasks.append(('behavior', task))
    start, errors = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        futures = {pool.submit(capacity_group if kind == 'capacity' else behavior_group, task):
                   (kind, task[:4]) for kind, task in tasks}
        for f in as_completed(futures):
            kind, task = futures[f]
            try:
                f.result()
                completed[kind] += 1
            except Exception as exc:
                errors.append(dict(kind=kind, task=task[:3], error=repr(exc)))
            for directory, category, total in ((OUT, 'capacity', 18), (BEHAVIOR_OUT, 'behavior', 72)):
                WRITE(directory / 'status.json', dict(completed=completed[category], total=total,
                      errors=[e for e in errors if e['kind'] == category],
                      elapsed_s=round(time.monotonic()-start, 1)))
            if sum(completed.values()) % 6 == 0 or errors:
                print(dict(completed=completed, errors=errors), flush=True)
    if errors:
        raise RuntimeError(errors)


def analyze():
    configure()
    confirm.prepare = prepare
    from research.www_component_ablation_mean_20261019 import confirm_analyze, behavior_analyze
    confirm_analyze.analyze()
    behavior.OUT = BEHAVIOR_OUT
    behavior.ARMS = ARMS
    behavior_analyze.OUT = BEHAVIOR_OUT / 'analysis'
    behavior_analyze.analyze(write_report=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if args.smoke:
        smoke()
    elif args.analyze:
        analyze()
    else:
        run(args.workers)
