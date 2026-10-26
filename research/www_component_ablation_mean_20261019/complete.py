"""Complete the frozen 24-window ablation grid in a new archive."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

from research.www_component_ablation_mean_20261019 import confirm, run as pilot

ROOT = pilot.ROOT
PREVIOUS = ROOT / 'artifacts/www_component_ablation_mean_20261019_r3'
OUT = ROOT / 'artifacts/www_component_ablation_mean_20261020_r4'
ARMS = confirm.ARMS + ('drop_candidate2', 'drop_candidate3', 'drop_candidate4')
READ = pilot.old.prior.base.read
WRITE = pilot.old.prior.base.write
DIGEST = pilot.old.prior.base.digest


def prepare():
    frozen = pilot.old.prepare()
    previous = READ(PREVIOUS / 'design.json')
    samples = previous['samples']
    assert len(samples) == 48
    assert {s['name'] for s in samples} <= {s['name'] for s in frozen['samples']}
    assert DIGEST(Path(pilot.policy.__file__)) == previous['policy_sha256']
    design = dict(previous, arms=ARMS, previous_design_sha256=DIGEST(PREVIOUS / 'design.json'),
                  runner_sha256=DIGEST(Path(__file__)),
                  timing_sha256=frozen['timing_sha256'],
                  input_sha256={s['input']: DIGEST(ROOT / s['input']) for s in samples},
                  reuse='Same 24 windows and frozen policy; copy existing paired rows; '
                        'simulate missing windows/rates only')
    normalized = json.loads(json.dumps(design))
    path = OUT / 'design.json'
    if path.exists():
        assert READ(path) == normalized, 'Frozen completion design changed'
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


def get_rate(source, kv, arm, rate, samples):
    target = OUT / 'rates' / pilot.old.rate_name(source, kv, arm, rate)
    wanted = {s['name'] for s in samples}
    if target.exists():
        data = READ(target)
    else:
        cached = PREVIOUS / 'rates' / target.name
        if not cached.exists():
            cached = confirm.archived_path(source, kv, arm, rate)
        checkpoint = OUT / 'checkpoints' / target.name
        if checkpoint.exists():
            rows = READ(checkpoint)['rows']
        else:
            rows = [] if cached is None else [r for r in READ(cached)['rows']
                                              if r['name'] in wanted]
        found = {r['name'] for r in rows}
        reused = len(rows)
        assert len(rows) == len(found) and found <= wanted
        for sample in samples:
            if sample['name'] not in found:
                rows.append(pilot.old.prior.simulate_one(sample, kv, arm, rate))
                WRITE(checkpoint, dict(rows=rows))
        rows.sort(key=lambda r: r['name'])
        data = dict(rows=rows, reused_from=str(cached.relative_to(ROOT)) if cached else None,
                    reused_sha256=DIGEST(cached) if cached else None,
                    newly_simulated_this_attempt=len(rows)-reused,
                    summary=confirm.summary(rows, rate))
        WRITE(target, data)
    assert {r['name'] for r in data['rows']} == wanted
    assert data['summary'] == confirm.summary(data['rows'], rate)
    return data['summary']


def group(task):
    source, kv, arm, samples = task
    prior_group = PREVIOUS / 'groups' / f'{source}_c{kv}_{arm}.json'
    if prior_group.exists():
        result = READ(prior_group)
        for rate in result['rates']:
            get_rate(source, kv, arm, rate, samples)
        result = dict(result, reused_from=str(prior_group.relative_to(ROOT)),
                      reused_sha256=DIGEST(prior_group))
        WRITE(OUT / 'groups' / prior_group.name, result)
        return result
    return confirm.run_group(task)


def smoke():
    samples = prepare()
    initialize()
    results = []
    for source, kv, arm, rate in (
            ('code', 1024, 'spare', 2.),
            ('conversation', 2048, 'drop_candidate2', 2.),
            ('code', 4096, 'drop_candidate3', 2.),
            ('conversation', 1024, 'drop_candidate4', 2.)):
        sample = next(s for s in samples if s['source'] == source and s['window'] == 2)
        cached = pilot.rate_path(source, kv, arm, rate)
        reference = next(r for r in READ(cached)['rows'] if r['name'] == sample['name'])
        current = pilot.old.prior.simulate_one(sample, kv, arm, rate)
        assert current['valid'] and current['trajectory_sha256'] == reference['trajectory_sha256']
        results.append(dict(source=source, kv=kv, arm=arm, name=sample['name'], parity=True))
    WRITE(OUT / 'smoke.json', dict(passed=True, checks=results))
    print('Completion smoke: four archived trajectories reproduced', flush=True)


def run(workers):
    samples = prepare()
    configure()
    assert READ(OUT / 'smoke.json')['passed']
    tasks = [(source, kv, arm, [s for s in samples if s['source'] == source])
             for source in pilot.SOURCES for kv in pilot.KVS for arm in ARMS]
    pending = [t for t in tasks if not (OUT / 'groups' / f'{t[0]}_c{t[1]}_{t[2]}.json').exists()]
    done, errors, start = len(tasks)-len(pending), [], time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        futures = {pool.submit(group, t): t[:3] for t in pending}
        for f in as_completed(futures):
            try:
                f.result()
                done += 1
            except Exception as exc:
                errors.append(dict(task=futures[f], error=repr(exc)))
            status = dict(completed=done, total=len(tasks), errors=errors,
                          elapsed_s=round(time.monotonic()-start, 1))
            WRITE(OUT / 'status.json', status)
            if done % 6 == 0 or errors:
                print(status, flush=True)
    if errors:
        raise RuntimeError(errors)


def analyze():
    configure()
    confirm.prepare = prepare
    from research.www_component_ablation_mean_20261019 import confirm_analyze
    confirm_analyze.analyze()


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
