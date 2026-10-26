"""Collect 2-QPS passive behavior for the archived no-arrival ablation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

from research.www_component_ablation_mean_20261019 import behavior, complete, run as pilot


ROOT = complete.ROOT
OUT = ROOT / 'artifacts/www_arrival_behavior_20261020_r1'
READ, WRITE, DIGEST = complete.READ, complete.WRITE, complete.DIGEST
RATE = 2.


def prepare():
    samples = complete.prepare()
    design = dict(samples=samples, arm='no_arrival', rate_qps=RATE,
                  kv_blocks=pilot.KVS, sources=pilot.SOURCES,
                  windows_per_source=24, behavioral_interval_ms=[20000, 60000],
                  parent_design_sha256=DIGEST(complete.OUT / 'design.json'),
                  policy_sha256=DIGEST(Path(pilot.policy.__file__)),
                  observer_sha256=DIGEST(Path(behavior.__file__)),
                  runner_sha256=DIGEST(Path(__file__)),
                  gpu_execution=False)
    design = json.loads(json.dumps(design))
    path = OUT / 'design.json'
    if path.exists():
        assert READ(path) == design
    else:
        WRITE(path, design)
    return samples


def group(task):
    source, kv, samples = task
    path = OUT / 'rates' / pilot.old.rate_name(source, kv, 'no_arrival', RATE)
    if path.exists():
        return READ(path)
    checkpoint = OUT / 'checkpoints' / path.name
    rows = READ(checkpoint)['rows'] if checkpoint.exists() else []
    found = {r['name'] for r in rows}
    assert len(found) == len(rows) and found <= {s['name'] for s in samples}
    for sample in samples:
        if sample['name'] not in found:
            row = behavior.replay(sample, kv, 'no_arrival', RATE)
            assert row['parity_sources']
            rows.append(row)
            WRITE(checkpoint, dict(rows=rows))
    rows.sort(key=lambda r: r['name'])
    result = dict(rows=rows, summary=complete.confirm.summary(rows, RATE))
    WRITE(path, result)
    return result


def smoke():
    samples = prepare()
    pilot.initialize()
    sample = next(s for s in samples if s['source'] == 'code' and s['window'] == 2)
    row = behavior.replay(sample, 1024, 'no_arrival', RATE)
    assert row['parity_sources']
    assert row['counters']['finish_conflict_rounds'] == 0
    WRITE(OUT / 'smoke.json', dict(passed=True, sample=sample['name'],
                                   trajectory_sha256=row['trajectory_sha256']))
    print('No-arrival behavior smoke passed', flush=True)


def run(workers):
    samples = prepare()
    assert READ(OUT / 'smoke.json')['passed']
    tasks = [(source, kv, [s for s in samples if s['source'] == source])
             for source in pilot.SOURCES for kv in pilot.KVS]
    pending = [t for t in tasks if not (OUT / 'rates' /
               pilot.old.rate_name(t[0], t[1], 'no_arrival', RATE)).exists()]
    done, errors, start = len(tasks)-len(pending), [], time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=pilot.initialize) as pool:
        futures = {pool.submit(group, t): t[:2] for t in pending}
        for future in as_completed(futures):
            try:
                future.result()
                done += 1
            except Exception as exc:
                errors.append(dict(task=futures[future], error=repr(exc)))
            WRITE(OUT / 'status.json', dict(completed=done, total=len(tasks),
                                            errors=errors, elapsed_s=round(time.monotonic()-start, 1)))
            print(dict(completed=done, total=len(tasks), errors=errors), flush=True)
    if errors:
        raise RuntimeError(errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    smoke() if args.smoke else run(args.workers)
