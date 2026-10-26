"""Frozen, paired TTFT/output-throughput pilot; native V1 CPU simulation."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/www_pareto_20261014_r1'
OLD = ROOT / 'artifacts/www_evaluation_completion_20261003_r1'
ARMS = ('arrival', 'srp', 'native', 'sarathi', 'past_future')
BUDGETS = (256, 512, 1024, 2048)
CAPACITIES = (768, 4096)
ANCHORS = {
    'code': ('2024-05-11 16:37', '2024-05-13 16:37', '2024-05-16 16:37'),
    'conversation': ('2024-05-13 16:37', '2024-05-15 16:37', '2024-05-18 16:37'),
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def prepare():
    target = OUT / 'design.json'
    if target.exists():
        return read(target)
    samples, histories, hashes = [], {}, {}
    fit = read(OLD / 'timing_profile.json')
    fit_ids = set(fit['training_request_ids'])
    all_ids = set()
    for source, anchors in ANCHORS.items():
        path = ROOT / f'data/azure_{"conv" if source == "conversation" else "code"}_1week.csv'
        hashes[str(path.relative_to(ROOT))] = digest(path)
        chosen = {a: [] for a in anchors}
        counts = (20, 25, 25)
        needed = {a: 512 + n * 128 for a, n in zip(anchors, counts)}
        with path.open(newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for index, row in enumerate(reader):
                for a in anchors:
                    if len(chosen[a]) >= needed[a] or row[0][:10] != a[:10] or row[0][:16] < a:
                        continue
                    prompt, output = int(row[1]), int(row[2])
                    if prompt > 0 and 0 < output <= 512 and prompt + output <= 8000:
                        chosen[a].append(dict(id=f'{source}:raw:{index}', prompt=prompt,
                                              output=output, raw_timestamp=row[0]))
                if all(len(chosen[a]) >= needed[a] for a in anchors):
                    break
        assert all(len(chosen[a]) == needed[a] for a in anchors)
        # Historical lengths precede validation and test requests in real time.
        histories[source] = chosen[anchors[0]][:512]
        for day, (a, n) in enumerate(zip(anchors, counts)):
            for k in range(n):
                raw = chosen[a][512 + k * 128:512 + k * 128 + 64]
                ids = {r['id'] for r in raw}
                assert len(ids) == 64 and not ids & (all_ids | fit_ids)
                all_ids.update(ids)
                name = f'{source}_d{day}_w{k:02d}'
                rows = [dict(**r, arrival_ms=i * 1000 / 5.25) for i, r in enumerate(raw)]
                dest = OUT / 'inputs' / (name + '.json')
                write(dest, rows)
                hashes[str(dest.relative_to(ROOT))] = digest(dest)
                samples.append(dict(name=name, source=source, day=a[:10], window=k,
                                    split='validation' if day == 0 else 'test',
                                    input=str(dest.relative_to(ROOT))))
    history_ids = {r['id'] for rows in histories.values() for r in rows}
    assert not history_ids & all_ids
    write(OUT / 'history.json', histories)
    write(OUT / 'timing_profile.json', fit)
    from research.www_evaluation_completion_20261003.simulate import SOURCE_FILES
    sources = SOURCE_FILES + ['research/www_evaluation_completion_20261003/timing.py',
                              'research/www_pareto_20261014/run.py']
    for rel in sources:
        hashes[rel] = digest(ROOT / rel)
        dst = OUT / 'source' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / rel).read_bytes())
    for p in (OUT / 'history.json', OUT / 'timing_profile.json'):
        hashes[str(p.relative_to(ROOT))] = digest(p)
    design = dict(samples=samples, arms=ARMS, budgets=BUDGETS, capacities=CAPACITIES,
                  chunk=256, concurrency=32, requests_per_window=64, arrival_rps=5.25,
                  expected_runs=len(samples) * len(ARMS) * len(BUDGETS) * len(CAPACITIES),
                  profile='retrospective GPU fitted timing, no added timing noise',
                  reference='native T1024 K256; paired per-window normalization',
                  selection='For each source/capacity and w=0,.1,...,1, minimize mean '
                            'paired normalized score on validation only; smallest budget breaks ties.',
                  inference='50 disjoint test segments per source from two dates; '
                            'uncertainty is conditional on these dates, not 50 independent days.',
                  baseline_scope='Native V1 and paper-rule adaptations on the same backend; '
                                 'not original full serving systems. Four budgets per family.',
                  request_overlap_validation_test=0, request_overlap_timing_fit=0,
                  performance_validated=False, hashes=hashes)
    write(target, design)
    return design


def initialize():
    global SIM, TIMING, HISTORY
    from research.www_evaluation_completion_20261003 import simulate as sim
    from research.www_evaluation_completion_20261003.timing import FittedTiming
    sim.digest = lru_cache(maxsize=None)(sim.digest)
    SIM = sim.simulate
    TIMING = FittedTiming(read(OUT / 'timing_profile.json'))
    HISTORY = {key: [r['output'] for r in rows] for key, rows in read(OUT / 'history.json').items()}


def run_sample(task):
    sample, capacity = task
    path = OUT / 'runs' / f"{sample['name']}_c{capacity}.json"
    if path.exists():
        saved = read(path)
        if len(saved) == len(ARMS) * len(BUDGETS) and all(r['complete'] for r in saved):
            return len(saved)
    requests = read(ROOT / sample['input'])
    seed = int.from_bytes(hashlib.sha256(sample['name'].encode()).digest()[:4], 'big')
    rows = []
    for arm in ARMS:
        for budget in BUDGETS:
            result = SIM(requests, arm=arm, capacity=capacity, budget=budget, chunk=256,
                         fitted_timing=TIMING, history=HISTORY[sample['source']], seed=seed,
                         output=None)
            m = result['metrics']
            # Check useful work, throughput denominator, and completion for every run.
            assert m['completed'] == m['requests'] == len(requests)
            expected = sum(r['output'] for r in requests) * 1000 / m['elapsed_simulated_ms']
            assert abs(m['output_throughput_tps'] - expected) < 1e-9
            rows.append(dict(**sample, capacity=capacity, arm=arm, budget=budget,
                             complete=True, seed=seed, metrics=m,
                             trajectory_sha256=result['trajectory_sha256']))
    write(path, rows)
    return len(rows)


def run(jobs):
    design = prepare()
    for rel, sha in design['hashes'].items():
        if digest(ROOT / rel) != sha:
            raise RuntimeError(f'Frozen file changed: {rel}')
    tasks = [(s, c) for s in design['samples'] for c in CAPACITIES]
    start, completed, failures = time.monotonic(), 0, []
    with ProcessPoolExecutor(max_workers=jobs, initializer=initialize) as pool:
        futures = {pool.submit(run_sample, task): (task[0]['name'], task[1]) for task in tasks}
        for future in as_completed(futures):
            try:
                completed += future.result()
            except Exception as exc:
                failures.append(dict(task=futures[future], error=repr(exc)))
            status = dict(completed=completed, expected=design['expected_runs'], failures=failures,
                          elapsed_s=time.monotonic() - start)
            write(OUT / 'status.json', status)
            print(json.dumps(status), flush=True)
    assert not failures and completed == design['expected_runs'], failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--jobs', type=int, default=12)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare()['expected_runs'])
    else:
        run(args.jobs)
