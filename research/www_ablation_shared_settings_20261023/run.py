"""Supplement three main-figure KV settings using the frozen n=8 ablation."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from research.www_component_ablation_mean_20261019 import confirm, run as pilot

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ROOT / 'artifacts/www_component_ablation_mean_20261020_r4'
OLD = ROOT / 'artifacts/www_biobjective_ablation_20261016_r1'
OUT = ROOT / 'artifacts/www_ablation_shared_settings_20261023_r1'
KVS = (3072, 3584, 5120)
ARMS = ('spare', 'srp', 'constant8', 'no_arrival')
THETAS = (2000., 4000.)
READ = pilot.old.prior.base.read
WRITE = pilot.old.prior.base.write
DIGEST = pilot.old.prior.base.digest


def prepare():
    ref = READ(PRIMARY / 'design.json')
    samples = [s for s in ref['samples'] if s['source'] == 'conversation']
    assert len(samples) == 24
    assert DIGEST(Path(pilot.policy.__file__)) == ref['policy_sha256']
    assert DIGEST(ROOT / 'artifacts/www_pareto_20261014_r1/timing_profile.json') == ref['timing_sha256']
    for sample in samples:
        assert DIGEST(ROOT / sample['input']) == ref['input_sha256'][sample['input']]
    for archive in (OLD, pilot.old.main.OUT, pilot.old.main.FINE, pilot.old.main.ARCHIVE):
        prior = READ(archive / 'design.json')
        assert prior['timing_sha256'] == ref['timing_sha256']
        assert {(s['name'], s['input']) for s in samples} <= {
            (s['name'], s['input']) for s in prior['samples']}
        for sample in samples:
            if 'input_sha256' in prior:
                assert prior['input_sha256'][sample['input']] == ref['input_sha256'][sample['input']]
    design = dict(samples=samples, source='conversation', kv_blocks=KVS, arms=ARMS,
                  theta_ms=THETAS, windows_per_source=24, gpu_execution=False,
                  policy_sha256=ref['policy_sha256'], timing_sha256=ref['timing_sha256'],
                  input_sha256={s['input']: ref['input_sha256'][s['input']] for s in samples},
                  reference_design_sha256=DIGEST(PRIMARY / 'design.json'),
                  capacity_rule=ref['capacity_rule'], refinement_qps=.02,
                  matched_qps=(2., 3., 4.),
                  selection='Three additional main-figure settings spanning intermediate and larger KV; '
                            'compare all four arms without changing n0=8; preserve every result.')
    design = json.loads(json.dumps(design))
    if (OUT / 'design.json').exists():
        assert READ(OUT / 'design.json') == design
    else:
        WRITE(OUT / 'design.json', design)
    return samples


def cached_path(kv, arm, rate):
    name = pilot.old.rate_name('conversation', kv, arm, rate)
    for archive in (PRIMARY, OLD):
        p = archive / 'rates' / name
        if p.exists():
            return p
    if arm == 'spare':
        return pilot.old.main.cached_path('conversation', kv, 'arrival', rate)
    return None


def initialize():
    pilot.initialize()
    pilot.THETA_MS = THETAS
    confirm.OUT = OUT
    confirm.get_rate = get_rate


def get_rate(source, kv, arm, rate, samples):
    assert source == 'conversation' and kv in KVS and arm in ARMS
    target = OUT / 'rates' / pilot.old.rate_name(source, kv, arm, rate)
    wanted = {s['name'] for s in samples}
    if target.exists():
        data = READ(target)
    else:
        cached = cached_path(kv, arm, rate)
        cached_rows = [] if cached is None else [r for r in READ(cached)['rows'] if r['name'] in wanted]
        checkpoint = OUT / 'checkpoints' / target.name
        rows = READ(checkpoint)['rows'] if checkpoint.exists() else cached_rows[:]
        found = {r['name'] for r in rows}
        assert len(found) == len(rows) and found <= wanted
        for sample in samples:
            if sample['name'] not in found:
                rows.append(pilot.old.prior.simulate_one(sample, kv, arm, rate))
                WRITE(checkpoint, dict(rows=rows))
        rows.sort(key=lambda row: row['name'])
        data = dict(rows=rows, summary=confirm.summary(rows, rate),
                    reused_from=str(cached.relative_to(ROOT)) if cached else None,
                    reused_sha256=DIGEST(cached) if cached else None,
                    reused_rows=len(cached_rows), newly_simulated=len(rows)-len(cached_rows))
        WRITE(target, data)
        print(json.dumps(dict(kv=kv, arm=arm, qps=rate, new=data['newly_simulated'],
                              ttft=round(data['summary']['mean_ttft_ms'], 2))), flush=True)
    assert {r['name'] for r in data['rows']} == wanted
    assert data['summary'] == confirm.summary(data['rows'], rate)
    return data['summary']


def smoke(samples):
    initialize()
    checks = []
    for kv, arm, rate in ((3072, 'spare', 4.), (4096, 'constant8', 4.),
                          (3072, 'srp', 3.), (3072, 'no_arrival', 3.)):
        sample = samples[0]
        path = cached_path(kv, arm, rate)
        old = next(r for r in READ(path)['rows'] if r['name'] == sample['name'])
        new = pilot.old.prior.simulate_one(sample, kv, arm, rate)
        for key in ('valid', 'trajectory_sha256', 'mean_ttft_ms', 'p99_tbt_ms', 'backlog_drift'):
            assert old[key] == new[key], (kv, arm, key)
        checks.append(dict(kv=kv, arm=arm, rate=rate, name=sample['name'],
                           archive=str(path.relative_to(ROOT)), trajectory_equal=True))
    WRITE(OUT / 'smoke.json', dict(passed=True, checks=checks))
    print('Four frozen trajectories reproduced', flush=True)


def group(kv, arm, samples):
    assert READ(OUT / 'smoke.json')['passed']
    initialize()
    result = confirm.run_group(('conversation', kv, arm, samples))
    for rate in (2., 3., 4.):
        get_rate('conversation', kv, arm, rate, samples)
    WRITE(OUT / 'done' / f'c{kv}_{arm}.json', dict(complete=True))
    print(json.dumps(result['capacities']), flush=True)


def launch(workers):
    tasks = [(kv, arm) for kv in KVS for arm in ARMS]
    def execute(task):
        kv, arm = task
        log = OUT / 'logs' / f'c{kv}_{arm}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as handle:
            result = subprocess.run([sys.executable, '-m', 'research.www_ablation_shared_settings_20261023.run',
                                     '--kv', str(kv), '--arm', arm], cwd=ROOT, stdout=handle, stderr=handle)
        if result.returncode:
            raise RuntimeError(f'{task}: see {log}')
        return task
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute, task) for task in tasks]
        for future in as_completed(futures):
            print('Completed', future.result(), flush=True)


def summarize(samples):
    names = {s['name'] for s in samples}
    capacities, latency, endpoints = [], [], 0
    for kv in (1024, 2048, 3072, 3584, 4096, 5120):
        archive = OUT if kv in KVS else PRIMARY
        for arm in ARMS:
            record = READ(archive / 'groups' / f'conversation_c{kv}_{arm}.json')
            for theta in THETAS:
                cap = record['capacities'][f'{theta:g}']
                for rate, expected in ((cap['last_pass_qps'], True), (cap['first_fail_qps'], False)):
                    rows = READ(archive / 'rates' / f'conversation_c{kv}_{arm}_r{rate:g}.json')['rows']
                    assert {r['name'] for r in rows} == names
                    metrics = confirm.summary(rows, rate)
                    assert bool(pilot.old.resource_ok(metrics, rate) and metrics['mean_ttft_ms'] <= theta) == expected
                    endpoints += 1
                capacities.append(dict(kv_blocks=kv, arm=arm, theta_s=theta/1000,
                                       capacity_qps=cap['last_pass_qps'], first_fail_qps=cap['first_fail_qps'],
                                       failure_causes=';'.join(cap['failure_causes']),
                                       archive=str(archive.relative_to(ROOT))))
            for rate in (2., 3., 4.):
                path = archive / 'rates' / f'conversation_c{kv}_{arm}_r{rate:g}.json'
                if not path.exists():
                    continue
                rows = READ(path)['rows']
                assert {r['name'] for r in rows} == names
                metrics = confirm.summary(rows, rate)
                latency.append(dict(kv_blocks=kv, arm=arm, qps=rate, **metrics,
                                    resource_ok=pilot.old.resource_ok(metrics, rate)))
    for name, rows in [('capacities', capacities), ('matched_latency', latency)]:
        path = OUT / 'analysis' / f'{name}.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    rates = [READ(p) for p in (OUT / 'rates').glob('*.json')]
    audit = dict(capacity_endpoints_checked=endpoints,
                 newly_simulated_windows=sum(r['newly_simulated'] for r in rates),
                 reused_windows=sum(r['reused_rows'] for r in rates),
                 all_valid=all(r['valid'] for data in rates for r in data['rows']),
                 new_groups=len(KVS)*len(ARMS), smoke_replays=4, gpu_execution=False)
    WRITE(OUT / 'analysis' / 'audit.json', audit)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--kv', type=int, choices=KVS)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    samples = prepare()
    if args.smoke:
        smoke(samples)
    elif args.run:
        launch(args.workers)
    elif args.kv and args.arm:
        group(args.kv, args.arm, samples)
    elif args.summarize:
        summarize(samples)
    else:
        parser.error('Choose --smoke, --run, --kv/--arm, or --summarize')
