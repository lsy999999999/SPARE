"""Paired parameter sensitivity: capacity and matched-load latency on identical KV."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from research.www_component_ablation_mean_20261019 import confirm, run as pilot

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/www_parameter_capacity_20261024_r1'
ARCHIVE = ROOT / 'artifacts/www_component_ablation_mean_20261020_r4'
READ = pilot.old.prior.base.read
WRITE = pilot.old.prior.base.write
DIGEST = pilot.old.prior.base.digest
KVS = (1024, 2048, 4096)
SOURCES = ('code', 'conversation')
THETAS = (2000., 4000.)
ARMS = {'spare': {}, 'n0_4': {'n0': 4}, 'n0_16': {'n0': 16},
        'alpha64': {'alpha': 64}, 'alpha256': {'alpha': 256},
        'H05': {'H': .5}, 'H2': {'H': 2},
        'tau05': {'tau': .5}, 'tau2': {'tau': 2}}
REFERENCE_RATES = (2., 1.5, 1., .5)


def prepare():
    ref = READ(ARCHIVE / 'design.json')
    assert DIGEST(Path(pilot.policy.__file__)) == ref['policy_sha256']
    timing = ROOT / 'artifacts/www_pareto_20261014_r1/timing_profile.json'
    assert DIGEST(timing) == ref['timing_sha256']
    for sample in ref['samples']:
        assert DIGEST(ROOT / sample['input']) == ref['input_sha256'][sample['input']]
    assert all(sum(s['source'] == src for s in ref['samples']) == 24 for src in SOURCES)
    sources = [Path(__file__), Path(pilot.policy.__file__),
               ROOT / 'research/www_biobjective_ablation_20261016/run.py',
               ROOT / 'research/www_evaluation_completion_20261003/simulate.py',
               ROOT / 'research/www_evaluation_completion_20261003/scheduler.py',
               ROOT / 'research/www_capacity_pareto_20261015/run.py',
               ROOT / 'research/v1_broad12h_20260925/policy.py',
               ROOT / 'research/v1_arrival_formula_20260924/policy.py']
    design = dict(samples=ref['samples'], kv_blocks=KVS, arms=ARMS, theta_ms=THETAS,
                  budget=1024, chunk=256, concurrency=32, horizon_ms=60000,
                  warm_ms=20000, backlog_interval_ms=(40000,60000),
                  p99_tbt_limit_ms=200, windows_per_source=24, gpu_execution=False,
                  input_sha256=ref['input_sha256'], timing_sha256=ref['timing_sha256'],
                  source_sha256={str(p.relative_to(ROOT)): DIGEST(p) for p in sources},
                  reference_design_sha256=DIGEST(ARCHIVE / 'design.json'),
                  capacity_rule='All 24 valid; mean window TTFT <= theta; mean window P99 TBT <= 200 ms; '
                                'mean backlog drift <= max(2,QPS); first failure on ascending tested grid.',
                  resolution_qps=.02,
                  latency_rule='For each workload/KV, use highest of 2,1.5,1,0.5 QPS within '
                               'all nine 2s capacities and passing all nine service checks.',
                  default_reuse='Only identical input/timing/policy defaults; verify trajectory parity first.')
    design = json.loads(json.dumps(design))
    path = OUT / 'design.json'
    if path.exists():
        assert READ(path) == design, 'Frozen design differs'
    else:
        WRITE(path, design)
    return design['samples']


def initialize():
    pilot.initialize()
    pilot.old.prior.ARMS.update({a: ('arrival', opt) for a, opt in ARMS.items()})


def rate_path(source, kv, arm, rate):
    return OUT / 'rates' / pilot.old.rate_name(source, kv, arm, rate)


def get_rate(source, kv, arm, rate, samples):
    target = rate_path(source, kv, arm, rate)
    wanted = {s['name'] for s in samples}
    if target.exists():
        data = READ(target)
    else:
        old = ARCHIVE / 'rates' / target.name if arm == 'spare' else None
        reused = [dict(r) for r in READ(old)['rows'] if r['name'] in wanted] if old and old.exists() else []
        checkpoint = OUT / 'checkpoints' / target.name
        rows = READ(checkpoint)['rows'] if checkpoint.exists() else reused[:]
        found = {r['name'] for r in rows}
        assert len(rows) == len(found) and found <= wanted
        for sample in samples:
            if sample['name'] not in found:
                rows.append(pilot.old.prior.simulate_one(sample, kv, arm, rate))
                WRITE(checkpoint, dict(rows=rows))
                WRITE(OUT / 'workers' / f'{source}_c{kv}_{arm}.json',
                      dict(source=source, kv=kv, arm=arm, qps=rate, rows=len(rows), updated_at=time.time()))
        rows.sort(key=lambda r: r['name'])
        data = dict(rows=rows, summary=confirm.summary(rows, rate),
                    reused_rows=len(reused), newly_simulated=len(rows)-len(reused),
                    reused_from=str(old.relative_to(ROOT)) if reused else None,
                    reused_sha256=DIGEST(old) if reused else None)
        WRITE(target, data)
    assert {r['name'] for r in data['rows']} == wanted
    assert data['summary'] == confirm.summary(data['rows'], rate)
    return data['summary']


def passed(summary, rate, theta):
    return pilot.old.resource_ok(summary, rate) and summary['mean_ttft_ms'] <= theta


def group(source, kv, arm, samples):
    path = OUT / 'groups' / f'{source}_c{kv}_{arm}.json'
    if path.exists():
        return READ(path)
    assert READ(OUT / 'smoke.json')['passed']
    scans = {}
    if arm == 'spare':
        old = READ(ARCHIVE / 'groups' / path.name)
        for qps in old['rates']:
            scans[qps] = get_rate(source, kv, arm, qps, samples)
    else:
        for qps in sorted(set(pilot.old.main.COARSE_RATES) | {.5}):
            scans[qps] = get_rate(source, kv, arm, qps, samples)
            if not passed(scans[qps], qps, max(THETAS)):
                break
    while passed(scans[max(scans)], max(scans), max(THETAS)):
        qps = 2*max(scans)
        assert qps <= 64
        scans[qps] = get_rate(source, kv, arm, qps, samples)
    for theta in THETAS:
        while True:
            low, high, _ = pilot.old.bounds(scans, theta)
            assert high is not None
            if high-low <= .0200001:
                break
            qps = round((low+high)*50)/100
            assert low < qps < high
            scans[qps] = get_rate(source, kv, arm, qps, samples)
    # These rates are included in the capacity grid before the final bound check.
    for qps in REFERENCE_RATES:
        scans[qps] = get_rate(source, kv, arm, qps, samples)
    capacities = {}
    for theta in THETAS:
        low, high, causes = pilot.old.bounds(scans, theta)
        assert high-low <= .0200001
        assert low > 0 and passed(scans[low], low, theta) and not passed(scans[high], high, theta)
        capacities[f'{theta:g}'] = dict(last_pass_qps=low, first_fail_qps=high,
                                        resolution_qps=round(high-low,4), failure_causes=causes)
    result = dict(source=source, kv_blocks=kv, arm=arm, capacities=capacities,
                  rates=sorted(scans), completed_at=time.time())
    WRITE(path, result)
    return result


def smoke(samples):
    initialize()
    checks = []
    for source, kv in (('code',1024), ('conversation',2048), ('conversation',4096)):
        sample = next(s for s in samples if s['source']==source)
        old = READ(ARCHIVE / 'rates' / pilot.old.rate_name(source,kv,'spare',2.))
        ref = next(r for r in old['rows'] if r['name']==sample['name'])
        row = pilot.old.prior.simulate_one(sample,kv,'spare',2.)
        for key in ('valid','trajectory_sha256','mean_ttft_ms','p99_tbt_ms','backlog_drift'):
            assert ref[key]==row[key], (source,kv,key)
        checks.append(dict(source=source,kv=kv,sample=sample['name'],parity=True))
    # Confirm the actual scheduler receives H, separately from policy coefficients.
    sim=pilot.old.prior.base.SIM
    scheduler_checks=[]
    for arm,opt in ARMS.items():
        scheduler,_=sim.make_scheduler('arrival',blocks=2048,budget=1024,chunk=256,policy_options=opt)
        assert scheduler.evaluation_options==opt
        horizon=opt.get('H',1.)
        assert scheduler.arrival_estimator.horizon_s==horizon
        scheduler_checks.append(dict(arm=arm,options=opt,horizon=horizon))
    WRITE(OUT/'smoke.json',dict(passed=True,parity=checks,options=scheduler_checks))
    print('Archive parity and parameter plumbing passed',flush=True)


def launch(workers,samples):
    lock=(OUT/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert READ(OUT/'smoke.json')['passed']
    tasks=[(s,k,a) for s in SOURCES for k in KVS for a in ARMS]
    pending=[t for t in tasks if not (OUT/'groups'/f'{t[0]}_c{t[1]}_{t[2]}.json').exists()]
    done=len(tasks)-len(pending); errors=[];start=time.monotonic()
    def worker(t):
        s,k,a=t
        log=OUT/'logs'/f'{s}_c{k}_{a}.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as f:
            result=subprocess.run([sys.executable,'-B','-m','research.www_parameter_capacity_20261024.run',
                                   '--source',s,'--kv',str(k),'--arm',a],cwd=ROOT,stdout=f,stderr=f)
        if result.returncode:
            raise RuntimeError(f'{t}: {log}')
        return t
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(worker,t):t for t in pending}
        for f in as_completed(futures):
            try:f.result();done+=1
            except Exception as exc:errors.append(dict(task=futures[f],error=str(exc)))
            status=dict(completed=done,total=len(tasks),errors=errors,elapsed_s=time.monotonic()-start)
            WRITE(OUT/'status.json',status);print(json.dumps(status),flush=True)
    if errors:raise RuntimeError(errors)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--workers',type=int,default=24)
    parser.add_argument('--source',choices=SOURCES)
    parser.add_argument('--kv',type=int,choices=KVS)
    parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args();samples=prepare()
    if args.smoke:smoke(samples)
    elif args.run:launch(args.workers,samples)
    elif args.source and args.kv and args.arm:
        initialize();print(json.dumps(group(args.source,args.kv,args.arm,
                           [s for s in samples if s['source']==args.source])),flush=True)
    else:parser.error('Select --smoke, --run or --source/--kv/--arm')
