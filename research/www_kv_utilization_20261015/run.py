"""Small controlled KV replay using existing inputs, timing and policies."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from unittest.mock import patch
from research.www_capacity_pareto_20261015 import run as base
from research.www_pressure_extended_20261015.run import generate

ROOT = base.ROOT
OUT = ROOT / 'artifacts/www_kv_utilization_20261015_r1'
ARMS = ('arrival','native','sarathi','sola_rule','uniboost_static','past_future','s3_proxy','apt_serve_kv_rule')
CAPACITIES = (768,1024,2048,4096)
RATE = 4.
HORIZON = 120000.

def initialize():
    import torch
    torch.set_num_threads(1)
    base.initialize()
    base.MAX_WALL_S = 180

def run_one(task):
    sample, capacity, arm = task
    output = OUT / 'runs' / f"{sample['name']}_c{capacity}_{arm}"
    if (output/'result.json').exists():
        return base.read(output/'result.json')
    requests = generate(sample, RATE, HORIZON)
    original = base.SIM.simulate
    def traced(*args, **kwargs):
        kwargs.update(output=output, stop_at_ms=HORIZON,
                      probe_times_ms=(60000.,), full_trace=False)
        return original(*args, **kwargs)
    seed = int.from_bytes(hashlib.sha256(sample['name'].encode()).digest()[:4], 'big')
    with patch.object(base.SIM, 'simulate', traced):
        result = base.simulate(requests, arm, capacity, 1024, sample['source'], seed)
    result.update(sample=sample['name'], source=sample['source'], rate=RATE,
                  input_sha256=base.digest(ROOT/sample['input']),
                  request_sha256=hashlib.sha256(__import__('json').dumps(requests, sort_keys=True).encode()).hexdigest())
    base.write(output/'result.json', result)
    base.write(output/'input.json', requests)
    return result

def main():
    old = base.read(base.OUT/'design.json')
    samples = [s for s in old['samples'] if s['split']=='test' and s['window']==0]
    assert len(samples)==4
    for s in samples:
        assert base.digest(ROOT/s['input'])==old['source_inputs'][s['input']]
    from research.www_evaluation_completion_20261003.simulate import SOURCE_FILES
    paths = SOURCE_FILES + ['research/www_legacy_baselines_20261006/scheduler.py',
        'research/www_capacity_pareto_20261015/run.py', 'research/www_pressure_extended_20261015/run.py',
        'research/www_kv_utilization_20261015/run.py']
    design = dict(samples=samples, arms=ARMS, capacities=CAPACITIES, rate=RATE,
        horizon_ms=HORIZON, measurement_ms=[60000,120000], budget=1024,chunk=256,
        concurrency=32, windows_per_source=2, gpu_execution=False,
        timing_sha256=base.digest(base.PREVIOUS/'timing_profile.json'),
        history_sha256=base.digest(base.PREVIOUS/'history.json'),
        source_sha256={p:base.digest(ROOT/p) for p in paths})
    if (OUT/'design.json').exists():
        assert base.read(OUT/'design.json')==__import__('json').loads(__import__('json').dumps(design))
    base.write(OUT/'design.json', design)
    tasks=[(s,c,a) for s in samples for c in CAPACITIES for a in ARMS]
    with ProcessPoolExecutor(max_workers=8, initializer=initialize) as pool:
        futures=[pool.submit(run_one,t) for t in tasks]
        for i,f in enumerate(as_completed(futures),1):
            r=f.result()
            if i%8==0 or i==len(tasks):
                base.write(OUT/'status.json',dict(completed=i,total=len(tasks),complete=i==len(tasks)))
                print(f'{i}/{len(tasks)}',flush=True)

if __name__=='__main__':
    main()
