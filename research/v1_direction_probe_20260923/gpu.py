"""Two-cell paired GPU direction probe; uses the already audited episode driver."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.v1_state_aware_20260923.common import read, write, digest, INPUT
from research.v1_state_aware_20260923 import run as infra
from research.v1_budget_20260922 import run as ref

HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/v1_direction_probe_20260923_gpu_r1'
MODEL = 'models/Qwen2.5-7B-Instruct'
METHODS = ('a_any_positive_saving', 'fixed_small', 'pressure_one', 'pressure_early',
           'pressure_min_delay', 'horizon8_first', 'horizon8_risk')
CONFIG = dict(budget=1024, chunk=256, concurrency=32)
ref.OUT = OUT


def sources():
    files = [HERE/n for n in ('policy.py', 'scheduler.py', 'gpu.py', 'PROTOCOL_CN.md', 'GPU_PROTOCOL_CN.md')]
    files += [ROOT/f'research/{folder}/{name}' for folder, names in
        [('v1_state_aware_20260923', ('policy.py', 'scheduler.py', 'run.py', 'common.py')),
         ('v1_budget_20260922', ('policy.py', 'scheduler.py', 'run.py')),
         ('v1_capacity_ablation_20260923', ('policy.py',))] for name in names]
    return {str(p.relative_to(ROOT)):digest(p) for p in files}


def checked_manifest():
    m = read(OUT/'manifest.json')
    assert m['sources'] == sources(), 'Source changed'
    assert all(digest(INPUT/name)==sha for name,sha in m['inputs'].items())
    return m


def save(engine, scheduler, key, method, reqs, phase, cell):
    cfg = dict(CONFIG, budget=512) if method == 'fixed_small' else CONFIG
    row = ref.episode(engine, scheduler, reqs, method, cfg, timeout=300)
    rounds = row.pop('rounds')
    row['metrics']['mean_e2e_ms'] = st.fmean(c['finish_ms']-c['arrival_ms'] for c in row['clocks'].values()) if row['status']=='complete' else None
    row['metrics']['deferred_prefill_tokens'] = sum(d['baseline_tokens']-d['actual_tokens']
        for r in rounds for d in r['reductions'] if d['outputs']==0)
    row['metrics']['risk_rejected_all'] = sum(r['probe'].get('risk_rejected_all', False) for r in rounds)
    row['metrics']['horizon_trigger_below_single_gate'] = sum(r['pressure_triggered'] and r['baseline_growth_blocks'] is not None
        and 4*r['baseline_growth_blocks']<max(r['free_before'],1) for r in rounds)
    path = OUT/'rounds'/(key+'.json.gz')
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', compresslevel=1) as f:
        json.dump(rounds, f)
    row.update(job=dict(key=key, method=method, config=cfg, phase=phase, cell=cell),
               rounds_path=str(path.relative_to(OUT)), rounds_sha256=digest(path))
    write(OUT/'runs'/(key+'.json'), row)
    assert row['status']=='complete', row['error']
    ref.audit_result(row)
    print(key, row['status'], row['metrics']['mean_ttft_ms'], flush=True)
    return row


def worker(cellkey):
    m = checked_manifest()
    cell = next(c for c in m['cells'] if c['key']==cellkey)
    g = cell['gpu']
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(g)
    assert sorted(os.sched_getaffinity(0))==m['affinity'][str(g)]
    import torch
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    assert str(torch.cuda.get_device_properties(0).uuid).lower()==m['hardware'][str(g)]['uuid'].removeprefix('GPU-').lower()
    engine = LLMEngine.from_engine_args(EngineArgs(model=MODEL, dtype='half', max_model_len=12288,
        max_num_seqs=64, max_num_batched_tokens=2048, enable_chunked_prefill=True,
        enable_prefix_caching=False, block_size=16, gpu_memory_utilization=.9,
        enforce_eager=True, disable_log_stats=True, seed=0, num_gpu_blocks_override=cell['capacity'],
        scheduler_cls='research.v1_direction_probe_20260923.scheduler.ProbeScheduler'))
    scheduler = engine.engine_core.engine_core.scheduler
    warm = ref.episode(engine, scheduler, [dict(id=f'w{i}', prompt=256, output=16, arrival_ms=0) for i in range(4)],
                       'pressure_one', CONFIG)
    assert warm['status']=='complete', warm['error']
    aa = [save(engine, scheduler, f'{cellkey}_aa{r}', 'pressure_one',
               read(INPUT/'code_calibration_original.json'), 'aa', cell) for r in range(2)]
    ratios = {metric:(max(xs)-min(xs))/st.fmean(xs) for metric,xs in
              [('mean_ttft_ms',[r['metrics']['mean_ttft_ms'] for r in aa]),
               ('elapsed_ms',[r['elapsed_ms'] for r in aa])]}
    write(OUT/f'aa_{cellkey}.json', dict(ratios=ratios, passed=max(ratios.values())<=.05))
    assert max(ratios.values())<=.05, ratios
    reqs = read(INPUT/'code_test0_original.json')
    shift = m['cells'].index(cell)
    order = METHODS[shift:]+METHODS[:shift]
    for repeat in range(2):
        for method in (order if repeat==0 else tuple(reversed(order))):
            key = f'{cellkey}_{method}_r{repeat}'
            write(OUT/f'worker_g{g}.json', dict(state='running', key=key, updated=time.time()))
            checked_manifest()
            save(engine, scheduler, key, method, reqs, 'compare', cell)
    write(OUT/f'worker_g{g}.json', dict(state='finished', updated=time.time()))


def run_cell(m, cell):
    g = cell['gpu']
    assert infra.idle(infra.hardware()[g]), f'GPU {g} no longer idle'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               VLLM_USE_V1='1', VLLM_ENABLE_V1_MULTIPROCESSING='0',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    path = OUT/'logs'/(cell['key']+'.log')
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['taskset', '-c', ','.join(map(str,m['affinity'][str(g)])), sys.executable,
           '-B', '-u', str(HERE/'gpu.py'), '--worker', cell['key']]
    with path.open('a') as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=f, stderr=subprocess.STDOUT)
        start = time.monotonic()
        try:
            while proc.poll() is None:
                external = infra.compute_pids(m['hardware'][str(g)]['uuid'])-{proc.pid}
                if external:
                    raise RuntimeError(f'External GPU processes on {g}: {external}; cell invalid')
                if time.monotonic()-start>1800:
                    raise RuntimeError(f'Cell watchdog: {cell}')
                time.sleep(3)
            assert proc.returncode==0, f'Worker {cell} exited {proc.returncode}; see {path}'
        finally:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=15)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()


def controller():
    OUT.mkdir(parents=True, exist_ok=True)
    master = infra.acquire(OUT/'controller.lock')
    locks = []
    try:
        assert not (OUT/'manifest.json').exists(), 'Existing pilot: inspect status rather than overwrite'
        h = infra.hardware()
        gpus = [g for g,v in h.items() if 'RTX 4090' in v['name'] and infra.idle(v)][:2]
        assert len(gpus)==2, 'Need two idle 4090s for paired cells'
        for g in gpus:
            locks.append(infra.acquire(f'/tmp/query-capacity-gpu{g}.lock'))
        cells = [dict(key=f'code_test0_k{cap}', capacity=cap, gpu=g) for cap,g in zip((768,2048),gpus)]
        m = dict(claim='quick exploratory GPU pilot on reused code window, not confirmation',
            methods=list(METHODS), cells=cells, sources=sources(),
            hardware={str(g):h[g] for g in gpus}, affinity=infra.cpu_assign(gpus),
            inputs={name:digest(INPUT/name) for name in ('code_test0_original.json','code_calibration_original.json')},
            versions={p:importlib.metadata.version(p) for p in ('vllm','torch','transformers')},
            model=MODEL, repeats=2, compare_expected=28, aa_expected=4)
        assert m['versions']['vllm']=='0.9.2'
        write(OUT/'manifest.json', m)
        for name in m['sources']:
            dest = OUT/'source'/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT/name).read_bytes())
        write(OUT/'status.json', dict(state='running', cells=cells, pid=os.getpid()))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_cell,m,c) for c in cells]
            for future in futures: future.result()
        rows = [read(p) for p in sorted((OUT/'runs').glob('*.json'))]
        assert len(rows)==32
        for row in rows: ref.audit_result(row)
        write(OUT/'summary.json', dict(rows=rows, accounting_audit_passed=True, manifest=m))
        write(OUT/'status.json', dict(state='finished', runs=len(rows), compare=28, aa=4))
    except Exception as exc:
        write(OUT/'status.json', dict(state='failed', error=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        for lock in locks: lock.close()
        master.close()


def launch():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'controller.log').open('a') as f:
        proc = subprocess.Popen([sys.executable, '-B', '-u', str(HERE/'gpu.py')], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    write(OUT/'launch.json', dict(pid=proc.pid, detached=True, output=str(OUT)))
    print('Started controller', proc.pid, str(OUT), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--worker'); p.add_argument('--launch', action='store_true')
    args = p.parse_args()
    if args.worker: worker(args.worker)
    elif args.launch: launch()
    else: controller()
