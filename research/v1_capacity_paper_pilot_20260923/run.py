"""Small paired GPU pilot, isolated per-cell workers and immutable provenance."""
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.v1_capacity_paper_pilot_20260923.common import *
from research.v1_state_aware_20260923 import run as infra
from research.v1_budget_20260922 import run as ref
ref.OUT = OUT


def checked_manifest():
    m = read(OUT/'manifest.json')
    assert m['sources'] == sources(), 'Source changed'
    assert m['history_sha256'] == digest(HISTORY)
    assert all(digest(INPUT/n) == v for n, v in m['inputs'].items())
    return m


def save(engine, scheduler, cell, arm, repeat, phase, reqs):
    prepare(scheduler, cell['block'].split('_')[0] if phase == 'compare' else 'code', repeat)
    method, cfg = ARMS[arm]
    key = f"{cell['key']}_{phase}_{arm}_r{repeat}"
    assert not (OUT/'runs'/(key+'.json')).exists(), key
    write(OUT/f"worker_g{cell['gpu']}.json", dict(state='running', key=key, updated=time.time()))
    row = ref.episode(engine, scheduler, reqs, method, cfg, timeout=300)
    rounds = row.pop('rounds')
    row['metrics']['mean_e2e_ms'] = (st.fmean(c['finish_ms']-c['arrival_ms'] for c in row['clocks'].values())
                                        if row['status'] == 'complete' else None)
    row['metrics']['gate_rejected'] = sum(r.get('gate_rejected', 0) for r in rounds)
    row['metrics']['deferred_prefill_tokens'] = sum(d['baseline_tokens']-d['actual_tokens']
        for r in rounds for d in r.get('reductions', ()) if d['outputs'] == 0)
    row['metrics']['low_free_round_fraction'] = st.fmean(r['free_before']/(r['total_blocks']-1) < .125 for r in rounds) if rounds else 0
    path = OUT/'rounds'/(key+'.json.gz')
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', compresslevel=1) as f: json.dump(rounds, f)
    row.update(job=dict(key=key, arm=arm, method=method, config=cfg, repeat=repeat,
        phase=phase, cell=cell), rounds_path=str(path.relative_to(OUT)), rounds_sha256=digest(path))
    write(OUT/'runs'/(key+'.json'), row)
    print(key, row['status'], row['metrics']['mean_ttft_ms'], flush=True)
    assert row['status'] == 'complete', row['error']
    ref.audit_result(row)
    return row


def worker(cellkey):
    m = checked_manifest()
    cell = next(c for c in m['cells'] if c['key'] == cellkey)
    g = cell['gpu']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(g)
    assert sorted(os.sched_getaffinity(0)) == m['affinity'][str(g)]
    dist = importlib.metadata.distribution('vllm')
    assert {n:digest(dist.locate_file('vllm/'+n)) for n in m['native_hashes']} == m['native_hashes']
    import torch
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    assert str(torch.cuda.get_device_properties(0).uuid).lower() == m['hardware'][str(g)]['uuid'].removeprefix('GPU-').lower()
    engine = LLMEngine.from_engine_args(EngineArgs(model=MODEL, dtype='half', max_model_len=12288,
        max_num_seqs=64, max_num_batched_tokens=2048, enable_chunked_prefill=True,
        enable_prefix_caching=False, block_size=16, gpu_memory_utilization=.9,
        enforce_eager=True, disable_log_stats=True, seed=0, num_gpu_blocks_override=cell['capacity'],
        scheduler_cls='research.v1_capacity_paper_pilot_20260923.scheduler.PilotScheduler'))
    scheduler = engine.engine_core.engine_core.scheduler
    warm = ref.episode(engine, scheduler, [dict(id=f'w{i}', prompt=256, output=16, arrival_ms=0) for i in range(4)],
                       'srp_matched', BASE)
    assert warm['status'] == 'complete', warm['error']
    aa = [save(engine, scheduler, cell, 'srp', r, 'aa', read(INPUT/'code_calibration_original.json')) for r in range(2)]
    ratios = {k:(max(xs)-min(xs))/st.fmean(xs) for k,xs in (
        ('mean_ttft_ms', [r['metrics']['mean_ttft_ms'] for r in aa]), ('elapsed_ms', [r['elapsed_ms'] for r in aa]))}
    write(OUT/f'aa_{cellkey}.json', dict(ratios=ratios, passed=max(ratios.values()) <= .05))
    assert max(ratios.values()) <= .05, ratios
    reqs = read(INPUT/(cell['block']+'_original.json'))
    names = list(ARMS)
    shift = m['cells'].index(cell) % len(names)
    order = names[shift:]+names[:shift]
    for repeat in range(2):
        for arm in (order if repeat == 0 else list(reversed(order))):
            checked_manifest()
            save(engine, scheduler, cell, arm, repeat, 'compare', reqs)
    write(OUT/f"worker_g{g}.json", dict(state='finished', cell=cellkey, updated=time.time()))


def run_cell(m, cell):
    g = cell['gpu']
    assert infra.idle(infra.hardware()[g]), f'GPU {g} no longer idle'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), CUDA_DEVICE_ORDER='PCI_BUS_ID',
        VLLM_USE_V1='1', VLLM_ENABLE_V1_MULTIPROCESSING='0', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    path = OUT/'logs'/(cell['key']+'.log')
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['taskset', '-c', ','.join(map(str,m['affinity'][str(g)])), sys.executable,
           '-B', '-u', str(HERE/'run.py'), '--worker', cell['key']]
    try:
        with path.open('a') as f:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=f, stderr=subprocess.STDOUT)
            start = time.monotonic()
            try:
                while proc.poll() is None:
                    external = infra.compute_pids(m['hardware'][str(g)]['uuid'])-{proc.pid}
                    if external: raise RuntimeError(f'External GPU processes: {external}; whole cell invalid')
                    if time.monotonic()-start > 3600: raise RuntimeError('Cell watchdog')
                    time.sleep(3)
                assert proc.returncode == 0, f'Worker exited {proc.returncode}; see {path}'
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try: proc.wait(timeout=15)
                    except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        write(OUT/'cells'/(cell['key']+'.json'), dict(state='complete', cell=cell))
    except Exception as exc:
        write(OUT/'cells'/(cell['key']+'.json'), dict(state='invalid', cell=cell, error=str(exc)))
        raise


def per_gpu(m, g):
    for c in m['cells']:
        if c['gpu'] == g: run_cell(m, c)


def controller():
    OUT.mkdir(parents=True, exist_ok=True)
    master = infra.acquire(OUT/'controller.lock')
    locks = []
    try:
        assert not (OUT/'manifest.json').exists(), 'Existing pilot: inspect; never overwrite'
        h = infra.hardware()
        gpus = [g for g in (6,7,2,1,0) if g in h and 'RTX 4090' in h[g]['name'] and infra.idle(h[g])][:3]
        assert len(gpus) == 3, 'Need three idle 4090s'
        for g in gpus: locks.append(infra.acquire(f'/tmp/query-capacity-gpu{g}.lock'))
        cells = [dict(key=f'{b}_k{c}', block=b, capacity=c, gpu=gpus[i%3])
                 for i,(b,c) in enumerate((b,c) for b in BLOCKS for c in CAPS)]
        dist = importlib.metadata.distribution('vllm')
        for source in ('code','conversation'): history(source)
        m = dict(claim='development pilot, reused windows, no confirmation or tuning',
            arms=ARMS, cells=cells, sources=sources(), hardware={str(g):h[g] for g in gpus},
            affinity=infra.cpu_assign(gpus), history_sha256=digest(HISTORY),
            inputs={b+'_original.json':digest(INPUT/(b+'_original.json')) for b in BLOCKS+('code_calibration',)},
            versions={p:importlib.metadata.version(p) for p in ('vllm','torch','transformers')},
            model=MODEL, model_revision=read(ROOT/'artifacts/v1_capacity_gpu_20260923_r1/manifest.json')['model_revision'],
            native_hashes={n:digest(dist.locate_file('vllm/'+n)) for n in (
                'v1/core/sched/scheduler.py','v1/core/block_pool.py','v1/core/kv_cache_manager.py','v1/core/single_type_kv_cache_manager.py')},
            repeats=2, compare_expected=96, aa_expected=12)
        assert m['versions']['vllm'] == '0.9.2'
        write(OUT/'manifest.json', m)
        for name in m['sources']:
            dest = OUT/'source'/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT/name).read_bytes())
        write(OUT/'status.json', dict(state='running', cells=cells, pid=os.getpid()))
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(per_gpu,m,g) for g in gpus]
            for future in futures: future.result()
        rows = [read(p) for p in sorted((OUT/'runs').glob('*.json'))]
        assert len(rows) == 108
        for row in rows: ref.audit_result(row)
        write(OUT/'status.json', dict(state='finished', runs=108, compare=96, aa=12, accounting_audit_passed=True))
    except Exception as exc:
        write(OUT/'status.json', dict(state='failed', error=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        for lock in locks: lock.close()
        master.close()


def launch():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'controller.log').open('a') as f:
        proc = subprocess.Popen([sys.executable, '-B', '-u', str(HERE/'run.py')], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    write(OUT/'launch.json', dict(pid=proc.pid, output=str(OUT)))
    print('Started controller', proc.pid, str(OUT), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--worker'); p.add_argument('--launch', action='store_true')
    args = p.parse_args()
    if args.worker: worker(args.worker)
    elif args.launch: launch()
    else: controller()
