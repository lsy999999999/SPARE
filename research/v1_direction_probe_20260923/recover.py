"""Recover only the externally interrupted cell with complete per-card pairs."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import statistics as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.v1_direction_probe_20260923 import gpu as base
from research.v1_state_aware_20260923.common import read, write, digest, INPUT

HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/v1_direction_probe_20260923_gpu_recovery_r1'
base.OUT = OUT
base.ref.OUT = OUT


def sources():
    return dict(base.sources(), **{str((HERE/n).relative_to(ROOT)):digest(HERE/n)
                                  for n in ('recover.py','RECOVERY_CN.md')})


def worker(key):
    m = read(OUT/'manifest.json')
    assert sources()==m['sources']
    assert all(digest(INPUT/n)==h for n,h in m['inputs'].items())
    cell = next(c for c in m['cells'] if c['key']==key)
    import torch
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    assert str(torch.cuda.get_device_properties(0).uuid).lower()==m['hardware'][str(cell['gpu'])]['uuid'].removeprefix('GPU-').lower()
    engine = LLMEngine.from_engine_args(EngineArgs(model=base.MODEL,dtype='half',max_model_len=12288,
        max_num_seqs=64,max_num_batched_tokens=2048,enable_chunked_prefill=True,
        enable_prefix_caching=False,block_size=16,gpu_memory_utilization=.9,enforce_eager=True,
        disable_log_stats=True,seed=0,num_gpu_blocks_override=768,
        scheduler_cls='research.v1_direction_probe_20260923.scheduler.ProbeScheduler'))
    scheduler = engine.engine_core.engine_core.scheduler
    warm = base.ref.episode(engine,scheduler,[dict(id=f'w{i}',prompt=256,output=16,arrival_ms=0) for i in range(4)],
                            'pressure_one',base.CONFIG)
    assert warm['status']=='complete', warm['error']
    aa = [base.save(engine,scheduler,f'{key}_aa{r}','pressure_one',
                   read(INPUT/'code_calibration_original.json'),'aa',cell) for r in range(2)]
    ratios = {metric:(max(xs)-min(xs))/st.fmean(xs) for metric,xs in
        [('mean_ttft_ms',[r['metrics']['mean_ttft_ms'] for r in aa]),('elapsed_ms',[r['elapsed_ms'] for r in aa])]}
    write(OUT/f'aa_{key}.json',dict(ratios=ratios,passed=max(ratios.values())<=.05))
    assert max(ratios.values())<=.05,ratios
    order = base.METHODS if cell['repeat']==0 else tuple(reversed(base.METHODS))
    for method in order:
        job = f'{key}_{method}_r{cell["repeat"]}'
        write(OUT/f'worker_g{cell["gpu"]}.json',dict(state='running',key=job,updated=time.time()))
        assert sources()==m['sources']
        base.save(engine,scheduler,job,method,read(INPUT/'code_test0_original.json'),'compare',cell)
    write(OUT/f'worker_g{cell["gpu"]}.json',dict(state='finished',updated=time.time()))


def run_cell(m,cell):
    g = cell['gpu']
    assert base.infra.idle(base.infra.hardware()[g])
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),CUDA_DEVICE_ORDER='PCI_BUS_ID',
        VLLM_USE_V1='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    path = OUT/'logs'/(cell['key']+'.log'); path.parent.mkdir(parents=True,exist_ok=True)
    cmd = ['taskset','-c',','.join(map(str,m['affinity'][str(g)])),sys.executable,'-B','-u',str(HERE/'recover.py'),'--worker',cell['key']]
    with path.open('a') as f:
        proc = subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        start = time.monotonic()
        try:
            while proc.poll() is None:
                external = base.infra.compute_pids(m['hardware'][str(g)]['uuid'])-{proc.pid}
                if external: raise RuntimeError(f'External GPU processes on {g}: {external}; cell invalid')
                if time.monotonic()-start>1500: raise RuntimeError('Recovery watchdog')
                time.sleep(3)
            assert proc.returncode==0,f'Worker failed: {path}'
        finally:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=15)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()


def controller():
    OUT.mkdir(parents=True,exist_ok=True)
    master = base.infra.acquire(OUT/'controller.lock'); locks=[]
    try:
        assert not (OUT/'manifest.json').exists(),'Do not overwrite existing recovery'
        h = base.infra.hardware()
        gpus = [g for g in (6,7,2,1) if g in h and 'RTX 4090' in h[g]['name'] and base.infra.idle(h[g])][:2]
        assert len(gpus)==2,'Need two idle cards'
        for g in gpus: locks.append(base.infra.acquire(f'/tmp/query-capacity-gpu{g}.lock'))
        cells = [dict(key=f'code_test0_k768_g{g}',capacity=768,gpu=g,repeat=i) for i,g in enumerate(gpus)]
        original = read(ROOT/'artifacts/v1_direction_probe_20260923_gpu_r1/manifest.json')
        m = dict(methods=list(base.METHODS),cells=cells,hardware={str(g):h[g] for g in gpus},
            affinity=base.infra.cpu_assign(gpus),sources=sources(),inputs=original['inputs'],
            versions=original['versions'],compare_expected=14,aa_expected=4,
            claim='recovery on two complete per-GPU method blocks; repeats across cards, matched comparisons within each card')
        write(OUT/'manifest.json',m)
        for n in m['sources']:
            p = OUT/'source'/n; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes((ROOT/n).read_bytes())
        write(OUT/'status.json',dict(state='running',cells=cells,pid=os.getpid()))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(run_cell,m,c) for c in cells]
            for f in futures: f.result()
        rows=[read(p) for p in sorted((OUT/'runs').glob('*.json'))]
        assert len(rows)==18
        for r in rows: base.ref.audit_result(r)
        write(OUT/'summary.json',dict(rows=rows,accounting_audit_passed=True,manifest=m))
        write(OUT/'status.json',dict(state='finished',compare=14,aa=4))
    except Exception as exc:
        write(OUT/'status.json',dict(state='failed',error=str(exc),traceback=traceback.format_exc()));raise
    finally:
        for lock in locks: lock.close()
        master.close()


def launch():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'controller.log').open('a') as f:
        proc=subprocess.Popen([sys.executable,'-B','-u',str(HERE/'recover.py')],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    write(OUT/'launch.json',dict(pid=proc.pid,detached=True))
    print('Started recovery',proc.pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.worker:worker(a.worker)
    elif a.launch:launch()
    else:controller()
