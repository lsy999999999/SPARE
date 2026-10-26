"""Small paired GPU experiment with frozen development selection and fresh holdout."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import random
import statistics as st
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_simple_formula_20260924.common import *
from research.v1_state_aware_20260923 import run as infra
from research.v1_budget_20260922 import run as ref
ref.OUT=OUT


def save(engine,scheduler,j,g):
    target=OUT/'runs'/f"{j['key']}.json"
    if target.exists():
        old=read(target);assert old['job']==j and old['status']=='complete';return old
    reqs=j.get('requests') or read(OUT/f"inputs/{j['case']}.json")
    write(OUT/f'worker_g{g}.json',dict(job=j['key'],state='running',updated=time.time()))
    result=ref.episode(engine,scheduler,reqs,j['method'],j['config'],timeout=180)
    rounds=result.pop('rounds');trace=OUT/'rounds'/f"{j['key']}.json.gz"
    trace.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(trace,'wt',compresslevel=1) as f:json.dump(rounds,f,separators=(',',':'))
    result['metrics']['mean_e2e_ms']=st.fmean(c['finish_ms']-c['arrival_ms'] for c in result['clocks'].values()) if result['status']=='complete' else None
    ns=[r['threshold'] for r in rounds if r.get('threshold') is not None and r['features']['pending_prefill']]
    result['metrics']['threshold_quantiles']=[ref.quantile(ns,q) for q in (0,.5,.95,1)] if ns else []
    result.update(job=j,gpu=g,rounds_path=str(trace.relative_to(OUT)),rounds_sha256=digest(trace),saved_at=time.time())
    write(target,result)
    assert result['status']=='complete',result['error']
    ref.audit_result(result)
    print(j['key'],round(result['metrics']['mean_ttft_ms'],2),flush=True)
    return result


def worker(taskpath,g):
    lock=infra.acquire(f'/tmp/query-capacity-gpu{g}.lock')
    m=read(OUT/'manifest.json');assert m['sources']==source_hashes()
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(g)
    assert sorted(os.sched_getaffinity(0))==m['affinity'][str(g)]
    import torch
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    assert str(torch.cuda.get_device_properties(0).uuid).lower()==m['hardware'][str(g)]['uuid'].removeprefix('GPU-').lower()
    task=read(taskpath)
    engine=LLMEngine.from_engine_args(EngineArgs(model=MODEL,dtype='half',max_model_len=12288,
        max_num_seqs=64,max_num_batched_tokens=2048,enable_chunked_prefill=True,
        enable_prefix_caching=False,block_size=16,gpu_memory_utilization=.9,
        enforce_eager=True,disable_log_stats=True,seed=0,num_gpu_blocks_override=task['capacity'],
        scheduler_cls='research.v1_simple_formula_20260924.scheduler.FormulaScheduler'))
    scheduler=engine.engine_core.engine_core.scheduler
    warm=[dict(id=f'w{i}',prompt=256,output=16,arrival_ms=0) for i in range(4)]
    for _ in range(2):
        r=ref.episode(engine,scheduler,warm,'threshold',dict(BASE,threshold='off'));assert r['status']=='complete'
    if not (OUT/f'calibration_g{g}.json').exists():
        cal=read(ROOT/'artifacts/v1_capacity_design_20260923/traces/code_calibration_original.json')
        results=[]
        for rep in range(2):
            j=dict(key=f'aa_g{g}_r{rep}',phase='aa',capacity=task['capacity'],method='threshold',
                   config=dict(BASE,threshold='off'),requests=cal)
            results.append(save(engine,scheduler,j,g))
        ratios=[(max(xs)-min(xs))/st.fmean(xs) for xs in
                ([r['metrics']['mean_ttft_ms'] for r in results],[r['elapsed_ms'] for r in results])]
        write(OUT/f'calibration_g{g}.json',dict(passed=max(ratios)<=.05,ratios=ratios))
        assert max(ratios)<=.05,('A/A noise',g,ratios)
    assert read(OUT/f'calibration_g{g}.json')['passed']
    for j in task['jobs']:save(engine,scheduler,j,g)
    write(OUT/f'worker_g{g}.json',dict(state='idle',updated=time.time()))
    lock.close()


def task_run(m,task,g):
    if all((OUT/'runs'/f"{j['key']}.json").exists() for j in task['jobs']):return
    assert infra.idle(infra.hardware()[g]),f'GPU {g} now busy; preserve pairing and resume later'
    path=OUT/'tasks'/f"{task['key']}.json";write(path,task)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),CUDA_DEVICE_ORDER='PCI_BUS_ID',VLLM_USE_V1='1',
        VLLM_ENABLE_V1_MULTIPROCESSING='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    cmd=['taskset','-c',','.join(map(str,m['affinity'][str(g)])),sys.executable,'-B','-u',str(HERE/'run.py'),
         '--worker',str(path),'--gpu',str(g)]
    logfile=OUT/'logs'/f"{task['key']}_g{g}.log";logfile.parent.mkdir(parents=True,exist_ok=True)
    with logfile.open('a') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                external=infra.compute_pids(m['hardware'][str(g)]['uuid'])-{proc.pid}
                if external:
                    write(OUT/'contamination'/f"{task['key']}.json",dict(external=sorted(external),task=task))
                    raise RuntimeError('External GPU task; comparison invalidated')
                time.sleep(3)
        finally:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
    assert proc.returncode==0,('Worker failed',logfile)


def jobs_for(cells,methods,phase,specs,repeat=0):
    groups={}
    for cell in cells:
        jobs=[]
        for name in methods:
            cfg=dict(BASE) if name=='native' else dict(BASE,threshold=specs[name])
            jobs.append(dict(key=f"{phase}_{cell['case']}_c{cell['capacity']}_{name}_r{repeat}",
                **cell,phase=phase,arm=name,repeat=repeat,method='native' if name=='native' else 'threshold',config=cfg))
        random.Random(20260924+int(object_digest([cell,phase,repeat])[:8],16)).shuffle(jobs)
        groups.setdefault(cell['capacity'],[]).extend(jobs)
    return groups


def stage(m,groups,phase):
    write(OUT/'status.json',dict(state='running',phase=phase,updated=time.time()))
    def consume(g):
        for cap,jobs in groups.items():
            if m['bindings'][str(cap)]==g:task_run(m,dict(key=f'{phase}_c{cap}',capacity=cap,jobs=jobs),g)
    with ThreadPoolExecutor(max_workers=len(m['gpus'])) as pool:
        for f in [pool.submit(consume,g) for g in m['gpus']]:f.result()


def scores(d):
    runs=[read(p) for p in (OUT/'runs').glob('dev_*.json')]
    cells={}
    for r in runs:
        j=r['job'];cells.setdefault((j['case'],j['capacity']),{}).setdefault(j['arm'],[]).append(r['metrics']['mean_ttft_ms'])
    return {arm:st.fmean(st.fmean(c[arm])/st.fmean(c['off'])-1 for c in cells.values())
            for arm in d['development_methods']}


def controller():
    lock=infra.acquire(OUT/'controller.lock')
    d=read(OUT/'design.json')
    assert read(OUT/'cpu_validation.json')['passed']
    for name,h in d['trace_hashes'].items():assert object_digest(read(OUT/f'inputs/{name}.json'))==h
    if (OUT/'manifest.json').exists():
        m=read(OUT/'manifest.json');assert m['sources']==source_hashes()
    else:
        h=infra.hardware();gs=[g for g,v in h.items() if '4090' in v['name'] and infra.idle(v)][:3];assert gs
        m=dict(gpus=gs,hardware={str(g):h[g] for g in gs},affinity=infra.cpu_assign(gs),sources=source_hashes(),
            bindings={str(c):gs[i%len(gs)] for i,c in enumerate((768,2048,4096,1536))},
            versions={p:importlib.metadata.version(p) for p in ('vllm','torch','transformers')},
            design_sha256=digest(OUT/'design.json'),model=MODEL,created=time.time())
        write(OUT/'manifest.json',m)
        for name in m['sources']:
            target=OUT/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    specs=dict(off='off',n16=16,n32=32,**d['candidates'])
    stage(m,jobs_for(d['development'],d['development_methods'],'dev',specs),'dev')
    if not (OUT/'frozen_formula.json').exists():
        score=scores(d);order=sorted(d['candidates'],key=score.get)
        if score[order[1]]-score[order[0]]<.02:
            stage(m,jobs_for(d['development'],['off']+order[:2],'dev',specs,1),'dev_repeat')
            score=scores(d);order=sorted(d['candidates'],key=score.get)
        eligible=[a for a in order if score[a]<=score[order[0]]+.005]
        winner=min(eligible,key=lambda a:(d['candidates'][a]['coefficients'][1]!=0,order.index(a)))
        write(OUT/'frozen_formula.json',dict(frozen=True,winner=winner,spec=specs[winner],scores=score,
            selection='development only; all holdout GPU runs follow this timestamp',time=time.time()))
    frozen=read(OUT/'frozen_formula.json');specs['formula']=frozen['spec']
    stage(m,jobs_for(d['holdout'],d['holdout_methods'],'holdout',specs),'holdout')
    write(OUT/'status.json',dict(state='complete',updated=time.time(),frozen=frozen))
    lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--gpu',type=int);p.add_argument('--launch',action='store_true')
    a=p.parse_args()
    if a.worker:worker(a.worker,a.gpu)
    elif a.launch:
        with (OUT/'controller.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',str(HERE/'run.py')],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print('Detached controller',child.pid)
    else:
        try:controller()
        except Exception:
            write(OUT/'status.json',dict(state='failed',error=traceback.format_exc(),updated=time.time()));raise
