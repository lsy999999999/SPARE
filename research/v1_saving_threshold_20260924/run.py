"""Detached, checkpointed GPU execution. Safe to reconnect and resume."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import gzip
import importlib.metadata
import os
from pathlib import Path
import queue
import random
import statistics as st
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_saving_threshold_20260924.common import *
from research.v1_state_aware_20260923 import run as infra
from research.v1_budget_20260922 import run as ref
ref.OUT=OUT


def check_manifest():
    m=read(OUT/'manifest.json')
    assert m['source_hashes']==sources(),'Frozen source changed; do not resume silently'
    assert digest(OUT/'inputs_manifest.json')==m['input_manifest_sha256']
    inputs=read(OUT/'inputs_manifest.json')
    for c in inputs['cases'].values():
        assert object_digest(read(INPUT/c['file']))==c['trace_digest']
    return m


def all_jobs():
    jobs=list(read(OUT/'inputs_manifest.json')['jobs'])
    for p in sorted((OUT/'plans').glob('*.json')):
        extension=read(p)
        assert extension.get('frozen',False),p
        jobs.extend(extension['jobs'])
    assert len({j['key'] for j in jobs})==len(jobs)
    return jobs


def pending(j):
    return not (OUT/'runs'/(j['key']+'.json')).exists()


def initialize():
    if (OUT/'manifest.json').exists():return check_manifest()
    assert read(OUT/'cpu/equivalence.json')['passed']==6
    inputs=read(OUT/'inputs_manifest.json')
    assert inputs['source_hashes']==sources()
    h=infra.hardware()
    gs=[g for g in (0,1,2,6,7) if g in h and '4090' in h[g]['name'] and infra.idle(h[g])][:4]
    assert gs,'No idle RTX4090'
    dist=importlib.metadata.distribution('vllm')
    native={n:digest(dist.locate_file('vllm/'+n)) for n in (
        'v1/core/sched/scheduler.py','v1/core/block_pool.py','v1/core/kv_cache_manager.py',
        'v1/core/single_type_kv_cache_manager.py')}
    m=dict(source_hashes=sources(),input_manifest_sha256=digest(OUT/'inputs_manifest.json'),
        hardware={str(g):h[g] for g in gs},gpus=gs,affinity=infra.cpu_assign(gs),
        versions={p:importlib.metadata.version(p) for p in ('vllm','torch','transformers')},
        native_hashes=native,model=MODEL,deadline=DEADLINE,created=time.time(),
        model_revision=read(ROOT/'artifacts/v1_capacity_gpu_20260923_r1/manifest.json')['model_revision'])
    for name in m['source_hashes']:
        p=OUT/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    write(OUT/'manifest.json',m)
    return m


def requests(j):
    if 'requests' in j:return j['requests']
    inputs=read(OUT/'inputs_manifest.json')
    return read(INPUT/inputs['cases'][j['case']]['file'])


def save_episode(engine,scheduler,j,g):
    path=OUT/'runs'/(j['key']+'.json')
    if path.exists():return read(path)
    write(OUT/f'worker_g{g}.json',dict(state='running',pid=os.getpid(),job=j,updated=time.time()))
    reqs=requests(j)
    timeout=max(300,reqs[-1]['arrival_ms']/1000+180)
    timeout=min(timeout,max(1,DEADLINE-time.time()))
    result=ref.episode(engine,scheduler,reqs,j['method'],j['config'],timeout=timeout)
    rounds=result.pop('rounds')
    result['metrics']['mean_e2e_ms']=st.fmean(c['finish_ms']-c['arrival_ms'] for c in result['clocks'].values()) if result['status']=='complete' else None
    result['metrics']['deferred_prefill_tokens']=sum(
        x['baseline_tokens']-x['actual_tokens'] for r in rounds for x in r['reductions'] if x['outputs']==0)
    result['metrics']['branch_applied']=sum(r.get('branch_applied',False) for r in rounds)
    for name in ('pressure','free_fraction','remaining_prefill','partial_prefill','decode_count','decode_boundary','recent_release'):
        xs=[r['features'][name] for r in rounds]
        result['metrics'][name+'_mean']=st.fmean(xs) if xs else None
    trace=OUT/'rounds'/(j['key']+'.json.gz');trace.parent.mkdir(parents=True,exist_ok=True)
    temp=trace.with_name(trace.name+'.tmp')
    with gzip.open(temp,'wt',compresslevel=1) as f:json.dump(rounds,f,separators=(',',':'))
    temp.replace(trace)
    result.update(job=j,gpu=g,gpu_uuid=read(OUT/'manifest.json')['hardware'][str(g)]['uuid'],
        rounds_path=str(trace.relative_to(OUT)),rounds_sha256=digest(trace),saved_at=time.time(),
        timeout_seconds=timeout)
    if result['status']=='complete':ref.audit_result(result)
    write(path,result)
    print(j['key'],result['status'],result['metrics']['mean_ttft_ms'],flush=True)
    return result


def worker(taskfile,g):
    lock=infra.acquire(f'/tmp/query-capacity-gpu{g}.lock')
    m=check_manifest();task=read(taskfile)
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(g)
    assert sorted(os.sched_getaffinity(0))==m['affinity'][str(g)]
    dist=importlib.metadata.distribution('vllm')
    assert {n:digest(dist.locate_file('vllm/'+n)) for n in m['native_hashes']}==m['native_hashes']
    jobs=[j for j in task['jobs'] if pending(j)]
    if not jobs:return
    import torch
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine
    expected=m['hardware'][str(g)]['uuid'].removeprefix('GPU-').lower()
    assert str(torch.cuda.get_device_properties(0).uuid).lower()==expected
    engine=LLMEngine.from_engine_args(EngineArgs(model=MODEL,dtype='half',max_model_len=12288,
        max_num_seqs=64,max_num_batched_tokens=2048,enable_chunked_prefill=True,
        enable_prefix_caching=False,block_size=16,gpu_memory_utilization=.9,
        enforce_eager=True,disable_log_stats=True,seed=0,num_gpu_blocks_override=task['capacity'],
        scheduler_cls='research.v1_saving_threshold_20260924.scheduler.FactorScheduler'))
    scheduler=engine.engine_core.engine_core.scheduler
    warm=ref.episode(engine,scheduler,[dict(id=f'w{i}',prompt=256,output=16,arrival_ms=0) for i in range(4)],
                     'threshold',dict(BASE,threshold='off'))
    assert warm['status']=='complete',warm['error']
    for j in jobs:
        if time.time()>=DEADLINE:break
        row=save_episode(engine,scheduler,j,g)
        if row['status']!='complete':
            # Saved failures are not retried. Remaining jobs resume in a fresh engine.
            raise SystemExit(75 if row['error'].startswith('TimeoutError:') else 70)
    write(OUT/f'worker_g{g}.json',dict(state='idle',pid=os.getpid(),updated=time.time()))
    lock.close()


def run_task(m,task,g):
    if not any(pending(j) for j in task['jobs']):return
    path=OUT/'tasks'/f"{task['key']}_g{g}_{object_digest(task)[:12]}.json"
    if path.exists():assert read(path)==task
    else:write(path,task)
    while any(pending(j) for j in task['jobs']) and time.time()<DEADLINE:
        while not infra.idle(infra.hardware()[g]):
            write(OUT/f'worker_g{g}.json',dict(state='waiting_for_idle',updated=time.time(),task=task['key']))
            if time.time()>=DEADLINE:return
            time.sleep(15)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),CUDA_DEVICE_ORDER='PCI_BUS_ID',VLLM_USE_V1='1',
            VLLM_ENABLE_V1_MULTIPROCESSING='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        logfile=OUT/'logs'/f"{task['key']}_g{g}.log";logfile.parent.mkdir(parents=True,exist_ok=True)
        cmd=['taskset','-c',','.join(map(str,m['affinity'][str(g)])),sys.executable,'-B','-u',
             str(HERE/'run.py'),'--worker',str(path),'--gpu',str(g)]
        with logfile.open('a') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            try:
                while proc.poll() is None:
                    external=infra.compute_pids(m['hardware'][str(g)]['uuid'])-{proc.pid}
                    if external:
                        write(OUT/'contamination'/f"{task['key']}_{time.time_ns()}.json",dict(
                            task=task,gpu=g,external=sorted(external),time=time.time()))
                        raise RuntimeError('External GPU process; task marked contaminated')
                    if time.time()>DEADLINE+30:raise RuntimeError('Global deadline')
                    time.sleep(3)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
        if proc.returncode==75:continue
        if proc.returncode!=0:
            raise RuntimeError(f"Worker exited {proc.returncode}: {logfile}")
        break


def calibration(m,g):
    done=OUT/f'calibration_g{g}.json'
    if done.exists():assert read(done)['passed'];return
    cal=read(ROOT/'artifacts/v1_capacity_design_20260923/traces/code_calibration_original.json')
    jobs=[dict(key=f'g{g}_smoke',phase='calibration',capacity=768,method='threshold',
        config=dict(BASE,threshold=1),requests=[dict(id=f's{i}',prompt=256+128*i,output=24,arrival_ms=0) for i in range(8)])]
    jobs += [dict(key=f'g{g}_aa{r}',phase='calibration',capacity=768,method='threshold',
                  config=dict(BASE,threshold='off'),requests=cal) for r in range(2)]
    run_task(m,dict(key=f'calibration_g{g}',capacity=768,jobs=jobs),g)
    rows=[read(OUT/'runs'/f'g{g}_aa{r}.json') for r in range(2)]
    assert all(r['status']=='complete' for r in rows)
    xs=[r['metrics']['mean_ttft_ms'] for r in rows]
    ys=[r['elapsed_ms'] for r in rows]
    ratios=[(max(x)-min(x))/st.fmean(x) for x in (xs,ys)]
    write(done,dict(passed=max(ratios)<=.05,ratios=ratios,time=time.time()))
    assert max(ratios)<=.05,('A/A instability',g,ratios)


def controller(stage):
    lock=infra.acquire(OUT/'controller.lock')
    m=initialize()
    write(OUT/'status.json',dict(state='running',pid=os.getpid(),stage=stage,started=time.time()))
    errors=[]
    try:
        with ThreadPoolExecutor(max_workers=len(m['gpus'])) as pool:
            checks={g:pool.submit(calibration,m,g) for g in m['gpus']}
            active=[]
            for g,f in checks.items():
                try:f.result();active.append(g)
                except Exception as exc:errors.append(dict(phase='calibration',gpu=g,error=str(exc)))
        assert active,('All calibration failed',errors)
        phases=['map','factor'] if stage=='main' else sorted({j['phase'] for j in all_jobs() if j['phase'] not in ('map','factor')})
        for phase in phases:
            jobs=[j for j in all_jobs() if j['phase']==phase]
            queues={g:queue.Queue() for g in active}
            binding_path=OUT/'capacity_gpu_bindings.json'
            bindings=read(binding_path) if binding_path.exists() else {}
            for cap in CAP_ORDER+sorted({j['capacity'] for j in jobs}-set(CAP_ORDER)):
                selected=[j for j in jobs if j['capacity']==cap and pending(j)]
                # Keep every comparison cell together and shuffle methods within it.
                groups={}
                for j in selected:groups.setdefault(j.get('case',j.get('group',j['key'])),[]).append(j)
                ordered=[]
                for name,group in sorted(groups.items()):
                    random.Random(20260924+int(object_digest([phase,name,cap])[:8],16)).shuffle(group)
                    ordered.extend(group)
                if ordered:
                    if str(cap) not in bindings:
                        bindings[str(cap)]=active[(CAP_ORDER.index(cap) if cap in CAP_ORDER else cap)%len(active)]
                    g=bindings[str(cap)]
                    if g not in active:
                        errors.append(dict(capacity=cap,error='Bound GPU unavailable; preserve same-card pairing'))
                        continue
                    queues[g].put(dict(key=f'{phase}_c{cap}',capacity=cap,jobs=ordered))
            write(binding_path,bindings)
            def consume(g):
                tasks=queues[g]
                while time.time()<DEADLINE:
                    try:task=tasks.get_nowait()
                    except queue.Empty:return
                    try:run_task(m,task,g)
                    except Exception as exc:
                        error=dict(task=task['key'],gpu=g,error=str(exc),traceback=traceback.format_exc())
                        write(OUT/'errors'/f"{task['key']}_{time.time_ns()}.json",error)
                        errors.append(error)
                    finally:tasks.task_done()
            with ThreadPoolExecutor(max_workers=len(active)) as pool:
                fs=[pool.submit(consume,g) for g in active]
                for f in fs:f.result()
        selected=[j for j in all_jobs() if j['phase'] in phases]
        missing=[j['key'] for j in selected if pending(j)]
        write(OUT/'status.json',dict(state='finished' if not missing and not errors else 'partial',pid=os.getpid(),
            stage=stage,ended=time.time(),missing=missing,errors=errors))
    except Exception as exc:
        write(OUT/'status.json',dict(state='failed',pid=os.getpid(),error=str(exc),traceback=traceback.format_exc(),time=time.time()))
        raise
    finally:lock.close()


def progress():
    jobs=all_jobs() if (OUT/'inputs_manifest.json').exists() else []
    summary={}
    for j in jobs:
        x=summary.setdefault(j['phase'],dict(planned=0,complete=0,failed=0,pending=0))
        x['planned']+=1
        p=OUT/'runs'/(j['key']+'.json')
        x['pending' if not p.exists() else 'complete' if read(p)['status']=='complete' else 'failed']+=1
    state=read(OUT/'status.json') if (OUT/'status.json').exists() else dict(state='not_started')
    workers={p.stem:read(p) for p in OUT.glob('worker_g*.json')}
    print(json.dumps(dict(controller=state,stages=summary,workers=workers,
        cpu=read(OUT/'cpu/status.json') if (OUT/'cpu/status.json').exists() else None),ensure_ascii=False,indent=2))


def launch(stage):
    OUT.mkdir(parents=True,exist_ok=True)
    # Read-only liveness check; the child additionally takes an exclusive flock.
    for p in (OUT/'launch.json',OUT/'status.json'):
        if p.exists():
            old=read(p);pid=old.get('pid')
            if pid and Path(f'/proc/{pid}/cmdline').exists():
                command=Path(f'/proc/{pid}/cmdline').read_bytes()
                if b'v1_saving_threshold_20260924/run.py' in command:
                    print('Controller already running:',pid);return
    with (OUT/'controller.log').open('a') as f:
        proc=subprocess.Popen([sys.executable,'-B','-u',str(HERE/'run.py'),'--controller','--stage',stage],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    write(OUT/'launch.json',dict(pid=proc.pid,stage=stage,time=time.time()))
    print('Detached controller:',proc.pid,'output:',OUT)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--controller',action='store_true')
    p.add_argument('--status',action='store_true');p.add_argument('--stage',default='main',choices=['main','extra'])
    p.add_argument('--worker');p.add_argument('--gpu',type=int)
    a=p.parse_args()
    if a.status:progress()
    elif a.worker:worker(a.worker,a.gpu)
    elif a.launch:launch(a.stage)
    else:controller(a.stage)
