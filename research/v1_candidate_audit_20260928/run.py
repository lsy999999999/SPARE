"""Frozen bounded GPU candidate audit, detached controller and monitored workers."""
import argparse
import gzip
import os
import random
import signal
import statistics as st
import subprocess
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_candidate_audit_20260928.common import *
from research.v1_candidate_audit_20260928.candidates import enumerate_actions
from research.v1_candidate_audit_20260928.audit import audit, measure
from research.v1_fixed_order_20260927.episode import episode


def freeze():
    assert not (OUT/'design.json').exists(),'Do not overwrite frozen design'
    groups=[]
    for source,capacity in [('conversation',768),('code',1536)]:
        reqs=read(OLD/f'inputs/{source}_w0_burst_stop.json')
        assert len(reqs)==64 and all(r['arrival_ms']==0 for r in reqs)
        write(OUT/f'inputs/{source}.json',reqs)
        for b,k in CONFIGS:
            groups.append(dict(key=f'{source}_c{capacity}_t{b}_k{k}',source=source,capacity=capacity,budget=b,chunk=k))
    write(OUT/'design.json',dict(frozen_at=time.time(),groups=groups,source_hashes=sources(),
        input_hashes={s:digest(OUT/f'inputs/{s}.json') for s in ('code','conversation')},
        states_per_group=2,discovery_repeats=2,confirmation_repeats=2,max_parameter_points=33,
        claim='Finite-pool single-action audit on reused all-arrived workloads; not closed-loop policy evaluation'))
    import shutil
    for rel in sources():
        dest=OUT/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)


def verify():
    d=read(OUT/'design.json');assert d['source_hashes']==sources(),'Frozen source mismatch'
    for source,sha in d['input_hashes'].items(): assert sha==digest(OUT/f'inputs/{source}.json')
    return d


def select_targets(rounds,budget,chunk):
    eligible=[]
    for row in rounds:
        c=enumerate_actions(row,budget,chunk)
        assert len(c['proposals'])<=33
        if len(c['actions'])<2 or not any(r['outputs']==0 for r in row['state']):continue
        eligible.append(dict(round=row['round'],fingerprint=row['state_fingerprint'],
                             free=row['free_before'],**c))
    pool=[c for c in eligible if c['new_plans']] or eligible
    selected=[]
    effective=[c for c in pool if c['k_matched']] or [c for c in pool if c['k_effective']]
    if effective:
        selected.append(dict(effective[len(effective)//2],reason='middle K-effective state, prefer equal q/growth'))
    redundant=[c for c in pool if not c['k_effective'] and c['round'] not in [s['round'] for s in selected]]
    if redundant:
        selected.append(dict(redundant[len(redundant)//2],reason='middle K-redundant state'))
    while len(selected)<2:
        other=sorted([c for c in pool if c['round'] not in [s['round'] for s in selected]],key=lambda c:(c['free'],c['round']))
        if not other:break
        selected.append(dict(other[int(.25*(len(other)-1))],reason='25th free rank fallback'))
    return dict(frozen_before_labels=True,eligible=len(eligible),with_new=sum(c['new_plans']>0 for c in eligible),
                k_effective=sum(c['k_effective'] for c in eligible),states=selected)


def save(engine,s,reqs,cfg,folder,name,meta,anchor=None):
    path=folder/'runs'/f'{name}.json';assert not path.exists()
    write(folder/'current.json',dict(name=name,state='running',updated=time.time()))
    result=episode(engine,s,reqs,'threshold',cfg,timeout=180)
    rows=result.pop('rounds'); result.update(config=cfg,**meta)
    rp=folder/'rounds'/f'{name}.json.gz';rp.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(rp,'wt',compresslevel=1) as f:json.dump(rows,f,separators=(',',':'))
    result.update(rounds_path=str(rp.relative_to(OUT)),rounds_sha256=digest(rp))
    write(path,result)
    assert result['status']=='complete',result['error']
    if cfg.get('branch'):result['label']=measure(result,rows,cfg['branch']['round'])
    result['audit']=audit(result,rows,anchor)
    write(path,result)
    write(folder/'current.json',dict(name=name,state='saved',updated=time.time()))
    print(name,round(result.get('label',{}).get('area_per_request_ms',result['metrics']['mean_ttft_ms']),4),flush=True)
    return result,rows


def worker(key,gpu):
    from research.v1_state_aware_20260923 import run as infra
    lock=infra.acquire(f'/tmp/query-capacity-gpu{gpu}.lock')
    assert infra.idle(infra.hardware()[gpu]),'GPU not idle'
    d=verify();group=next(g for g in d['groups'] if g['key']==key)
    folder=OUT/'groups'/key;manifest=read(OUT/'manifest.json')
    assert not (folder/'status.json').exists()
    status=dict(group=group,gpu=gpu,pid=os.getpid(),state='running',updated=time.time())
    write(folder/'status.json',status)
    os.sched_setaffinity(0,manifest['affinity'][str(gpu)])
    try:
        import torch
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.engine.llm_engine import LLMEngine
        assert str(torch.cuda.get_device_properties(0).uuid).lower()==manifest['hardware'][str(gpu)]['uuid'].removeprefix('GPU-').lower()
        engine=LLMEngine.from_engine_args(EngineArgs(model=MODEL,dtype='half',max_model_len=8176,
            max_num_seqs=64,max_num_batched_tokens=2048,enable_chunked_prefill=True,enable_prefix_caching=False,
            block_size=16,gpu_memory_utilization=.9,enforce_eager=True,disable_log_stats=True,seed=0,
            num_gpu_blocks_override=group['capacity'],scheduler_cls='research.v1_fixed_order_20260927.probe_scheduler.ProbeScheduler'))
        s=engine.engine_core.engine_core.scheduler
        s.probe_enabled_context=True;s.probe_times_context=None;s.probe_target_context=None
        reqs=read(OUT/f"inputs/{group['source']}.json")
        cfg=dict(budget=group['budget'],chunk=group['chunk'],concurrency=32,threshold=spec())
        warm=episode(engine,s,reqs,'threshold',cfg,timeout=180);assert warm['status']=='complete'
        aa=[save(engine,s,reqs,cfg,folder,f'aa_{r}',dict(phase='aa',repeat=r))[0] for r in range(2)]
        vals=[r['metrics']['mean_ttft_ms'] for r in aa]
        spread=(max(vals)-min(vals))/st.fmean(vals)
        write(folder/'aa.json',dict(relative_spread=spread,passed=spread<=.05,values=vals))
        assert spread<=.05,'A/A failed'
        _,anchor=save(engine,s,reqs,cfg,folder,'anchor',dict(phase='anchor'))
        selected=select_targets(anchor,group['budget'],group['chunk'])
        write(folder/'selected.json',selected)
        times=[r['features']['arrival_time_s'] for r in anchor]
        for target in selected['states']:
            index=target['round'];s.probe_times_context=times;s.probe_target_context=index
            order=list(target['actions']);random.Random(20260928+index).shuffle(order)
            costs={a['id']:[] for a in order}
            for repeat in range(2):
                for a in (order if repeat==0 else list(reversed(order))):
                    branch=dict(round=index,fingerprint=target['fingerprint'],
                        threshold=dict(spec(),forced_action=dict(budget=a['budget'],chunk=a['chunk'])),after_threshold=spec())
                    result,_=save(engine,s,reqs,dict(cfg,branch=branch),folder,f's{index}_discovery_{a["id"]}_r{repeat}',
                        dict(phase='discovery',target=index,action=a,repeat=repeat),anchor)
                    costs[a['id']].append(result['label']['area_per_request_ms'])
            old=[a for a in order if 'old' in a['families'] or 'base' in a['families']]
            best_old=min(old,key=lambda a:(st.fmean(costs[a['id']]),a['id']))
            best_new=min(order,key=lambda a:(st.fmean(costs[a['id']]),a['id']))
            current=next(a for a in order if a['id']==target['current_id'])
            winners=dict(current=current['id'],best_old=best_old['id'],best_expanded=best_new['id'])
            write(folder/f's{index}_winners.json',dict(frozen_before_confirmation=True,winners=winners,discovery_costs=costs))
            confirm=list({a['id']:a for a in (current,best_old,best_new)}.values())
            random.Random(20261028+index).shuffle(confirm)
            for repeat in range(2):
                for a in (confirm if repeat==0 else list(reversed(confirm))):
                    branch=dict(round=index,fingerprint=target['fingerprint'],
                        threshold=dict(spec(),forced_action=dict(budget=a['budget'],chunk=a['chunk'])),after_threshold=spec())
                    save(engine,s,reqs,dict(cfg,branch=branch),folder,f's{index}_confirm_{a["id"]}_r{repeat}',
                        dict(phase='confirm',target=index,action=a,repeat=repeat),anchor)
        write(folder/'status.json',dict(status,state='complete',states=len(selected['states']),updated=time.time()))
    except BaseException:
        write(folder/'status.json',dict(status,state='failed',error=traceback.format_exc(),updated=time.time()))
        raise
    finally:lock.close()


def controller():
    from research.v1_state_aware_20260923 import run as infra
    from research.v1_fixed_order_20260927.run import descendants
    lock=infra.acquire(OUT/'controller.lock');d=verify()
    validation=read(OUT/'validation.json')
    assert validation['passed'] and validation['source_hashes']==d['source_hashes']
    assert not (OUT/'manifest.json').exists()
    hw=infra.hardware();gpus=[g for g in (0,1,2,6,7,3) if g in hw and 'RTX 4090' in hw[g]['name']]
    import importlib.metadata
    dist=importlib.metadata.distribution('vllm')
    assert dist.version=='0.9.2'
    manifest=dict(pid=os.getpid(),created=time.time(),deadline=time.time()+10800,
                  hardware={str(g):hw[g] for g in gpus},affinity=infra.cpu_assign(gpus),design_sha=digest(OUT/'design.json'))
    manifest['versions']={p:importlib.metadata.version(p) for p in ('vllm','torch','transformers')}
    manifest['native_hashes']={p:digest(dist.locate_file('vllm/'+p)) for p in
        ('v1/core/sched/scheduler.py','v1/core/block_pool.py','v1/core/kv_cache_manager.py')}
    write(OUT/'manifest.json',manifest);pending=list(d['groups']);running={};finished=[]
    while pending or running:
        now=time.time()
        for gpu,t in list(running.items()):
            p=t['proc'];folder=OUT/'groups'/t['key']
            foreign=infra.compute_pids(hw[gpu]['uuid'])-descendants(p.pid) if p.poll() is None else set()
            reason='foreign_gpu_process' if foreign else 'group_deadline' if now-t['start']>5400 else None
            if now>=manifest['deadline']:reason='controller_deadline'
            if reason and p.poll() is None:
                write(folder/'exclusion.json',dict(reason=reason,foreign_pids=sorted(foreign),updated=now))
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
            if p.poll() is not None:
                t['log'].close();status=read(folder/'status.json') if (folder/'status.json').exists() else {}
                ok=p.returncode==0 and status.get('state')=='complete' and not (folder/'exclusion.json').exists()
                finished.append(dict(group=t['key'],gpu=gpu,accepted=ok,exit_code=p.returncode));del running[gpu]
        if now>=manifest['deadline'] and not running:break
        if now<manifest['deadline']:
            current=infra.hardware()
            for gpu in gpus:
                if not pending or gpu in running or not infra.idle(current[gpu]):continue
                group=pending.pop(0);folder=OUT/'groups'/group['key'];folder.mkdir(parents=True,exist_ok=True)
                log=(folder/'worker.log').open('w')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',VLLM_USE_V1='1',
                         VLLM_ENABLE_V1_MULTIPROCESSING='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                p=subprocess.Popen([sys.executable,'-B',str(HERE/'run.py'),'--worker',group['key'],'--gpu',str(gpu)],
                    cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[gpu]=dict(proc=p,key=group['key'],start=time.time(),log=log)
        write(OUT/'status.json',dict(state='running',finished=finished,pending=[g['key'] for g in pending],
            running={str(g):dict(group=t['key'],pid=t['proc'].pid) for g,t in running.items()},updated=time.time()))
        time.sleep(3)
    write(OUT/'status.json',dict(state='complete' if len(finished)==len(d['groups']) and all(g['accepted'] for g in finished) else 'finished_with_gaps',
        finished=finished,pending=[g['key'] for g in pending],running={},updated=time.time()))
    lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--worker');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.freeze:freeze()
    elif a.worker:worker(a.worker,a.gpu)
    else:controller()
