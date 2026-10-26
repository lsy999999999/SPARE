"""Detached, bounded, idle-only GPU pilot with matched-state branches."""
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.v1_dynamic_actions_20260927.common import *
from research.v1_dynamic_actions_20260927.episode import episode
from research.v1_dynamic_actions_20260927.policy import select
from research.v1_dynamic_actions_20260927.audit import audit
from research.v1_budget_20260922.policy import State
from research.v1_budget_20260922.run import quantile
from research.v1_state_aware_20260923 import run as infra


def verify_design():
    d = read(OUT/'design.json')
    assert d['source_hashes'] == sources(), 'Source changed after freeze'
    for case, sha in d['input_hashes'].items():
        assert digest(OUT/f'inputs/{case}.json') == sha
    assert digest(HERE/'PROTOCOL_CN.md') == d['protocol_sha256']
    return d


def save_episode(engine, scheduler, reqs, config, folder, name, meta):
    path = folder/'runs'/f'{name}.json'
    assert not path.exists()
    write(folder/'current.json', dict(state='running', name=name, updated=time.time()))
    result = episode(engine, scheduler, reqs, 'threshold', config, timeout=180)
    rounds = result.pop('rounds')
    result.update(config=config, **meta)
    if result['status'] == 'complete':
        es = [c['finish_ms']-c['arrival_ms'] for c in result['clocks'].values()]
        result['metrics'].update(mean_e2e_ms=st.fmean(es), p95_e2e_ms=quantile(es,.95))
        result['metrics']['dynamic_nonstandard_rounds'] = sum(
            r['intervened'] and (r['budget'],r['chunk']) not in [(256,256),(512,256),(1024,64),(512,64)] for r in rounds)
    rp = folder/'rounds'/f'{name}.json.gz'
    rp.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(rp,'wt',compresslevel=1) as stream:
        json.dump(rounds,stream,separators=(',',':'))
    result.update(rounds_path=str(rp.relative_to(OUT)), rounds_sha256=digest(rp))
    write(path,result)
    assert result['status'] == 'complete', result['error']
    result['audit'] = audit(result,rounds)
    write(path,result)
    write(folder/'current.json',dict(state='saved',name=name,updated=time.time(),ttft_ms=result['metrics']['mean_ttft_ms']))
    print(name, 'TTFT',round(result['metrics']['mean_ttft_ms'],3),flush=True)
    return result,rounds


def warm(engine,scheduler,reqs,config):
    result = episode(engine,scheduler,reqs,'threshold',config,timeout=180)
    assert result['status'] == 'complete', result['error']


def run_probes(engine,s,reqs,folder,meta):
    s.probe_enabled_context=True
    s.probe_times_context=None
    s.probe_target_context=None
    config=dict(BASE,threshold=spec('current'))
    # The second calibration episode is already fully warmed but retain a separate anchor.
    anchor,anchor_rows=save_episode(engine,s,reqs,config,folder,'anchor',dict(meta,phase='anchor'))
    eligible=[]
    for row in anchor_rows:
        if row['baseline'] is None or not row['baseline'] or not any(r['outputs']==0 for r in row['state']):
            continue
        states=[State(**r) for r in row['state']]
        actions=[]
        for arm in ARMS:
            choice=select(states,**BASE,free=row['free_before'],total=row['total_blocks'],block_size=16,
                          max_model_len=8176,threshold=spec(arm),recent=dict(arrival_rate=row['features']['arrival_rate']))
            actions.append(dict(arm=arm,budget=choice['budget'],chunk=choice['chunk'],
                                tokens=choice['selected'].tokens,growth=choice['selected'].growth))
        if all(a['tokens']==actions[0]['tokens'] for a in actions[1:]):
            continue
        actions.append(dict(arm='base',budget=1024,chunk=256,tokens=row['baseline'],growth=row['baseline_growth_blocks']))
        unique=[]
        for action in actions:
            same=next((a for a in unique if a['tokens']==action['tokens']),None)
            if same is None:
                unique.append(dict(action,aliases=[action['arm']]))
            else:
                same['aliases'].append(action['arm'])
        eligible.append(dict(round=row['round'],fingerprint=row['state_fingerprint'],
                             pressure=row['features']['pressure'],actions=unique))
    eligible.sort(key=lambda r:(r['pressure'],r['round']))
    targets=[]
    for q in (.25,.75):
        if not eligible:break
        candidate=eligible[int(q*(len(eligible)-1))]
        if candidate not in targets:targets.append(candidate)
    write(folder/'selected_states.json',dict(frozen_before_branch_outcomes=True,eligible_states=len(eligible),states=targets))
    times=[r['features']['arrival_time_s'] for r in anchor_rows]
    summary=[]
    for target in targets:
        index=target['round']
        s.probe_times_context=times
        s.probe_target_context=index
        actions=list(target['actions'])
        random.Random(20260927+index).shuffle(actions)
        for repeat in (0,1):
            for action in (actions if repeat==0 else list(reversed(actions))):
                forced=dict(spec('current'),forced_action={k:action[k] for k in ('budget','chunk')})
                cfg=dict(BASE,threshold=spec('current'),branch=dict(round=index,fingerprint=target['fingerprint'],threshold=forced))
                name=f"branch_s{index}_{action['arm']}_r{repeat}"
                result,rows=save_episode(engine,s,reqs,cfg,folder,name,
                    dict(meta,phase='branch',arm=action['arm'],aliases=action['aliases'],repeat=repeat,target=index))
                assert all(rows[i]['state_fingerprint']==anchor_rows[i]['state_fingerprint'] for i in range(index+1))
                assert rows[index]['branch_applied'] and rows[index]['scheduled']==action['tokens']
                branch_ms=(rows[index]['probe_decision_wall']-result['start_wall'])*1000
                pending=[r['rid'] for r in rows[index]['state'] if r['outputs']==0]
                assert pending and all(result['clocks'][rid]['first_ms']>=branch_ms for rid in pending)
                area=sum(result['clocks'][rid]['first_ms']-branch_ms for rid in pending)
                result.update(branch_ms=branch_ms,residual_waiting_area_ms=area,pending=len(pending),prefix_matched=True)
                write(folder/'runs'/f'{name}.json',result)
                summary.append(dict(target=index,arm=action['arm'],aliases=action['aliases'],repeat=repeat,
                                    budget=action['budget'],chunk=action['chunk'],area_ms_request=area,
                                    pending=len(pending),saving=rows[index]['baseline_growth_blocks']-action['growth']))
    write(folder/'probes.json',dict(states=len(targets),branches=len(summary),results=summary))
    s.probe_times_context=None
    s.probe_target_context=None
    s.probe_enabled_context=False


def worker(key,gpu):
    lock=infra.acquire(f'/tmp/query-capacity-gpu{gpu}.lock')
    assert infra.idle(infra.hardware()[gpu]), 'GPU is no longer idle'
    design=verify_design()
    manifest=read(OUT/'manifest.json')
    group=next(g for g in design['groups'] if g['key']==key)
    folder=OUT/'groups'/key
    assert not (folder/'status.json').exists()
    status=dict(state='running',group=group,gpu=gpu,pid=os.getpid(),updated=time.time())
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
            num_gpu_blocks_override=group['capacity'],
            scheduler_cls='research.v1_dynamic_actions_20260927.probe_scheduler.ProbeScheduler'))
        s=engine.engine_core.engine_core.scheduler
        cfg=dict(BASE,threshold=spec('current'))
        warm(engine,s,[dict(id=f'w{i}',prompt=256,output=16,arrival_ms=0) for i in range(4)],cfg)
        burst=read(OUT/f"inputs/{group['source']}_w1_burst_stop.json")
        s.probe_enabled_context=True
        warm(engine,s,burst,cfg)
        aa=[save_episode(engine,s,burst,cfg,folder,f'aa_r{r}',dict(group=key,gpu=gpu,phase='aa',repeat=r))[0] for r in (0,1)]
        ttfts=[r['metrics']['mean_ttft_ms'] for r in aa]
        spread=(max(ttfts)-min(ttfts))/st.fmean(ttfts)
        write(folder/'calibration.json',dict(passed=spread<=.05,relative_spread=spread,ttft_ms=ttfts))
        assert spread<=.05, 'A/A calibration > 5%'
        run_probes(engine,s,burst,folder,dict(group=key,gpu=gpu))
        for shape in ('steady','burst_stop'):
            reqs=read(OUT/f"inputs/{group['source']}_w1_{shape}.json")
            warm(engine,s,reqs,cfg)
            order=list(ARMS)
            random.Random(20260927+group['capacity']+sum(map(ord,group['source']+shape))).shuffle(order)
            for repeat in (0,1):
                for arm in (order if repeat==0 else list(reversed(order))):
                    save_episode(engine,s,reqs,dict(BASE,threshold=spec(arm)),folder,f'{shape}_{arm}_r{repeat}',
                        dict(group=key,gpu=gpu,phase='formal',shape=shape,arm=arm,repeat=repeat))
        write(folder/'status.json',dict(status,state='complete',formal_runs=16,updated=time.time()))
    except BaseException:
        write(folder/'status.json',dict(status,state='failed',error=traceback.format_exc(),updated=time.time()))
        raise
    finally:
        lock.close()


def descendants(pid):
    raw=subprocess.check_output(['ps','-eo','pid=,ppid='],text=True)
    tree=[tuple(map(int,line.split())) for line in raw.splitlines() if line.strip()]
    own={pid}
    while True:
        new=own|{p for p,pp in tree if pp in own}
        if new==own:return own
        own=new


def controller():
    lock=infra.acquire(OUT/'controller.lock')
    design=verify_design()
    assert read(OUT/'cpu_validation.json')['passed']
    assert not (OUT/'manifest.json').exists(), 'No silent restart over existing evidence'
    hardware=infra.hardware()
    gpus=[g for g in (0,1,2,6,7,3) if g in hardware and 'RTX 4090' in hardware[g]['name']]
    manifest=dict(created=time.time(),hardware={str(g):hardware[g] for g in gpus},affinity=infra.cpu_assign(gpus),
                  design_sha256=digest(OUT/'design.json'),pid=os.getpid(),deadline=time.time()+3*3600)
    write(OUT/'manifest.json',manifest)
    pending=list(design['groups'])
    running={}
    finished=[]
    while pending or running:
        now=time.time()
        for gpu,task in list(running.items()):
            proc=task['proc'];folder=OUT/'groups'/task['group']['key']
            foreign=infra.compute_pids(hardware[gpu]['uuid'])-descendants(proc.pid) if proc.poll() is None else set()
            reason='foreign_gpu_process' if foreign else ('group_deadline' if now-task['started']>3600 else None)
            if now>manifest['deadline']:reason='controller_deadline'
            if reason and proc.poll() is None:
                write(folder/'exclusion.json',dict(reason=reason,foreign_pids=sorted(foreign),updated=now))
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
            if proc.poll() is not None:
                task['log'].close()
                status=read(folder/'status.json') if (folder/'status.json').exists() else {}
                accepted=proc.returncode==0 and status.get('state')=='complete' and not (folder/'exclusion.json').exists()
                finished.append(dict(group=task['group']['key'],gpu=gpu,accepted=accepted,exit_code=proc.returncode))
                del running[gpu]
                subprocess.run([sys.executable,'-B',str(HERE/'report.py')],cwd=ROOT,check=True)
        if now<manifest['deadline']:
            current_hw=infra.hardware()
            for gpu in gpus:
                if not pending or gpu in running or not infra.idle(current_hw[gpu]):continue
                # Worker rechecks and acquires the shared lock before any GPU use.
                group=pending.pop(0)
                folder=OUT/'groups'/group['key'];folder.mkdir(parents=True,exist_ok=True)
                log=(folder/'worker.log').open('w')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',VLLM_USE_V1='1',
                         VLLM_ENABLE_V1_MULTIPROCESSING='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                proc=subprocess.Popen([sys.executable,'-B',str(HERE/'run.py'),'--worker',group['key'],'--gpu',str(gpu)],
                                      cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[gpu]=dict(proc=proc,log=log,group=group,started=time.time())
        write(OUT/'status.json',dict(state='running',pending=[g['key'] for g in pending],finished=finished,
              running={str(g):dict(group=t['group']['key'],pid=t['proc'].pid,started=t['started']) for g,t in running.items()},updated=time.time()))
        if now>=manifest['deadline'] and not running:break
        time.sleep(3)
    passed=len(finished)==len(design['groups']) and all(g['accepted'] for g in finished)
    write(OUT/'status.json',dict(state='complete' if passed else 'finished_with_gaps',finished=finished,pending=[g['key'] for g in pending],running={},updated=time.time()))
    subprocess.run([sys.executable,'-B',str(HERE/'report.py')],cwd=ROOT,check=True)
    lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.worker:worker(a.worker,a.gpu)
    else:controller()
