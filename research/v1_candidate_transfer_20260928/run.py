"""Focused contrasts with existing strict replay and monitored GPU infrastructure."""
import argparse, os, random, statistics as st, sys, time, traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_candidate_transfer_20260928.common import *
from research.v1_candidate_audit_20260928.candidates import enumerate_actions
from research.v1_dynamic_actions_20260927.policy import inverse_budget
from research.v1_budget_20260922.policy import State
from research.v1_fixed_order_20260927.episode import episode
from research.v1_candidate_audit_20260928 import run as infrastructure

# Reuse functions without changing the frozen predecessor files. Controller children
# execute this directory's run.py; source verification covers both dependency trees.
infrastructure.OUT=OUT
infrastructure.HERE=HERE
infrastructure.verify=verify
save=infrastructure.save

def select_targets(rounds,budget,chunk):
    holes=[]; matched=[]; normal=0
    for row in rounds:
        c=enumerate_actions(row,budget,chunk)
        if not c['actions'] or not any(r['outputs']==0 for r in row['state']):continue
        normal+=1
        actions=c['actions']; current=next(a for a in actions if a['id']==c['current_id'])
        base=next(a for a in actions if 'base' in a['families'])
        old_nonbase=[a for a in actions if 'old' in a['families'] and a['tokens']!=base['tokens']]
        target=dict(round=row['round'],fingerprint=row['state_fingerprint'],free=row['free_before'],
                    actual_current=[row['budget'],row['chunk']],threshold=c['threshold'],
                    base_growth=base['growth'],current_id=c['current_id'])
        if not old_nonbase and base['growth']>=c['threshold']:
            b=inverse_budget([State(**r) for r in row['state']],chunk,base['growth']-c['threshold'],budget,16,32,8176)
            candidate=next((a for a in actions if [b,chunk] in a['aliases'] and a['tokens']!=base['tokens']),None)
            if candidate:
                assert current['tokens']==base['tokens']
                # Execute the actual inverse(n) parameters, not an earlier alias.
                candidate=dict(candidate,budget=b,chunk=chunk)
                old_guarded=[p for p in c['proposals'] if 'old' in p['families'] and p['guard'] and p['saving'] is not None]
                shortfall=min((c['threshold']-p['saving'] for p in old_guarded),default=None)
                holes.append(dict(target,actions=[current,candidate],contrast=dict(kind='hole',
                    left=current['id'],right=candidate['id'],inverse_budget=b,old_min_shortfall=shortfall)))
        pairs=[]
        for i,a in enumerate(actions):
            for b in actions[i+1:]:
                if a['q']!=b['q'] or a['growth']!=b['growth']:continue
                for x in a['aliases']:
                    for y in b['aliases']:
                        if x[0]!=y[0] or x[1]==y[1]:continue
                        hi,lo=(a,b) if x[1]>y[1] else (b,a)
                        kh,kl=max(x[1],y[1]),min(x[1],y[1])
                        pairs.append(((-kh/kl,-x[0],-kh,kl,hi['id'],lo['id']),
                                      dict(hi,budget=x[0],chunk=kh),dict(lo,budget=x[0],chunk=kl)))
        if pairs:
            _,hi,lo=min(pairs,key=lambda p:p[0])
            aa={a['id']:a for a in (hi,lo,current)}
            # Preserve chosen K alias when current is one member of the contrast.
            aa.update({hi['id']:hi,lo['id']:lo})
            matched.append(dict(target,actions=list(aa.values()),contrast=dict(kind='k_matched',left=hi['id'],right=lo['id'],
                budget=hi['budget'],high_K=hi['chunk'],low_K=lo['chunk'],q=hi['q'],growth=hi['growth'])))
    selected={}
    for pool in (holes,matched):
        if not pool:continue
        t=pool[len(pool)//2];index=t['round']
        if index not in selected:selected[index]=dict(t,contrasts=[],actions=[])
        s=selected[index];s['contrasts'].append(t['contrast'])
        merged={a['id']:a for a in s['actions']}
        merged.update({a['id']:a for a in t['actions']});s['actions']=list(merged.values())
        s.pop('contrast',None)
    return dict(frozen_before_labels=True,normal=normal,hole_states=len(holes),k_matched_states=len(matched),
                states=list(selected.values()))

def worker(key,gpu):
    from research.v1_state_aware_20260923 import run as infra
    lock=infra.acquire(f'/tmp/query-capacity-gpu{gpu}.lock')
    assert infra.idle(infra.hardware()[gpu]),'GPU not idle'
    d=verify();g=next(g for g in d['groups'] if g['key']==key)
    folder=OUT/'groups'/key;manifest=read(OUT/'manifest.json')
    assert not (folder/'status.json').exists()
    status=dict(group=g,gpu=gpu,pid=os.getpid(),state='running',updated=time.time())
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
            num_gpu_blocks_override=g['capacity'],scheduler_cls='research.v1_fixed_order_20260927.probe_scheduler.ProbeScheduler'))
        s=engine.engine_core.engine_core.scheduler
        s.probe_enabled_context=True;s.probe_times_context=None;s.probe_target_context=None
        reqs=read(OUT/f"inputs/{g['input']}.json")
        cfg=dict(budget=g['budget'],chunk=g['chunk'],concurrency=32,threshold=spec())
        warm=episode(engine,s,reqs,'threshold',cfg,timeout=180);assert warm['status']=='complete'
        aa=[save(engine,s,reqs,cfg,folder,f'aa_{r}',dict(phase='aa',repeat=r))[0] for r in range(2)]
        vals=[r['metrics']['mean_ttft_ms'] for r in aa];spread=(max(vals)-min(vals))/st.fmean(vals)
        write(folder/'aa.json',dict(relative_spread=spread,passed=spread<=.05,values=vals));assert spread<=.05,'A/A failed'
        _,anchor=save(engine,s,reqs,cfg,folder,'anchor',dict(phase='anchor'))
        selected=select_targets(anchor,g['budget'],g['chunk']);write(folder/'selected.json',selected)
        times=[r['features']['arrival_time_s'] for r in anchor]
        for target in selected['states']:
            index=target['round'];s.probe_times_context=times;s.probe_target_context=index
            order=list(target['actions']);random.Random(20260929+index).shuffle(order)
            for repeat in range(REPEATS):
                for a in (order if repeat%2==0 else list(reversed(order))):
                    branch=dict(round=index,fingerprint=target['fingerprint'],
                        threshold=dict(spec(),forced_action=dict(budget=a['budget'],chunk=a['chunk'])),after_threshold=spec())
                    save(engine,s,reqs,dict(cfg,branch=branch),folder,f's{index}_transfer_{a["id"]}_r{repeat}',
                        dict(phase='transfer',target=index,action=a,repeat=repeat),anchor)
        write(folder/'status.json',dict(status,state='complete',states=len(selected['states']),updated=time.time()))
    except BaseException:
        write(folder/'status.json',dict(status,state='failed',error=traceback.format_exc(),updated=time.time()));raise
    finally:lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.worker:worker(a.worker,a.gpu)
    else:infrastructure.controller()
