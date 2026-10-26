"""Freeze native-CPU semantic branch states for deterministic all-arrived GPU replay.

The CPU supplies physical states only, never a GPU timing label. All requests
arrive at time zero so prefix identity does not depend on wall-clock arrivals.
"""
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_saving_threshold_20260924.common import *
from research.v1_saving_threshold_20260924.scheduler import FactorScheduler
from research.v1_budget_20260922.test_scheduler import scheduler


def action_for(row,n):
    if n!='off' and row['baseline_growth_blocks'] and len(row['state'])>=2:
        for d in row['candidate_diagnostics']:
            if d['guard'] and d['saving'] is not None and d['saving']>=n:
                return d['tokens']
    return row['baseline']


def collect(case,cap):
    from vllm import SamplingParams
    from vllm.v1.request import Request,RequestStatus
    reqs=[dict(r,arrival_ms=0) for r in read(INPUT/'traces'/f'{case}.json')]
    s=scheduler(blocks=cap,cls=FactorScheduler)
    s.configure('threshold',**BASE,threshold=16)
    targets={r['id']:r['output'] for r in reqs}
    for i,r in enumerate(reqs):
        s.add_request(Request(r['id'],[100]*r['prompt'],None,None,None,
            SamplingParams(max_tokens=r['output']),None,None,arrival_time=float(i)))
    available=[]
    for step in range(100000):
        out=s.schedule();row=s.round_rows[-1]
        if row['baseline'] is not None and row['features']['pending_prefill']>=4:
            actions={}
            for n in (16,1,4,32,64,'off'):
                action=action_for(row,n)
                actions.setdefault(object_digest(action),dict(n=n,action=action))
            if len(actions)>1:
                available.append(dict(round=step,fingerprint=row['state_fingerprint'],
                    features=row['features'],actions=list(actions.values()),
                    baseline_action=action_for(row,16),state=row['state'],
                    candidate_diagnostics=row['candidate_diagnostics']))
        done=[]
        for rid in out.num_scheduled_tokens:
            r=s.requests[rid]
            if r.num_computed_tokens==r.num_tokens:r.append_output_token_ids(100)
            if len(r.output_token_ids)==targets[rid]:done.append(rid)
        s.finish_requests(done,RequestStatus.FINISHED_LENGTH_CAPPED)
        if not s.requests:break
    assert not s.requests,(case,cap)
    # Four predeclared pressure ranks, with distinct rounds and separated prefix positions.
    available.sort(key=lambda x:(x['features']['pressure'],x['round']))
    selected=[]
    for fraction in (.10,.35,.65,.90):
        if not available:break
        index=min(len(available)-1,int(fraction*(len(available)-1)))
        target=available[index]
        if any(x['round']==target['round'] for x in selected):continue
        selected.append(target)
    return reqs,selected,len(available)


def main():
    path=OUT/'plans/branches.json'
    if path.exists():print('Already frozen:',path);return
    jobs=[];states=[]
    for cap in (768,2048,4096):
        for case in ('code_rate1','conversation_rate1'):
            reqs,selected,total=collect(case,cap)
            anchor=f'branch_anchor_{case}_c{cap}'
            jobs.append(dict(key=anchor,phase='branch',case=case+'_synchronous',capacity=cap,
                requests=reqs,method='threshold',config=dict(BASE,threshold=16),threshold=16,
                group=anchor,role='anchor'))
            for s in selected:
                stateid=f'{case}_c{cap}_r{s["round"]}'
                state=dict(s,id=stateid,source_case=case,capacity=cap,anchor=anchor,alternatives=[],
                           candidate_state_count=total)
                variants=[a for a in s['actions'] if a['action']!=s['baseline_action']][:2]
                for a in variants:
                    key=f'branch_{stateid}_n{a["n"]}'
                    branch=dict(round=s['round'],fingerprint=s['fingerprint'],threshold=a['n'])
                    jobs.append(dict(key=key,phase='branch',case=case+'_synchronous',capacity=cap,
                        requests=reqs,method='threshold',config=dict(BASE,threshold=16,branch=branch),
                        threshold=a['n'],group=anchor,role='alternative',state_id=stateid,
                        expected_action=a['action']))
                    state['alternatives'].append(key)
                states.append(state)
            print(case,cap,'states',len(selected),'opportunities',total,flush=True)
    write(path,dict(frozen=True,claim='same-state single-action branches; synchronous controlled workload',
        states=states,jobs=jobs,source_sha256=digest(Path(__file__)),policy_sources=sources(),
        state_origin='native CPU allocator; GPU state identity must pass before using timing labels'))
    print('Frozen states',len(states),'GPU runs',len(jobs),flush=True)


if __name__=='__main__':main()
