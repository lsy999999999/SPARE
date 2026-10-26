"""Native allocator CPU loops plus dynamic inverse correctness and prefix identity."""
import math
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_completion_order_20260927.common import *
from research.v1_completion_order_20260927.policy import select,inverse_budget
from research.v1_broad12h_20260925.policy import select as original_select
from research.v1_budget_20260922.policy import State,project


def main():
    import os
    # Set library variables before native simulator imports.
    os.environ.setdefault('OMP_NUM_THREADS','1')
    os.environ.setdefault('MKL_NUM_THREADS','1')
    from research.v1_budget_20260922.test_scheduler import scheduler
    from research.v1_completion_order_20260927.probe_scheduler import ProbeScheduler
    from vllm.v1.request import Request,RequestStatus
    from vllm import SamplingParams
    rng=random.Random(20260927)
    for _ in range(100):
        rows=[State(str(i),0,rng.randrange(16,800),1000,0,0,False) for i in range(rng.randrange(2,9))]
        k=rng.randrange(16,257);budget=rng.randrange(32,513);target=rng.randrange(0,33)
        actual=inverse_budget(rows,k,target,budget,16,32,8176)
        expected=max(t for t in range(budget+1) if project(rows,t,k,10**12,16,32,8176).growth<=target)
        assert actual==expected
    d=read(OUT/'design.json')
    records=[];example=None;nonstandard=0

    def replay(reqs,capacity,arm,dt=.05,probe=False,times=None,target=None,branch=None):
        nonlocal example,nonstandard
        s=scheduler(blocks=capacity,cls=ProbeScheduler);s.max_model_len=8176
        clock=[0.]
        s.probe_enabled_context=probe;s.probe_times_context=times;s.probe_target_context=target
        s.configure('threshold',**BASE,threshold=spec(arm),branch=branch)
        if not probe:
            s.arrival_clock=lambda:clock[0];s.arrival_origin=0
        cursor=0;targets={r['id']:r['output'] for r in reqs}
        for step in range(24000):
            while cursor<len(reqs) and reqs[cursor]['arrival_ms']/1000<=clock[0]:
                r=reqs[cursor]
                s.add_request(Request(r['id'],[100]*r['prompt'],None,None,None,SamplingParams(max_tokens=r['output']),None,None,arrival_time=clock[0]))
                cursor+=1
            if not s.requests:
                if cursor==len(reqs):break
                clock[0]=reqs[cursor]['arrival_ms']/1000;continue
            out=s.schedule();row=s.round_rows[-1]
            states=[State(**r) for r in row['state']]
            assert row['free_before']+sum(r.blocks for r in states)==capacity-1
            if arm=='current' and not row['branch_applied']:
                old=original_select(states,**BASE,free=row['free_before'],total=capacity,block_size=16,max_model_len=8176,
                                    threshold=spec('current'),recent=dict(arrival_rate=row['features']['arrival_rate']))
                assert old['threshold']==row['threshold']
                if old['selected'] is not None:assert old['selected'].tokens==row['scheduled']
            if row['intervened'] and arm in ('nonstall','work','spt','spt_time') and not row['branch_applied']:
                if sum(row['baseline'].get(r.rid,0) for r in states if r.outputs==0)>0:
                    assert sum(row['scheduled'].get(r.rid,0) for r in states if r.outputs==0)>0
            if row['intervened']:
                assert row['baseline_growth_blocks']-row['actual_growth_blocks']>=row['threshold']
                if (row['budget'],row['chunk']) not in [(256,256),(512,256),(1024,64),(512,64)]:nonstandard+=1
            done=[]
            for rid in out.num_scheduled_tokens:
                r=s.requests[rid]
                if r.num_computed_tokens==r.num_tokens:r.append_output_token_ids(100)
                if len(r.output_token_ids)==targets[rid]:done.append(rid)
            s.finish_requests(done,RequestStatus.FINISHED_LENGTH_CAPPED)
            clock[0]+=dt
        assert cursor==len(reqs) and not s.requests,(capacity,arm)
        return s.round_rows

    # All 12 physical scenarios × 6 methods. Artificial step time never predicts GPU latency.
    for group in d['groups']:
        for shape in ('steady','burst_stop'):
            reqs=read(OUT/f"inputs/{group['source']}_w1_{shape}.json")
            for arm in ARMS:
                rows=replay(reqs,group['capacity'],arm)
                record=dict(group=group['key'],shape=shape,arm=arm,rounds=len(rows),passed=True)
                records.append(record)
                write(OUT/'simulator/status.json',dict(completed=len(records),expected=72,last=record))
                print('CPU',len(records),'/72',group['key'],shape,arm,len(rows),flush=True)
    # A separate native same-prefix check exercises the one-action forced mechanism.
    reqs=[dict(id=str(i),prompt=600+i*13,output=24,arrival_ms=0) for i in range(20)]
    prefix_counts=[]
    for origin in ('current','dynamic'):
        anchor=replay(reqs,192,origin,probe=True)
        candidates=[r for r in anchor if r['intervened']]
        assert candidates
        row=candidates[len(candidates)//2];target=row['round'];times=[r['features']['arrival_time_s'] for r in anchor]
        for b,k in [(1024,256),(row['budget'],row['chunk'])]:
            forced=dict(spec('current'),forced_action=dict(budget=b,chunk=k))
            branch=dict(round=target,fingerprint=row['state_fingerprint'],threshold=forced,after_threshold=spec('current'))
            rows=replay(reqs,192,origin,probe=True,times=times,target=target,branch=branch)
            assert all(rows[i]['state_fingerprint']==anchor[i]['state_fingerprint'] for i in range(target+1))
            assert all(rows[i]['scheduled']==anchor[i]['scheduled'] for i in range(target))
            assert rows[target]['scheduled']==(row['baseline'] if b==1024 and k==256 else row['scheduled'])
        prefix_counts.append(dict(origin=origin,matched_rounds=target+1))
    result=dict(passed=True,loops=len(records),rounds=sum(r['rounds'] for r in records),
                inverse_checks=100,nonstandard_action_rounds=nonstandard,matched_prefix_checks=prefix_counts,
                source_hashes=sources(),design_sha256=digest(OUT/'design.json'),
                caveat='CPU uses native physical allocator with artificial clocks; no GPU performance prediction.')
    assert nonstandard>0
    write(OUT/'cpu_validation.json',result)
    write(OUT/'simulator/results.json',records)
    print(result,flush=True)


if __name__=='__main__':main()
