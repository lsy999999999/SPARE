"""Native CPU allocator correctness only; artificial clocks do not predict latency."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_efficient_points_20260927.common import *


def main():
    from research.v1_budget_20260922.test_scheduler import scheduler
    from research.v1_efficient_points_20260927.scheduler import DynamicScheduler
    from research.v1_efficient_points_20260927 import policy
    from research.v1_efficient_points_20260927.timing import GRID,CONTEXTS,CHUNKS,requests
    from research.v1_broad12h_20260925.policy import select as original
    from research.v1_budget_20260922.policy import State
    from vllm.v1.request import Request,RequestStatus
    from vllm import SamplingParams
    derived='--derived' in sys.argv
    if not derived:
        # Fixtures exercise new actions before any GPU measurements exist.
        fixture=dict(pairs=[(248,256),(496,256),(1024,64),(511,64),(240,64),(256,256),(512,256),(512,64)],
            contexts=dict(fresh=0,long=2048,mixed=1024),
            curves={f'{ctx}_k{k}':[dict(q=q,ms=20+10*((q+255)//256)) for q in GRID] for ctx in CONTEXTS for k in CHUNKS})
        policy.calibration=lambda:fixture
    results=[]
    for cap in (192,768,1536):
        for arm in ARMS:
            s=scheduler(blocks=cap,cls=DynamicScheduler);s.max_model_len=8176
            clock=[0.];s.arrival_clock=lambda:clock[0]
            s.configure('threshold',**BASE,threshold=spec(arm))
            reqs=[dict(id=str(i),prompt=(128,800,1800,2500)[i%4]+i,output=24) for i in range(20)]
            targets={r['id']:r['output'] for r in reqs}
            for r in reqs:s.add_request(Request(r['id'],[100]*r['prompt'],None,None,None,SamplingParams(max_tokens=r['output']),None,None,arrival_time=0.))
            for step in range(10000):
                if not s.requests:break
                out=s.schedule();row=s.round_rows[-1];states=[State(**r) for r in row['state']]
                assert row['free_before']+sum(r.blocks for r in states)==cap-1
                ref=original(states,**BASE,free=row['free_before'],total=cap,block_size=16,max_model_len=8176,
                    threshold=spec('current'),recent=dict(arrival_rate=row['features']['arrival_rate']))
                assert ref['threshold']==row['threshold']
                if arm=='current' and ref['selected'] is not None:assert ref['selected'].tokens==row['scheduled']
                if row['intervened']:assert row['baseline_growth_blocks']-row['actual_growth_blocks']>=row['threshold']
                done=[]
                for rid in out.num_scheduled_tokens:
                    r=s.requests[rid]
                    if r.num_computed_tokens==r.num_tokens:r.append_output_token_ids(100)
                    if len(r.output_token_ids)==targets[rid]:done.append(rid)
                s.finish_requests(done,RequestStatus.FINISHED_LENGTH_CAPPED)
                clock[0]+=.05
            assert not s.requests
            result=dict(capacity=cap,arm=arm,rounds=len(s.round_rows),interventions=sum(r['intervened'] for r in s.round_rows))
            results.append(result);print('CPU',result,flush=True)
    # Every profile shape/action must really yield requested actual q and reset.
    profiles=0
    for ctx,target in CONTEXTS.items():
        fingerprints=set()
        for q in GRID:
            for k in CHUNKS:
                s=scheduler(blocks=2048,cls=DynamicScheduler);s.max_model_len=8176;s.arrival_clock=lambda:0.
                s.configure('threshold',**BASE,threshold=dict(spec('current'),profile_action=dict(budget=1024,chunk=256)))
                for r in requests(ctx):s.add_request(Request(r['id'],[100]*r['prompt'],None,None,None,SamplingParams(max_tokens=r['output']),None,None,arrival_time=0.))
                for i in range(target+1):
                    if i==target:s.threshold_spec=dict(spec('current'),profile_action=dict(budget=q,chunk=k))
                    out=s.schedule()
                    for rid in out.num_scheduled_tokens:
                        r=s.requests[rid]
                        if r.num_computed_tokens==r.num_tokens:r.append_output_token_ids(100)
                row=s.round_rows[-1];assert sum(row['scheduled'].values())==q
                fingerprints.add(row['state_fingerprint']);assert len(fingerprints)==1
                s.finish_requests(list(s.requests),RequestStatus.FINISHED_ABORTED)
                assert not s.requests and s.kv_cache_manager.block_pool.get_num_free_blocks()==2047
                profiles+=1
    result=dict(passed=True,loops=results,profile_checks=profiles,source_hashes=sources(),
        design_sha256=digest(OUT/'design.json'),calibration_sha256=digest(OUT/'calibration.json') if derived else None,
        caveat='Native physical allocator and synthetic clock: correctness only, not a latency simulator')
    write(OUT/('derived_validation.json' if derived else 'cpu_validation.json'),result)
    print('PASSED',len(results),'closed loops;',profiles,'profile shapes')


if __name__=='__main__':main()
