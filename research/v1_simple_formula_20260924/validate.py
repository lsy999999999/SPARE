"""Native V1 CPU simulator: allocator, guards, formula, and FCFS differential."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_simple_formula_20260924.common import *
from research.v1_simple_formula_20260924.policy import observable
from research.v1_simple_formula_20260924.scheduler import FormulaScheduler
from research.v1_budget_20260922.test_scheduler import scheduler, add, advance
from research.v1_budget_20260922.policy import State
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus
from vllm import SamplingParams
from research.v1_saving_threshold_20260924.policy import threshold_value


def main():
    d=read(OUT/'design.json')
    rows=[State('a',16,64,64,0,1,True),State('b',0,16,16,0,0,False),
          State('c',100,101,64,37,7,True)]
    f=observable(rows,10,16)
    assert f['work_free_ratio']==2 and abs(f['remaining_cv']-.5)<1e-12
    assert observable([],7,16)==dict(work_free_ratio=7,remaining_cv=0)
    # Genuine native path agrees with unmodified native scheduler in allocation/order.
    for cap in (48,96):
        a=scheduler(blocks=cap,cls=FormulaScheduler);b=scheduler(blocks=cap,cls=Scheduler)
        a.configure('native',256,256,8)
        for s in (a,b):add(s,[257,129,385,99,400,12])
        for _ in range(10000):
            x,y=a.schedule(),b.schedule()
            assert x.num_scheduled_tokens==y.num_scheduled_tokens
            assert a.kv_cache_manager.block_pool.get_num_free_blocks()==b.kv_cache_manager.block_pool.get_num_free_blocks()
            advance(a,x);advance(b,y)
            if not a.requests:break
        assert not a.requests and not b.requests
    results=[]
    for cap in (768,2048,4096):
        for name,spec in d['candidates'].items():
            for source in ('code','conversation'):
                reqs=read(OUT/f'inputs/{source}_dev.json')
                s=scheduler(blocks=cap,cls=FormulaScheduler)
                s.configure('threshold',**BASE,threshold=spec)
                targets={r['id']:r['output'] for r in reqs}
                for i,r in enumerate(reqs):
                    s.add_request(Request(r['id'],[100]*r['prompt'],None,None,None,
                        SamplingParams(max_tokens=r['output']),None,None,arrival_time=float(i)))
                for step in range(10000):
                    out=s.schedule();r=s.round_rows[-1]
                    assert r['free_after']>=0
                    assert r['threshold']==threshold_value(spec,r['features'])
                    if r['intervened']:
                        assert r['baseline_growth_blocks']-r['actual_growth_blocks']>=r['threshold']
                    done=[]
                    for rid in out.num_scheduled_tokens:
                        request=s.requests[rid]
                        if request.num_computed_tokens==request.num_tokens:request.append_output_token_ids(100)
                        if len(request.output_token_ids)==targets[rid]:done.append(rid)
                    s.finish_requests(done,RequestStatus.FINISHED_LENGTH_CAPPED)
                    if not s.requests:break
                assert not s.requests,(cap,name,source)
                results.append(dict(capacity=cap,formula=name,source=source,rounds=len(s.round_rows),
                    interventions=sum(r['intervened'] for r in s.round_rows),
                    preemptions=sum(r['preemptions'] for r in s.round_rows)))
    write(OUT/'cpu_validation.json',dict(passed=True,native_differential=2,closed_loops=results,
        clock='All arrived, unit scheduler steps. No latency prediction or GPU coefficient selection.'))
    print('CPU validation passed:',len(results),'closed loops')


if __name__=='__main__':main()
