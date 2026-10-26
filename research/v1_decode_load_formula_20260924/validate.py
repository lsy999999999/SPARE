"""CPU native allocator checks, sign checks, c=0 ablation equivalence."""
from pathlib import Path
import math
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_decode_load_formula_20260924.common import *
from research.v1_decode_load_formula_20260924.policy import observable,formula,select
from research.v1_decode_load_formula_20260924.scheduler import FormulaScheduler
from research.v1_simple_formula_20260924.policy import select as previous_select
from research.v1_budget_20260922.test_scheduler import scheduler
from research.v1_budget_20260922.policy import State
from research.v1_saving_threshold_20260924.policy import threshold_value
from vllm.v1.request import Request, RequestStatus
from vllm import SamplingParams


def main():
    d=read(OUT/'design.json')
    rows=[State('p',16,64,64,0,1,True),State('q',0,16,16,0,0,False),
          State('d',100,101,64,37,7,True)]
    f=observable(rows,10,29,16)
    assert f['work_free_ratio']==2 and f['decoded_kv_fraction']==.25
    assert observable([],7,29,16)['decoded_kv_headroom']==1
    for c in (0,16,32):
        for x in (0,.01,.1,.3,1,10):
            ns=[threshold_value(formula(c),dict(work_free_ratio=x,decoded_kv_headroom=1-z)) for z in (0,.25,.5,.75,1)]
            assert ns==sorted(ns,reverse=True)
        for z in (0,.3,1):
            ns=[threshold_value(formula(c),dict(work_free_ratio=x,decoded_kv_headroom=1-z)) for x in (0,.01,.1,.3,1,10)]
            assert ns==sorted(ns)
    results=[];equivalence=0
    for cap in (768,1536,2048):
        for name,spec in dict(old=d['old_formula'],**d['candidates']).items():
            for source in ('code','conversation'):
                reqs=read(OUT/f'inputs/{source}_dev.json')
                s=scheduler(blocks=cap,cls=FormulaScheduler);s.configure('threshold',**BASE,threshold=spec)
                targets={r['id']:r['output'] for r in reqs}
                for i,r in enumerate(reqs):
                    s.add_request(Request(r['id'],[100]*r['prompt'],None,None,None,
                        SamplingParams(max_tokens=r['output']),None,None,arrival_time=float(i)))
                for step in range(10000):
                    out=s.schedule();r=s.round_rows[-1]
                    assert 0<=r['free_after']<cap
                    assert r['threshold']==threshold_value(spec,r['features'])
                    if r['intervened']:assert r['baseline_growth_blocks']-r['actual_growth_blocks']>=r['threshold']
                    if name=='old':
                        oldspec=dict(kind='linear',intercept=8,features=['work_free_ratio','remaining_cv'],coefficients=[128,0])
                        old=previous_select([State(**row) for row in r['state']],1024,256,r['free_before'],cap,16,32,12288,oldspec)
                        if old['selected'] is not None:assert old['selected'].tokens==r['scheduled']
                        assert old['threshold']==r['threshold'];equivalence+=1
                    done=[]
                    for rid in out.num_scheduled_tokens:
                        request=s.requests[rid]
                        if request.num_computed_tokens==request.num_tokens:request.append_output_token_ids(100)
                        if len(request.output_token_ids)==targets[rid]:done.append(rid)
                    s.finish_requests(done,RequestStatus.FINISHED_LENGTH_CAPPED)
                    if not s.requests:break
                assert not s.requests,(cap,name,source)
                results.append(dict(capacity=cap,formula=name,source=source,rounds=len(s.round_rows),
                    interventions=sum(r['intervened'] for r in s.round_rows),preemptions=sum(r['preemptions'] for r in s.round_rows)))
    write(OUT/'cpu_validation.json',dict(passed=True,old_equivalence_rounds=equivalence,closed_loops=results,
        monotonicity_checks=True,clock='Synchronous unit steps only; no performance labels.'))
    print('Passed',len(results),'native loops and',equivalence,'old-formula equivalence rounds')


if __name__=='__main__':main()
