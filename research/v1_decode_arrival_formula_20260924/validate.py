"""Native allocator replay with controlled causal arrival clocks; not a timing model."""
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_decode_arrival_formula_20260924.common import *
from research.v1_decode_arrival_formula_20260924.policy import ArrivalRate,formula,select
from research.v1_decode_arrival_formula_20260924.scheduler import FormulaScheduler
from research.v1_simple_formula_20260924.policy import select as previous_select
from research.v1_decode_load_formula_20260924.policy import select as previous_decode_select,formula as previous_decode_formula
from research.v1_arrival_formula_20260924.policy import select as previous_arrival_select,formula as previous_arrival_formula
from research.v1_budget_20260922.test_scheduler import scheduler
from research.v1_budget_20260922.policy import State
from vllm.v1.request import Request,RequestStatus
from vllm import SamplingParams


def main():
    rate=ArrivalRate(1)
    rate.observe(0,100);assert rate.at(0)==100
    assert math.isclose(rate.at(math.log(2)),50)
    rate.observe(1,200);assert math.isclose(rate.at(2),100*math.exp(-2)+200*math.exp(-1))
    try:rate.at(1)
    except AssertionError:pass
    else:raise AssertionError('Estimator accepted backward time')
    rows=[State(str(i),0,1024,1024,0,0,False) for i in range(8)]
    ns=[select(rows,1024,256,400,768,16,32,12288,formula(1),dict(arrival_rate=a))['threshold'] for a in (0,100,1000,10000)]
    assert ns==sorted(ns,reverse=True)
    for rate_value in (0,100,1000,10000):
        thresholds=[select(rows,1024,256,400,768,16,32,12288,formula(tau,k),dict(arrival_rate=rate_value))['threshold']
                    for tau,k in ((1,0),(1,32),(0,32))]
        assert thresholds==sorted(thresholds)
    s=scheduler(blocks=768,cls=FormulaScheduler)
    clock=[0.];s.arrival_clock=lambda:clock[0]
    s.configure('threshold',**BASE,threshold=formula(1));s.arrival_estimator.observe(0,100)
    clock[0]=2;s.configure('threshold',**BASE,threshold=formula(1))
    assert s.arrival_estimator.at(0)==0 and not s.arrival_ids
    d=read(OUT/'design.json');results=[];equivalence=0;rate_checks=0;control_checks=0
    for cell in d['cells']:
        for name,spec in d['arms'].items():
            reqs=read(OUT/f"inputs/{cell['case']}.json")
            s=scheduler(blocks=cell['capacity'],cls=FormulaScheduler)
            clock=[0.];s.arrival_clock=lambda:clock[0];s.configure('threshold',**BASE,threshold=spec)
            cursor=0;events=[];targets={r['id']:r['output'] for r in reqs}
            for step in range(20000):
                while cursor<len(reqs) and reqs[cursor]['arrival_ms']/1000<=clock[0]:
                    r=reqs[cursor]
                    s.add_request(Request(r['id'],[100]*r['prompt'],None,None,None,
                        SamplingParams(max_tokens=r['output']),None,None,arrival_time=clock[0]))
                    events.append((clock[0],r['prompt']));cursor+=1
                if not s.requests:
                    if cursor==len(reqs):break
                    clock[0]=reqs[cursor]['arrival_ms']/1000;continue
                out=s.schedule();row=s.round_rows[-1]
                expected=sum(prompt*math.exp(-(clock[0]-t)) for t,prompt in events)
                assert math.isclose(row['features']['arrival_rate'],expected,rel_tol=1e-10,abs_tol=1e-9)
                assert row['features']['arrival_observed_tokens']==sum(p for _,p in events)
                rate_checks+=1
                work=sum(max(0,r['prompt']-r['computed']) for r in row['state'] if r['outputs']==0)
                held=sum(r['blocks'] for r in row['state'] if r['outputs']>0)
                n=max(1,min(100000,math.ceil(8+128*row['free_before']/(1+(work+spec['tau_s']*expected)/16)
                                            +spec['coefficients'][1]*(1-held/(cell['capacity']-1)))))
                assert n==row['threshold']
                if name=='old':
                    oldspec=dict(kind='linear',intercept=8,features=['work_free_ratio','remaining_cv'],coefficients=[128,0])
                    old=previous_select([State(**r) for r in row['state']],1024,256,row['free_before'],cell['capacity'],16,32,12288,oldspec)
                    assert old['threshold']==row['threshold']
                    if old['selected'] is not None:assert old['selected'].tokens==row['scheduled']
                    equivalence+=1
                if name in ('decode','arrival'):
                    selector=previous_decode_select if name=='decode' else previous_arrival_select
                    previous_spec=previous_decode_formula(32) if name=='decode' else previous_arrival_formula(1)
                    old=selector([State(**r) for r in row['state']],1024,256,row['free_before'],cell['capacity'],16,32,12288,
                                 previous_spec,dict(arrival_rate=expected))
                    assert old['threshold']==row['threshold']
                    if old['selected'] is not None:assert old['selected'].tokens==row['scheduled']
                    control_checks+=1
                if row['intervened']:assert row['baseline_growth_blocks']-row['actual_growth_blocks']>=row['threshold']
                done=[]
                for rid in out.num_scheduled_tokens:
                    r=s.requests[rid]
                    if r.num_computed_tokens==r.num_tokens:r.append_output_token_ids(100)
                    if len(r.output_token_ids)==targets[rid]:done.append(rid)
                s.finish_requests(done,RequestStatus.FINISHED_LENGTH_CAPPED)
                clock[0]+=.05
            assert cursor==len(reqs) and not s.requests,(cell,name)
            results.append(dict(**cell,arm=name,rounds=len(s.round_rows),
                interventions=sum(r['intervened'] for r in s.round_rows),preemptions=sum(r['preemptions'] for r in s.round_rows)))
    write(OUT/'cpu_validation.json',dict(passed=True,closed_loops=results,old_equivalence_rounds=equivalence,
        causal_rate_checks=rate_checks,decode_arrival_equivalence_rounds=control_checks,
        sources=source_hashes(),design_sha256=digest(OUT/'design.json'),
        clock='Fixed 50 ms simulator clock only for arrival causality; not calibrated performance.'))
    print('Passed',len(results),'closed loops,',equivalence,'old equivalence rounds,',rate_checks,'causal arrival checks')


if __name__=='__main__':main()
