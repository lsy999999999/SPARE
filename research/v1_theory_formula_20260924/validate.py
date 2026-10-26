"""CPU native allocator checks, sign checks, c=0 ablation equivalence."""
from pathlib import Path
import math
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_theory_formula_20260924.common import *
from research.v1_theory_formula_20260924.policy import values,arms,select
from research.v1_theory_formula_20260924.scheduler import FormulaScheduler
from research.v1_simple_formula_20260924.policy import select as previous_select
from research.v1_decode_load_formula_20260924.policy import select as previous_decode_select
from research.v1_theory_formula_20260924.analysis.report import reconstruct_threshold
from research.v1_budget_20260922.test_scheduler import scheduler
from research.v1_budget_20260922.policy import State
from research.v1_saving_threshold_20260924.policy import threshold_value
from vllm.v1.request import Request, RequestStatus
from vllm import SamplingParams


def main():
    d=read(OUT/'design.json')
    for name,spec in d['arms'].items():
        for work in (0,16,2048,16384):
            ns=[threshold_value(spec,values(free,work,128,2048)) for free in (0,1,16,64,512)]
            assert ns==sorted(ns),name
    results=[];equivalence=0
    for cap in (768,1536):
        for name,spec in d['arms'].items():
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
                    assert r['threshold']==reconstruct_threshold(spec,r)
                    if r['intervened']:assert r['baseline_growth_blocks']-r['actual_growth_blocks']>=r['threshold']
                    if name=='old':
                        oldspec=dict(kind='linear',intercept=8,features=['work_free_ratio','remaining_cv'],coefficients=[128,0])
                        old=previous_select([State(**row) for row in r['state']],1024,256,r['free_before'],cap,16,32,12288,oldspec)
                        if old['selected'] is not None:assert old['selected'].tokens==r['scheduled']
                        assert old['threshold']==r['threshold'];equivalence+=1
                    if name in ('decode','constant16'):
                        old=previous_decode_select([State(**row) for row in r['state']],1024,256,r['free_before'],cap,16,32,12288,spec)
                        if old['selected'] is not None:assert old['selected'].tokens==r['scheduled']
                        assert old['threshold']==r['threshold']
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
        monotonicity_checks=True,clock='Synchronous unit steps only; no performance labels.',
        sources=source_hashes(),trace_hashes=d['trace_hashes'],design_sha256=digest(OUT/'design.json')))
    print('Passed',len(results),'native loops and',equivalence,'old-formula equivalence rounds')


if __name__=='__main__':main()
