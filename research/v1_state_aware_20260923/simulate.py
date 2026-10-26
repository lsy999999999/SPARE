"""Detached CPU allocation experiments; deliberately no uncalibrated TTFT claim."""
import gzip
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_state_aware_20260923.common import *


def diagnose_old():
    target=OUT/'old_diagnostics.json'
    if target.exists(): return
    result=[]
    for path in sorted((OLD/'runs').glob('*.json')):
        row=read(path); j=row['job']
        if j.get('block') not in BLOCKS or j.get('capacity') not in CAPS or j.get('arrival')!='original': continue
        if j['method'] not in ('a_frozen','a_any_positive_saving') or j.get('repeat')!=0: continue
        trace=OLD/row['rounds_path']
        if not trace.exists(): continue
        assert digest(trace)==row['rounds_sha256']
        with gzip.open(trace,'rt') as f: rounds=json.load(f)
        bins={}
        for r in rounds:
            g=r['baseline_growth_blocks']; free=r['free_before']
            bucket='fallback' if g is None else 'zero' if g==0 else (
                'demand_below_quarter_free' if 4*g<max(free,1) else 'demand_at_least_quarter_free')
            b=bins.setdefault(bucket,dict(rounds=0,interventions=0,preemptions=0,recomputed_tokens=0))
            b['rounds']+=1; b['interventions']+=r['intervened']
            b['preemptions']+=r['preemptions']; b['recomputed_tokens']+=r['recomputed_tokens']
        result.append(dict(job=j,metrics=row['metrics'],bins=bins,trace_sha256=digest(trace)))
    write(target,dict(records=result,missing=not bool(result),claim='descriptive old states; not causal counterfactual effects'))


def run(block,cap,method,cfg):
    from vllm import SamplingParams
    from vllm.v1.request import Request, RequestStatus
    from research.v1_budget_20260922.test_scheduler import scheduler
    from research.v1_state_aware_20260923.scheduler import StateScheduler
    s=scheduler(blocks=cap,cls=StateScheduler)
    s.configure(method,**cfg)
    reqs=read(INPUT/(block+'_original.json'))
    targets={r['id']:r['output'] for r in reqs}
    for i,r in enumerate(reqs):
        s.add_request(Request(r['id'],[100]*r['prompt'],None,None,None,
            SamplingParams(max_tokens=r['output']),None,None,arrival_time=float(i)))
    stats=dict(rounds=0,preemptions=0,recomputed_tokens=0,interventions=0,max_consecutive_reductions=0)
    action_hash=__import__('hashlib').sha256()
    started=time.monotonic()
    while s.requests and stats['rounds']<100000 and time.monotonic()-started<180:
        out=s.schedule(); r=s.round_rows[-1]
        for key,field in [('rounds',None),('preemptions','preemptions'),('recomputed_tokens','recomputed_tokens'),('interventions','intervened')]:
            stats[key]+=1 if field is None else r[field]
        stats['max_consecutive_reductions']=max(stats['max_consecutive_reductions'],r['max_consecutive_reductions'])
        action_hash.update(json.dumps(out.num_scheduled_tokens,sort_keys=True).encode())
        done=[]
        for rid in out.num_scheduled_tokens:
            req=s.requests[rid]
            if req.num_computed_tokens==req.num_tokens: req.append_output_token_ids(100)
            if len(req.output_token_ids)>=targets[rid]: done.append(rid)
        s.finish_requests(done,RequestStatus.FINISHED_LENGTH_CAPPED)
        s.round_rows.clear()
    return dict(block=block,capacity=cap,method=method,config=cfg,complete=not s.requests,
                remaining_requests=len(s.requests),stats=stats,action_sha256=action_hash.hexdigest(),
                cpu_wall_s=time.monotonic()-started,clock='unit scheduler steps; all arrivals at start; not latency')


def main():
    check_manifest(); diagnose_old()
    variants=[(m,config(m)) for m in METHODS[:4]]
    for p in ('1/8','1/4','1/2'):
        variants.append(('pressure_one',dict(CONFIG,pressure=p,saving='1/2')))
        for save in ('1/4','1/2'):
            variants.append(('pressure_relative',dict(CONFIG,pressure=p,saving=save)))
    expected=len(BLOCKS)*len(CAPS)*len(variants)
    count=0
    for block in BLOCKS:
        for cap in CAPS:
            for i,(method,cfg) in enumerate(variants):
                key=f'{block}_k{cap}_v{i}'
                path=OUT/'simulation'/(key+'.json')
                state('simulation_status',state='running',job=key,saved=count,expected=expected)
                if path.exists():
                    r=read(path); assert (r['block'],r['capacity'],r['method'],r['config'])==(block,cap,method,cfg)
                else:
                    r=run(block,cap,method,cfg); write(path,r)
                count+=1
                print(key,r['complete'],r['stats'],flush=True)
    rows=[read(p) for p in sorted((OUT/'simulation').glob('*.json'))]
    write(OUT/'simulation_summary.json',dict(complete=True,runs=len(rows),expected=expected,
        noncompletion=[r for r in rows if not r['complete']],
        performance_calibrated=False,gpu_parameters_selected_by='predeclared constants, not synthetic performance',rows=rows))
    state('simulation_status',state='finished',saved=count,expected=expected)


if __name__=='__main__': main()
