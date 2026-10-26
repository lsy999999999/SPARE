"""Native allocator simulation and prior-policy equivalence; artificial clocks only."""
import argparse,math,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_broad12h_20260925.common import *
from research.v1_broad12h_20260925.policy import formula,select,ArrivalRate
from research.v1_broad12h_20260925.scheduler import FormulaScheduler
from research.v1_pressure_gate_20260925.policy import select as prior_select
from research.v1_budget_20260922.test_scheduler import scheduler
from research.v1_budget_20260922.policy import State
from vllm.v1.request import Request,RequestStatus
from vllm import SamplingParams

def replay(cell,arm,spec,dt=.05):
    reqs=read(OUT/f"inputs/{cell['case']}.json");s=scheduler(blocks=cell['capacity'],cls=FormulaScheduler)
    s.max_model_len=min(12288,(cell['capacity']-1)*16)
    clock=[0.];s.arrival_clock=lambda:clock[0];s.configure('threshold',**BASE,threshold=spec)
    cursor=0;events=[];targets={r['id']:r['output'] for r in reqs};checks=0;first={};bins=dict(crowded=0,partial_prefill=0,decode_dominant=0,recomputing=0)
    for step in range(24000):
        while cursor<len(reqs) and reqs[cursor]['arrival_ms']/1000<=clock[0]:
            r=reqs[cursor];s.add_request(Request(r['id'],[100]*r['prompt'],None,None,None,SamplingParams(max_tokens=r['output']),None,None,arrival_time=clock[0]));events.append((clock[0],r['prompt']));cursor+=1
        if not s.requests:
            if cursor==len(reqs):break
            clock[0]=reqs[cursor]['arrival_ms']/1000;continue
        out=s.schedule();row=s.round_rows[-1];expected=sum(prompt*math.exp(-(clock[0]-t)) for t,prompt in events)
        assert math.isclose(row['features']['arrival_rate'],expected,rel_tol=1e-10,abs_tol=1e-8)
        work=sum(max(0,r['prompt']-r['computed']) for r in row['state'] if r['outputs']==0)
        held=sum(r['blocks'] for r in row['state'] if r['outputs']>0)
        assert row['free_before']+sum(r['blocks'] for r in row['state'])==cell['capacity']-1
        x=row['free_before']/(1+(work+spec['tau_s']*expected)/16);u=1-held/(cell['capacity']-1);g=x/(1+x)
        ext={'decoded_kv_headroom':u,'pressure_decoded_headroom':u*g,'pressure_gate':g}[spec['features'][1]]
        n=max(1,min(100000,math.ceil(8+128*x+spec['coefficients'][1]*ext)))
        assert n==row['threshold']
        prior=prior_select([State(**r) for r in row['state']],1024,256,row['free_before'],cell['capacity'],16,32,s.max_model_len,spec,dict(arrival_rate=expected))
        assert prior['threshold']==n
        if prior['selected'] is not None:assert prior['selected'].tokens==row['scheduled']
        checks+=1
        if row['intervened']:assert row['baseline_growth_blocks']-row['actual_growth_blocks']>=n
        bins['crowded']+=row['free_before']/(cell['capacity']-1)<.1
        bins['partial_prefill']+=any(0<r['computed']<r['prompt'] and r['outputs']==0 for r in row['state'])
        bins['decode_dominant']+=held/(cell['capacity']-1)>.5
        bins['recomputing']+=row['features']['recomputing']>0
        done=[]
        for rid in out.num_scheduled_tokens:
            r=s.requests[rid]
            if r.num_computed_tokens==r.num_tokens:r.append_output_token_ids(100);first.setdefault(rid,clock[0])
            if len(r.output_token_ids)==targets[rid]:done.append(rid)
        s.finish_requests(done,RequestStatus.FINISHED_LENGTH_CAPPED);clock[0]+=dt
    assert cursor==len(reqs) and not s.requests,(cell,arm)
    return dict(**cell,arm=arm,dt=dt,passed=True,rounds=len(s.round_rows),equivalence_checks=checks,
        interventions=sum(r['intervened'] for r in s.round_rows),preemptions=sum(r['preemptions'] for r in s.round_rows),state_coverage=bins,
        clock_note='Artificial step time; no GPU latency claim.')

def main(full):
    d=read(OUT/'design.json');assert read(OUT/'theory_checks.json')['passed']
    if full:
        cells=d['cells'];base=OUT/'simulator/full';status=OUT/'simulator/status.json'
    else:
        # Cover each capacity, each source including natural relative gaps, and all stress types.
        chosen={}
        for i,cap in enumerate(d['capacities']):
            c=next(c for c in d['cells'] if c['capacity']==cap and c['source']==d['source_families'][i%4] and c['shape']=='accelerating')
            chosen[c['case'],cap]=c
        for c in d['cells']:
            if c['family']=='synthetic' and c['capacity']==512 or c['shape']=='trace_relative' and c['capacity']==768:
                chosen[c['case'],c['capacity']]=c
        cells=list(chosen.values());base=OUT/'simulator/preflight';status=OUT/'simulator/preflight_status.json'
    results=[];expected=len(cells)*len(d['methods'])
    for cell in cells:
        for arm in d['methods']:
            key=f"{cell['case']}_c{cell['capacity']}_{arm}";path=base/(key+'.json')
            if path.exists():r=read(path)
            else:r=replay(cell,arm,d['arms'][arm]);write(path,r)
            assert r['passed'];results.append(r)
            write(status,dict(state='running',completed=len(results),expected=expected,updated=time.time(),last=key))
    if full:
        # Repeat selected endpoints with two alternative clocks as sensitivity of decisions only.
        sensitivity=[]
        for cell in [c for c in cells if c['family']=='synthetic' and c['capacity'] in (512,4096)]:
            for arm in ('old','pressure_constant'):
                for dt in (.02,.1):
                    r=replay(cell,arm,d['arms'][arm],dt);sensitivity.append(r);write(OUT/f"simulator/sensitivity/{cell['case']}_c{cell['capacity']}_{arm}_{dt}.json",r)
        write(OUT/'simulator/full_complete.json',dict(passed=True,loops=len(results),sensitivity_loops=len(sensitivity),rounds=sum(r['rounds'] for r in results)))
    else:
        write(OUT/'cpu_validation.json',dict(passed=True,loops=len(results),rounds=sum(r['rounds'] for r in results),
            sources=source_hashes(),design_sha256=digest(OUT/'design.json'),theory_checks_sha256=digest(OUT/'theory_checks.json')))
    write(status,dict(state='complete',completed=len(results),expected=expected,updated=time.time()))
    print('Passed',len(results),'closed loops;',sum(r['rounds'] for r in results),'rounds',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');a=p.parse_args();main(a.full)
