"""Independent recomputation from token clocks, physical records and inputs."""
import gzip
import math
from pathlib import Path
import statistics as st
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_saving_threshold_20260924.common import *


def audit_run(path,inputs):
    r=read(path);j=r['job']
    expected=j.get('requests')
    if expected is None:expected=read(INPUT/inputs['cases'][j['case']]['file'])
    assert r['requests']==expected,(path,'inputs')
    assert digest(OUT/r['rounds_path'])==r['rounds_sha256'],path
    with gzip.open(OUT/r['rounds_path'],'rt') as f:rounds=json.load(f)
    assert len(rounds)==r['metrics']['round_count']
    if r['status']!='complete':
        assert r['metrics']['mean_ttft_ms'] is None
        return dict(key=j['key'],status=r['status'],checked_partial=True,error=r['error'])
    arrivals={q['id']:q for q in expected};ttft=[];e2e=[];events=[]
    for rid,q in arrivals.items():
        c=r['clocks'][rid];times=c['token_times_ms']
        assert len(times)==q['output']==c['tokens']
        assert c['arrival_ms']==q['arrival_ms']
        assert times[0]==c['first_ms'] and times[-1]==c['finish_ms']
        assert times[0]>=c['arrival_ms']
        assert all(b>=a for a,b in zip(times,times[1:]))
        ttft.append(times[0]-c['arrival_ms']);e2e.append(times[-1]-c['arrival_ms'])
        events.extend([(c['arrival_ms'],1),(times[0],-1)])
    assert math.isclose(st.fmean(ttft),r['metrics']['mean_ttft_ms'],abs_tol=1e-7)
    assert math.isclose(st.fmean(e2e),r['metrics']['mean_e2e_ms'],abs_tol=1e-7)
    waiting=0;area=0.;previous=0.
    for clock,delta in sorted(events):
        area+=waiting*(clock-previous);waiting+=delta;previous=clock
        assert waiting>=0
    assert waiting==0 and math.isclose(area,sum(ttft),rel_tol=1e-10,abs_tol=1e-5)
    scheduled=0;recomputed=0;interventions=0
    for index,row in enumerate(rounds):
        assert row['round']==index
        assert 0<=row['free_after']<=row['usable_blocks']
        allocation=row['scheduled'];scheduled+=sum(allocation.values())
        assert sum(allocation.values())<=row['budget']
        assert set(allocation)<=set(arrivals)
        states={s['rid']:s for s in row['state']}
        if row['verified']:
            physical=sum(max(0,math.ceil((states[rid]['computed']+n)/16)-states[rid]['blocks'])
                         for rid,n in allocation.items())
            assert physical==row['free_before']-row['free_after']==row['actual_growth_blocks']
        if row['baseline'] is not None:
            for rid,n in row['baseline'].items():
                s=states[rid];remaining=s['tokens']-s['computed']
                protected=(s['outputs']>0 and remaining==1) or (s['outputs']==0 and n==remaining)
                if protected:assert allocation.get(rid,0)>=n
        if row['intervened']:
            assert row['threshold'] is not None
            assert row['baseline_growth_blocks']-row['actual_growth_blocks']>=row['threshold']
        recomputed+=row['recomputed_tokens'];interventions+=row['intervened']
    necessary=sum(q['prompt']+q['output']-1 for q in expected)
    assert scheduled-necessary==recomputed,(j['key'],scheduled,necessary,recomputed)
    assert recomputed==r['metrics']['recomputed_tokens']
    assert interventions==r['metrics']['interventions']
    return dict(key=j['key'],status='complete',ttft_ms=st.fmean(ttft),waiting_area_ms=area,
                scheduled_tokens=scheduled,necessary_tokens=necessary,recomputed_tokens=recomputed,
                rounds=len(rounds),physical_and_guards=True)


def main():
    inputs=read(OUT/'inputs_manifest.json');m=read(OUT/'manifest.json')
    assert digest(OUT/'inputs_manifest.json')==m['input_manifest_sha256']
    assert m['source_hashes']==sources()
    results=[];errors=[]
    for p in sorted((OUT/'runs').glob('*.json')):
        try:results.append(audit_run(p,inputs))
        except Exception as exc:errors.append(dict(file=p.name,error=repr(exc)))
    for alias in inputs['aliases']:
        a=inputs['cases'][alias['job']['case']]
        canonical=next(j for j in inputs['jobs'] if j['key']==alias['canonical'])
        b=inputs['cases'][canonical['case']]
        assert a['trace_digest']==b['trace_digest']
        assert alias['job']['capacity']==canonical['capacity']
        assert alias['job']['threshold']==canonical['threshold']
    result=dict(passed=not errors,runs_checked=len(results),errors=errors,results=results,
        aliases_checked=len(inputs['aliases']),claim='independent arithmetic recomputation, not external review')
    write(OUT/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)
    assert not errors,errors[:3]


if __name__=='__main__':main()
