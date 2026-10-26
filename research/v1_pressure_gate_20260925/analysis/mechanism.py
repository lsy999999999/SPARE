"""Offline matched-state diagnosis; temporal association is not causal attribution."""
import csv,gzip,json,math,statistics as st
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'artifacts/v1_decode_arrival_confirm_20260925_r1'
OUT=ROOT/'artifacts/v1_pressure_gate_20260925_r1'

def main():
    results=[]
    for p in sorted((OLD/'runs').glob('*.json')):
        r=json.loads(p.read_text());j=r['job']
        if j['phase']!='compare' or j['arm']!='combined':continue
        rows=json.load(gzip.open(OLD/r['rounds_path'],'rt'))
        blocked=[];changed=[];eligible=0;restored=0
        for i,row in enumerate(rows):
            if row['baseline'] is None:continue
            f=row['features'];x=row['free_before']/(1+(f['remaining_prefill']+f['arrival_rate'])/16)
            u=1-f['decoded_kv_blocks']/(row['total_blocks']-1)
            ns=[max(1,min(100000,math.ceil(8+128*x+delta))) for delta in (0,32*u,32*u*x/(1+x))]
            def action(n):
                if len(row['state'])>=2 and row['baseline_growth_blocks']>0:
                    for c in row['candidate_diagnostics']:
                        if c['tokens'] is not None and c['guard'] and c['saving']>=n:return c['tokens']
                return row['baseline']
            a,d,g=map(action,ns)
            assert d==row['scheduled'] and ns[1]==row['threshold']
            eligible+=action(1)!=row['baseline']
            if a!=d:changed.append(i)
            if a!=row['baseline'] and d==row['baseline']:
                blocked.append(dict(round=i,x=x,free_fraction=row['free_before']/(row['total_blocks']-1),
                    d_headroom=u,n_a=ns[0],n_da=ns[1],n_gate=ns[2],
                    gate_restores=g!=row['baseline'],next10_preemptions=sum(v['preemptions'] for v in rows[i+1:i+11])))
                restored+=g!=row['baseline']
        results.append(dict(key=j['key'],case=j['case'],capacity=j['capacity'],repeat=j['repeat'],
            eligible_rounds=eligible,changed_vs_a=len(changed),d_blocks_a=len(blocked),gate_restores=restored,
            blocked_x_median=st.median(v['x'] for v in blocked) if blocked else None,
            blocked_free_fraction_median=st.median(v['free_fraction'] for v in blocked) if blocked else None,
            blocked_headroom_median=st.median(v['d_headroom'] for v in blocked) if blocked else None,
            blocked_low_x=sum(v['x']<1 for v in blocked),
            blocked_followed_preemption10=sum(v['next10_preemptions']>0 for v in blocked),
            blocked_events=blocked,metrics=r['metrics']))
    (OUT/'analysis').mkdir(parents=True,exist_ok=True)
    (OUT/'analysis/prior_mechanism.json').write_text(json.dumps(dict(
        scope='All 24 combined trajectories from prior 17:17 experiment. Same-state action reconstruction; next10 is retrospective diagnostic only, overlapping windows, not causal evidence.',
        runs=results),indent=2)+'\n')
    for cap in (768,1536):
        for case in ('code_accelerating','conversation_steady'):
            rr=[r for r in results if r['capacity']==cap and r['case']==case]
            print(cap,case,[{k:r[k] for k in ('d_blocks_a','gate_restores','blocked_x_median','blocked_free_fraction_median','blocked_headroom_median','blocked_followed_preemption10')} for r in rr])

if __name__=='__main__':main()
