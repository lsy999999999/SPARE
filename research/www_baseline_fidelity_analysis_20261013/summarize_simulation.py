"""Summarize all frozen CPU comparisons and decision diagnostics."""
from pathlib import Path
import csv
import gzip
import json
import statistics as st
import sys
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.www_baseline_fidelity_20261013.common import *
rows=read(OUT/'simulation/results.json')
assert len(rows)==432 and read(OUT/'simulation/status.json')['state']=='complete'
summary=[];effects=[];diagnostics=[]
metrics=['ttft_ms_mean','ttft_ms_p99','e2e_ms_mean','tpot_ms_mean','token_itl_ms_p99','preemptions','recomputed_tokens','slo_ttft5000_maxitl200_goodput_rps','cpu_schedule_including_audit_ms_mean']
for profile in ('resource','resource_staircase'):
    for arm in ARMS:
        selected=[r for r in rows if r['profile']==profile and r['arm']==arm]
        summary.append(dict(profile=profile,arm=arm,cells=len(selected),**{m:st.fmean(r['metrics'][m] for r in selected) for m in metrics}))
    for old,new in [('past_future','past_future_multi'),('sola_rule','sola_calibrated')]:
        pairs={r['cell']:r for r in rows if r['profile']==profile and r['arm']==old}
        changes=[100*(pairs[r['cell']]['metrics']['ttft_ms_mean']-r['metrics']['ttft_ms_mean'])/pairs[r['cell']]['metrics']['ttft_ms_mean'] for r in rows if r['profile']==profile and r['arm']==new]
        effects.append(dict(profile=profile,old=old,new=new,ttft_paired_improvement_pct=st.fmean(changes),wins=sum(v>0 for v in changes),losses=sum(v<0 for v in changes)))
    for arm in ('past_future_multi','sola_calibrated'):
        count=Counter();gamma=[]
        for path in (OUT/'simulation'/profile).glob(f'*/{arm}/iterations.jsonl.gz'):
            with gzip.open(path,'rt') as f:
                for line in f:
                    r=json.loads(line);count['rounds']+=1
                    if arm=='past_future_multi':
                        count['sample_rounds_'+str(r.get('pf_samples',0))]+=1
                        count['gate_checks']+=r.get('pf_gate_checked',0)
                        count['gate_rejections']+=r.get('pf_gate_rejected',0)
                    else:
                        count['phase_'+str(r.get('sola_phase'))]+=1
                        count['fallback_'+str(r.get('sola_fallback'))]+=1
                        count['held_running_rounds']+=r.get('sola_held_running',0)>0
                        count['constraint_satisfied_rounds']+=bool(r.get('sola_constraint_satisfied'))
                        if 'sola_gamma' in r:gamma.append(r['sola_gamma'])
        diagnostics.append(dict(profile=profile,arm=arm,counters=dict(count),gamma_mean=st.fmean(gamma) if gamma else None))
analysis=OUT/'simulation/analysis';analysis.mkdir(exist_ok=True)
for name,data in [('aggregate',summary),('new_vs_old',effects)]:
    with (analysis/(name+'.csv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
write(analysis/'summary.json',dict(complete=True,runs=len(rows),aggregate=summary,new_vs_old=effects,diagnostics=diagnostics,scope='synthetic mechanism replay; not GPU performance',analysis_source_sha256=digest(Path(__file__))))
print(json.dumps(dict(new_vs_old=effects,diagnostics=diagnostics),indent=2))
