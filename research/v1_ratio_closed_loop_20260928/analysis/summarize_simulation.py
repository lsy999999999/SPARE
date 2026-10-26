"""Report synthetic timing sensitivity without treating it as GPU measurement."""
import csv,statistics as st,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from research.v1_ratio_closed_loop_20260928.common import *
from research.v1_ratio_closed_loop_20260928.policy import pairs
d=verify();s=read(OUT/'simulation_summary.json')
assert s['complete'] and s['runs']==528 and s['source_hashes']==d['source_hashes']
rows=s['rows'];fields=('source','window','shape','budget','chunk','profile')
index={tuple(r[k] for k in fields)+(r['arm'],):r for r in rows}
comparisons=[]
for r in rows:
    if r['arm']=='fixed64':continue
    base=index[tuple(r[k] for k in fields)+('fixed64',)]
    same=pairs(r['budget'],r['chunk'],r['arm'])==pairs(r['budget'],r['chunk'],'fixed64')
    if same:assert r['trajectory_sha256']==base['trajectory_sha256'] and r['different_from_fixed64']==0
    comparisons.append(dict(**{k:r[k] for k in fields},arm=r['arm'],same_candidates=same,
        improvement_pct=100*(base['mean_ttft_ms']-r['mean_ttft_ms'])/base['mean_ttft_ms'],
        base_ttft_ms=base['mean_ttft_ms'],arm_ttft_ms=r['mean_ttft_ms'],
        changed_rounds=r['different_from_fixed64'],rounds=r['rounds'],
        selected_small=sum(r['selected_candidates'][str(i)] for i in (2,3)),
        baseline_preemptions=base['preemptions'],arm_preemptions=r['preemptions'],
        baseline_recomputed=base['recomputed_tokens'],arm_recomputed=r['recomputed_tokens']))
def stats(rs):
    return dict(scenes=len(rs),mean_improvement_pct=st.fmean(r['improvement_pct'] for r in rs),
        min_improvement_pct=min(r['improvement_pct'] for r in rs),max_improvement_pct=max(r['improvement_pct'] for r in rs),
        changed_rounds=sum(r['changed_rounds'] for r in rs),rounds=sum(r['rounds'] for r in rs),
        with_changed_action=sum(r['changed_rounds']>0 for r in rs))
groups=[]
for profile in ('resource','resource_staircase'):
    for arm in ARMS[1:]:
        rr=[r for r in comparisons if r['profile']==profile and r['arm']==arm]
        groups.append(dict(profile=profile,arm=arm,all=stats(rr),changed_configs=stats([r for r in rr if not r['same_candidates']]),
            by_config=[dict(budget=t,chunk=k,**stats([r for r in rr if r['budget']==t and r['chunk']==k])) for t,k in SIM_CONFIGS]))
write(OUT/'analysis/simulation_comparisons.json',comparisons)
write(OUT/'analysis/simulation_analysis.json',dict(complete=True,runs=len(rows),rounds=sum(r['rounds'] for r in rows),
    all_equivalence_controls_passed=True,equivalent_controls=sum(r['same_candidates'] for r in comparisons),
    timing_claim='synthetic-uncalibrated',groups=groups))
with (OUT/'analysis/simulation_comparisons.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
print(json.dumps(groups,ensure_ascii=False,indent=2))
