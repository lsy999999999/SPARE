"""Independent request/round audit and old/new paired comparisons."""
from pathlib import Path
import csv
import gzip
import json
import random
import statistics as st
import sys
from collections import defaultdict,Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.www_baseline_fidelity_20261013.common import *


def ci(values,seed=20261013):
    rng=random.Random(seed)
    draws=sorted(st.fmean(rng.choice(values) for _ in values) for _ in range(10000))
    return [draws[249],draws[9749]]


def main():
    design=read(OUT/'design.json')
    summary=read(OUT/'analysis/summary.json')
    assert summary['runs']==864 and summary['groups']==36
    runs={};counts=defaultdict(Counter)
    failures=[];quality=[];prediction_errors=[]
    for cell in design['cells']:
        group=read(OUT/'groups'/f"{cell['key']}.json")
        folder=OUT/group['folder']
        quality.append(dict(cell=cell['key'],attempt=group['attempt'],gpu=group['gpu'],aa=read(folder/'aa.json')))
        for arm in ARMS:
            runs[cell['key'],arm]=[]
            for rep in range(4):
                r=read(folder/'runs'/f'{arm}_r{rep}.json')
                assert r['status']=='complete' and r['metrics']['completed']==128
                with gzip.open(folder/r['rounds_path'],'rt') as f:trace=json.load(f)
                executed=sum(sum(row['scheduled'].values()) for row in trace)
                recompute=sum(row['recomputed_tokens'] for row in trace)
                expected=sum(q['prompt_tokens']+q['output_tokens']-1 for q in r['requests'])
                assert executed-recompute==expected,(cell['key'],arm,rep,executed,recompute,expected)
                assert abs(st.fmean(q['ttft_ms'] for q in r['requests'])-r['metrics']['ttft_ms_mean'])<1e-8
                gaps=[]
                for q in r['requests']:
                    clock=r['clocks'][q['request_id']]
                    ts=clock['token_times_ms']
                    assert len(ts)==q['output_tokens']
                    assert ts==sorted(ts)
                    assert abs(ts[0]-q['arrival_ms']-q['ttft_ms'])<1e-8
                    gaps.extend(b-a for a,b in zip(ts,ts[1:]))
                if gaps:assert abs(st.fmean(gaps)-r['metrics']['token_itl_ms_mean'])<1e-8
                good=sum(q['ttft_ms']<=5000 and (q['max_itl_ms'] or 0)<=200 for q in r['requests'])
                assert abs(good/128-r['metrics']['slo_ttft5000_maxitl200_attainment'])<1e-12
                runs[cell['key'],arm].append(r['metrics'])
                for row in trace:
                    assert 0<=row['free_after']<=cell['capacity']-1
                    counts[arm]['rounds']+=1
                    counts[arm]['held_running_rounds']+=row.get('sola_held_running',0)>0
                    counts[arm]['multi_sample_rounds']+=row.get('pf_samples',0)>1
                    counts[arm]['gate_rejections']+=row.get('pf_gate_rejected',0)
                    if arm=='sola_calibrated':
                        if 'sola_predicted_prefill_ms' in row:
                            pred=row['sola_predicted_prefill_ms']+row['sola_predicted_decode_ms']
                            prediction_errors.append(abs(pred-row['step_ms'])/row['step_ms'])
                        counts[arm]['phase_'+str(row.get('sola_phase'))]+=1
                        counts[arm]['fallback_'+str(row.get('sola_fallback'))]+=1
                        counts[arm]['constraint_satisfied_rounds']+=bool(row.get('sola_constraint_satisfied'))
    comparisons=[];cell_effects=[]
    pairs=[('past_future','past_future_multi'),('sola_rule','sola_calibrated'),('past_future_multi','arrival'),('sola_calibrated','arrival')]
    for old,new in pairs:
        for metric in ('ttft_ms_mean','e2e_ms_mean','token_itl_ms_p99','preemptions','recomputed_tokens','slo_ttft5000_maxitl200_goodput_rps'):
            effects=[];windows=defaultdict(list)
            relative=metric not in ('preemptions','recomputed_tokens')
            for cell in design['cells']:
                a=st.fmean(r[metric] for r in runs[cell['key'],old]);b=st.fmean(r[metric] for r in runs[cell['key'],new])
                effect=100*(a-b)/a if relative and a!=0 else a-b
                effects.append(effect);windows[cell['window']].append(effect)
                if metric=='ttft_ms_mean':cell_effects.append(dict(cell=cell['key'],window=cell['window'],capacity=cell['capacity'],shape=cell['shape'],old=old,new=new,old_mean=a,new_mean=b,reduction_pct=effect))
            comparisons.append(dict(old=old,new=new,metric=metric,measure='old_minus_new_percent' if relative else 'old_minus_new_absolute',mean=st.fmean(effects),ci95=ci([st.fmean(v) for v in windows.values()]),wins_lower=sum(x>0 for x in effects),losses_lower=sum(x<0 for x in effects)))
    for path in OUT.glob('attempts/*/*/attempt_status.json'):
        record=read(path)
        if record['state']!='complete':failures.append(record)
    result=dict(audited=True,formal_runs=864,requests=864*128,calibration=read(OUT/'calibration/cost_model.json'),aggregate=summary['aggregate'],comparisons=comparisons,mechanisms={a:dict(c) for a,c in counts.items()},quality=quality,excluded_attempts=failures,online_cost_validation=dict(rows=len(prediction_errors),mape=st.fmean(prediction_errors) if prediction_errors else None),analysis_source_sha256=digest(Path(__file__)))
    write(OUT/'analysis/fidelity_summary.json',result)
    with (OUT/'analysis/new_vs_old_cells.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(cell_effects[0]));w.writeheader();w.writerows(cell_effects)
    print(json.dumps(dict(audited=True,runs=864,excluded_attempts=len(failures),ttft_comparisons=[r for r in comparisons if r['metric']=='ttft_ms_mean']),indent=2))

if __name__=='__main__':main()
