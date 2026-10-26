"""Factor responses, grouped simple-model fitting, and frozen follow-up plans."""
import argparse
import copy
import csv
import gzip
import itertools
import math
from pathlib import Path
import statistics as st
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from research.v1_saving_threshold_20260924.common import *
from research.v1_saving_threshold_20260924.policy import threshold_value
from research.v1_budget_20260922.policy import State,project


def load():
    manifest=read(OUT/'inputs_manifest.json')
    physical={p.stem:read(p) for p in (OUT/'runs').glob('*.json')}
    expanded=list(physical.values())
    aliases=list(manifest['aliases'])
    for plan in sorted((OUT/'plans').glob('*.json')):
        aliases.extend(read(plan).get('aliases',[]))
    for alias in aliases:
        if alias['canonical'] in physical:
            row=dict(physical[alias['canonical']],job=alias['job'],alias_of=alias['canonical'])
            expanded.append(row)
    return manifest,physical,expanded


def group_rows(rows,phase):
    result={}
    for r in rows:
        j=r['job']
        if j['phase']!=phase or r['status']!='complete':continue
        result.setdefault((j['case'],j['capacity']),{})[str(j.get('threshold'))]=r
    return result


def map_data(rows):
    result=[]
    for (case,cap),group in group_rows(rows,'map').items():
        if not all(str(n) in group for n in THRESHOLDS):continue
        base=group['off']['metrics']['mean_ttft_ms']
        result.append(dict(case=case,source=case.split('_')[0],capacity=cap,
            features=dict(capacity=cap-1),
            losses={str(n):group[str(n)]['metrics']['mean_ttft_ms']/base-1 for n in THRESHOLDS},
            ttft={str(n):group[str(n)]['metrics']['mean_ttft_ms'] for n in THRESHOLDS}))
    return result


def best_constant_map(data):
    n=min(THRESHOLDS,key=lambda n:(st.fmean(d['losses'][str(n)] for d in data),str(n)))
    return dict(kind='constant',n=n)


def map_loss(data,spec):
    return st.fmean(d['losses'][str(threshold_value(spec,d['features']) or 'off')] for d in data)


def fit_capacity(data):
    best=best_constant_map(data);score=map_loss(data,best)
    caps=sorted({d['capacity']-1 for d in data})
    for cut in [(a+b)/2 for a,b in zip(caps,caps[1:])]:
        left=[d for d in data if d['capacity']-1<=cut];right=[d for d in data if d not in left]
        a,b=best_constant_map(left),best_constant_map(right)
        spec=dict(kind='tree',tree=dict(feature='capacity',cut=cut,left=dict(n=a['n']),right=dict(n=b['n'])))
        value=map_loss(data,spec)
        if value<score-1e-12:best,score=spec,value
    return best


def state_action(d,n):
    if n is not None and d['features']['growth']>0 and len(d['state'])>=2:
        for c in d['candidate_diagnostics']:
            if c['guard'] and c['saving'] is not None and c['saving']>=n:return object_digest(c['tokens'])
    return d['base_action_hash']


def state_loss(d,spec):
    action=state_action(d,threshold_value(spec,d['features']))
    return d['action_losses'].get(action,math.inf)


def average_loss(data,spec):
    return st.fmean(state_loss(d,spec) for d in data)


def constant_state(data):
    return min((dict(kind='constant',n=n) for n in THRESHOLDS),key=lambda s:average_loss(data,s))


def stump(data,feature):
    best=constant_state(data);score=average_loss(data,best)
    values=sorted({d['features'][feature] for d in data})
    for cut in [(a+b)/2 for a,b in zip(values,values[1:])]:
        left=[d for d in data if d['features'][feature]<=cut]
        right=[d for d in data if d['features'][feature]>cut]
        if min(len(left),len(right))<3:continue
        a,b=constant_state(left),constant_state(right)
        spec=dict(kind='tree',tree=dict(feature=feature,cut=cut,left=dict(n=a['n']),right=dict(n=b['n'])))
        value=average_loss(data,spec)
        if value<score-1e-12:best,score=spec,value
    return best


def regression(data,features):
    x=[];y=[]
    for d in data:
        options=list(range(1,max(2,d['features']['growth']+2)))
        losses=[d['action_losses'].get(state_action(d,n),math.inf) for n in options]
        best=min(losses)
        good=[n for n,v in zip(options,losses) if v<=best+1e-12]
        # Interior of the admissible interval, rather than pretending an endpoint is identified.
        target=math.sqrt(min(good)*max(good))
        x.append([1]+[math.log1p(max(0,d['features'][f])) for f in features])
        y.append(math.log1p(target))
    coefficients=np.linalg.lstsq(np.asarray(x),np.asarray(y),rcond=None)[0]
    return dict(kind='loglinear',features=list(features),intercept=float(coefficients[0]),
                coefficients=list(map(float,coefficients[1:])))


def cv(data,fit):
    scores=[]
    for group in sorted({d['anchor'] for d in data}):
        train=[d for d in data if d['anchor']!=group];test=[d for d in data if d['anchor']==group]
        if not train:continue
        spec=fit(train)
        scores.extend(state_loss(d,spec) for d in test)
    return st.fmean(scores) if scores else math.inf


def branch_data(physical):
    p=OUT/'plans/branches.json'
    if not p.exists():return [],[]
    plans=[read(q) for q in sorted((OUT/'plans').glob('branches*.json'))]
    results=[];audit=[];cache={}
    for original in [s for plan in plans for s in plan['states']]:
        d=copy.deepcopy(original)
        anchor=physical.get(d['anchor'])
        if anchor is None or anchor['status']!='complete':continue
        if d['anchor'] not in cache:
            with gzip.open(OUT/anchor['rounds_path'],'rt') as f:cache[d['anchor']]=json.load(f)
        rounds=cache[d['anchor']]
        identity=d['round']<len(rounds) and rounds[d['round']]['state_fingerprint']==d['fingerprint']
        if not identity:
            audit.append(dict(state=d['id'],valid=False,reason='anchor_CPU_GPU_state_mismatch'));continue
        raw=project([State(**r) for r in d['state']],1024,256,d['features']['free'],16,32,12288)
        d['base_action_hash']=object_digest(raw.tokens)
        baseline=anchor['metrics']['mean_ttft_ms']
        d['action_losses']={object_digest(d['baseline_action']):0.0}
        d['action_gains_ms']={object_digest(d['baseline_action']):0.0}
        valid=True
        for key in d['alternatives']:
            r=physical.get(key)
            if r is None or r['status']!='complete' or r['metrics']['branch_applied']!=1:
                valid=False;break
            with gzip.open(OUT/r['rounds_path'],'rt') as f:rr=json.load(f)
            if rr[d['round']]['state_fingerprint']!=d['fingerprint'] or rr[d['round']]['scheduled']!=r['job']['expected_action']:
                valid=False;break
            action=object_digest(r['job']['expected_action'])
            delta=r['metrics']['mean_ttft_ms']-baseline
            d['action_losses'][action]=delta/baseline
            d['action_gains_ms'][action]=-delta
        audit.append(dict(state=d['id'],valid=valid,reason=None if valid else 'missing_or_invalid_alternative'))
        if valid:
            intervals=[]
            for n in range(1,max(2,d['features']['growth']+2)):
                action=state_action(d,n)
                if intervals and intervals[-1]['action_hash']==action:
                    intervals[-1]['max_n']=n
                else:intervals.append(dict(min_n=n,max_n=n,action_hash=action))
            for interval in intervals:
                interval['loss']=d['action_losses'].get(interval['action_hash'])
                interval['extends_to_infinity']=interval['action_hash']==d['base_action_hash'] and interval is intervals[-1]
            d['threshold_action_intervals']=intervals
            results.append(d)
    return results,audit


def ablate(spec,data):
    result=copy.deepcopy(spec)
    if spec['kind']=='constant':return result,None
    if spec['kind']=='tree':
        feature=spec['tree']['feature'];mid=st.median(d['features'][feature] for d in data)
        n=spec['tree']['left' if mid<=spec['tree']['cut'] else 'right']['n']
        return dict(kind='constant',n=n),feature
    feature=spec['features'][0];mid=st.median(d['features'][feature] for d in data)
    result['intercept']+=result['coefficients'][0]*math.log1p(max(mid,0))
    result['features']=result['features'][1:];result['coefficients']=result['coefficients'][1:]
    return result,feature


def fit_models(manifest,physical,rows):
    path=OUT/'fit.json'
    if path.exists():return read(path)
    maps=map_data(rows)
    assert len(maps)>=18,('Not enough complete capacity cells',len(maps))
    capacity=fit_capacity(maps);constant=best_constant_map(maps)
    capacity_cv=[];constant_cv=[]
    for source in sorted({d['source'] for d in maps}):
        train=[d for d in maps if d['source']!=source];test=[d for d in maps if d['source']==source]
        capacity_cv.append(map_loss(test,fit_capacity(train)));constant_cv.append(map_loss(test,best_constant_map(train)))
    states,audit=branch_data(physical)
    features=['capacity','pressure','free_fraction','remaining_over_free','pending_prefill',
              'partial_prefill','decode_count','decode_boundary','computed_prefix','block_slack','mean_remaining','recent_release']
    candidates=[]
    if len(states)>=12:
        baseline_cv=cv(states,constant_state)
        for feature in features:
            for family in ('stump','loglinear'):
                fitter=(lambda ds,f=feature:stump(ds,f)) if family=='stump' else (lambda ds,f=feature:regression(ds,[f]))
                score=cv(states,fitter)
                if math.isfinite(score):candidates.append(dict(family=family,features=[feature],cv_loss=score,spec=fitter(states)))
        top=sorted(candidates,key=lambda c:c['cv_loss'])[:3]
        for pair in itertools.combinations(sorted({c['features'][0] for c in top}),2):
            score=cv(states,lambda ds,fs=pair:regression(ds,fs))
            if math.isfinite(score):candidates.append(dict(family='loglinear',features=list(pair),cv_loss=score,spec=regression(states,pair)))
        candidates.sort(key=lambda c:(c['cv_loss'],len(c['features'])))
    else:baseline_cv=None
    # A state formula requires positive grouped validation information; otherwise use the measured capacity relation.
    if candidates and candidates[0]['cv_loss']<baseline_cv-.0005:
        winner=candidates[0];spec=winner['spec'];origin='same_state_grouped_cv';fitdata=states
    else:
        spec=capacity;origin='capacity_response_map';fitdata=maps
    removed,feature=ablate(spec,fitdata)
    result=dict(frozen=True,selected=spec,ablation=removed,ablated_feature=feature,origin=origin,
        map_complete_cells=len(maps),capacity_formula=capacity,constant_formula=constant,
        capacity_source_holdout_loss=st.fmean(capacity_cv),constant_source_holdout_loss=st.fmean(constant_cv),
        branch_states=len(states),branch_audit=audit,branch_constant_cv=baseline_cv,
        state_candidates=candidates,state_dataset=states,analysis_sha256=digest(Path(__file__)),
        selection='state formula only if grouped CV improves constant by >0.0005 normalized TTFT; otherwise capacity relation',
        interpretation='exploratory fitted formula; empirical selection is not a global TTFT theorem')
    write(path,result)
    return result


def make_followups(manifest,fit,rows):
    path=OUT/'plans/formula.json'
    if not path.exists():
        jobs=[];aliases=[];skipped=[]
        for cap in CAP_ORDER:
            for case,c in manifest['cases'].items():
                if c['role']!='heldout':continue
                specs=[('formula',fit['selected']),('ablated',fit['ablation']),('off','off')]
                seen={}
                for name,spec in specs:
                    j=dict(key=f'formula_{case}_c{cap}_{name}',phase='formula',case=case,capacity=cap,
                        method='threshold',config=dict(BASE,threshold=spec),threshold=spec,arm=name)
                    if c['max_total_tokens']>min(12288,(cap-1)*16):
                        skipped.append(j);continue
                    key=object_digest(spec)
                    if key in seen:aliases.append(dict(job=j,canonical=seen[key]))
                    else:seen[key]=j['key'];jobs.append(j)
        write(path,dict(frozen=True,jobs=jobs,aliases=aliases,skipped=skipped,fit_sha256=digest(OUT/'fit.json')))
    path=OUT/'plans/refine.json'
    if not path.exists():
        data=map_data(rows);bycap={}
        for cap in sorted({d['capacity'] for d in data}):
            group=[d for d in data if d['capacity']==cap]
            bycap[cap]=-min(st.fmean(d['losses'][str(n)] for d in group) for n in THRESHOLDS)
        gaps=[]
        for a,b in zip(sorted(bycap),sorted(bycap)[1:]):
            priority=abs(bycap[a]-bycap[b]) + (.25 if (bycap[a]>=.05)!=(bycap[b]>=.05) else 0)
            middle=int(round((a+b)/128))*64
            if a<middle<b:gaps.append((priority,middle,a,b))
        selected=sorted(gaps,reverse=True)[:4]
        jobs=[]
        for _,cap,a,b in selected:
            for case in ('code_rate1','conversation_rate1'):
                if manifest['cases'][case]['max_total_tokens']>(cap-1)*16:continue
                for n in THRESHOLDS:
                    jobs.append(dict(key=f'refine_{case}_c{cap}_n{n}',phase='refine',case=case,capacity=cap,
                        method='threshold',config=dict(BASE,threshold=n),threshold=n,bracket=[a,b]))
        write(path,dict(frozen=True,jobs=jobs,selected_gaps=selected,
                       rule='top four adjacent capacity gaps by gain change plus 5% transition bonus; exploratory'))


def summary(manifest,physical,rows):
    records=[]
    for r in rows:
        j=r['job'];m=r['metrics']
        records.append(dict(key=j['key'],phase=j['phase'],case=j.get('case'),capacity=j['capacity'],
            threshold=json.dumps(j.get('threshold')),arm=j.get('arm'),status=r['status'],
            alias_of=r.get('alias_of'),ttft_ms=m['mean_ttft_ms'],e2e_ms=m.get('mean_e2e_ms'),
            p95_tbt_ms=m['p95_tbt_ms'],preemptions=m['preemptions'],recompute=m['recomputed_tokens'],
            interventions=m['interventions'],elapsed_ms=r['elapsed_ms'],
            deferred_prefill_tokens=m.get('deferred_prefill_tokens'),
            rounds=m.get('round_count'),step_ms_mean=m.get('step_ms_mean'),step_ms_total=m.get('step_ms_total'),
            decision_ms_total=m.get('decision_ms_total'),audit_ms_total=m.get('audit_ms_total'),
            pressure_mean_posthoc=m.get('pressure_mean'),free_fraction_mean_posthoc=m.get('free_fraction_mean'),
            partial_prefill_mean_posthoc=m.get('partial_prefill_mean'),decode_count_mean_posthoc=m.get('decode_count_mean'),
            recent_release_mean_posthoc=m.get('recent_release_mean')))
    output=OUT/'analysis';output.mkdir(parents=True,exist_ok=True)
    if records:
        with (output/'runs.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    maps=map_data(rows)
    tables=[]
    for cap in sorted({d['capacity'] for d in maps}):
        ds=[d for d in maps if d['capacity']==cap]
        gains={str(n):-100*st.fmean(d['losses'][str(n)] for d in ds) for n in THRESHOLDS}
        tables.append(dict(capacity=cap,complete_cells=len(ds),gain_percent=gains,
                           best_threshold=max(gains,key=gains.get)))
    factors=[]
    gr=group_rows(rows,'factor')
    for source in ('code','conversation'):
        for factor in ('density','burst','prompt','output'):
            for cap in (768,2048,4096):
                low=gr.get((f'{source}_{factor}_low',cap),{})
                high=gr.get((f'{source}_{factor}_high',cap),{})
                if not all(str(n) in low and str(n) in high for n in (1,16,'off')):continue
                def gain(g,n):return 1-g[str(n)]['metrics']['mean_ttft_ms']/g['off']['metrics']['mean_ttft_ms']
                factors.append(dict(source=source,factor=factor,capacity=cap,
                    low_gain_n1=100*gain(low,1),high_gain_n1=100*gain(high,1),
                    low_gain_n16=100*gain(low,16),high_gain_n16=100*gain(high,16),
                    interaction_n1_pp=100*(gain(high,1)-gain(low,1)),
                    interaction_n16_pp=100*(gain(high,16)-gain(low,16)),
                    low_n1_advantage_over_n16_percent=100*(1-low['1']['metrics']['mean_ttft_ms']/low['16']['metrics']['mean_ttft_ms']),
                    high_n1_advantage_over_n16_percent=100*(1-high['1']['metrics']['mean_ttft_ms']/high['16']['metrics']['mean_ttft_ms']),
                    low_best=min(low,key=lambda n:low[n]['metrics']['mean_ttft_ms']),
                    high_best=min(high,key=lambda n:high[n]['metrics']['mean_ttft_ms'])))
    write(output/'summary.json',dict(physical_runs=len(physical),complete=sum(r['status']=='complete' for r in physical.values()),
        failed=[key for key,r in physical.items() if r['status']!='complete'],capacity_map=tables,
        factor_interactions=factors,formula=read(OUT/'fit.json')['selected'] if (OUT/'fit.json').exists() else None))
    lines=['# 因素与容量扫描：滚动分析','',f'已保存物理运行 {len(physical)} 次；完整容量cell {len(maps)} 个。',
           '以下为探索性单次运行，正改善表示相对同cell不干预降低TTFT。','',
           '| KV blocks | 完整cell | n=1 | n=4 | n=16 | n=32 | n=64 | 最好取值 |',
           '|---:|---:|---:|---:|---:|---:|---:|---|']
    for t in tables:
        lines.append('| '+ ' | '.join([str(t['capacity']),str(t['complete_cells'])]+[
            f"{t['gain_percent'][str(n)]:+.1f}%" for n in (1,4,16,32,64)]+[t['best_threshold']])+' |')
    lines+=['','控制因素完成配对数：'+str(len(factors)),'','完整数值见 runs.csv 和 summary.json。']
    (output/'PROGRESS_RESULTS_CN.md').write_text('\n'.join(lines)+'\n')
    print('Summary:',len(physical),'runs;',len(maps),'map cells;',len(factors),'factor pairs',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit',action='store_true');a=p.parse_args()
    manifest,physical,rows=load()
    if a.fit:
        fitted=fit_models(manifest,physical,rows);make_followups(manifest,fitted,rows)
        print('Selected formula:',json.dumps(fitted['selected']),fitted['origin'],flush=True)
    summary(manifest,physical,rows)
