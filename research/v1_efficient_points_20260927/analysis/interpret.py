"""Post-run independent file/decision audit and descriptive mechanisms."""
import csv,gzip,json,math,statistics as st,sys
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from research.v1_efficient_points_20260927.common import *
from research.v1_efficient_points_20260927.audit import audit
from research.v1_efficient_points_20260927.policy import select
from research.v1_budget_20260922.policy import State
from research.v1_efficient_points_20260927.report import csv_write


def main():
    design=read(OUT/'design.json');table=read(OUT/'calibration.json')
    assert sources()==design['source_hashes']
    assert digest(HERE/'PROTOCOL_CN.md')==design['protocol_sha256']
    for name,sha in design['input_hashes'].items():assert digest(OUT/f'inputs/{name}.json')==sha
    for filename in ['cpu_validation.json','derived_validation.json']:
        validation=read(OUT/filename);assert validation['passed'] and validation['source_hashes']==sources()
        assert validation['design_sha256']==digest(OUT/'design.json')
    assert read(OUT/'derived_validation.json')['calibration_sha256']==digest(OUT/'calibration.json')
    timings=[];jumps=[]
    for group in design['profile_groups']:
        folder=OUT/'groups'/group['key'];assert read(folder/'status.json')['state']=='complete'
        assert digest(folder/'profile_summary.json')==table['profile_summary_hashes'][group['key']]
        rs=[read(p) for p in (folder/'profile').glob('*.json')];assert len(rs)==720
        for ctx in design['contexts']:
            rr=[r for r in rs if r['context']==ctx]
            assert len({r['fingerprint'] for r in rr})==1
            assert len({json.dumps(r['prefix'],sort_keys=True) for r in rr})==1
            for r in rr:
                assert r['q']==sum(r['row']['scheduled'].values())
                assert math.isclose(r['non_schedule_ms'],r['step_ms']-r['schedule_ms'])
                if not r['warmup']:timings.append(dict(group=group['key'],context=ctx,chunk=r['chunk'],q=r['q'],repeat=r['repeat'],
                    non_schedule_ms=r['non_schedule_ms'],scheduled_requests=r['scheduled_requests'],decode_requests=r['decode_requests']))
            for k in design['chunks']:
                for q in (256,512):
                    a=[r for r in rr if r['chunk']==k and r['q']==q and not r['warmup']]
                    b=[r for r in rr if r['chunk']==k and r['q']==q+1 and not r['warmup']]
                    assert len(a)==len(b)==4
                    x=st.median(r['non_schedule_ms'] for r in a);y=st.median(r['non_schedule_ms'] for r in b)
                    jumps.append(dict(group=group['key'],context=ctx,chunk=k,q=q,base_ms=x,next_ms=y,
                        increase_pct=100*(y/x-1),base_requests=a[0]['scheduled_requests'],next_requests=b[0]['scheduled_requests']))
    for ctx in design['contexts']:
        for k in design['chunks']:
            for row in table['curves'][f'{ctx}_k{k}']:
                vals=[]
                for group in design['profile_groups']:
                    rr=[r for r in timings if r['group']==group['key'] and r['context']==ctx and r['chunk']==k and r['q']==row['q']]
                    assert len(rr)==4;vals.append(st.median(r['non_schedule_ms'] for r in rr))
                assert math.isclose(st.median(vals),row['ms'],rel_tol=1e-12)
    audits=[];mechanisms=[];trajectory=[]
    for group in design['groups']:
        folder=OUT/'groups'/group['key']
        if (folder/'exclusion.json').exists():continue
        assert read(folder/'status.json')['state']=='complete'
        assert read(folder/'calibration.json')['passed']
        signatures={}
        for path in sorted((folder/'runs').glob('*.json')):
            result=read(path);assert digest(OUT/result['rounds_path'])==result['rounds_sha256']
            with gzip.open(OUT/result['rounds_path'],'rt') as f:rows=json.load(f)
            checked=audit(result,rows);audits.append(dict(group=group['key'],run=path.stem,**checked))
            if result['phase']!='formal':continue
            assert result['calibration_sha256']==digest(OUT/'calibration.json')
            arm=result['arm'];changed=shortfall=extra=ties=0;counts=Counter();diffs=[];prediction_errors=[];steps=[];sched=[];matches_half=0
            signatures[(result['shape'],arm,result['repeat'])]=[(r['state'],r['free_before'],r['scheduled'],r['actual_growth_blocks'],r['preemptions']) for r in rows]
            for row in rows:
                if row['intervened']:counts[(row['budget'],row['chunk'])]+=1
                if arm=='current':continue
                if row['baseline'] is None:continue
                states=[State(**r) for r in row['state']]
                ref=select(states,**BASE,free=row['free_before'],total=row['total_blocks'],block_size=16,max_model_len=8176,
                    threshold=spec('current'),recent=dict(arrival_rate=row['features']['arrival_rate']))
                if ref['selected'].tokens!=row['scheduled']:
                    changed+=1
                    delta=ref['selected'].growth-row['actual_growth_blocks'];diffs.append(delta)
                    shortfall+=delta<0;extra+=delta>0;ties+=delta==0
                if arm=='points_rate':
                    half=select(states,**BASE,free=row['free_before'],total=row['total_blocks'],block_size=16,max_model_len=8176,
                        threshold=spec('points_half'),recent=dict(arrival_rate=row['features']['arrival_rate']))
                    matches_half+=half['selected'].tokens==row['scheduled']
                d=next((d for d in row['candidate_diagnostics'] if d.get('tokens')==row['scheduled']),None)
                if row['intervened'] and d is not None:
                    actual=row['step_ms']-row['schedule_ms']
                    prediction_errors.append(abs(d['predicted_ms']-actual)/actual*100)
                    steps.append(row['step_ms']);sched.append(row['schedule_ms'])
            mechanisms.append(dict(group=group['key'],shape=result['shape'],arm=arm,repeat=result['repeat'],rounds=len(rows),
                interventions=sum(counts.values()),actions={f'{b},{k}':n for (b,k),n in counts.items()},
                differs_current=changed,saves_less_than_current=shortfall,saves_more_than_current=extra,saves_same_as_current=ties,
                delta_saved_blocks_sum=sum(diffs),matches_half=matches_half,
                prediction_mape_pct=st.fmean(prediction_errors) if prediction_errors else None,
                prediction_med_ape_pct=st.median(prediction_errors) if prediction_errors else None,
                selected_steps_ms=sum(steps),selected_schedule_ms=sum(sched)))
        for shape in ('steady','burst_stop'):
            for rep in (0,1):
                a=signatures[(shape,'current',rep)];b=signatures[(shape,'points_fixed',rep)]
                trajectory.append(dict(group=group['key'],shape=shape,repeat=rep,current_rounds=len(a),fixed_rounds=len(b),
                    identical_physical_trajectory=a==b))
        print('Audited',group['key'],flush=True)
    csv_write(OUT/'analysis/timing_samples.csv',timings);csv_write(OUT/'analysis/timing_jumps.csv',jumps)
    csv_write(OUT/'analysis/fixed_trajectory.csv',trajectory)
    write(OUT/'analysis/mechanisms.json',mechanisms)
    summary=dict(passed=True,episodes=len(audits),rounds=sum(r['rounds'] for r in audits),
        profiling_records=2880,measured_timing_records=len(timings),jump_comparisons=len(jumps),
        positive_jumps=sum(r['increase_pct']>0 for r in jumps),min_jump_pct=min(r['increase_pct'] for r in jumps),
        max_jump_pct=max(r['increase_pct'] for r in jumps),formal_episodes=len(mechanisms),
        fixed_changes=sum(r['differs_current'] for r in mechanisms if r['arm']=='points_fixed'),
        fixed_identical_trajectories=sum(r['identical_physical_trajectory'] for r in trajectory),
        excluded=[dict(group=g['key'],**read(OUT/'groups'/g['key']/'exclusion.json')) for g in design['groups'] if (OUT/'groups'/g['key']/'exclusion.json').exists()],
        timing_note='non_schedule includes non-scheduler host costs; some q boundaries change request count')
    write(OUT/'analysis/independent_audit.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
