"""Read-only replay of frozen outcomes and mechanism diagnostics; no GPU use."""
import csv
import gzip
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.v1_fixed_order_20260927.common import *
from research.v1_fixed_order_20260927.audit import audit
from research.v1_completion_order_20260927.policy import completion_work, completion_time, remaining_work
from research.v1_fixed_order_20260927.run import verify_design
from research.v1_budget_20260922.policy import State
from research.v1_budget_20260922.run import quantile
from research.v1_dynamic_actions_20260927.analysis.interpret import features


def rows_for(result):
    p=OUT/result['rounds_path']
    assert digest(p)==result['rounds_sha256']
    with gzip.open(p,'rt') as f:return json.load(f)


def csv_write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    design=verify_design()
    audits=[];phases=Counter();formal=[];branches=[];branch_timing=[]
    for group in design['groups']:
        folder=OUT/'groups'/group['key']
        assert read(folder/'status.json')['state']=='complete'
        assert read(folder/'calibration.json')['passed']
        assert not (folder/'exclusion.json').exists()
        anchors={a:rows_for(read(folder/f'runs/anchor_{a}.json')) for a in ('current','dyn_spt')}
        targets=read(folder/'selected_states.json')['states']
        for target in targets:
            row=anchors[target['origin']][target['round']]
            assert row['state_fingerprint']==target['fingerprint']
            states=[State(**r) for r in row['state']]
            for action in target['actions']:
                f,details=features(row,action['tokens'],action['growth'])
                rem=remaining_work(states,action['tokens'])
                branches.append(dict(group=group['key'],target=target['target'],origin=target['origin'],arm=action['arm'],
                    aliases='/'.join(action['aliases']),budget=action['budget'],chunk=action['chunk'],free=row['free_before'],
                    prefill_progress=sum(action['tokens'].get(r.rid,0) for r in states if r.outputs==0),
                    work_score=sum(rem),spt_score=completion_work(rem),
                    **f,requests=details))
        for p in sorted((folder/'runs').glob('*.json')):
            result=read(p);rows=rows_for(result)
            audits.append(dict(path=str(p.relative_to(OUT)),**audit(result,rows)))
            phases[result['phase']]+=1
            if result['phase']=='branch':
                index=result['branch_round'];anchor=anchors[result['origin']]
                assert all(r['state_fingerprint']==anchor[i]['state_fingerprint'] for i,r in enumerate(rows[:index+1]))
                assert all(r['scheduled']==anchor[i]['scheduled'] for i,r in enumerate(rows[:index]))
                assert all(r['features']['arrival_time_s']==anchor[i]['features']['arrival_time_s'] for i,r in enumerate(rows[:index+1]))
                action=next(a for t in targets if t['target']==result['target'] for a in t['actions'] if a['arm']==result['arm'])
                assert rows[index]['scheduled']==action['tokens'] and rows[index]['branch_applied']
                start=(rows[index]['probe_decision_wall']-result['start_wall'])*1000
                pending=[r['rid'] for r in rows[index]['state'] if r['outputs']==0]
                area=sum(result['clocks'][rid]['first_ms']-start for rid in pending)
                assert math.isclose(area,result['residual_waiting_area_ms'],rel_tol=1e-12)
                assert result['config']['branch']['after_threshold']['candidate_mode']=='current'
                row=rows[index]
                first_area=len(pending)*(row['step_end_ms']-start)
                later_area=sum(max(0,result['clocks'][rid]['first_ms']-row['step_end_ms']) for rid in pending)
                assert math.isclose(first_area+later_area,area,rel_tol=1e-10,abs_tol=1e-6)
                branch_timing.append(dict(group=group['key'],target=result['target'],arm=result['arm'],
                    aliases='/'.join(result['aliases']),repeat=result['repeat'],budget=row['budget'],chunk=row['chunk'],
                    executed_tokens=sum(row['scheduled'].values()),scheduled_requests=len(row['scheduled']),pending=len(pending),
                    step_ms=row['step_ms'],schedule_ms=row['schedule_ms'],non_schedule_ms=row['step_ms']-row['schedule_ms'],
                    first_round_wait_ms_per_request=first_area/64,later_wait_ms_per_request=later_area/64,
                    total_wait_ms_per_request=area/64))
            if result['phase']!='formal':continue
            fs=[];counts=Counter();ratios=[];streak=max_streak=0;weighted_overhead=0;ref_deltas=[];misalignment=Counter()
            for row in rows:
                pending=[r['rid'] for r in row['state'] if r['outputs']==0]
                base_progress=sum((row['baseline'] or {}).get(rid,0) for rid in pending)
                progress=sum(row['scheduled'].get(rid,0) for rid in pending)
                weighted_overhead+=len(pending)*row['decision_ms']/64
                tiny=bool(row['intervened'] and base_progress>=256 and 0<progress<=16)
                streak=streak+1 if tiny else 0;max_streak=max(max_streak,streak)
                if row['intervened']:
                    counts['interventions']+=1
                    f,_=features(row,row['scheduled'],row['actual_growth_blocks']);fs.append(f)
                    if base_progress>0:
                        counts['base_can_progress']+=1
                        counts['zero_progress']+=progress==0
                        counts['tiny_progress']+=tiny
                        counts['at_most_eighth_progress']+=progress/base_progress<=1/8
                        ratios.append(progress/base_progress)
                if row['features'].get('different_from_reference') and row['baseline'] is not None:
                    from research.v1_dynamic_actions_20260927.policy import select as reference_select
                    states=[State(**r) for r in row['state']]
                    ref=reference_select(states,budget=1024,chunk=256,free=row['free_before'],total=row['total_blocks'],
                        block_size=16,concurrency=32,max_model_len=8176,
                        threshold=dict(result['config']['threshold'],candidate_mode='current'),
                        recent={'arrival_rate':row['features']['arrival_rate']})['selected']
                    dq=sum(row['scheduled'].values())-sum(ref.tokens.values())
                    ref_deltas.append(dq)
                    q=sum(row['scheduled'].values());rq=sum(ref.tokens.values())
                    misalignment['changed']+=1
                    misalignment['aligned_ref_small_extra']+=rq%256==0 and 0<dq<=16
                    misalignment['same_growth']+=row['actual_growth_blocks']==ref.growth
                    misalignment['per_request_no_less']+=all(row['scheduled'].get(r.rid,0)>=ref.tokens.get(r.rid,0) for r in states)
                    misalignment['ref_base_but_intervened']+=row['features'].get('reference_saving')==0 and row['intervened']
            assert counts['zero_progress']==result['metrics']['no_prefill_progress_rounds']
            formal.append(dict(group=group['key'],shape=result['shape'],arm=result['arm'],repeat=result['repeat'],
                rounds=len(rows),interventions=counts['interventions'],base_can_progress=counts['base_can_progress'],
                zero_progress=counts['zero_progress'],tiny_progress=counts['tiny_progress'],
                max_tiny_streak=max_streak,at_most_eighth_progress=counts['at_most_eighth_progress'],
                mean_progress_fraction=st.fmean(ratios) if ratios else None,
                saving_sum=sum(f['saving'] for f in fs),saving_over_n_sum=sum(f['saving_over_n'] for f in fs),
                partial_after_sum=sum(f['partial_after'] for f in fs),affected_sum=sum(f['affected'] for f in fs),
                withheld_tokens_sum=sum(f['withheld_tokens'] for f in fs),
                decision_ms_total=result['metrics']['decision_ms_total'],decision_waiting_area_ms_per_request=weighted_overhead,
                reference_changed=len(ref_deltas),mean_extra_tokens_vs_reference=st.fmean(ref_deltas) if ref_deltas else None,
                aligned_ref_small_extra=misalignment['aligned_ref_small_extra'],same_growth_as_reference=misalignment['same_growth'],
                per_request_no_less=misalignment['per_request_no_less'],ref_base_but_intervened=misalignment['ref_base_but_intervened']))
        print('audited',group['key'],len(audits),flush=True)
    write(OUT/'analysis/local_reaudit.json',dict(passed=True,episodes=len(audits),phases=dict(phases),
        rounds=sum(a['rounds'] for a in audits),hashes_passed=True,branch_prefix_and_area_passed=True,audits=audits))
    write(OUT/'analysis/formal_mechanisms.json',formal);csv_write(OUT/'analysis/formal_mechanisms.csv',formal)
    write(OUT/'analysis/branch_timing.json',branch_timing);csv_write(OUT/'analysis/branch_timing.csv',branch_timing)
    write(OUT/'analysis/branch_features.json',branches)
    csv_write(OUT/'analysis/branch_features.csv',[{k:v for k,v in b.items() if k!='requests'} for b in branches])
    print(dict(passed=True,episodes=len(audits),phases=dict(phases)),flush=True)


if __name__=='__main__':main()
