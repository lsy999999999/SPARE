"""Read-only replay of frozen outcomes and mechanism diagnostics; no GPU use."""
import csv
import gzip
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.v1_completion_order_20260927.common import *
from research.v1_completion_order_20260927.audit import audit
from research.v1_completion_order_20260927.policy import completion_work, completion_time, remaining_work
from research.v1_completion_order_20260927.run import verify_design
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
    design=verify_design();model=read(OUT/'service_model.json')['coefficients']
    audits=[];phases=Counter();formal=[];branches=[]
    for group in design['groups']:
        folder=OUT/'groups'/group['key']
        assert read(folder/'status.json')['state']=='complete'
        assert read(folder/'calibration.json')['passed']
        assert not (folder/'exclusion.json').exists()
        anchors={a:rows_for(read(folder/f'runs/anchor_{a}.json')) for a in ('current','dynamic')}
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
                    work_score=sum(rem),spt_score=completion_work(rem),time_score=completion_time(states,action['tokens'],256,model),
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
            if result['phase']!='formal':continue
            fs=[];counts=Counter();ratios=[];streak=max_streak=0;errors=[];weighted_overhead=0
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
                if row['preemptions']==0:
                    q=sum(row['scheduled'].values())
                    attention=sum(row['scheduled'].get(r['rid'],0)*r['computed']+
                        row['scheduled'].get(r['rid'],0)*(row['scheduled'].get(r['rid'],0)+1)/2 for r in row['state'])
                    predicted=model[0]+model[1]*q+model[2]*attention/1e6
                    observed=row['step_ms']-row['schedule_ms']
                    assert observed>0
                    errors.append(abs(predicted-observed)/observed)
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
                model_relative_error_median=st.median(errors),model_relative_error_p95=quantile(errors,.95)))
        print('audited',group['key'],len(audits),flush=True)
    write(OUT/'analysis/local_reaudit.json',dict(passed=True,episodes=len(audits),phases=dict(phases),
        rounds=sum(a['rounds'] for a in audits),hashes_passed=True,branch_prefix_and_area_passed=True,audits=audits))
    write(OUT/'analysis/formal_mechanisms.json',formal);csv_write(OUT/'analysis/formal_mechanisms.csv',formal)
    write(OUT/'analysis/branch_features.json',branches)
    csv_write(OUT/'analysis/branch_features.csv',[{k:v for k,v in b.items() if k!='requests'} for b in branches])
    print(dict(passed=True,episodes=len(audits),phases=dict(phases)),flush=True)


if __name__=='__main__':main()
