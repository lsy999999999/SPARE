"""Use deduplication savings to cover prompt/output compositions in branch labels."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_saving_threshold_20260924.common import *
from research.v1_saving_threshold_20260924.build_branches import collect


def main():
    path=OUT/'plans/branches_factors.json'
    if path.exists():print('Already frozen',path);return
    jobs=[];states=[]
    for cap in (768,4096):
        for case in ('code_output_high','conversation_output_high','code_prompt_low','conversation_prompt_low'):
            reqs,available,total=collect(case,cap)
            selected=available[:1]+available[-1:] if len(available)>1 else available
            anchor=f'branch_anchor_{case}_c{cap}'
            jobs.append(dict(key=anchor,phase='branch',case=case+'_synchronous',capacity=cap,
                requests=reqs,method='threshold',config=dict(BASE,threshold=16),threshold=16,
                group=anchor,role='anchor'))
            for s in selected:
                stateid=f'{case}_c{cap}_r{s["round"]}'
                state=dict(s,id=stateid,source_case=case,capacity=cap,anchor=anchor,
                           alternatives=[],candidate_state_count=total)
                for a in [a for a in s['actions'] if a['action']!=s['baseline_action']][:2]:
                    key=f'branch_{stateid}_n{a["n"]}'
                    jobs.append(dict(key=key,phase='branch',case=case+'_synchronous',capacity=cap,
                        requests=reqs,method='threshold',config=dict(BASE,threshold=16,
                            branch=dict(round=s['round'],fingerprint=s['fingerprint'],threshold=a['n'])),
                        threshold=a['n'],group=anchor,role='alternative',state_id=stateid,expected_action=a['action']))
                    state['alternatives'].append(key)
                states.append(state)
            print(case,cap,'states',len(selected),flush=True)
    original=read(OUT/'plans/branches.json')
    assert len(jobs)+len(original['jobs'])<=72
    write(path,dict(frozen=True,jobs=jobs,states=states,source_sha256=digest(Path(__file__)),
        policy_sources=sources(),claim='controlled prompt/output branch expansion within original 72-run budget'))
    print('Additional states',len(states),'runs',len(jobs),'all branch runs',len(jobs)+len(original['jobs']))


if __name__=='__main__':main()
