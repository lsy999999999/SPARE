"""Post-observation CPU ablation: risk ranking over total-budget cuts only.

This is an explicitly adaptive follow-up after GPU risk ranking selected many
chunk=64 actions. It is not part of the original 40-run CPU matrix or GPU pilot.
"""
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.v1_direction_probe_20260923 import run
from research.v1_direction_probe_20260923.scheduler import ProbeScheduler
from research.v1_budget_20260922.policy import project
from research.v1_state_aware_20260923.common import read,write,digest


def main():
    original=ProbeScheduler._select

    def selector(self,method,rows,*args):
        choice=original(self,method,rows,*args)
        budget,chunk,free,total,size,concurrency,maxlen=args[:7]
        details=choice['candidate_diagnostics']
        if not details:
            return choice
        eligible=[d for d in details if d['rejection'] is None and d['chunk']==chunk
                  and d['risk']<choice['probe']['base_risk']]
        choice.update(budget=budget,chunk=chunk,selected=choice['base'],changed=False)
        choice['probe'].pop('chosen',None)
        choice['probe']['budget_only_ablation']=True
        if eligible:
            chosen=min(eligible,key=lambda d:(d['risk'],d['deferred_tokens'],d['deferred_requests']))
            plan=project(rows,chosen['budget'],chosen['chunk'],free,size,concurrency,maxlen)
            choice.update(budget=chosen['budget'],chunk=chosen['chunk'],selected=plan,
                          changed=plan.tokens!=choice['base'].tokens)
            choice['probe']['chosen']=chosen
        return choice

    run.global_out(ROOT/'artifacts/v1_direction_probe_20260923_budget_only_r1')
    rows=[]
    for block,cap in run.CELLS:
        with patch.object(ProbeScheduler,'_select',selector):
            r=run.run(block,cap,'horizon8_risk')
        r['variant']='horizon8_risk_budget_only'
        write(run.OUT/'runs'/(r['key']+'.json'),r)
        base=read(ROOT/'artifacts/v1_direction_probe_20260923_r1/runs'/(r['key']+'.json'))
        rows.append(dict(block=block,capacity=cap,complete=r['complete'],
                         original=base['stats'],budget_only=r['stats']))
        print(block,cap,'preemptions',base['stats']['preemptions'],'->',r['stats']['preemptions'],
              'first token rounds',base['stats']['mean_first_token_round'],'->',r['stats']['mean_first_token_round'],flush=True)
    write(run.OUT/'summary.json',dict(claim='adaptive CPU-only candidate-family ablation; no GPU performance claim',
        sources={str((Path(__file__).parent/n).relative_to(ROOT)):digest(Path(__file__).parent/n)
                 for n in ('budget_only_probe.py','policy.py','scheduler.py','run.py')},rows=rows))


if __name__=='__main__':main()
