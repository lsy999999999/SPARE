"""Selection reads development clocks only, never TBT or confirmation outcomes."""
import statistics as st
from research.v1_theory_formula_20260924.common import OUT,read,digest
from research.v1_budget_20260922 import run as ref


def select_winner(design):
    ref.OUT=OUT
    scores={n:[] for n in design['methods']};hashes={}
    for cell in design['development']:
        means={}
        for arm in design['methods']:
            values=[]
            for repeat in (0,1):
                key=f"dev_{cell['case']}_c{cell['capacity']}_{arm}_r{repeat}"
                path=OUT/'runs'/f'{key}.json';r=read(path)
                assert r['status']=='complete' and r['job']['arm']==arm
                ref.audit_result(r)
                values.append(r['metrics']['mean_ttft_ms']);hashes[key]=digest(path)
            means[arm]=st.fmean(values)
        for arm in design['methods']:scores[arm].append(means[arm]/means['old']-1)
    scores={a:st.fmean(v) for a,v in scores.items()}
    best=min(scores.values())
    winner=next(a for a in design['tie_order'] if scores[a]<=best+design['tie_margin'])
    return dict(winner=winner,scores=scores,development_hashes=hashes,
                criterion='mean TTFT only; equal cells; frozen simplicity tie rule')
