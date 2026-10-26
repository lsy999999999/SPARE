"""Screen old single-action evidence; freeze small GPU development/holdout design."""
import csv
from datetime import datetime
import math
from pathlib import Path
import statistics as st
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.v1_simple_formula_20260924.common import *
from research.v1_simple_formula_20260924.policy import observable, formula
from research.v1_budget_20260922.policy import State
from research.v1_saving_threshold_20260924.analyze import state_action
from research.v1_saving_threshold_20260924.policy import threshold_value


def screen():
    data = read(OLD/'fit.json')['state_dataset']
    for d in data:
        d['features'].update(observable([State(**r) for r in d['state']], d['features']['free'], 16))
    def losses(spec):
        return [d['action_losses'].get(state_action(d, threshold_value(spec,d['features'])), math.inf) for d in data]
    groups = sorted({d['anchor'] for d in data})
    grid = []
    for a in (1, 8, 16, 32):
        for b in (8, 16, 32, 64, 128, 256):
            for k in (0, -16, 16):
                spec = formula(a,b,k); ls=losses(spec)
                if all(math.isfinite(x) for x in ls):
                    grid.append(dict(spec=spec, losses=ls, loss=st.fmean(ls)))
    def family_cv(two):
        candidates=[d for d in grid if (d['spec']['coefficients'][1]!=0)==two]
        scores=[]
        for group in groups:
            train=[i for i,d in enumerate(data) if d['anchor']!=group]
            test=[i for i,d in enumerate(data) if d['anchor']==group]
            best=min(candidates,key=lambda c:st.fmean(c['losses'][i] for i in train))
            scores.extend(best['losses'][i] for i in test)
        return st.fmean(scores)
    cv1,cv2=family_cv(False),family_cv(True)
    one=sorted((d for d in grid if d['spec']['coefficients'][1]==0),key=lambda d:d['loss'])
    chosen=[]; signatures=set()
    for d in one:
        sig=tuple(state_action(s,threshold_value(d['spec'],s['features'])) for s in data)
        if sig in signatures:continue
        signatures.add(sig);chosen.append(d)
        if len(chosen)==3:break
    if cv2 < cv1-.0005:
        chosen[2]=min((d for d in grid if d['spec']['coefficients'][1]!=0),key=lambda d:d['loss'])
    candidates={f'formula{i+1}':d['spec'] for i,d in enumerate(chosen)}
    result=dict(states=len(data),groups=len(groups),cv_one=cv1,cv_two=cv2,
        adopt_second_feature_margin=.0005,selected=candidates,
        grid=[{k:v for k,v in d.items() if k!='losses'} for d in grid],
        warning='Old single-action branches only shortlist hypotheses, not closed-loop performance.',
        x_quantiles=[sorted(d['features']['work_free_ratio'] for d in data)[int(q*(len(data)-1))] for q in (0,.25,.5,.75,1)])
    write(OUT/'screen.json',result)
    return candidates


def main():
    assert not (OUT/'design.json').exists(), 'Use frozen design; do not regenerate'
    candidates=screen()
    cases={}
    for source in ('code','conversation'):
        rows=read(OLD/f'inputs/traces/{source}_rate1.json')
        cases[source+'_dev']=rows
    # Prespecified new minutes. Select first 64 valid raw rows, no outcome filtering.
    anchors={'code':'2024-05-15 11:17','conversation':'2024-05-17 11:17'}
    provenance={}
    oldids={r['id'] for p in (OLD/'inputs/traces').glob('*.json') for r in read(p)}
    for source,minute in anchors.items():
        path=ROOT/f'data/azure_{"conv" if source=="conversation" else "code"}_1week.csv'
        chosen=[]
        with path.open(newline='') as f:
            reader=csv.reader(f);next(reader)
            for index,row in enumerate(reader):
                if row[0][:16]==minute:
                    chosen.append((index,row))
                    if len(chosen)==64:break
        assert len(chosen)==64, (source,minute,len(chosen))
        first=datetime.fromisoformat(chosen[0][1][0])
        rows=[dict(id=f'{source}:raw:{i}',prompt=int(r[1]),output=int(r[2]),
                   arrival_ms=(datetime.fromisoformat(r[0])-first).total_seconds()*1000) for i,r in chosen]
        assert not oldids & {r['id'] for r in rows}
        assert max(r['prompt']+r['output'] for r in rows)<(768-1)*16
        cases[source+'_holdout']=rows
        provenance[source]=dict(path=str(path.relative_to(ROOT)),anchor=minute,source_sha256=digest(path),
                                row_range=[chosen[0][0],chosen[-1][0]])
    for level in ('low','high'):
        cases['code_output_'+level]=[dict(r,output=max(1,r['output']//2) if level=='low' else r['output']*2)
                                     for r in cases['code_holdout']]
    for name,rows in cases.items():write(OUT/f'inputs/{name}.json',rows)
    dev=[dict(case=s+'_dev',capacity=c) for c in (768,2048,4096) for s in ('code','conversation')]
    test=[dict(case=s+'_holdout',capacity=c) for c in (768,1536,2048,4096) for s in ('code','conversation')]
    test += [dict(case='code_output_'+level,capacity=2048) for level in ('low','high')]
    write(OUT/'design.json',dict(frozen=True,candidates=candidates,development=dev,holdout=test,
        development_methods=['off','n16']+list(candidates),holdout_methods=['off','n16','n32','formula','native'],
        objective='Equal-cell mean(TTFT/off - 1); choose using development only. Simplicity tie within 0.005.',
        repeats=1,aa_tolerance=.05,provenance=provenance,
        trace_hashes={n:object_digest(r) for n,r in cases.items()},
        repeat_rule='If development winner is within 2 percentage points of another formula, repeat their six development cells once before freezing.',
        holdout_claim='Disjoint new minutes in same source days; output variants reuse code holdout IDs.',
        baseline='native = FCFS V1 schedule with matched 1024/256/32 configuration and audit instrumentation; off = reordered base.'))
    print('Candidates:',candidates,'development',len(dev),'holdout',len(test))


if __name__=='__main__':main()
