"""Cheap exact-allocation factor grid; does not claim GPU timing benefits."""
import itertools
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_saving_threshold_20260924.common import *
from research.v1_saving_threshold_20260924.policy import select
from research.v1_budget_20260922.policy import State


def main():
    results=[]
    for free,nd,boundary,npref,remaining,computed in itertools.product(
            [16,32,64,128,256,512],[0,8,24],[False,True],[4,12],[128,512,2048],[0,257]):
        dc=256 if boundary else 255
        rows=[State(f'd{i}',dc,dc+1,200,dc+1-200,16,True) for i in range(nd)]
        rows += [State(f'p{i}',computed,computed+remaining,computed+remaining,0,
                       (computed+15)//16,computed>0) for i in range(npref)]
        if sum(r.running for r in rows)>32:
            continue
        total=sum(r.blocks for r in rows)+free+1
        choices={}
        details=None
        for n in THRESHOLDS:
            x=select(rows,1024,256,free,total,16,32,12288,n)
            choices[str(n)]=dict(budget=x['budget'],chunk=x['chunk'],changed=x['changed'],
                growth=x['selected'].growth if x['selected'] else None)
            if details is None:details=x
        results.append(dict(free=free,total=total,decode_count=nd,decode_at_boundary=boundary,
            prefill_count=npref,remaining=remaining,computed=computed,choices=choices,
            base_feasible=details['base'] is not None,
            base_growth=details['base'].growth if details['base'] else None,
            maximum_eligible_saving=max([d['saving'] for d in details['candidate_diagnostics']
                if d['guard'] and d['saving'] is not None]+[0])))
    pairs=[]
    index={(r['free'],r['decode_count'],r['prefill_count'],r['remaining'],r['computed'],r['decode_at_boundary']):r for r in results}
    for key,a in index.items():
        if key[-1] or key[1]==0:continue
        b=index[(*key[:-1],True)]
        if a['base_growth']!=b['base_growth'] or a['choices']!=b['choices']:
            pairs.append(dict(within_block=a,at_boundary=b))
    write(OUT/'cpu/geometry.json',dict(states=len(results),threshold_evaluations=len(results)*len(THRESHOLDS),
        interpretation='exact candidate allocation geometry, not TTFT labels',rows=results,
        boundary_effect_pairs=len(pairs),boundary_examples=pairs[:8]))
    print('Exact allocation states',len(results),'threshold evaluations',len(results)*6,'boundary effect pairs',len(pairs))


if __name__=='__main__':main()
