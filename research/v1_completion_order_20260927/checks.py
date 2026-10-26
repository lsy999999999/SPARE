"""Small mathematical and recorded-regression checks, independent of GPU timing."""
import gzip
import itertools
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_completion_order_20260927.common import *
from research.v1_completion_order_20260927.policy import select, completion_work, completion_time
from research.v1_dynamic_actions_20260927.policy import select as previous_select
from research.v1_budget_20260922.policy import State


def main():
    count=0
    for sizes in [(1,2,3),(0,7,2,7),(2,3,5,8),(0,0,5),(1,100,101)]:
        def total_completion(order):
            time=total=0
            for duration in order:
                time+=duration;total+=time
            return total
        assert completion_work(sizes)==min(total_completion(p) for p in itertools.permutations(sizes))
        count+=1
    assert sum([0,10])==sum([5,5]) and completion_work([0,10])<completion_work([5,5])
    rows=[State('a',0,20,20,0,0,False),State('b',0,50,50,0,0,False)]
    # At zero step overhead and unit token time, timing score is exactly
    # current waiting area plus optimal ideal serial continuation area.
    assert completion_time(rows,{'a':10},256,[0,1,0])==2*10+completion_work([10,50])
    previous=ROOT/'artifacts/v1_dynamic_actions_20260927_r1'
    result=read(previous/'groups/code_c2048/runs/steady_dynamic_r0.json')
    with gzip.open(previous/result['rounds_path'],'rt') as f:recorded=json.load(f)
    equivalence=0;guard_changes=0
    for row in recorded[::7]+[recorded[79],recorded[80]]:
        states=[State(**r) for r in row['state']]
        kwargs=dict(**BASE,free=row['free_before'],total=row['total_blocks'],block_size=16,max_model_len=8176,
                    recent=dict(arrival_rate=row['features']['arrival_rate']))
        for arm in ('current','dynamic'):
            a=previous_select(states,threshold=spec(arm),**kwargs)
            b=select(states,threshold=spec(arm),**kwargs)
            assert a['threshold']==b['threshold'] and (a['budget'],a['chunk'])==(b['budget'],b['chunk'])
            assert (None if a['selected'] is None else a['selected'].tokens)==(None if b['selected'] is None else b['selected'].tokens)
            equivalence+=1
        dynamic=select(states,threshold=spec('dynamic'),**kwargs)
        for arm in ('nonstall','work','spt','spt_time'):
            chosen=select(states,threshold=spec(arm),**kwargs)
            assert chosen['threshold']==dynamic['threshold']
            if chosen['base'] is None:continue
            ids=[r.rid for r in states if r.outputs==0]
            if sum(chosen['base'].tokens.get(rid,0) for rid in ids)>0:
                assert sum(chosen['selected'].tokens.get(rid,0) for rid in ids)>0
            if row['round']==79 and arm=='nonstall':
                assert sum(dynamic['selected'].tokens.get(rid,0) for rid in ids)==0
                assert chosen['selected'].tokens!=dynamic['selected'].tokens
                guard_changes+=1
    assert guard_changes==1
    out=dict(passed=True,serial_optimality_cases=count,legacy_equivalence_states=equivalence,
             reproduced_stall_guard_change=True)
    write(OUT/'mathematical_checks.json',out)
    print(out)


if __name__=='__main__':main()
