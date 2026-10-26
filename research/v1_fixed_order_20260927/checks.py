"""Frozen-state invariants, reference equivalence and candidate coverage."""
import gzip
import itertools
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_fixed_order_20260927.common import *
from research.v1_fixed_order_20260927.policy import select
from research.v1_dynamic_actions_20260927.policy import select as original_select,old_pairs
from research.v1_completion_order_20260927.policy import remaining_work,completion_work
from research.v1_budget_20260922.policy import State


def main():
    states=[State(str(i),0,4096,4096,0,0,False) for i in range(12)]
    args=dict(**BASE,free=127,total=128,block_size=16,max_model_len=8176,recent={'arrival_rate':0})
    a=select(states,threshold=spec('current'),**args)
    b=select(states,threshold=spec('half_first'),**args)
    assert a['budget']==256 and b['budget']==512
    assert a['base'].growth-a['selected'].growth==48
    assert b['base'].growth-b['selected'].growth==32
    assert a['threshold']==b['threshold']
    previous=ROOT/'artifacts/v1_completion_order_20260927_r1'
    differences=Counter();n=0;pareto_changed=0
    for path in sorted((previous/'groups').glob('*/runs/*_current_r0.json')):
        result=read(path)
        if result['phase']!='formal':continue
        with gzip.open(previous/result['rounds_path'],'rt') as f:recorded=json.load(f)
        for row in recorded[::11]:
            states=[State(**r) for r in row['state']]
            args=dict(**BASE,free=row['free_before'],total=row['total_blocks'],block_size=16,max_model_len=8176,
                      recent=dict(arrival_rate=row['features']['arrival_rate']))
            legacy=original_select(states,threshold=spec('current'),**args)
            choices={arm:select(states,threshold=spec(arm),**args) for arm in ARMS}
            assert legacy['threshold']==choices['current']['threshold']
            assert (legacy['budget'],legacy['chunk'])==(choices['current']['budget'],choices['current']['chunk'])
            signature=lambda c: None if c['selected'] is None else c['selected'].tokens
            assert signature(legacy)==signature(choices['current'])
            for arm,choice in choices.items():
                assert choice['threshold']==legacy['threshold']
                if arm in ('half_first','spt4') and choice['changed']:
                    assert (choice['budget'],choice['chunk']) in old_pairs(1024,256)
            pools=[[(d['budget'],d['chunk'],d['tokens']) for d in choices[arm].get('candidate_diagnostics',[])]
                   for arm in ('dyn_spt','dyn_save','dyn_pareto')]
            assert pools[0]==pools[1]==pools[2]
            if legacy['base'] is not None:
                ref=legacy['selected'];pending=[r for r in states if r.outputs==0]
                ref_progress=sum(ref.tokens.get(r.rid,0) for r in pending)
                for arm in ('dyn_save','dyn_pareto'):
                    c=choices[arm]
                    if c['changed']:
                        assert legacy['base'].growth-c['selected'].growth>=max(legacy['threshold'],legacy['base'].growth-ref.growth)
                p=choices['dyn_pareto']['selected']
                assert p.growth<=ref.growth
                assert sum(p.tokens.get(r.rid,0) for r in pending)>=ref_progress
                assert completion_work(remaining_work(states,p.tokens))<=completion_work(remaining_work(states,ref.tokens))
                assert all(p.tokens.get(r.rid,0)>=ref.tokens.get(r.rid,0) for r in states if r.outputs>0)
                pareto_changed+=p.tokens!=ref.tokens
            for lhs,rhs in itertools.combinations(ARMS,2):
                differences[lhs+'/'+rhs]+=signature(choices[lhs])!=signature(choices[rhs])
            n+=1
    assert differences['current/half_first']>0 and differences['current/spt4']>0 and pareto_changed>0
    result=dict(passed=True,states=n,all_thresholds_equal=True,dynamic_pools_equal=True,
        reference_invariants_passed=True,pareto_changed=pareto_changed,differences=dict(differences),
        ideal_aligned_savings=dict(quarter=48,half=32),outcomes_used=False)
    write(OUT/'mathematical_checks.json',result);print(result)


if __name__=='__main__':main()
