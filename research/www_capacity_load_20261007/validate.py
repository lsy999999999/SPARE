"""CPU allocation checks of the actual GPU scheduler before hardware launch."""
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.www_capacity_load_20261007.common import *

sys.path.insert(0, str(OUT / 'source'))
from research.www_capacity_load_20261007.scheduler import GPUBaselineScheduler
from research.www_capacity_load_20261007.policy import select
from research.v1_budget_20260922.policy import State
from research.www_evaluation_completion_20261003 import simulate as sim
from writing.www.code.run_simulator import simulate as frozen_simulate


def main():
    design = read(OUT / 'design.json')
    assert design['source_hashes'] == source_hashes()
    reqs = [dict(id=f'r{i:03}', prompt=(128,800,1800,2500)[i%4]+i,
                 output=24, arrival_ms=float(i//5)*40) for i in range(20)]
    history = [4,8,16,24,32,64]
    references = {p:frozen_simulate(reqs, capacity=192, profile=p, history=history)
                  for p in ('resource','resource_staircase')}
    results = {}
    with patch.object(sim, 'FormulaScheduler', GPUBaselineScheduler), \
         patch.object(sim, 'ComparisonScheduler', GPUBaselineScheduler), \
         patch.object(sim, 'BudgetScheduler', GPUBaselineScheduler):
        for arm in ARMS:
            engine_arm = 'arrival' if arm in OPTIONS else arm
            for profile in ('resource','resource_staircase'):
                result = sim.simulate(reqs, arm=engine_arm, capacity=192,
                    history=history, profile=profile, policy_options=OPTIONS.get(arm, {}))
                m=result['metrics']
                assert result['complete'] and m['completed']==len(reqs)
                assert m['executed_tokens']-m['recomputed_tokens']==sum(r['prompt']+r['output']-1 for r in reqs)
                if arm=='arrival':
                    assert result['trajectory_sha256']==references[profile]['trajectory_sha256']
                    assert m['preemptions']>0
                results[f'{arm}/{profile}']=dict(rounds=m['rounds'],preemptions=m['preemptions'],trajectory=result['trajectory_sha256'])
    # A block-boundary fixture: 15+one token reuses an existing block; replacing
    # the metric must preserve the original physical plan and guard checks.
    rows=[State(str(i),15,1024,1024,0,1,True) for i in range(4)]
    args=(rows,1024,256,256,261,16,32,8176)
    exact=select(*args,options={'mode':'constant','n':1})
    approx=select(*args,options={'mode':'constant','n':1,'saving_basis':'tokens'})
    assert exact['base'].growth==approx['base'].growth
    for axes in ('budget','chunk'):
        choice=select(*args,options={'candidate_axes':axes,'mode':'constant','n':1})
        assert (choice['chunk']==256 if axes=='budget' else choice['budget']==1024)
        assert choice['candidates']==(2 if axes=='budget' else 1)
    write(OUT/'validation.json',dict(passed=True,created=time.time(),
        source_hashes=source_hashes(),checks=results,
        scope='actual GPU scheduler on native CPU allocator; default matches frozen trajectories including recovery; no hardware latency claim'))
    print(json.dumps(dict(passed=True, trajectories=len(results), default_matches_frozen=True)))


if __name__=='__main__':
    main()
