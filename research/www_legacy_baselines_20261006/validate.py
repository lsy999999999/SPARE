"""CPU allocation checks of the actual GPU scheduler before hardware launch."""
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.www_legacy_baselines_20261006.common import *

sys.path.insert(0, str(OUT / 'source'))
from research.www_legacy_baselines_20261006.scheduler import GPUBaselineScheduler
from research.www_legacy_baselines_20261006.policy import select
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
    original_make = sim.make_scheduler
    def make(arm="arrival", blocks=768, budget=1024, chunk=256, **kwargs):
        kwargs.update(blocks=blocks, budget=budget, chunk=chunk)
        if arm not in GPUBaselineScheduler.LEGACY:
            return original_make(arm=arm, **kwargs)
        with patch.object(sim, "BudgetScheduler", GPUBaselineScheduler):
            scheduler, clock = original_make(arm="native", **kwargs)
        scheduler.history_context = kwargs.get('history', ())
        scheduler.oracle_lengths_context = kwargs.get('oracle_lengths') or {}
        scheduler.configure(arm, kwargs.get('budget', 1024), kwargs.get('chunk', 256),
                            kwargs.get('concurrency', 32))
        return scheduler, clock
    sim.ARMS = tuple(sim.ARMS) + tuple(GPUBaselineScheduler.LEGACY)
    sim.ORACLE_ARMS = tuple(sim.ORACLE_ARMS) + ('ltr_oracle',)
    sim.ADAPTATION.update({arm: ('rule_adapter', ()) for arm in GPUBaselineScheduler.LEGACY})
    sim.make_scheduler = make
    results = {}
    with patch.object(sim, 'FormulaScheduler', GPUBaselineScheduler), \
         patch.object(sim, 'ComparisonScheduler', GPUBaselineScheduler), \
         patch.object(sim, 'BudgetScheduler', GPUBaselineScheduler):
        for arm in ARMS:
            engine_arm = 'arrival' if arm in OPTIONS else arm
            for profile in ('resource','resource_staircase'):
                result = sim.simulate(reqs, arm=engine_arm, capacity=384 if arm in GPUBaselineScheduler.LEGACY else 192,
                    history=history, profile=profile, policy_options=OPTIONS.get(arm, {}))
                m=result['metrics']
                assert result['complete'] and m['completed']==len(reqs)
                assert m['executed_tokens']-m['recomputed_tokens']==sum(r['prompt']+r['output']-1 for r in reqs)
                if arm=='arrival':
                    assert result['trajectory_sha256']==references[profile]['trajectory_sha256']
                    assert m['preemptions']>0
                results[f'{arm}/{profile}']=dict(rounds=m['rounds'],preemptions=m['preemptions'],trajectory=result['trajectory_sha256'])
    from types import SimpleNamespace as NS
    scheduler, _ = make('orca_max', history=history, blocks=768)
    req = NS(request_id='r', num_prompt_tokens=160, output_token_ids=[])
    assert scheduler._reservation(req) == 138
    scheduler, _ = make('s3_proxy', history=history, blocks=768)
    assert scheduler._reservation(req) == 23
    req.output_token_ids = [1] * 205
    assert scheduler._reservation(req) == 36
    # First-fit skips a head reservation that exceeds the residual capacity.
    scheduler, _ = make('s3_proxy', history=history, blocks=32)
    from vllm.v1.core.sched.request_queue import FCFSRequestQueue
    big = NS(request_id='big',num_prompt_tokens=800,output_token_ids=[],arrival_time=0)
    small = NS(request_id='small',num_prompt_tokens=16,output_token_ids=[],arrival_time=0)
    scheduler.waiting = FCFSRequestQueue([big, small])
    scheduler._order()
    assert [r.request_id for r in scheduler.waiting] == ['small']
    assert [r.request_id for r in scheduler.legacy_held] == ['big']
    scheduler, _ = make('ltr_proxy', history=history, blocks=768)
    scheduler.waiting = FCFSRequestQueue([big, small])
    scheduler.legacy_quantum['small'] = 10
    scheduler._order()
    assert list(scheduler.waiting)[0].request_id == 'small'
    try:
        make('ltr_oracle', history=history, blocks=768)
    except ValueError:
        pass
    else:
        raise AssertionError('oracle input must be explicit')
    write(OUT/'validation.json' ,dict(passed=True,created=time.time(),
        source_hashes=source_hashes(),checks=results,
        scope='actual GPU scheduler on native CPU allocator; default matches frozen trajectories including recovery; no hardware latency claim'))
    print(json.dumps(dict(passed=True, trajectories=len(results), default_matches_frozen=True)))


if __name__=='__main__':
    main()
