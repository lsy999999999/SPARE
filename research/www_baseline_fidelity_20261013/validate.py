"""Exercise the actual GPU adapter on the native allocator before measurement."""
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.www_baseline_fidelity_20261013.common import *
sys.path.insert(0,str(OUT/'source'))
from research.www_baseline_fidelity_20261013.simulate import run
from writing.www.code.run_simulator import simulate as frozen


def main():
    reqs=[dict(id=f'r{i:03}',prompt=(128,800,1800,2500)[i%4]+i,
               output=24,arrival_ms=float(i//5)*40) for i in range(20)]
    history=[4,8,16,24,32,64]
    results={}
    for profile in ('resource','resource_staircase'):
        ref=frozen(reqs,capacity=192,profile=profile,history=history,seed=20260929)
        for arm in ARMS:
            result=run(reqs,arm,192,history,profile=profile,
                       output=OUT/'validation'/f'{arm}_{profile}')
            m=result['metrics']
            assert result['complete'] and m['completed']==len(reqs)
            assert m['executed_tokens']-m['recomputed_tokens']==sum(r['prompt']+r['output']-1 for r in reqs)
            if arm=='arrival':assert result['trajectory_sha256']==ref['trajectory_sha256']
            results[arm+'/'+profile]=dict(rounds=m['rounds'],preemptions=m['preemptions'],trajectory=result['trajectory_sha256'])
    # An observed prefix must not depend on a request's hidden stopping length.
    from unittest.mock import patch
    from research.www_baseline_fidelity_20261013.scheduler import GPUBaselineScheduler
    from research.www_evaluation_completion_20261003 import simulate as sim
    from vllm import SamplingParams
    from vllm.v1.request import Request
    def first_plan(method,hidden):
        with patch.object(sim,'BudgetScheduler',GPUBaselineScheduler):
            scheduler,clock=sim.make_scheduler('native',blocks=192)
        scheduler.fidelity_cost_path=OUT/'calibration/cost_model.json'
        scheduler.history_context=history
        scheduler.configure(method,1024,256,32)
        for i in range(8):
            scheduler.add_request(Request(str(i),[100]*128,None,None,None,
                SamplingParams(max_tokens=hidden),None,None,arrival_time=0.))
        return scheduler.schedule().num_scheduled_tokens
    for method in GPUBaselineScheduler.NEW:
        assert first_plan(method,24)==first_plan(method,2000)
    write(OUT/'validation.json',dict(passed=True,created=time.time(),source_hashes=source_hashes(),
        checks=results,hidden_output_cap_invariance=True,
        scope='actual GPU scheduler on native allocator; conservation; SPARE frozen parity; not GPU latency'))
    print(json.dumps(dict(passed=True,trajectories=len(results))))

if __name__=='__main__':main()
