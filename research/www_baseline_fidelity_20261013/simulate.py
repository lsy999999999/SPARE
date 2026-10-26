"""Native CPU allocator uses exactly the new GPU scheduler, including feedback."""
import json
from pathlib import Path
from unittest.mock import patch
from research.www_evaluation_completion_20261003 import simulate as sim
from research.www_baseline_fidelity_20261013.scheduler import GPUBaselineScheduler
from research.www_baseline_fidelity_20261013.common import ARMS, METHOD, OUT
from simulator_core.timing import ResourceTiming

ORIGINAL_MAKE=sim.make_scheduler


def run(requests,arm,capacity,history,*,profile='resource',seed=20260929,output=None,cost_model=None):
    if cost_model is None:cost_model=json.loads((OUT/'calibration/cost_model.json').read_text())
    capture={}
    def make(arm='arrival',blocks=768,budget=1024,chunk=256,**kwargs):
        with patch.object(sim,'BudgetScheduler',GPUBaselineScheduler):
            scheduler,clock=ORIGINAL_MAKE('native',blocks,budget,chunk,**kwargs)
        scheduler.history_context=kwargs.get('history',())
        scheduler.seed_context=kwargs.get('seed',0)
        scheduler.fidelity_cost_model=cost_model
        if arm=='arrival':
            from research.v1_broad12h_20260925.policy import formula
            scheduler.configure('threshold',budget,chunk,32,threshold=formula(1))
        else:scheduler.configure(METHOD[arm],budget,chunk,32)
        capture['scheduler']=scheduler
        return scheduler,clock
    timing=ResourceTiming(dict(measurement_scope='synthetic-uncalibrated',parameters=sim.TIMING_PARAMETERS))
    class Timing:
        def estimate(self,batch):
            import math
            duration,details=timing.estimate(batch)
            if profile=='resource_staircase':
                duration+=5*math.ceil(sum(capture['scheduler'].round_rows[-1]['scheduled'].values())/256)
            capture['scheduler'].round_rows[-1]['step_ms']=duration
            return duration,details
    adaptation=dict(sim.ADAPTATION,**{a:('expanded_rule_adaptation',()) for a in ARMS})
    with patch.object(sim,'make_scheduler',make),patch.object(sim,'ARMS',tuple(set(sim.ARMS)|set(ARMS))),patch.object(sim,'ADAPTATION',adaptation):
        return sim.simulate(requests,arm=arm,capacity=capacity,history=history,seed=seed,
                            profile=profile,output=output,fitted_timing=Timing(),full_trace=True)
