"""Compare pressure signals on identical CPU states without changing actions."""
from collections import Counter
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.v1_direction_probe_20260923 import run
from research.v1_direction_probe_20260923.policy import forecast, select
from research.v1_direction_probe_20260923.scheduler import ProbeScheduler
from research.v1_state_aware_20260923.common import read, write


def main():
    original = ProbeScheduler._select
    observed = []

    def selector(self, method, rows, *args):
        choice = original(self, method, rows, *args)
        budget, chunk, free, total, size, concurrency, maxlen = args[:7]
        base = choice['base']
        data = dict(normal=base is not None)
        if base is not None:
            data.update(g0=base.growth, free=free, decode=sum(r.decode for r in rows))
            for h in (4, 8):
                trajectory = forecast(rows, base, h, budget, chunk, size, concurrency, maxlen)
                data[f'h{h}'] = dict(demand=trajectory[-1],
                    trigger=trajectory[-1]>free,
                    constant_trigger=h*base.growth>=max(free,1),
                    constant_strict_trigger=h*base.growth>free)
            # Observe all candidates on this exact state, independent of execution.
            risk = select('horizon8_risk', rows, *args)
            candidates = [d for d in risk['candidate_diagnostics'] if d['rejection'] is None]
            data.update(risk_selected=risk['changed'],
                eligible_candidates=len(candidates),
                all_catch_up=bool(candidates) and all(d['risk']>=risk['probe']['base_risk'] for d in candidates))
        observed.append(data)
        return choice

    base_out = ROOT/'artifacts/v1_direction_probe_20260923_r1'
    run.global_out(ROOT/'artifacts/v1_direction_probe_20260923_shadow_r1')
    results = []
    for block, cap in run.CELLS:
        observed.clear()
        with patch.object(ProbeScheduler, '_select', selector):
            row = run.run(block, cap, 'pressure_one')
        old = read(base_out/'runs'/(row['key']+'.json'))
        assert row['action_sha256']==old['action_sha256']
        normal = [r for r in observed if r['normal']]
        summary = dict(block=block, capacity=cap, normal_rounds=len(normal),
            eligible_candidate_rounds=sum(r['eligible_candidates']>0 for r in normal),
            all_candidates_catch_up_rounds=sum(r['all_catch_up'] for r in normal))
        for h in (4, 8):
            hs = [r[f'h{h}'] for r in normal]
            actionable = [r[f'h{h}'] for r in normal if r['g0']>0]
            summary[f'h{h}'] = dict(
                same_trigger=sum(x['trigger']==x['constant_trigger'] for x in hs),
                same_trigger_fraction=sum(x['trigger']==x['constant_trigger'] for x in hs)/len(hs),
                actionable_rounds=len(actionable),
                actionable_same_trigger_fraction=sum(x['trigger']==x['constant_trigger'] for x in actionable)/len(actionable),
                differences_beyond_equal_boundary=sum(x['trigger']!=x['constant_strict_trigger'] for x in hs),
                demand_equals_h_g0=sum(r[f'h{h}']['demand']==h*r['g0'] for r in normal),
                future_pressure_with_g0_zero=sum(r['g0']==0 and r[f'h{h}']['trigger'] for r in normal))
        results.append(summary)
        write(run.OUT/'states'/(row['key']+'.json'), observed)
        print(json.dumps(summary), flush=True)
    write(run.OUT/'summary.json', dict(claim='shadow signals on identical pressure_one trajectories; not causal effects', rows=results))


if __name__ == '__main__': main()
