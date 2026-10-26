import math
import tempfile
import unittest
from pathlib import Path
from research.v1_budget_20260922.policy import State
from research.v1_broad12h_20260925.policy import select as frozen_select, formula
from research.www_evaluation_completion_20261003.policy import select
from research.www_evaluation_completion_20261003.simulate import simulate
from writing.www.code.run_simulator import simulate as frozen_simulate


class EvaluationTests(unittest.TestCase):
    def test_frozen_trajectory_equivalence_and_early_exit(self):
        reqs=[dict(id=f'r{i:03}',prompt=(128,800,1800,2500)[i%4]+i,
                   output=24,arrival_ms=float(i//5)*40) for i in range(20)]
        for profile in ('resource','resource_staircase'):
            a=frozen_simulate(reqs,capacity=192,profile=profile)
            for options in ({},{'early_exit':True,'diagnostics':False}):
                b=simulate(reqs,capacity=192,profile=profile,policy_options=options)
                self.assertEqual(a['trajectory_sha256'],b['trajectory_sha256'])
                self.assertGreater(b['metrics']['preemptions'],0)

    def test_finish_guard_changes_an_actual_decision(self):
        rows=[State(str(i),0,p,p,0,0,False) for i,p in enumerate([200,400,800,1600])]
        args=(rows,1024,256,300,301,16,32,8176)
        guarded=select(*args,options={'mode':'constant','n':1})
        unguarded=select(*args,options={'mode':'constant','n':1,'protect_finish':False})
        # First candidate preserves the first request, so use a finish at a later queue position.
        rows=[State(str(i),0,p,p,0,0,i<2) for i,p in enumerate([800,200,1600,2000])]
        args=(rows,1024,256,300,301,16,32,8176)
        a=select(*args,options={'mode':'constant','n':1})
        b=select(*args,options={'mode':'constant','n':1,'protect_finish':False})
        self.assertEqual(a['selected'].tokens.get('1'),200)
        self.assertLess(b['selected'].tokens.get('1',0),200)

    def test_controls_drain_under_pressure(self):
        reqs=[dict(id=f'r{i}',prompt=100+i*80,output=30,arrival_ms=0.) for i in range(12)]
        for options in ({'mode':'fixed','fixed_budget':256}, {'mode':'occupancy'},
                        {'protect_finish':False}, {'mode':'constant','n':16},
                        {'H':.5,'tau':2}, {'n0':16,'alpha':256}):
            r=simulate(reqs,capacity=96,policy_options=options)
            self.assertEqual(r['metrics']['completed'],len(reqs))
            self.assertEqual(r['metrics']['executed_tokens']-r['metrics']['recomputed_tokens'],
                             sum(x['prompt']+x['output']-1 for x in reqs))

    def test_threshold_example(self):
        rows=[State(str(i),0,p,p,0,0,False) for i,p in enumerate([1024,2048,2048,3072])]
        args=(rows,1024,256,256,257,16,32,8176)
        self.assertEqual(select(*args,recent={'arrival_rate':8192})['threshold'],40)
        self.assertEqual(select(*args,recent={'arrival_rate':8192},options={'tau':0})['threshold'],72)


if __name__=='__main__':unittest.main()
