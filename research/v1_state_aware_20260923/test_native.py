import unittest
from research.v1_budget_20260922.test_scheduler import scheduler, add, advance
from research.v1_capacity_ablation_20260923.scheduler import CapacityScheduler
from research.v1_state_aware_20260923.scheduler import StateScheduler
from research.v1_state_aware_20260923.common import METHODS, config


class NativeTests(unittest.TestCase):
    def test_unchanged_control_actions_and_accounting(self):
        for method in METHODS[:4]:
            a=scheduler(blocks=96,cls=CapacityScheduler)
            b=scheduler(blocks=96,cls=StateScheduler)
            for s in (a,b):
                s.configure(method,**config(method)); add(s,[513,600,251,23,449,375,67,100])
            for _ in range(5000):
                x,y=a.schedule(),b.schedule()
                self.assertEqual(x.num_scheduled_tokens,y.num_scheduled_tokens)
                for key in ('free_after','recomputed_tokens','preemptions','intervened'):
                    self.assertEqual(a.round_rows[-1][key],b.round_rows[-1][key])
                advance(a,x); advance(b,y)
                if not a.requests: break
            self.assertFalse(a.requests); self.assertFalse(b.requests)

    def test_new_policies_with_real_allocator_and_recovery(self):
        changes=preemptions=0
        for method in METHODS[4:]:
            for cap in (96,160):
                s=scheduler(blocks=cap,cls=StateScheduler)
                s.configure(method,**config(method))
                add(s,[513,600,251,23,449,375,67,100,599,501,243,183])
                for _ in range(10000):
                    out=s.schedule()
                    r=s.round_rows[-1]
                    self.assertGreaterEqual(r['free_after'],0)
                    if r['intervened']:
                        self.assertTrue(r['pressure_triggered'])
                        self.assertGreater(r['baseline_growth_blocks'],r['actual_growth_blocks'])
                    changes+=r['intervened']; preemptions+=r['preemptions']
                    advance(s,out)
                    if not s.requests: break
                self.assertFalse(s.requests,(method,cap))
        self.assertGreater(changes,0); self.assertGreater(preemptions,0)


if __name__=='__main__': unittest.main()
