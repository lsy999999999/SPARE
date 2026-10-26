"""CPU execution of the real V1 scheduler and allocator; no GPU model needed."""
import random
import unittest
from research.v1_budget_20260922.test_scheduler import scheduler,add,advance
from research.v1_capacity_ablation_20260923.scheduler import CapacityScheduler


class NativeTests(unittest.TestCase):
    def test_frozen_a_and_shadow_native_equivalence(self):
        for original,current in (('budget','a_frozen'),('srp','srp_matched'),('srp','a_shadow')):
            for capacity in (96,160):
                a=scheduler(original,blocks=capacity,budget=1024)
                a.configure(original,1024,256,32)
                b=scheduler(blocks=capacity,budget=1024,cls=CapacityScheduler)
                b.configure(current,1024,256,32)
                for s in (a,b): add(s,[513,600,251,23,449,375,67,100])
                for _ in range(4000):
                    x,y=a.schedule(),b.schedule()
                    self.assertEqual(x.num_scheduled_tokens,y.num_scheduled_tokens)
                    self.assertEqual(a.kv_cache_manager.block_pool.get_num_free_blocks(),
                                     b.kv_cache_manager.block_pool.get_num_free_blocks())
                    advance(a,x); advance(b,y)
                    if not a.requests: break
                self.assertFalse(a.requests,(original,capacity))
                self.assertFalse(b.requests)

    def test_new_methods_closed_loop_native_allocation(self):
        rng=random.Random(230923)
        changes=preemptions=0
        for method in ('occupancy','occupancy_finish_guard','a_no_finish_guard','a_any_positive_saving'):
            for capacity in (96,160):
                s=scheduler(blocks=capacity,budget=1024,cls=CapacityScheduler)
                s.configure(method,1024,256,32,.7)
                add(s,[rng.randrange(17,650) for _ in range(12)])
                for _ in range(5000):
                    out=s.schedule()  # checks physical block growth and guards each round
                    row=s.round_rows[-1]
                    changes+=row['intervened']; preemptions+=row['preemptions']
                    advance(s,out)
                    if not s.requests: break
                self.assertFalse(s.requests,(method,capacity))
        self.assertGreater(changes,0)
        self.assertGreater(preemptions,0)


if __name__=='__main__': unittest.main()
