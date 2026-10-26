import random
import unittest
from research.v1_budget_20260922.policy import State, choose
from research.v1_capacity_ablation_20260923.policy import select


class PolicyTests(unittest.TestCase):
    def test_frozen_a_exact_choices(self):
        rng=random.Random(20260923)
        for _ in range(500):
            total=rng.choice((768,1024,1536,2048,4096,7445))
            rows=[State(str(i),0,n,n,0,0,False) for i,n in enumerate(sorted(rng.randrange(1,8000) for _ in range(40)))]
            free=rng.randrange(total)
            expected=choose(rows,1024,256,free,total,16,32,12288)
            actual=select('a_frozen',rows,1024,256,free,total,16,32,12288)
            self.assertEqual({k:actual[k] for k in expected},expected)
            if total>=4096: self.assertFalse(actual['changed'])

    def test_finish_ablation_changes_behavior(self):
        rows=[State(str(i),0,100,100,0,0,False) for i in range(6)]
        rows += [State('long',0,1000,1000,0,0,False)]
        full=select('a_frozen',rows,1024,256,767,768,16,32,12288)
        ablated=select('a_no_finish_guard',rows,1024,256,767,768,16,32,12288)
        self.assertFalse(full['changed'])
        self.assertTrue(ablated['changed'])
        self.assertLess(ablated['selected'].tokens.get('5',0),100)

    def test_positive_saving_ablation_changes_behavior(self):
        rows=[State('finish',0,1,1,0,0,False),State('long',0,1000,1000,0,0,False)]
        full=select('a_frozen',rows,1024,256,767,768,16,32,12288)
        ablated=select('a_any_positive_saving',rows,1024,256,767,768,16,32,12288)
        self.assertFalse(full['changed'])
        self.assertTrue(ablated['changed'])
        self.assertEqual(ablated['selected'].tokens['finish'],1)

    def test_occupancy_excludes_null_and_has_no_projection(self):
        low=select('occupancy',[],1024,256,11,101,16,32,12288,.9)
        high=select('occupancy',[],1024,256,10,101,16,32,12288,.9)
        self.assertEqual(low['budget'],1024)
        self.assertEqual(high['budget'],512)
        self.assertFalse(high['projected'])
        self.assertEqual(high['candidates'],0)

    def test_occupancy_guard_protects_first_response(self):
        rows=[State(str(i),0,100,100,0,0,False) for i in range(6)]
        result=select('occupancy_finish_guard',rows,1024,256,60,768,16,32,12288,.9)
        self.assertEqual(result['budget'],1024)
        self.assertEqual(result['rejected_guard'],1)

    def test_preemption_handed_to_native(self):
        rows=[State('decode',16,17,16,1,1,True)]
        for method in ('a_frozen','a_no_finish_guard','a_any_positive_saving','occupancy_finish_guard'):
            result=select(method,rows,1024,256,0,768,16,32,12288)
            self.assertIsNone(result['selected'])
            self.assertEqual(result['budget'],1024)


if __name__=='__main__': unittest.main()
