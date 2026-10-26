import unittest
from research.v1_budget_20260922.policy import State, project
from research.v1_state_aware_20260923.policy import select


class PolicyTests(unittest.TestCase):
    def choose(self,rows,free,method='pressure_one',**kwargs):
        return select(method,rows,1024,256,free,4096,16,32,12288,**kwargs)

    def test_blocks_are_private_and_use_existing_slack(self):
        rows=[State('a',0,17,17,0,0,False),State('b',0,17,17,0,0,False)]
        self.assertEqual(project(rows,1024,256,100,16,32,12288).growth,4)
        self.assertEqual(project([State('x',20,30,30,0,2,True)],1024,256,100,16,32,12288).growth,0)

    def test_pressure_exact_boundary_and_low_pressure_bypass(self):
        rows=[State(str(i),0,1000,1000,0,0,False) for i in range(4)]
        self.assertFalse(self.choose(rows,257)['changed'])
        self.assertTrue(self.choose(rows,256)['changed'])

    def test_relative_gate_has_distinct_actions(self):
        # First response consumes 200 of the reduced 256 budget. The candidate
        # saves only 5 blocks, insufficient for half of baseline growth (23).
        rows=[State('finish',0,200,200,0,0,False),State('long',0,1000,1000,0,0,False)]
        a=self.choose(rows,40)
        b=self.choose(rows,40,'pressure_relative')
        self.assertTrue(a['changed']); self.assertFalse(b['changed'])
        self.assertEqual(a['selected'].tokens['finish'],200)

    def test_growth_zero_and_native_preemption(self):
        rows=[State('x',16,17,16,1,1,True)]
        self.assertIsNone(self.choose(rows,0)['selected'])
        rows=[State('x',15,16,15,1,1,True)]
        self.assertFalse(self.choose(rows,100)['pressure_triggered'])

    def test_total_capacity_does_not_change_local_decision(self):
        rows=[State(str(i),0,1000,1000,0,0,False) for i in range(4)]
        for method in ('pressure_one','pressure_relative'):
            a=self.choose(rows,200,method)
            b=select(method,rows,1024,256,200,768,16,32,12288)
            self.assertEqual(a,b)


if __name__=='__main__': unittest.main()
