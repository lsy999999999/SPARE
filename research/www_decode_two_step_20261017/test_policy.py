import unittest

from research.v1_budget_20260922.policy import State
from research.www_decode_two_step_20261017.policy import next_decode_blocks, select
from research.www_evaluation_completion_20261003.policy import select as spare_select


class TwoStepTests(unittest.TestCase):
    def test_next_token_uses_existing_block_space(self):
        rows = [State('within', 4, 5, 2, 2, 1, True),
                State('edge', 15, 16, 2, 13, 1, True)]
        self.assertEqual(next_decode_blocks(rows, {'within': 1, 'edge': 1}, 16), 1)

    def test_normal_policy_is_unchanged_without_decode(self):
        rows = [State(str(i), 0, p, p, 0, 0, False)
                for i, p in enumerate((200, 400, 800, 1600))]
        args = (rows, 1024, 256, 300, 301, 16, 32, 8176)
        self.assertEqual(select(*args)['selected'].tokens,
                         spare_select(*args)['selected'].tokens)

    def test_two_step_guard_can_reject_a_candidate(self):
        rows = [State(str(i), 15, 16, 10, 1, 1, True) for i in range(24)]
        rows += [State('p1', 0, 2000, 2000, 0, 0, False),
                 State('p2', 0, 2000, 2000, 0, 0, False)]
        args = (rows, 1024, 256, 38, 39, 16, 32, 8176)
        options = {'mode': 'constant', 'n': 1}
        original = spare_select(*args, options=options)
        guarded = select(*args, options=options)
        self.assertEqual((original['budget'], original['chunk']), (256, 256))
        self.assertEqual((guarded['budget'], guarded['chunk']), (1024, 64))
        self.assertEqual(guarded['two_step_vetoes'], 2)
        self.assertTrue(guarded['two_step_action_changed'])


if __name__ == '__main__':
    unittest.main()
