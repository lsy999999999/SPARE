"""Focused checks for independent scheduling actions."""
import unittest
from unittest.mock import patch

from research.v1_budget_20260922.policy import State
from research.www_rl_standalone_20261019 import policy, run


class StandalonePolicyTest(unittest.TestCase):
    def test_action_controls_prefill_budget_without_spare_selector(self):
        rows = [
            State('decode', 100, 101, 100, 1, 7, True),
            State('prefill_a', 0, 600, 600, 0, 0, False),
            State('prefill_b', 0, 600, 600, 0, 0, False),
        ]
        args = (rows, 1024, 256, 2048, 4096, 16, 32, 8176)
        with patch('research.www_evaluation_completion_20261003.policy.select',
                   side_effect=AssertionError('SPARE selector called')):
            narrow = run.independent_select(*args, options={
                'mode': 'fixed', 'standalone_action': 1})
            wide = run.independent_select(*args, options={
                'mode': 'fixed', 'standalone_action': 9})
        self.assertEqual(narrow['budget'], 1)
        self.assertEqual(wide['budget'], 1024)
        self.assertEqual(narrow['selected'].tokens['decode'], 1)
        self.assertEqual(wide['selected'].tokens['decode'], 1)
        self.assertLess(sum(narrow['selected'].tokens.values()),
                        sum(wide['selected'].tokens.values()))
        self.assertEqual(narrow['rejected_guard'], 0)
        self.assertEqual(narrow['rejected_saving'], 0)
        self.assertEqual(policy.ACTIONS[1], (0., 64))
        only_prefill = run.independent_select(rows[1:], 1024, 256, 2048,
                                              4096, 16, 32, 8176,
                                              options={'mode': 'fixed',
                                                       'standalone_action': 1})
        self.assertEqual(only_prefill['budget'], 1024)

    def test_prefill_completion_receives_reward(self):
        row = dict(step_ms=100., rl_pending_first_tokens=1,
                   rl_decode_count=0, scheduled={}, state=[], allocations=[])
        stalled = policy.reward_of(row)
        row['allocations'] = [dict(outputs=0, before=192, tokens=64,
                                   prompt=256)]
        self.assertEqual(policy.reward_of(row) - stalled, 5.)


if __name__ == '__main__':
    unittest.main()
