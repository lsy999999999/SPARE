"""Focused checks of independent v2 observations and actions."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from research.v1_budget_20260922.policy import State
from research.www_rl_standalone_v2_20261019 import policy


class PolicyTest(unittest.TestCase):
    def test_state_uses_observed_requests(self):
        def request(prompt, computed, outputs, arrival):
            return NS(num_prompt_tokens=prompt, num_computed_tokens=computed,
                      output_token_ids=[1] * outputs, arrival_time=arrival)

        scheduler = NS(
            kv_cache_config=NS(num_blocks=2048),
            kv_cache_manager=NS(block_pool=NS(get_num_free_blocks=lambda: 1000)),
            waiting=[request(200, 0, 0, 0.)],
            running=[request(1000, 950, 0, 1.), request(800, 801, 1, 1.)],
            arrival_estimator=NS(value=1000.), arrival_clock=lambda: 2.5)
        state = policy.state_of(scheduler)
        self.assertEqual(len(state), policy.STATE_SIZE)
        self.assertEqual(state[10:14].tolist(), [0.5, 0.5, 0., 0.])
        self.assertEqual(state[21], .5)

    def test_action_does_not_call_spare_selector(self):
        rows = [State('decode', 100, 101, 100, 1, 7, True),
                State('prefill', 0, 600, 600, 0, 0, False)]
        args = (rows, 1024, 256, 2048, 4096, 16, 32, 8176)
        with patch('research.www_evaluation_completion_20261003.policy.select',
                   side_effect=AssertionError('SPARE selector called')):
            choice = policy.select_action(*args, options={
                'mode': 'fixed', 'standalone_action': 5})
        self.assertEqual((choice['budget'], choice['chunk']), (256, 256))
        self.assertEqual(choice['rejected_guard'], 0)
        self.assertEqual(choice['rejected_saving'], 0)

    def test_overdue_wait_increases_penalty(self):
        row = dict(step_ms=100., rl_pending_ages_s=[.5], rl_decode_count=0,
                   scheduled={}, state=[])
        fresh = policy.reward_of(row)
        row['rl_pending_ages_s'] = [2.5]
        self.assertLess(policy.reward_of(row), fresh)


if __name__ == '__main__':
    unittest.main()
