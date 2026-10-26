"""Test zero/negative savings acceptance and retained feasibility/protection."""
import unittest
from unittest.mock import patch

from research.www_component_ablation_mean_20261019 import no_savings_policy as policy
from research.v1_budget_20260922.policy import State, Plan


def select(first_growth, options, protected=False, infeasible=False):
    rows = ([State('a', 100, 101, 64, 16, 7, True),
             State('b', 0, 1024, 1024, 0, 0, True)] if protected else
            [State('a', 0, 1024, 1024, 0, 0, True),
             State('b', 0, 1024, 1024, 0, 0, True)])
    baseline = Plan({'a': 1 if protected else 128, 'b': 128}, 10)
    first = None if infeasible else Plan({'b': 127}, first_growth)
    second = Plan({'a': 1 if protected else 64, 'b': 64}, 5)
    plans = [baseline, first, second, second, second]
    with patch.object(policy, 'project', side_effect=plans), \
         patch.object(policy, 'features', return_value={}):
        return policy.select(rows, 1024, 256, 100, 1024, 16, 32, 8176,
                             options=dict(mode='constant', n=1, **options))


class NoSavingsChecks(unittest.TestCase):
    def test_zero_savings_condition_is_deleted(self):
        self.assertEqual(select(10, {})['budget'], 512)
        result = select(10, {'disable_saving_condition': True})
        self.assertEqual(result['budget'], 256)
        self.assertEqual(result['rejected_saving'], 0)

    def test_negative_savings_accepted_unlike_threshold_zero(self):
        result = select(11, {'disable_saving_condition': True})
        self.assertEqual(result['budget'], 256)
        self.assertEqual(result['candidate_diagnostics'][0]['saving'], -1)
        self.assertEqual(result['rejected_saving'], 0)

    def test_decode_protection_remains_enabled(self):
        result = select(10, {'disable_saving_condition': True}, protected=True)
        self.assertEqual(result['budget'], 512)
        self.assertEqual(result['selected'].tokens['a'], 1)
        self.assertGreater(result['rejected_guard'], 0)

    def test_infeasible_candidate_remains_rejected(self):
        result = select(10, {'disable_saving_condition': True}, infeasible=True)
        self.assertEqual(result['budget'], 512)
        self.assertEqual(result['rejected_infeasible'], 1)


if __name__ == '__main__':
    unittest.main()
