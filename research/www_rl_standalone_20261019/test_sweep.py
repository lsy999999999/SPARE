"""Capacity gates use the current mean-TTFT rule."""
import unittest

from research.www_rl_standalone_20261019.sweep import passed


class MeanCapacityRuleTest(unittest.TestCase):
    def test_legacy_p95_pass_does_not_override_mean_ttft(self):
        summary = dict(passed=True, failed_samples=0, mean_ttft_ms=2100.,
                       p99_tbt_ms=80., backlog_drift=1.)
        self.assertFalse(passed(summary, 3.))
        summary['mean_ttft_ms'] = 1900.
        self.assertTrue(passed(summary, 3.))


if __name__ == '__main__':
    unittest.main()
