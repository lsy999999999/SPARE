"""Checks for the revised mean-TTFT RL comparison."""
import unittest

from research.www_online_rl_20261017 import analyze_mean


class MeanCapacityTests(unittest.TestCase):
    def test_p95_is_not_part_of_current_capacity_rule(self):
        summary = dict(failed_samples=0, mean_ttft_ms=1900,
                       p95_ttft_ms=30000, p99_tbt_ms=150,
                       backlog_drift=3)
        self.assertTrue(analyze_mean.passed(summary, 4, 2000))
        self.assertFalse(analyze_mean.passed(summary, 4, 1800))
        self.assertFalse(analyze_mean.passed(dict(summary, p99_tbt_ms=201), 4, 2000))
        self.assertFalse(analyze_mean.passed(dict(summary, backlog_drift=5), 4, 2000))

    def test_fixed_kv_capacity_uses_first_failure(self):
        paths = analyze_mean.records()
        self.assertEqual(analyze_mean.capacity(paths['spare', 'conversation', 12032], 2000),
                         4.96)
        self.assertEqual(analyze_mean.capacity(paths['rl', 'conversation', 12032], 2000),
                         4.84)
        rows = {row['kv_blocks']: row for row in analyze_mean.comparison(paths)}
        self.assertEqual(len(rows), 27)
        self.assertAlmostEqual(rows[2048]['capacity_gain_pct'], -1.3966480447)
        self.assertAlmostEqual(rows[12032]['capacity_gain_pct'], -6.7058823529)
        for kv in (1792, 2048, 2816, 3072, 3840, 4096, 12032):
            self.assertLess(rows[kv]['capacity_gain_pct'], 0)


if __name__ == '__main__':
    unittest.main()
