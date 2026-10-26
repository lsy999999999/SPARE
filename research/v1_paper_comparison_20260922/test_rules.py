import random
import unittest
from math import ceil

from research.v1_paper_comparison_20260922.rules import HistoryPredictor, future_peak_blocks


class RuleTests(unittest.TestCase):
    def test_breakpoint_peak_matches_exhaustive_timeline(self):
        rng = random.Random(42)
        for _ in range(100):
            states = [(rng.randrange(1, 130), g, g+rng.randrange(1, 80))
                      for g in [rng.randrange(30) for _ in range(rng.randrange(1, 12))]]
            expected = max(sum(ceil((p+g+t)/16) for p,g,length in states if length-g>=t)
                           for t in range(1, max(length-g for _,g,length in states)+1))
            self.assertEqual(expected, future_peak_blocks(states))

    def test_conditional_samples_and_history_updates(self):
        predictor = HistoryPredictor([2, 4, 8], 12)
        self.assertEqual({predictor.sample(4, 100) for _ in range(30)}, {8})
        self.assertEqual(predictor.sample(8, 100), 100)
        predictor.observe(12)
        self.assertEqual(predictor.sample(8, 100), 12)
        self.assertEqual(predictor.sample(8, 10), 10)

    def test_peak_accounts_for_released_memory(self):
        self.assertEqual(future_peak_blocks([(160, 0, 1), (16, 0, 32)]), 13)


if __name__ == '__main__':
    unittest.main()
