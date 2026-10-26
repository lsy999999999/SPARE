import unittest
from research.v1_budget_20260922.policy import State, project
from research.v1_budget_20260922.test_scheduler import scheduler, add, advance
from research.v1_state_aware_20260923.scheduler import StateScheduler
from research.v1_direction_probe_20260923.scheduler import ProbeScheduler
from research.v1_direction_probe_20260923.policy import METHODS, forecast, select

CONFIG = dict(budget=1024, chunk=256, concurrency=32)


class ProbeTests(unittest.TestCase):
    def test_old_paths_identical(self):
        for method in METHODS[:3]:
            a = scheduler(blocks=96, cls=StateScheduler)
            b = scheduler(blocks=96, cls=ProbeScheduler)
            for s in (a, b):
                s.configure(method, **CONFIG)
                add(s, [513, 600, 251, 23, 449, 375, 67, 100])
            while a.requests:
                x, y = a.schedule(), b.schedule()
                self.assertEqual(x.num_scheduled_tokens, y.num_scheduled_tokens)
                self.assertEqual(a.round_rows[-1]['free_after'], b.round_rows[-1]['free_after'])
                advance(a, x); advance(b, y)
            self.assertFalse(b.requests)

    def test_forecast_matches_no_completion_native_growth(self):
        s = scheduler(blocks=1024, cls=StateScheduler)
        s.configure('srp_matched', budget=96, chunk=32, concurrency=8)
        add(s, [33, 71, 112])
        s._order()
        rows = [State(r.request_id, 0, r.num_tokens, r.num_prompt_tokens, 0, 0, False)
                for r in s.waiting]
        first = project(rows, 96, 32, 1023, 16, 8, 12288)
        expected = forecast(rows, first, 8, 96, 32, 16, 8, 12288)
        actual = []
        for _ in range(8):
            out = s.schedule()
            actual.append(1023-s.round_rows[-1]['free_after'])
            advance(s, out, target=100)
        self.assertEqual(expected, actual)

    def test_future_block_boundaries_matter_when_current_growth_is_zero(self):
        def demand(computed):
            rows = [State('d', computed, computed+1, 1, computed, 1, True)]
            first = project(rows, 16, 16, 8, 16, 8, 12288)
            self.assertEqual(first.growth, 0)
            return forecast(rows, first, 4, 16, 16, 16, 8, 12288)[-1]
        self.assertEqual(demand(12), 0)
        self.assertEqual(demand(14), 1)

    def test_new_methods_complete_with_native_physical_guards(self):
        for method in METHODS[3:]:
            s = scheduler(blocks=96, cls=ProbeScheduler)
            s.configure(method, **CONFIG)
            add(s, [513, 600, 251, 23, 449, 375, 67, 100, 599, 501, 243, 183])
            for _ in range(10000):
                out = s.schedule()
                row = s.round_rows[-1]
                self.assertGreaterEqual(row['free_after'], 0)
                if row['intervened']:
                    self.assertGreater(row['baseline_growth_blocks'], row['actual_growth_blocks'])
                    if method.endswith('_risk'):
                        self.assertLess(row['probe']['chosen']['risk'], row['probe']['base_risk'])
                advance(s, out)
                if not s.requests:
                    break
            self.assertFalse(s.requests, method)


if __name__ == '__main__':
    unittest.main()
