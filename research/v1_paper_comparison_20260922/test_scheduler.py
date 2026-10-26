"""Tests against native V1 KV allocator, on CPU; no GPU inference required."""
import unittest

from research.v1_budget_20260922.test_scheduler import scheduler, add, advance
from research.v1_paper_comparison_20260922.scheduler import ComparisonScheduler


def make(method, blocks=96, budget=256, history=(24,)):
    s = scheduler(blocks=blocks, budget=budget, cls=ComparisonScheduler)
    s.history_context = history
    s.seed_context = 42
    s.configure(method, budget, budget, 8)
    return s


class SchedulerTests(unittest.TestCase):
    def test_frozen_a_and_srp_are_action_equivalent(self):
        for method in ('budget', 'srp'):
            a, b = scheduler(method, blocks=48), make(method, blocks=48)
            for s in (a, b):
                add(s, [257, 129, 385, 99, 400, 12])
            for _ in range(3000):
                x, y = a.schedule(), b.schedule()
                self.assertEqual(x.num_scheduled_tokens, y.num_scheduled_tokens)
                self.assertEqual(a.round_rows[-1]['free_after'], b.round_rows[-1]['free_after'])
                advance(a, x)
                advance(b, y)
                if not a.requests:
                    break
            self.assertFalse(a.requests)
            self.assertFalse(b.requests)

    def test_sarathi_decode_then_partial_then_waiting(self):
        s = make('sarathi_v1', budget=64)
        add(s, [8, 200])
        x = s.schedule()
        self.assertEqual(x.num_scheduled_tokens, {'0': 8, '1': 56})
        advance(s, x)
        # Simulate the native running order having a partial prefill first.
        s.running.reverse()
        x = s.schedule()
        self.assertEqual(list(x.num_scheduled_tokens), ['0', '1'])
        self.assertEqual(x.num_scheduled_tokens, {'0': 1, '1': 63})

    def test_future_gate_waits_then_releases_fifo_head(self):
        s = make('past_future_v1', blocks=10, budget=64, history=(24,))
        add(s, [96, 96])
        checked = rejected = 0
        for _ in range(300):
            x = s.schedule()
            checked += s.round_rows[-1]['gate_checked']
            rejected += s.round_rows[-1]['gate_rejected']
            self.assertEqual(len(s.requests), len(s.running)+len(s.waiting))
            advance(s, x)
            if not s.requests:
                break
        self.assertFalse(s.requests)
        self.assertGreater(checked, 0)
        self.assertGreater(rejected, 0)
        self.assertEqual(list(s.predictor.history), [24, 24, 24])

    def test_prediction_does_not_read_replay_stop_length(self):
        a, b = make('past_future_v1'), make('past_future_v1')
        add(a, [100, 200])
        add(b, [100, 200])
        for r in b.requests.values():
            r.max_tokens = 9999
        self.assertEqual(a.schedule().num_scheduled_tokens, b.schedule().num_scheduled_tokens)
        self.assertEqual(a.paper_round, b.paper_round)

    def test_future_gate_native_recovery_and_queue_restoration(self):
        s = make('past_future_v1', blocks=24, history=(1,))
        add(s, [160, 128, 96, 80, 48])
        for _ in range(3000):
            x = s.schedule()
            self.assertEqual(len(s.requests), len(s.running)+len(s.waiting))
            self.assertEqual(len({r.request_id for r in s.running} |
                                 {r.request_id for r in s.waiting}), len(s.requests))
            advance(s, x)
            if not s.requests:
                break
        self.assertFalse(s.requests)


if __name__ == '__main__':
    unittest.main()
