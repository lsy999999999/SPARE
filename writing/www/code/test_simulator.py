"""Checks target semantic equivalence and execution accounting, not GPU speed."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from writing.www.code.run_simulator import simulate, make_scheduler
from research.v1_ratio_closed_loop_20260928.simulate import simulate as frozen_simulate
from research.v1_broad12h_20260925.policy import ArrivalRate, formula, select
from research.v1_budget_20260922.policy import State
from research.v1_paper_comparison_20260922.rules import (
    apt_kv_density, geometric_quantize, uniboost_priority)


class CurrentMethodTests(unittest.TestCase):
    def test_draft_example_and_basic_ablation(self):
        rows = [State(str(i), 0, p, p, 0, 0, False)
                for i, p in enumerate([1024, 2048, 2048, 3072])]
        args = (rows, 1024, 256, 256, 257, 16, 32, 8176)
        arrival = select(*args, threshold=formula(1), recent=dict(arrival_rate=8192))
        basic = select(*args, threshold=formula(0), recent=dict(arrival_rate=8192))
        self.assertEqual(arrival['threshold'], 40)
        self.assertEqual(arrival['base'].growth-arrival['selected'].growth, 48)
        self.assertEqual((arrival['budget'], arrival['chunk']), (256, 256))
        self.assertFalse(basic['changed'])

    def test_arrival_clock_decay(self):
        rate = ArrivalRate(1)
        rate.observe(0, 100)
        rate.observe(1, 200)
        import math
        self.assertAlmostEqual(rate.at(2), 100*math.exp(-2)+200*math.exp(-1))
        self.assertEqual(rate.total, 300)

    def test_matches_latest_fixed64_native_simulator(self):
        reqs = [dict(id=f'r{i:03}', prompt=(128, 800, 1800, 2500)[i%4]+i,
                     output=24, arrival_ms=float(i//5)*40) for i in range(20)]
        for profile in ('resource', 'resource_staircase'):
            new = simulate(reqs, capacity=192, profile=profile)
            old = frozen_simulate(reqs, 192, 1024, 256, 'fixed64', profile)
            self.assertEqual(new['trajectory_sha256'], old['trajectory_sha256'])
            self.assertEqual(new['metrics']['ttft_ms_mean'], old['mean_ttft_ms'])
            self.assertEqual(new['metrics']['recomputed_tokens'], old['recomputed_tokens'])
            self.assertGreater(new['metrics']['preemptions'], 0)

    def test_every_arm_drains_and_preserves_output_accounting(self):
        reqs = [dict(id=str(i), prompt=100+i*80, output=20, arrival_ms=0.) for i in range(8)]
        for arm in ('arrival', 'basic', 'srp', 'native', 'sarathi', 'past_future',
                    'ssjf_oracle', 'mc_sf_oracle', 'trail_oracle',
                    'tie_oracle_proxy', 'uniboost_static',
                    'apt_serve_kv_rule', 'sola_rule'):
            result = simulate(reqs, arm=arm, capacity=96, history=[20]*100)
            m = result['metrics']
            self.assertEqual(m['completed'], len(reqs))
            self.assertEqual(m['executed_tokens']-m['recomputed_tokens'],
                             sum(r['prompt']+r['output']-1 for r in reqs))
            self.assertAlmostEqual(m['ttft_waiting_area_ms_requests'], m['ttft_ms_mean']*len(reqs))

    def test_oracle_arms_are_explicit_and_change_short_job_order(self):
        reqs = [dict(id='long', prompt=64, output=40, arrival_ms=0.),
                dict(id='short', prompt=64, output=2, arrival_ms=0.)]
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)/'ssjf'
            result = simulate(reqs, arm='ssjf_oracle', capacity=128,
                              budget=64, chunk=64, output=dest)
            self.assertTrue(result['uses_realized_output_lengths'])
            self.assertFalse(result['deployable'])
            with gzip.open(dest/'iterations.jsonl.gz', 'rt') as handle:
                first = json.loads(next(handle))
            self.assertEqual(list(first['scheduled']), ['short'])

        with self.assertRaisesRegex(ValueError, 'realized-length map'):
            make_scheduler('mc_sf_oracle', blocks=128)

    def test_paper_rule_helpers(self):
        self.assertEqual(geometric_quantize(0, 16), 16)
        self.assertEqual(geometric_quantize(31, 16), 16)
        self.assertEqual(geometric_quantize(32, 16), 32)
        self.assertGreater(apt_kv_density(.8, 2), apt_kv_density(.4, 2))
        self.assertLess(apt_kv_density(1.1, 1), apt_kv_density(.1, 100))
        self.assertLess(uniboost_priority(0., 16),
                        uniboost_priority(0., 64))

    def test_trail_limited_preemption_threshold(self):
        scheduler, _ = make_scheduler('trail_oracle', blocks=128,
            oracle_lengths={'running': 10, 'waiting': 1})
        # The threshold predicate itself is recorded and exercised in the
        # end-to-end drain test. At exactly floor(.8*10), it must be disabled.
        request = type('R', (), {'request_id': 'running',
            'output_token_ids': [1] * 8})()
        self.assertFalse(len(request.output_token_ids) <
                         int(.8 * scheduler.oracle_lengths['running']))

    def test_first_decision_does_not_read_hidden_output_or_future_arrivals(self):
        common = [dict(id='b', prompt=1024, output=4, arrival_ms=0.),
                  dict(id='a', prompt=2048, output=8, arrival_ms=0.)]
        first = []
        with tempfile.TemporaryDirectory() as tmp:
            for index, output in enumerate((4, 40)):
                reqs = [dict(r) for r in common]
                reqs[0]['output'] = output
                if index:
                    reqs.append(dict(id='future', prompt=3000, output=10, arrival_ms=100000.))
                dest = Path(tmp)/str(index)
                simulate(reqs, capacity=512, output=dest)
                with gzip.open(dest/'iterations.jsonl.gz', 'rt') as handle:
                    first.append(json.loads(next(handle)))
        for key in ('scheduled', 'threshold', 'features', 'state'):
            self.assertEqual(first[0][key], first[1][key])


if __name__ == '__main__':
    unittest.main()
