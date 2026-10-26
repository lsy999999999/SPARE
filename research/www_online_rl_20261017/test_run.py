"""Focused checks for the frozen online RL adapter."""
import unittest

from research.www_online_rl_20261017 import run


class RLAdapterTests(unittest.TestCase):
    def test_frozen_policy_is_deterministic_and_falls_back_to_spare(self):
        policy = run.QPolicy(table={
            '0:1:0': {'q': [-2., -1., -4., -5., -6., -7.],
                      'visits': [4, 3, 3, 3, 3, 3]},
            '0:2:0': {'q': [-2., 10., 0., 0., 0., 0.],
                      'visits': [2, 1, 0, 0, 0, 0]},
        })
        self.assertEqual(policy.act('0:1:0'), 1)
        self.assertEqual(policy.act('0:1:0'), 1)
        self.assertEqual(policy.act('0:2:0'), 0)
        self.assertEqual(policy.act('unseen'), 0)

    def test_training_and_test_windows_do_not_overlap(self):
        train = {s['name'] for s in run.samples('validation') if s['window'] < 16}
        test = {s['name'] for s in run.samples('test')}
        self.assertEqual(len(train), 32)
        self.assertEqual(len(test), 100)
        self.assertFalse(train & test)

    def test_spare_action_reproduces_original_trajectory(self):
        sample = next(s for s in run.samples('validation') if s['source'] == 'code')
        policy = run.QPolicy()
        policy.act = lambda state: 0
        actual = run.simulate(sample, 2048, 2., policy)
        run.base.initialize()
        expected = run.base.simulate(run.base.generate(sample, 2.), 'arrival',
                                     2048, 1024, sample['source'], 0)
        self.assertEqual(actual['trajectory_sha256'], expected['trajectory_sha256'])
        self.assertEqual(actual['metrics']['ttft_ms_mean'],
                         expected['metrics']['ttft_ms_mean'])


if __name__ == '__main__':
    unittest.main()
