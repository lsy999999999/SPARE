"""Check that potential conflicts are not confused with changed decisions."""
from copy import deepcopy
import unittest

from research.www_component_ablation_mean_20261019.behavior import Observer


def fixture(protect=True, later=False, threshold=2):
    safe = {'f': 32, 'd': 1, 'b': 64}
    threat = {'d': 1, 'b': 64}
    ds = [dict(budget=b, chunk=256, tokens=t, acceptance_saving=s,
               guard=(not protect or 'f' in t))
          for b, t, s in ((256, safe if later else threat, 4),
                          (512, threat if later else safe, 3))]
    eligible = [d for d in ds if d['guard'] and d['acceptance_saving'] >= threshold]
    baseline = {'f': 32, 'd': 1, 'b': 128}
    selected = eligible[0]['tokens'] if eligible else baseline
    return dict(preemptions=0, recomputed_tokens=0, free_before=100,
                state=[dict(rid='f', prompt=64, computed=32, tokens=64, outputs=0),
                       dict(rid='d', prompt=64, computed=80, tokens=81, outputs=16),
                       dict(rid='b', prompt=512, computed=0, tokens=512, outputs=0)],
                baseline=baseline, baseline_growth_blocks=10, actual_growth_blocks=6,
                scheduled=selected, candidate_diagnostics=ds, threshold=threshold)


class BehaviorChecks(unittest.TestCase):
    def test_guard_prevents_first_admissible_conflict(self):
        full = Observer('spare')
        full.inspect(fixture(), 20000, {})
        self.assertEqual(full.counts['finish_conflict_rounds'], 0)
        self.assertEqual(full.counts['finish_potential_rounds'], 1)
        self.assertEqual(full.counts['finish_toggle_changed_rounds'], 1)
        removed = Observer('no_finish_guard')
        removed.inspect(fixture(protect=False), 20000, {'protect_finish': False})
        self.assertEqual(removed.counts['finish_conflict_rounds'], 1)
        self.assertEqual(removed.counts['finish_toggle_changed_rounds'], 1)

    def test_later_conflict_does_not_change_first_choice(self):
        observer = Observer('spare')
        observer.inspect(fixture(later=True), 20000, {})
        self.assertEqual(observer.counts['finish_potential_rounds'], 1)
        self.assertEqual(observer.counts['finish_toggle_changed_rounds'], 0)

    def test_below_saving_requirement_is_not_potential_conflict(self):
        observer = Observer('spare')
        observer.inspect(fixture(threshold=10), 20000, {})
        self.assertEqual(observer.counts['finish_potential_rounds'], 0)
        self.assertEqual(observer.counts['finish_toggle_changed_rounds'], 0)

    def test_measurement_interval_and_passive_observation(self):
        row = fixture()
        before = deepcopy(row)
        observer = Observer('spare')
        observer.inspect(row, 19999, {})
        observer.inspect(row, 60000, {})
        self.assertEqual(observer.counts['rounds'], 0)
        observer.inspect(row, 20000, {})
        self.assertEqual(observer.counts['rounds'], 1)
        self.assertEqual(row, before)


if __name__ == '__main__':
    unittest.main()
