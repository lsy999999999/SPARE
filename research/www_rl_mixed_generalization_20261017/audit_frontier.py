"""Check every plotted capacity against the archived 50-window rate records."""
from __future__ import annotations

import json

from research.www_rl_mixed_generalization_20261017 import frontier, run, sweep


def main():
    checked = 0
    for source in ('code', 'conversation'):
        cohort = sweep.cohort_for('original', source)
        signature = sweep.source_hash(cohort)
        for arm, seeds in (('spare', (0,)), ('rl_mixed', (1017, 2027, 3037))):
            for seed in seeds:
                path = (frontier.OUT / 'groups' /
                        f'original_{source}_c2048_{arm}_seed{seed}.json')
                group = run.old.base.read(path)
                assert group['stage'] == 'complete'
                assert group['input_sha256'] == signature
                assert len(group['thresholds']) == len(frontier.LIMITS_MS)
                for point in group['thresholds']:
                    limit = point['mean_ttft_limit_ms']
                    passed = point['capacity_qps']
                    failed = point['next_failed_qps']
                    assert (passed == 0 and point['below_minimum_measured_rate']
                            or passed > 0 and failed - passed <= .020001)
                    for rate, expected in ((passed, True), (failed, False)):
                        if rate == 0:
                            continue
                        original = (sweep.original_records()['spare', source, 2048].get(rate)
                                    if arm == 'spare' else None)
                        record_path = original or sweep.rate_path(
                            'original', source, 2048, arm, seed, rate)
                        assert record_path.exists(), record_path
                        summary, fresh = sweep.get_rate(
                            'original', source, 2048, arm, seed, rate,
                            cohort, signature)
                        assert not fresh
                        assert frontier.threshold_passed(summary, rate, limit) == expected
                        checked += 1
    print(json.dumps({'groups': 8, 'thresholds_per_group': len(frontier.LIMITS_MS),
                      'raw_rate_checks': checked, 'status': 'passed'}))


if __name__ == '__main__':
    main()
