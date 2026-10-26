"""Audit every resumed group and service-capacity boundary."""
from __future__ import annotations

from . import run


def main():
    run.prepare()
    assert run.prior.base.read(run.OUT / 'smoke.json')['passed']
    status = run.prior.base.read(run.OUT / 'status.json')
    assert status['stage'] == 'complete' and status['complete'] == status['total'] == 40
    assert not status['errors']
    count = 0
    window_rows = 0
    for source in ('code', 'conversation'):
        for kv in run.CAPACITIES:
            for arm in run.OPTIONS:
                path = run.OUT / 'groups' / f'{source}_c{kv}_{arm}.json'
                group = run.prior.base.read(path)
                assert (group['source'], group['kv_blocks'], group['arm']) == (source, kv, arm)
                assert group['windows'] == 50
                rates = group['tested_rates']
                assert rates == sorted(rates) and .5 in rates
                passed = []
                for rate in rates:
                    record = run.prior.base.read(
                        run.OUT / 'rates' / f'{source}_c{kv}_{arm}_r{rate:g}.json')
                    rows = record['rows']
                    assert len(rows) == len({r['name'] for r in rows}) == 50
                    assert all((r['source'], r['capacity'], r['arm'], r['rate']) ==
                               (source, kv, arm, rate) for r in rows)
                    assert record['summary'] == run.prior.archive.summary(rows, rate)
                    passed.append(record['summary']['passed'])
                    count += 1
                    window_rows += len(rows)
                assert passed == sorted(passed, reverse=True)
                capacity = max((rate for rate, ok in zip(rates, passed) if ok), default=0.)
                failure = next((rate for rate, ok in zip(rates, passed) if not ok), None)
                assert capacity == group['capacity_qps']
                assert failure == group['next_failed_qps']
    assert count == 355 and window_rows == 17750
    trace = run.prior.base.read(run.OUT / 'analysis/trace/metadata.json')
    assert trace['base_growth_blocks'] - trace['spare_growth_blocks'] == 48
    assert trace['threshold'] == 39
    report = dict(passed=True, groups=40, rates=count, window_rows=window_rows,
                  first_divergence_blocks_saved=48)
    run.prior.base.write(run.OUT / 'validation.json', report)
    print(report)


if __name__ == '__main__':
    main()
