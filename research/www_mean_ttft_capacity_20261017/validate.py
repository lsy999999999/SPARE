"""Check that the delivered plot CSV is complete and traceable to rate records."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from functools import lru_cache

from research.www_mean_ttft_capacity_20261017 import analyze, run


def rows(name):
    with (run.OUT / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


@lru_cache(maxsize=None)
def record(path):
    return json.loads((run.ROOT / path).read_text())


def main():
    status = json.loads((run.OUT / 'status.json').read_text())
    expected_groups = len(run.CAPACITIES) * len(run.METHODS) * len(run.SOURCES)
    assert status['stage'] == 'complete' and not status['errors']
    assert status['groups_complete'] == status['groups_total'] == expected_groups
    assert len(list((run.OUT / 'groups').glob('*.json'))) == expected_groups

    plotted = rows('plot_data_all.csv')
    pareto = rows('pareto_points.csv')
    by_group = defaultdict(list)
    for point in plotted:
        key = (int(point['kv_blocks']), point['arm'])
        assert key[0] in run.CAPACITIES and key[1] in run.METHODS
        assert point['windows_per_workload'] == '50'
        assert point['scan_complete'] == 'True'
        code = float(point['code_capacity_qps'])
        conversation = float(point['conversation_capacity_qps'])
        assert abs(float(point['mean_capacity_qps']) - (code + conversation) / 2) < 1e-9
        for source, rate in (('code', code), ('conversation', conversation)):
            path = point[f'{source}_rate_record_path']
            if not rate:
                assert not path
                continue
            data = record(path)
            assert len(data['rows']) == 50
            assert len({row['name'] for row in data['rows']}) == 50
            summary = data['summary']
            assert abs(float(point[f'{source}_mean_ttft_ms']) - summary['mean_ttft_ms']) < 1e-8
            assert abs(float(point[f'{source}_p99_tbt_ms']) - summary['p99_tbt_ms']) < 1e-8
            assert abs(float(point[f'{source}_backlog_drift']) - summary['backlog_drift']) < 1e-8
            assert int(point[f'{source}_failed_samples']) == summary.get('failed_samples', 0) == 0
            assert summary['p99_tbt_ms'] <= 200
            assert summary['backlog_drift'] <= max(2, rate)
            assert summary['mean_ttft_ms'] <= float(point['mean_ttft_limit_ms']) + 1e-8
        by_group[key].append(point)

    assert set(by_group) == {(kv, arm) for kv in run.CAPACITIES for arm in run.METHODS}
    assert len(pareto) == sum(len(points) - 1 for points in by_group.values())
    for points in by_group.values():
        assert points[-1]['is_plot_extension'] == 'True'
        assert float(points[-1]['mean_ttft_limit_ms']) == analyze.PLOT_MAX_MS
        assert all(p['is_plot_extension'] == 'False' for p in points[:-1])
        assert all(float(a['mean_ttft_limit_ms']) < float(b['mean_ttft_limit_ms'])
                   for a, b in zip(points, points[1:]))
        assert all(float(a['mean_capacity_qps']) <= float(b['mean_capacity_qps'])
                   for a, b in zip(points, points[1:]))
        assert points[-1]['mean_capacity_qps'] == points[-2]['mean_capacity_qps']
    for page in range(1, 5):
        for suffix in ('.png', '.pdf'):
            path = run.OUT / f'mean_ttft_capacity_page{page}{suffix}'
            assert path.is_file() and path.stat().st_size > 0
    print(f'validated {len(plotted)} plot CSV rows, {len(pareto)} observed points, '
          f'{expected_groups} complete groups, and four PNG/PDF pages')


if __name__ == '__main__':
    main()
