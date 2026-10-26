"""Verify completed capacity boundaries, reused evidence, and passive replays."""
import csv
import json
import math

from research.www_component_ablation_mean_20261019 import complete, behavior

READ, WRITE = complete.READ, complete.WRITE


def audit():
    status = READ(complete.OUT / 'status.json')
    assert status['completed'] == status['total'] == 84 and not status['errors']
    design = READ(complete.OUT / 'design.json')
    assert complete.DIGEST(complete.ROOT / 'research/www_component_ablation_mean_20261019/policy.py') == design['policy_sha256']
    assert all(complete.DIGEST(complete.ROOT / path) == sha for path, sha in design['input_sha256'].items())
    names = {source: {s['name'] for s in design['samples'] if s['source'] == source}
             for source in complete.pilot.SOURCES}
    boundaries = 0
    points = list(csv.DictReader((complete.OUT / 'analysis/points.csv').open()))
    assert len(points) == 336
    for row in points:
        for field in ('capacity_qps', 'next_fail_qps'):
            rate = float(row[field])
            if rate == 0:
                continue
            path = complete.OUT / 'rates' / complete.pilot.old.rate_name(
                row['source'], int(row['kv_blocks']), row['arm'], rate)
            data = READ(path)
            raw = data['rows']
            assert len(raw) == 24 and {r['name'] for r in raw} == names[row['source']]
            assert all(r['valid'] for r in raw)
            values = {k: sum(r[k] for r in raw)/24 for k in
                      ('mean_ttft_ms', 'p99_tbt_ms', 'backlog_drift')}
            assert all(math.isclose(v, data['summary'][k], abs_tol=1e-8) for k, v in values.items())
            passing = (values['mean_ttft_ms'] <= float(row['theta_s'])*1000
                       and values['p99_tbt_ms'] <= 200 and values['backlog_drift'] <= max(2, rate))
            assert passing == (field == 'capacity_qps'), (row, field)
            boundaries += 1
    reused_rows = 0
    total_rows = 0
    newly_simulated = 0
    for path in (complete.OUT / 'rates').glob('*.json'):
        data = READ(path)
        assert len(data['rows']) == 24
        total_rows += 24
        newly_simulated += data['newly_simulated_this_attempt']
        if data['reused_from']:
            source = complete.ROOT / data['reused_from']
            assert complete.DIGEST(source) == data['reused_sha256']
            prior = {r['name']: r for r in READ(source)['rows']}
            for r in data['rows']:
                if r['name'] in prior:
                    for k in ('trajectory_sha256', 'mean_ttft_ms', 'p99_tbt_ms', 'backlog_drift'):
                        assert r[k] == prior[r['name']][k]
                    reused_rows += 1
    paired_behavior = unmatched_behavior = 0
    main_replays = 0
    for path in (behavior.OUT / 'rates').glob('*.json'):
        current = READ(path)['rows']
        rate = current[0]['rate']
        archive = complete.OUT / 'rates' / path.name
        if archive.exists():
            prior = {r['name']: r for r in READ(archive)['rows']}
            assert len(prior) == 24
            for r in current:
                reference = prior[r['name']]
                for k in ('trajectory_sha256', 'mean_ttft_ms', 'p99_tbt_ms', 'backlog_drift',
                          'preemptions', 'recomputed_tokens'):
                    assert r[k] == reference[k], (path.name, r['name'], k)
                paired_behavior += 1
                main_replays += rate == 2
        else:
            unmatched_behavior += len(current)
    assert main_replays == 54*24
    result = dict(capacity_groups=84, capacity_points=len(points),
                  verified_boundary_endpoints=boundaries, capacity_window_records=total_rows,
                  reused_window_records_verified=reused_rows,
                  new_window_simulations_this_run=newly_simulated,
                  behavioral_replays_vs_completed_capacity=paired_behavior,
                  behavioral_replays_without_matching_capacity_rate=unmatched_behavior,
                  main_2qps_replays_verified=main_replays,
                  input_hashes_verified=True, frozen_policy_hash_verified=True,
                  errors=[])
    WRITE(complete.OUT / 'analysis/final_audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    audit()
