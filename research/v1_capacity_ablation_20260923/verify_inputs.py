"""Validate the frozen design, traces, matrix and unchanged reference A."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/v1_capacity_design_20260923'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((OUT/'input_manifest.json').read_text())
    config = json.loads((HERE/'experiment.json').read_text())
    assert manifest['config'] == config
    assert manifest['design_sha256'] == digest(HERE/'experiment.json')
    assert manifest['builder_sha256'] == digest(HERE/'prepare.py')
    assert manifest['previous_manifest_sha256'] == digest(
        ROOT/'research/state_conditioned_prefill_selection_20260918/data_manifest.json')
    old = json.loads((ROOT/'artifacts/v1_paper_comparison_multi_20260922_r3/manifest.json').read_text())
    reference = {}
    for name in ('policy.py', 'scheduler.py', 'run.py'):
        key = 'research/v1_budget_20260922/'+name
        reference[key] = digest(ROOT/key)
        assert reference[key] == old['frozen_reference_hashes'][key]
    seen = set()
    for block, payload in manifest['blocks'].items():
        rows = payload['rows']
        assert len(rows) == config['requests_per_role'][payload['role']]
        assert payload['source_sha256'] == config['sources'][payload['source']]['sha256']
        assert payload['summary']['within_model_context']
        for row in rows:
            assert row['id'] not in seen
            seen.add(row['id'])
        for profile in config['arrival_profiles']:
            requests = json.loads((OUT/'traces'/f'{block}_{profile}.json').read_text())
            assert len(requests) == len(rows)
            for request, row in zip(requests, rows):
                assert (request['id'], request['prompt'], request['output']) == (
                    row['id'], row['prompt'], row['output'])
                expected = row['arrival_ms'] if profile == 'original' else (
                    row['arrival_ms']/payload['summary']['arrival_span_ms']*20)
                assert abs(request['arrival_ms']-expected) < 1e-9
            assert requests[0]['arrival_ms'] == 0
    jobs = manifest['jobs']
    assert len(jobs) == len({job['id'] for job in jobs}) == 936
    assert {stage: sum(j['stage']==stage for j in jobs) for stage in manifest['planned_counts']} == {
        'tune': 96, 'core': 576, 'ablation': 216, 'shadow': 48}
    assert manifest['total_planned'] == len(jobs)+sum(manifest['validation_counts'].values()) == 999
    for job in jobs:
        assert job['input_feasible']
        payload = manifest['blocks'][job['block']]
        assert payload['summary']['max_single_request_blocks'] <= job['capacity']-1
        assignment = config['tuning_gpu_assignment'] if job['stage']=='tune' else config['test_gpu_assignment']
        assert job['gpu'] == assignment[job['block']]
        assert payload['role'] == ('dev' if job['stage']=='tune' else 'test')
    assert len(seen) == 512
    print(json.dumps(dict(status='passed', windows=len(manifest['blocks']), unique_requests=len(seen),
        trace_files=20, enumerated_jobs=len(jobs), total_planned=999,
        current_a_matches_previous_frozen_source=True, measured_runs=0,
        limitation='Overlap checked against the linked historical data manifest only'), indent=2))


if __name__ == '__main__':
    main()
