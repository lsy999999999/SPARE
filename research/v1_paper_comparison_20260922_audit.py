"""Independent artifact-only audit for the 2026-09-22 three-GPU comparison.

Usage: python3 research/v1_paper_comparison_20260922_audit.py ARTIFACT_DIR
No experiment imports or GPU dependency.
"""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys

root = Path(sys.argv[1])
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
m = read(root / 'manifest.json')
s = read(root / 'summary.json')
sel = read(root / 'selection.json')
rows = [read(p) for p in sorted((root / 'runs').glob('*.json'))]
assert dict(collections.Counter(r['job']['phase'] for r in rows)) == m['expected']
assert len({r['job']['key'] for r in rows}) == len(rows)
assert read(root / 'aa_check.json')['passed']
for name, digest in m['source_hashes'].items():
    assert sha(root / 'source' / name) == digest
for r in rows:
    j = r['job']
    assert j['gpu_uuid'] == m['hardware_by_gpu'][str(j['gpu'])]['uuid']
    assert sha(root / r['rounds_path']) == r['rounds_sha256']
    assert len(r['requests']) == r['request_count']
    assert set(r['clocks']) == {q['id'] for q in r['requests']}
    assert sum(c['finish_ms'] is not None for c in r['clocks'].values()) == r['completed']
    for q in r['requests']:
        c = r['clocks'][q['id']]
        assert c['arrival_ms'] == q['arrival_ms']
        assert c['tokens'] == len(c['token_times_ms'])
        assert c['first_ms'] == (c['token_times_ms'][0] if c['tokens'] else None)
        assert all(a <= b for a, b in zip(c['token_times_ms'], c['token_times_ms'][1:]))
        if r['status'] == 'complete':
            assert c['tokens'] == q['output'] and c['finish_ms'] is not None
    if r['status'] == 'complete':
        ttft = st.fmean(c['first_ms'] - c['arrival_ms'] for c in r['clocks'].values())
        assert math.isclose(ttft, r['metrics']['mean_ttft_ms'], abs_tol=1e-6)
    else:
        assert r['metrics']['mean_ttft_ms'] is None
    throughput = sum(c['tokens'] for c in r['clocks'].values()) / (r['elapsed_ms'] / 1000)
    assert math.isclose(throughput, r['metrics']['throughput_tokens_s'], rel_tol=1e-10)
    with gzip.open(root / r['rounds_path'], 'rt') as f:
        rounds = json.load(f)
    assert len(rounds) == r['metrics']['round_count']
    assert sum(x['preemptions'] for x in rounds) == r['metrics']['preemptions']
    assert sum(x['recomputed_tokens'] for x in rounds) == r['metrics']['recomputed_tokens']
    for x in rounds:
        assert 0 <= x['free_after'] <= x['total_blocks']
        assert sum(x['scheduled'].values()) <= x['budget']
    if j['phase'] == 'compare':
        assert j['config'] == sel[j['regime']][j['arm']]

comp = [r for r in rows if r['job']['phase'] == 'compare']
groups = collections.defaultdict(list)
for r in comp:
    groups[r['job']['regime'], r['job']['block']].append(r)
assert len(groups) == 12
for (regime, block), rs in groups.items():
    assert len(rs) == 8
    assert len({r['job']['gpu'] for r in rs}) == 1
    assert {(r['job']['arm'], r['job']['repeat']) for r in rs} == {
        (a, rep) for a in m['arms'] for rep in (0, 1)}
    assert all(r['requests'] == rs[0]['requests'] for r in rs)
    bi = sorted(b for b, spec in m['blocks'].items() if spec['role'] == 'compare').index(block)
    assert rs[0]['job']['gpu'] == m['comparison_gpu'][regime][bi]
for metric in s['metrics']:
    rs = [r for r in comp if r['job']['regime'] == metric['regime'] and r['job']['arm'] == metric['arm']]
    assert len(rs) == metric['runs'] == 8
    assert metric['complete'] == all(r['status'] == 'complete' for r in rs)
    if metric['complete']:
        for key in ('mean_ttft_ms', 'throughput_tokens_s', 'preemptions', 'recomputed_tokens'):
            assert math.isclose(st.fmean(r['metrics'][key] for r in rs), metric[key], abs_tol=1e-6)
for regime, methods in s['comparisons'].items():
    for arm, data in methods.items():
        for pair in data['pairs']:
            rs = groups[regime, pair['block']]
            a = [r for r in rs if r['job']['arm'] == 'budget']
            b = [r for r in rs if r['job']['arm'] == arm]
            if pair['complete']:
                gain = st.fmean(r['metrics']['mean_ttft_ms'] for r in b) - st.fmean(r['metrics']['mean_ttft_ms'] for r in a)
                assert math.isclose(gain, pair['gain_ms'], abs_tol=1e-6)
        if data['complete']:
            assert math.isclose(st.fmean(p['gain_ms'] for p in data['pairs']), data['mean_gain_ms'], abs_tol=1e-6)
            assert sum(p['gain_ms'] > 0 for p in data['pairs']) == data['positive_blocks']
print(json.dumps(dict(passed=True, records=len(rows), comparison_runs=len(comp),
    comparison_groups=len(groups), failed_runs=[r['job']['key'] for r in rows if r['status'] != 'complete'],
    checked=['frozen source hashes', 'trace hashes and block bounds', 'per-request clocks and outputs',
             'raw TTFT and throughput recomputation', 'same inputs and GPU within each block',
             'selected configuration use', 'summary and paired-gain recomputation']), indent=2))
