"""Rebuild saved decisions and clocks; keep CPU and GPU timing claims separate."""
import gzip
import math
import statistics as st
from research.v1_dynamic_actions_20260927.common import *
from research.v1_dynamic_actions_20260927.policy import select
from research.v1_budget_20260922.policy import State
from research.v1_budget_20260922.run import quantile


def audit(result, rounds=None):
    assert result['status'] == 'complete' and result['completed'] == len(result['requests'])
    if rounds is None:
        path = OUT/result['rounds_path']
        assert digest(path) == result['rounds_sha256']
        with gzip.open(path, 'rt') as f:
            rounds = json.load(f)
    spec0 = result['config']['threshold']
    branch = result['config'].get('branch')
    events = []
    for row in rounds:
        states = [State(**r) for r in row['state']]
        assert row['free_before'] + sum(r.blocks for r in states) == row['usable_blocks']
        events += row['arrival_events']
        now = row['features']['arrival_time_s']
        assert all(e['time_s'] <= now for e in events)
        rate = sum(e['prompt']*math.exp(-(now-e['time_s'])) for e in events)
        assert math.isclose(rate, row['features']['arrival_rate'], rel_tol=1e-9, abs_tol=1e-7)
        spec1 = branch['threshold'] if branch and row['round'] == branch['round'] else spec0
        choice = select(states, **BASE, free=row['free_before'], total=row['total_blocks'],
                        block_size=16, max_model_len=8176, threshold=spec1,
                        recent=dict(arrival_rate=rate))
        assert choice['threshold'] == row['threshold']
        assert (choice['budget'], choice['chunk']) == (row['budget'], row['chunk'])
        if choice['selected'] is not None:
            assert choice['selected'].tokens == row['scheduled']
            assert choice['selected'].growth == row['actual_growth_blocks']
            assert row['preemptions'] == 0
        if row['intervened']:
            assert row['baseline_growth_blocks']-row['actual_growth_blocks'] >= row['threshold']
        if choice['base'] is not None:
            for r in states:
                n = choice['base'].tokens.get(r.rid, 0)
                if n and (r.decode or (r.outputs == 0 and n == r.remaining)):
                    assert row['scheduled'].get(r.rid, 0) >= n
        assert sum(row['scheduled'].values()) <= row['budget']
    ttft, e2e, tbt = [], [], []
    for req in result['requests']:
        c = result['clocks'][req['id']]
        assert c['tokens'] == req['output'] == len(c['token_times_ms'])
        assert c['first_ms'] == c['token_times_ms'][0] >= req['arrival_ms']
        assert c['finish_ms'] >= c['token_times_ms'][-1]
        ttft.append(c['first_ms']-req['arrival_ms'])
        e2e.append(c['finish_ms']-req['arrival_ms'])
        tbt += [b-a for a,b in zip(c['token_times_ms'],c['token_times_ms'][1:])]
    for name, values in [('ttft',ttft),('tbt',tbt),('e2e',e2e)]:
        assert math.isclose(result['metrics']['mean_'+name+'_ms'], st.fmean(values), rel_tol=1e-10)
        assert math.isclose(result['metrics']['p95_'+name+'_ms'], quantile(values,.95), rel_tol=1e-10)
    executed = sum(sum(r['scheduled'].values()) for r in rounds)
    unique = sum(r['prompt']+r['output']-1 for r in result['requests'])
    assert executed-unique == sum(r['recomputed_tokens'] for r in rounds)
    assert math.isclose(sum(ttft), len(ttft)*result['metrics']['mean_ttft_ms'])
    return dict(passed=True, rounds=len(rounds), arrival_events=len(events),
                executed_tokens=executed, recomputed_tokens=executed-unique)
