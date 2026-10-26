"""Recompute saved GPU clocks and allocation accounting without rerunning policy."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/v1_direction_probe_20260923_gpu_final'


def quantile(xs, q):
    xs = sorted(xs)
    x = (len(xs)-1)*q
    i = int(x)
    return xs[i] + (xs[min(i+1,len(xs)-1)]-xs[i])*(x-i)


def main():
    results = []
    signal_rows = []
    for path in sorted((OUT/'runs').glob('*.json')):
        r = json.loads(path.read_text())
        assert r['status']=='complete', path.name
        c = r['clocks']
        ttft = []; e2e = []; tbt = []
        for req in r['requests']:
            x = c[req['id']]
            assert x['arrival_ms']==req['arrival_ms']
            assert x['tokens']==req['output']==len(x['token_times_ms'])
            assert x['first_ms']==x['token_times_ms'][0]
            assert req['arrival_ms']<=x['submitted_ms']<=x['first_ms']<=x['finish_ms']
            ttft.append(x['first_ms']-req['arrival_ms'])
            e2e.append(x['finish_ms']-req['arrival_ms'])
            tbt.extend(b-a for a,b in zip(x['token_times_ms'],x['token_times_ms'][1:]))
        for key,value in [('mean_ttft_ms',st.fmean(ttft)),('p95_ttft_ms',quantile(ttft,.95)),
                          ('mean_e2e_ms',st.fmean(e2e)),('p95_tbt_ms',quantile(tbt,.95))]:
            assert math.isclose(value,r['metrics'][key],abs_tol=1e-6), (path.name,key)
        trace_path = OUT/r['rounds_path']
        assert hashlib.sha256(trace_path.read_bytes()).hexdigest()==r['rounds_sha256']
        with gzip.open(trace_path,'rt') as f:
            rounds = json.load(f)
        assert len(rounds)==r['metrics']['round_count']
        for row in rounds:
            assert row['free_before']>=0 and row['free_after']>=0
            assert sum(row['scheduled'].values())<=row['budget']
            if row['verified']:
                assert row['preemptions']==0
                physical_growth=sum(x['blocks_after']-x['blocks_before'] for x in row['allocations'])
                assert physical_growth==row['free_before']-row['free_after']
            for d in row['reductions']:
                assert not (d['outputs']>0 and d['remaining']==1)
                assert not (d['outputs']==0 and d['baseline_tokens']==d['remaining'])
            if row['intervened']:
                assert row['baseline_growth_blocks']>row['actual_growth_blocks']
                if r['job']['method'].endswith('_risk'):
                    assert row['probe']['chosen']['risk']<row['probe']['base_risk']
        for key in ('preemptions','recomputed_tokens'):
            assert sum(x[key] for x in rounds)==r['metrics'][key]
        actual_work = sum(sum(x['scheduled'].values()) for x in rounds)
        useful_work = sum(req['prompt']+req['output']-1 for req in r['requests'])
        assert actual_work-useful_work==r['metrics']['recomputed_tokens'], path.name
        assert math.isclose(sum(x['decision_ms'] for x in rounds),r['metrics']['decision_ms_total'],abs_tol=1e-6)
        results.append(dict(key=r['job']['key'], verified_rounds=sum(x['verified'] for x in rounds),
                            actual_work=actual_work, useful_work=useful_work))
        if r['job']['phase']=='compare' and r['job']['method'].startswith('horizon8'):
            xs = [x for x in rounds if x['baseline_growth_blocks'] and 'base_trajectory' in x['probe']]
            simple = lambda x: 8*x['baseline_growth_blocks']>=max(x['free_before'],1)
            signal_rows.append(dict(key=r['job']['key'], capacity=r['job']['cell']['capacity'],
                method=r['job']['method'], eligible_rounds=len(xs),
                agreement=sum(simple(x)==x['pressure_triggered'] for x in xs)/len(xs),
                early_only=sum(simple(x) and not x['pressure_triggered'] for x in xs),
                horizon_only=sum(not simple(x) and x['pressure_triggered'] for x in xs)))
    assert len(results)==34, len(results)
    report = dict(passed=True, runs=len(results),
        scope='saved-clock and physical-allocation accounting recheck; not external independent review',
        rows=results, gpu_same_state_signals=signal_rows)
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,runs=len(results),gpu_same_state_signals=signal_rows),indent=2))


if __name__ == '__main__': main()
