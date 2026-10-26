"""Reproducible bounded CPU direction probe, deliberately no millisecond model."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics as st
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.v1_state_aware_20260923.common import read, write, digest, INPUT
from research.v1_direction_probe_20260923.policy import METHODS

HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/v1_direction_probe_20260923_r1'
CELLS = [(block, cap) for block in ('code_test0', 'conversation_test0') for cap in (768, 2048)]
CONFIG = dict(budget=1024, chunk=256, concurrency=32)


def run(block, cap, method):
    from vllm import SamplingParams
    from vllm.v1.request import Request, RequestStatus
    from research.v1_budget_20260922.test_scheduler import scheduler
    from research.v1_direction_probe_20260923.scheduler import ProbeScheduler
    s = scheduler(blocks=cap, cls=ProbeScheduler)
    cfg = dict(CONFIG, budget=512) if method == 'fixed_small' else CONFIG
    s.configure(method, **cfg)
    reqs = read(INPUT/(block+'_original.json'))
    targets = {r['id']: r['output'] for r in reqs}
    clocks = {r['id']: dict(first=None, finish=None, output_tokens=0) for r in reqs}
    for i, r in enumerate(reqs):
        s.add_request(Request(r['id'], [100]*r['prompt'], None, None, None,
            SamplingParams(max_tokens=r['output']), None, None, arrival_time=float(i)))
    action_hash = hashlib.sha256()
    start = time.monotonic()
    for step in range(1, 100001):
        out = s.schedule()
        action_hash.update(json.dumps(out.num_scheduled_tokens, sort_keys=True).encode())
        done = []
        for rid in out.num_scheduled_tokens:
            req = s.requests[rid]
            if req.num_computed_tokens == req.num_tokens:
                req.append_output_token_ids(100)
                clocks[rid]['output_tokens'] += 1
                if clocks[rid]['first'] is None:
                    clocks[rid]['first'] = step
            if len(req.output_token_ids) >= targets[rid]:
                done.append(rid)
                clocks[rid]['finish'] = step
        s.finish_requests(done, RequestStatus.FINISHED_LENGTH_CAPPED)
        if not s.requests or time.monotonic()-start > 180:
            break
    complete = not s.requests
    rounds = s.round_rows
    sums = {name: sum(r[key] for r in rounds) for name, key in
            [('preemptions', 'preemptions'), ('recomputed_tokens', 'recomputed_tokens'),
             ('interventions', 'intervened'), ('pressure_rounds', 'pressure_triggered'),
             ('recovery_rounds', 'recovery_fallback')]}
    deferred = [d for r in rounds for d in r['reductions'] if d['outputs'] == 0]
    sums.update(deferred_tokens=sum(d['baseline_tokens']-d['actual_tokens'] for d in deferred),
                deferred_request_rounds=len(deferred), rounds=len(rounds))
    means = dict(mean_first_token_round=st.fmean(c['first'] for c in clocks.values()) if complete else None,
                 mean_finish_round=st.fmean(c['finish'] for c in clocks.values()) if complete else None,
                 decision_us_mean=1000*st.fmean(r['decision_ms'] for r in rounds))
    upcoming = [any(x['preemptions'] for x in rounds[i+1:i+9]) for i in range(len(rounds))]
    eligible = [i for i,r in enumerate(rounds) if r['baseline_growth_blocks'] is not None
                and r['baseline_growth_blocks'] > 0]
    events = dict(eligible_rounds=len(eligible),
        untriggered_then_preemption_8=sum(not rounds[i]['pressure_triggered'] and upcoming[i] for i in eligible),
        horizon_trigger_below_single_gate=sum(r['pressure_triggered'] and r['baseline_growth_blocks'] is not None
            and 4*r['baseline_growth_blocks'] < max(r['free_before'],1) for r in rounds),
        risk_rejected_all=sum(r['probe'].get('risk_rejected_all',False) for r in rounds),
        chosen_configs=dict(Counter(f"{r['budget']}/{r['chunk']}" for r in rounds if r['intervened'])))
    if complete:
        assert all(c['output_tokens'] == targets[rid] and c['finish'] >= c['first'] > 0
                   for rid, c in clocks.items())
    key = f'{block}_k{cap}_{method}'
    trace = OUT/'rounds'/(key+'.json.gz')
    trace.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(trace, 'wt', compresslevel=1) as f:
        json.dump(rounds, f)
    return dict(key=key, block=block, capacity=cap, method=method, config=cfg,
                complete=complete, stats=dict(**sums, **means), events=events, clocks=clocks,
                cpu_wall_s=time.monotonic()-start, action_sha256=action_hash.hexdigest(),
                rounds_path=str(trace.relative_to(OUT)), rounds_sha256=digest(trace))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    global_out(args.output)
    source_files = list(HERE.glob('*.py')) + [HERE/'PROTOCOL_CN.md']
    source_files += [ROOT/f'research/{folder}/{name}' for folder, names in
        [('v1_state_aware_20260923', ('policy.py', 'scheduler.py')),
         ('v1_budget_20260922', ('policy.py', 'scheduler.py', 'test_scheduler.py')),
         ('v1_capacity_ablation_20260923', ('policy.py',))] for name in names]
    manifest = dict(methods=list(METHODS), cells=CELLS,
        claim='CPU physical-allocation exploration; all-at-start arrivals; unit rounds, not GPU TTFT',
        sources={str(p.relative_to(ROOT)):digest(p) for p in source_files},
        inputs={b:digest(INPUT/(b+'_original.json')) for b in {b for b,_ in CELLS}})
    manifest = json.loads(json.dumps(manifest))
    path = OUT/'manifest.json'
    if path.exists():
        assert read(path) == manifest, 'Changed source: use a new output directory'
    else:
        write(path, manifest)
    old = read(ROOT/'artifacts/v1_state_aware_20260923_r1/simulation_summary.json')['rows']
    results = []
    for block, cap in CELLS:
        for method in METHODS:
            key = f'{block}_k{cap}_{method}'
            path = OUT/'runs'/(key+'.json')
            result = read(path) if path.exists() else run(block, cap, method)
            if method in METHODS[:3]:
                matches = [r for r in old if r['block']==block and r['capacity']==cap
                           and r['method']==method and (method!='pressure_one' or r['config']['pressure']=='1/4')]
                assert len(matches) == 1
                assert result['action_sha256'] == matches[0]['action_sha256'], key
                result['old_action_hash_matched'] = True
            write(path, result)
            results.append(result)
            write(OUT/'status.json', dict(complete=False, saved=len(results), expected=len(CELLS)*len(METHODS), current=key))
            print(key, result['complete'], result['stats'], flush=True)
    for result in results:
        assert digest(OUT/result['rounds_path']) == result['rounds_sha256']
        with gzip.open(OUT/result['rounds_path'], 'rt') as f:
            trace = json.load(f)
        for key in ('preemptions', 'recomputed_tokens'):
            assert sum(r[key] for r in trace) == result['stats'][key]
    write(OUT/'summary.json', dict(manifest=manifest, rows=results,
        all_complete=all(r['complete'] for r in results), accounting_audit_passed=True))
    write(OUT/'status.json', dict(complete=True, saved=len(results), expected=len(CELLS)*len(METHODS)))


def global_out(path):
    global OUT
    OUT = path.resolve()


if __name__ == '__main__':
    main()
