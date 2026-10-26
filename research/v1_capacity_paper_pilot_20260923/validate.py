"""Native allocator action equivalence plus a small untimed CPU replay."""
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.v1_capacity_paper_pilot_20260923.common import *
from research.v1_capacity_paper_pilot_20260923.scheduler import PilotScheduler
from research.v1_state_aware_20260923.scheduler import StateScheduler
from research.v1_paper_comparison_20260922.scheduler import ComparisonScheduler, PAPER_METHODS
from research.v1_budget_20260922.test_scheduler import scheduler, add, advance


def equivalence():
    passed = 0
    for arm, (method, cfg) in ARMS.items():
        cls = ComparisonScheduler if method in PAPER_METHODS else StateScheduler
        for cap in (48, 96):
            a, b = [scheduler(blocks=cap, cls=c) for c in (PilotScheduler, cls)]
            for s in (a, b):
                s.history_context = [24]*1000
                s.seed_context = 42
                s.configure(method, **cfg)
                add(s, [257, 129, 385, 99, 400, 12])
            for _ in range(4000):
                x, y = a.schedule(), b.schedule()
                assert x.num_scheduled_tokens == y.num_scheduled_tokens, (arm, cap)
                for name in ('free_after', 'preemptions', 'recomputed_tokens', 'budget', 'chunk'):
                    assert a.round_rows[-1][name] == b.round_rows[-1][name], (arm, cap, name)
                advance(a, x); advance(b, y)
                if not a.requests: break
            assert not a.requests and not b.requests, (arm, cap)
            if method == 'past_future_v1':
                assert list(a.predictor.history) == list(b.predictor.history)
            passed += 1
    return passed


def simulate(block, cap, arm):
    from vllm import SamplingParams
    from vllm.v1.request import Request, RequestStatus
    s = scheduler(blocks=cap, cls=PilotScheduler)
    prepare(s, block.split('_')[0], 0)
    method, cfg = ARMS[arm]
    s.configure(method, **cfg)
    reqs = read(INPUT/(block+'_original.json'))
    targets = {r['id']: r['output'] for r in reqs}
    for i, r in enumerate(reqs):
        s.add_request(Request(r['id'], [100]*r['prompt'], None, None, None,
            SamplingParams(max_tokens=r['output']), None, None, arrival_time=float(i)))
    start = time.monotonic()
    action = hashlib.sha256()
    first, finished = {}, {}
    for step in range(1, 100001):
        out = s.schedule()
        action.update(json.dumps(out.num_scheduled_tokens, sort_keys=True).encode())
        done = []
        for rid in out.num_scheduled_tokens:
            r = s.requests[rid]
            if r.num_computed_tokens == r.num_tokens:
                r.append_output_token_ids(100)
                first.setdefault(rid, step)
            if len(r.output_token_ids) == targets[rid]:
                done.append(rid); finished[rid] = step
        s.finish_requests(done, RequestStatus.FINISHED_LENGTH_CAPPED)
        if not s.requests or time.monotonic()-start > 180: break
    assert not s.requests and len(finished) == len(reqs), (block, cap, arm)
    rounds = s.round_rows
    return dict(block=block, capacity=cap, arm=arm, complete=True, rounds=len(rounds),
        sums={k: sum(r.get(k, 0) for r in rounds) for k in
              ('preemptions', 'recomputed_tokens', 'intervened', 'gate_rejected')},
        action_sha256=action.hexdigest(), first_round=first, finish_round=finished,
        elapsed_s=time.monotonic()-start)


if __name__ == '__main__':
    target = OUT/'cpu'
    target.mkdir(parents=True, exist_ok=True)
    assert not (target/'summary.json').exists(), 'Do not overwrite finished CPU pilot'
    equal = equivalence()
    write(target/'equivalence.json', dict(passed=equal, expected=16, sources=sources()))
    print('Action equivalence passed:', equal, flush=True)
    rows = []
    for block in BLOCKS:
        for cap in CAPS:
            for arm in ARMS:
                r = simulate(block, cap, arm)
                rows.append(r)
                write(target/'runs'/f'{block}_{cap}_{arm}.json', r)
                print(block, cap, arm, r['sums'], flush=True)
    write(target/'summary.json', dict(passed=len(rows), expected=48, rows=rows,
        sources=sources(), history_sha256=digest(HISTORY),
        inputs={b:digest(INPUT/(b+'_original.json')) for b in BLOCKS}))
