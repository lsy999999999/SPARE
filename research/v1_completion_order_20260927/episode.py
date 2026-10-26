"""Recorded synchronous episode; same clocks/metrics as the existing GPU runner."""
import time
import statistics as st
from research.v1_budget_20260922.run import quantile

def episode(engine, scheduler, reqs, method, config, timeout=300):
    from vllm import SamplingParams
    scheduler.configure(method, **config)
    clocks = {r['id']: dict(arrival_ms=r['arrival_ms'], first_ms=None,
                          finish_ms=None, tokens=0, token_times_ms=[]) for r in reqs}
    cursor = 0
    error = None
    start = time.perf_counter()
    try:
        while cursor < len(reqs) or engine.has_unfinished_requests():
            now = (time.perf_counter()-start)*1000
            if now > timeout*1000:
                raise TimeoutError('Predeclared episode deadline exceeded')
            while cursor < len(reqs) and reqs[cursor]['arrival_ms'] <= now:
                r = reqs[cursor]
                clocks[r['id']]['submitted_ms'] = now
                engine.add_request(r['id'], {'prompt_token_ids': [100]*r['prompt']},
                    SamplingParams(temperature=0., max_tokens=r['output'], ignore_eos=True,
                                   detokenize=False))
                cursor += 1
            if engine.has_unfinished_requests():
                step_start = time.perf_counter()
                outputs = engine.step()
                now = (time.perf_counter()-start)*1000
                scheduler.round_rows[-1]['step_ms'] = (time.perf_counter()-step_start)*1000
                scheduler.round_rows[-1]['step_end_ms'] = now
                for out in outputs:
                    c = clocks[out.request_id]
                    n = len(out.outputs[0].token_ids) if out.outputs else 0
                    if n > c['tokens']:
                        assert n == c['tokens']+1, 'Expected synchronous single-token output'
                        c['tokens'] = n
                        c['token_times_ms'].append(now)
                        if c['first_ms'] is None:
                            c['first_ms'] = now
                    if out.finished:
                        c['finish_ms'] = now
            else:
                time.sleep(min(.005, max(0, (reqs[cursor]['arrival_ms']-now)/1000)))
        assert all(clocks[r['id']]['tokens']==r['output'] and clocks[r['id']]['finish_ms'] is not None for r in reqs)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    elapsed = (time.perf_counter()-start)*1000
    rounds = scheduler.round_rows
    complete = sum(clocks[r['id']]['finish_ms'] is not None and clocks[r['id']]['tokens']==r['output'] for r in reqs)
    ttft = [c['first_ms']-c['arrival_ms'] for c in clocks.values() if c['first_ms'] is not None]
    tbt = [b-a for c in clocks.values() for a, b in zip(c['token_times_ms'], c['token_times_ms'][1:])]
    metrics = dict(completion_rate=complete/len(reqs),
                   mean_ttft_ms=st.fmean(ttft) if error is None else None,
                   p95_ttft_ms=quantile(ttft, .95) if error is None else None,
                   mean_tbt_ms=st.fmean(tbt) if tbt else None, p95_tbt_ms=quantile(tbt, .95),
                   throughput_tokens_s=sum(c['tokens'] for c in clocks.values())/(elapsed/1000),
                   preemptions=sum(r['preemptions'] for r in rounds),
                   recomputed_tokens=sum(r['recomputed_tokens'] for r in rounds),
                   interventions=sum(r['intervened'] for r in rounds),
                   candidates=sum(r['candidates'] for r in rounds),
                   verified_rounds=sum(r['verified'] for r in rounds),
                   recovery_fallbacks=sum(r['recovery_fallback'] for r in rounds),
                   peak_kv_fraction=max((1-r['free_after']/r['total_blocks'] for r in rounds), default=0),
                   round_count=len(rounds))
    for field in ('decision_ms', 'native_ms', 'audit_ms', 'schedule_ms', 'step_ms'):
        xs = [r[field] for r in rounds if field in r]
        metrics[field+'_mean'] = st.fmean(xs) if xs else None
        metrics[field+'_p95'] = quantile(xs, .95)
        metrics[field+'_total'] = sum(xs)
    return dict(status='complete' if error is None else 'failed', error=error,
                completed=complete, request_count=len(reqs), clocks=clocks,
                requests=reqs, metrics=metrics, elapsed_ms=elapsed, rounds=rounds, start_wall=start)

