"""Paired same-host scheduler CPU timing for SPARE and frozen RL-Mixed."""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import time

from research.www_rl_mixed_generalization_20261017 import run, sweep


OUT = run.OUT / 'cost_same_host_20261018.json'
KV = 2048
RATE = 3.
SEED = 1017


def main():
    if OUT.exists():
        result = run.old.base.read(OUT)
        assert result['stage'] == 'complete'
        print(json.dumps(result['summary'], indent=2))
        return
    run.old.base.initialize()
    weights = run.selected_weights(SEED)
    rows = []
    for source in ('code', 'conversation'):
        cohort = sweep.cohort_for('original', source)
        signature = sweep.source_hash(cohort)
        for index, sample in enumerate(cohort):
            outcomes = {}
            order = ('spare', 'rl_mixed') if index % 2 == 0 else ('rl_mixed', 'spare')
            for arm in order:
                started = time.monotonic()
                if arm == 'spare':
                    requests = run.old.base.generate(sample, RATE)
                    seed = int.from_bytes(hashlib.sha256(
                        sample['name'].encode()).digest()[:4], 'big')
                    result = run.old.base.simulate(
                        requests, 'arrival', KV, 1024, source, seed)
                else:
                    actor = run.learned.DQNPolicy(weights=weights)
                    result = run.simulate(sample, KV, RATE, actor)
                outcomes[arm] = dict(
                    schedule_mean_ms=result['metrics']['cpu_schedule_including_audit_ms_mean'],
                    wall_s=time.monotonic() - started,
                    mean_ttft_ms=run.old.base.probe(result)['mean_ttft_ms'])
            rows.append(dict(source=source, sample=sample['name'],
                             input_sha256=signature, rate_qps=RATE,
                             execution_order=list(order), **outcomes))
            if (index + 1) % 10 == 0:
                print(f'{source} {index + 1}/50', flush=True)
    summary = {}
    for source in ('code', 'conversation', 'all'):
        chosen = [r for r in rows if source == 'all' or r['source'] == source]
        spare = statistics.fmean(r['spare']['schedule_mean_ms'] for r in chosen)
        rl = statistics.fmean(r['rl_mixed']['schedule_mean_ms'] for r in chosen)
        summary[source] = dict(windows=len(chosen), spare_schedule_ms=spare,
                               rl_schedule_ms=rl, rl_over_spare=rl / spare,
                               paired_median_ratio=statistics.median(
                                   r['rl_mixed']['schedule_mean_ms'] /
                                   r['spare']['schedule_mean_ms'] for r in chosen))
    payload = dict(stage='complete', kv_blocks=KV, rate_qps=RATE,
                   rl_seed=SEED, host=os.uname().nodename,
                   metric='cpu_schedule_including_audit_ms_mean',
                   execution='paired 50-window source cohorts, alternating order',
                   input_sha256={s: sweep.source_hash(sweep.cohort_for('original', s))
                                 for s in ('code', 'conversation')},
                   rows=rows, summary=summary)
    run.old.base.write(OUT, payload)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
