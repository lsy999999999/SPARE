"""Paired held-out validation of online decision cost and scheduler outcomes."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import statistics
import time

from research.www_online_rl_20261017 import run


def main():
    import torch
    torch.set_num_threads(1)
    checkpoint = run.base.read(run.OUT / 'checkpoint.json')
    run.base.initialize()
    validation = [s for s in run.samples('validation') if s['window'] >= 16]
    assert len(validation) == 8
    rows = []
    for kv in (1024, 4096):
        for rate in (.5, 4.):
            for sample in validation:
                seed = int.from_bytes(hashlib.sha256(sample['name'].encode()).digest()[:4], 'big')
                requests = run.base.generate(sample, rate)
                started = time.monotonic()
                spare = run.base.simulate(requests, 'arrival', kv, 1024,
                                          sample['source'], seed)
                spare_wall = time.monotonic()-started
                started = time.monotonic()
                rl = run.simulate(sample, kv, rate, run.QPolicy(table=checkpoint['q_table']))
                rl_wall = time.monotonic()-started
                a, b = run.base.probe(spare), run.base.probe(rl)
                rows.append(dict(source=sample['source'], sample=sample['name'],
                                 kv_blocks=kv, rate=rate,
                                 spare_ttft_ms=a['mean_ttft_ms'], rl_ttft_ms=b['mean_ttft_ms'],
                                 spare_p99_tbt_ms=a['p99_tbt_ms'], rl_p99_tbt_ms=b['p99_tbt_ms'],
                                 spare_backlog_drift=a['backlog_drift'],
                                 rl_backlog_drift=b['backlog_drift'],
                                 spare_schedule_ms=spare['metrics']['cpu_schedule_including_audit_ms_mean'],
                                 rl_schedule_ms=rl['metrics']['cpu_schedule_including_audit_ms_mean'],
                                 spare_wall_s=spare_wall, rl_wall_s=rl_wall))
    groups = defaultdict(list)
    for row in rows:
        groups[row['kv_blocks'], row['rate']].append(row)
    summary = []
    for (kv, rate), cohort in sorted(groups.items()):
        summary.append(dict(kv_blocks=kv, rate=rate, windows=len(cohort),
            mean_spare_ttft_ms=statistics.fmean(r['spare_ttft_ms'] for r in cohort),
            mean_rl_ttft_ms=statistics.fmean(r['rl_ttft_ms'] for r in cohort),
            mean_spare_p99_tbt_ms=statistics.fmean(r['spare_p99_tbt_ms'] for r in cohort),
            mean_rl_p99_tbt_ms=statistics.fmean(r['rl_p99_tbt_ms'] for r in cohort),
            mean_spare_backlog_drift=statistics.fmean(r['spare_backlog_drift'] for r in cohort),
            mean_rl_backlog_drift=statistics.fmean(r['rl_backlog_drift'] for r in cohort),
            mean_spare_schedule_ms=statistics.fmean(r['spare_schedule_ms'] for r in cohort),
            mean_rl_schedule_ms=statistics.fmean(r['rl_schedule_ms'] for r in cohort),
            total_spare_wall_s=sum(r['spare_wall_s'] for r in cohort),
            total_rl_wall_s=sum(r['rl_wall_s'] for r in cohort)))
    result = dict(split='held-out validation windows 16-19 per source',
                  checkpoint_sha256=run.base.digest(run.OUT / 'checkpoint.json'),
                  rows=rows, summary=summary)
    run.base.write(run.OUT / 'cost_validation.json', result)
    for row in summary:
        print(row)


if __name__ == '__main__':
    main()
