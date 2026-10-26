"""Resumable paired capacity scan for standalone RL and SPARE."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
from pathlib import Path

from research.www_kv_extreme_stress_20261016 import run as sim
from research.www_online_rl_20261017 import analyze_mean
from research.www_rl_standalone_20261019 import run


OUT = run.OUT
COARSE = (.25, .5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 8., 10., 12.)


def passed(summary, rate):
    return (summary['failed_samples'] == 0
            and summary['mean_ttft_ms'] <= 2000
            and summary['p99_tbt_ms'] <= 200
            and summary['backlog_drift'] <= max(2, rate))


def cohort_for(source):
    cohort = [sample for sample in run.old.samples('test')
              if sample['source'] == source]
    assert len(cohort) == 50
    return cohort


@lru_cache(maxsize=1)
def prior_records():
    return analyze_mean.records()


def get_rate(source, kv, arm, seed, rate, cohort, signature, weights=None):
    original = (prior_records().get(('spare', source, kv), {}).get(rate)
                if arm == 'spare' else None)
    if original is not None:
        record = run.old.base.read(original)
        assert {row['name'] for row in record['rows']} == {
            sample['name'] for sample in cohort}
        return record['summary']
    path = OUT / 'rates' / f'{source}_c{kv}_{arm}_seed{seed}_r{rate:g}.json'
    checkpoint_sha = (None if arm == 'spare' else run.old.base.digest(
        OUT / run.old.base.read(OUT / 'checkpoints' /
                                f'seed{seed}_status.json')['selected']['path']))
    if path.exists():
        record = run.old.base.read(path)
        assert record['input_sha256'] == signature
        assert record['checkpoint_sha256'] == checkpoint_sha
    else:
        if arm == 'spare':
            if not hasattr(run.old.base, 'SIM'):
                sim.initialize()
            rows = [sim.simulate_sample(sample, kv, 'arrival', rate)
                    for sample in cohort]
        else:
            assert weights is not None
            rows = [run.eval_one(sample, kv, rate, weights) for sample in cohort]
        rows.sort(key=lambda row: row['name'])
        summary = run.old.prior.old.summary(rows, rate)
        record = dict(rows=rows, summary=summary,
                      mean_rule_passed=passed(summary, rate),
                      input_sha256=signature,
                      checkpoint_sha256=checkpoint_sha, windows=50)
        run.old.base.write(path, record)
    assert len(record['rows']) == 50
    assert record['summary'] == run.old.prior.old.summary(record['rows'], rate)
    if 'mean_rule_passed' in record:
        assert record['mean_rule_passed'] == passed(record['summary'], rate)
    return record['summary']


def scan(source, kv, arm, seed):
    assert arm in ('spare', 'rl_standalone')
    cohort = cohort_for(source)
    signature = hashlib.sha256(''.join(
        run.old.base.digest(run.ROOT / sample['input']) for sample in cohort
    ).encode()).hexdigest()
    weights = run.selected_weights(seed) if arm == 'rl_standalone' else None
    passed_rate, failed_rate, checks = 0., None, []
    for rate in COARSE:
        summary = get_rate(source, kv, arm, seed, rate, cohort, signature, weights)
        checks.append(dict(rate_qps=rate, passed=passed(summary, rate),
                           mean_ttft_ms=summary['mean_ttft_ms'],
                           p99_tbt_ms=summary['p99_tbt_ms'],
                           backlog_drift=summary['backlog_drift'],
                           failed_samples=summary['failed_samples']))
        print(f'{arm} seed={seed} {source} KV={kv} QPS={rate:g} '
              f'passed={checks[-1]["passed"]}', flush=True)
        if checks[-1]['passed']:
            passed_rate = rate
        else:
            failed_rate = rate
            break
    if failed_rate is None:
        raise RuntimeError('no failed rate in coarse grid')
    while failed_rate - passed_rate > .0200001:
        rate = round((failed_rate + passed_rate) * 50) / 100
        if not passed_rate < rate < failed_rate:
            break
        summary = get_rate(source, kv, arm, seed, rate, cohort, signature, weights)
        ok = passed(summary, rate)
        checks.append(dict(rate_qps=rate, passed=ok,
                           mean_ttft_ms=summary['mean_ttft_ms'],
                           p99_tbt_ms=summary['p99_tbt_ms'],
                           backlog_drift=summary['backlog_drift'],
                           failed_samples=summary['failed_samples']))
        print(f'{arm} seed={seed} {source} KV={kv} QPS={rate:g} '
              f'passed={ok}', flush=True)
        if ok:
            passed_rate = rate
        else:
            failed_rate = rate
    result = dict(source=source, kv_blocks=kv, arm=arm, seed=seed,
                  capacity_qps=passed_rate, next_failed_qps=failed_rate,
                  resolution_qps=round(failed_rate - passed_rate, 4),
                  windows=50, input_sha256=signature,
                  rule='mean TTFT <= 2s; P99 TBT <= 200ms; 40-60s '
                       'backlog drift <= max(2, QPS); all windows valid',
                  checks=checks)
    target = OUT / 'groups' / f'{source}_c{kv}_{arm}_seed{seed}.json'
    run.old.base.write(target, result)
    print(f'capacity={passed_rate:.3f} QPS; result={target}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=('code', 'conversation'), required=True)
    parser.add_argument('--kv', type=int, required=True)
    parser.add_argument('--arm', choices=('spare', 'rl_standalone'), required=True)
    parser.add_argument('--seed', type=int, default=1017)
    parser.add_argument('--out-dir', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out_dir.resolve()
    run.OUT = OUT
    scan(args.source, args.kv, args.arm, args.seed)
