"""Paired Conversation workload check at a KV outside RL training."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

from research.www_online_rl_20261017 import extreme_mean, run
from research.www_kv_extreme_stress_20261016 import run as sim


def spare_one():
    import torch
    torch.set_num_threads(1)
    sim.initialize()
    source, kv, rate = 'conversation', extreme_mean.KV, 3.5
    cohort = [sample for sample in run.samples('test') if sample['source'] == source]
    assert len(cohort) == 50
    path = extreme_mean.OUT / 'rates' / f'{source}_c{kv}_arrival_r{rate:g}.json'
    if path.exists():
        record = run.base.read(path)
    else:
        rows = [sim.simulate_sample(sample, kv, 'arrival', rate) for sample in cohort]
        record = dict(rows=rows, summary=run.prior.old.summary(rows, rate))
        run.base.write(path, record)
    assert {row['name'] for row in record['rows']} == {sample['name'] for sample in cohort}
    assert record['summary'] == run.prior.old.summary(record['rows'], rate)
    return record['summary']['mean_ttft_ms']


def main():
    with ProcessPoolExecutor(max_workers=2) as pool:
        spare = pool.submit(spare_one)
        rl = pool.submit(extreme_mean.one, ('conversation', 3.5))
        print(dict(spare_mean_ttft_ms=spare.result(),
                   rl_mean_ttft_ms=rl.result()[2]), flush=True)


if __name__ == '__main__':
    main()
