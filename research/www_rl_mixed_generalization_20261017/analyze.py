"""Summarize matched SPARE and frozen mixed-RL capacity scans."""
from __future__ import annotations

import csv
from collections import defaultdict

from research.www_rl_mixed_generalization_20261017 import run


OUT = run.OUT
ANALYSIS = OUT / 'analysis'
SEEDS = (1017, 2027, 3037)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_groups():
    groups = {}
    for path in (OUT / 'groups').glob('*.json'):
        row = run.old.base.read(path)
        assert row['stage'] == 'complete'
        key = (row['workload'], row['source'], row['kv_blocks'],
               row['arm'], row['seed'])
        assert key not in groups
        groups[key] = row
    return groups


def comparisons(groups):
    rows = []
    for (workload, source, kv, arm, seed), rl in sorted(groups.items()):
        if arm != 'rl_mixed':
            continue
        spare = groups[workload, source, kv, 'spare', 0]
        assert rl['input_sha256'] == spare['input_sha256']
        a, b = spare['capacity_qps'], rl['capacity_qps']
        rows.append(dict(workload=workload, source=source, kv_blocks=kv,
                         seed=seed, spare_capacity_qps=a,
                         rl_capacity_qps=b, rl_gain_pct=100 * (b / a - 1),
                         spare_next_failed_qps=spare['next_failed_qps'],
                         rl_next_failed_qps=rl['next_failed_qps'],
                         input_sha256=rl['input_sha256']))
    return rows


def mean_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['workload'], row['source'], row['kv_blocks']].append(row)
    result = []
    for (workload, source, kv), values in sorted(groups.items()):
        assert {v['seed'] for v in values} == set(SEEDS)
        values.sort(key=lambda r: r['seed'])
        result.append(dict(workload=workload, source=source, kv_blocks=kv,
                           spare_capacity_qps=values[0]['spare_capacity_qps'],
                           rl_mean_capacity_qps=sum(v['rl_capacity_qps'] for v in values) / 3,
                           rl_mean_gain_pct=sum(v['rl_gain_pct'] for v in values) / 3,
                           rl_min_gain_pct=min(v['rl_gain_pct'] for v in values),
                           rl_max_gain_pct=max(v['rl_gain_pct'] for v in values),
                           seed1017_qps=values[0]['rl_capacity_qps'],
                           seed2027_qps=values[1]['rl_capacity_qps'],
                           seed3037_qps=values[2]['rl_capacity_qps']))
    return result


def report(rows, means):
    lines = [
        '# Capacity-aware RL-Mixed versus SPARE', '',
        'All numbers below are CPU-simulator point estimates using the same '
        '50 test windows per condition. Capacity requires mean TTFT <= 2 s, '
        'P99 TBT <= 200 ms, backlog drift <= max(2, offered QPS), and no '
        'failed window. Brackets are refined to <= 0.02 QPS. RL was trained '
        'on original Code/Conversation windows at KV 1024/2048/4096/8192 '
        'and QPS 0.5/2/4/6, then frozen for every test below. The prompt '
        'composition tests reuse the same checkpoint.', '',
        '| Workload | KV | Source | SPARE QPS | RL mean QPS | RL mean gain | '
        'RL seed range |',
        '|---|---:|---|---:|---:|---:|---:|',
    ]
    for row in means:
        lines.append('|{workload}|{kv_blocks}|{source}|{spare_capacity_qps:.3f}|'
                     '{rl_mean_capacity_qps:.3f}|{rl_mean_gain_pct:+.2f}%|'
                     '{rl_min_gain_pct:+.2f}% to {rl_max_gain_pct:+.2f}%|'.format(**row))
    lines.extend(['', 'For each row, gain = 100 * (RL / SPARE - 1). '
                  'The seed range is training-seed variation, not a confidence interval.', ''])
    for source in ('code', 'conversation'):
        original = [r for r in means if r['workload'] == 'original'
                    and r['source'] == source]
        if original:
            trained = [r['rl_mean_gain_pct'] for r in original
                       if r['kv_blocks'] in (2048, 4096)]
            heldout = [r['rl_mean_gain_pct'] for r in original
                       if r['kv_blocks'] in (3072, 6144)]
            extreme = next(r['rl_mean_gain_pct'] for r in original
                           if r['kv_blocks'] == 12032)
            lines.append(f'{source}: mean gain at trained KV = '
                         f'{sum(trained)/len(trained):+.2f}%; '
                         f'held-out in-range KV = {sum(heldout)/len(heldout):+.2f}%; '
                         f'out-of-range KV 12032 = {extreme:+.2f}%.')
    lines.extend(['', 'Prompt-composition robustness at fixed KV 2048:'])
    for seed in SEEDS:
        values = [r for r in rows if r['seed'] == seed
                  and r['workload'] != 'original']
        if values:
            worst = min(values, key=lambda row: row['rl_gain_pct'])
            lines.append(f'- Seed {seed}: worst relative capacity '
                         f'{worst["rl_gain_pct"]:+.2f}% in {worst["workload"]}.')
    lines.extend(['', 'Training and selection:'])
    for seed in SEEDS:
        status = run.old.base.read(OUT / 'checkpoints' / f'seed{seed}_status.json')
        selected = status['selected']
        timing = ('post-resume wall time' if status['resumed_from_episode']
                  else 'training wall time')
        lines.append(f'- Seed {seed}: {status["episodes"]} episodes, '
                     f'{status["steps"]} final step counter '
                     f'(initialized at 1,000,000 on resume), '
                     f'{status["gradient_updates"]} post-resume gradient updates, '
                     f'{status["wall_s"]:.1f} s {timing}; '
                     f'selected episode {selected["episode"]} '
                     f'using held-out development windows '
                     f'(score {selected["dev_score"]:.4f}).')
        if status['resumed_from_episode']:
            lines.append('  Replay buffer and optimizer were reset after episode 512 '
                         'because the first development-score adapter only '
                         'accepted 50-window groups. The episode-512 model '
                         'weights were preserved; first-stage wall time is '
                         'additional to the number above.')
    lines.extend(['', '[Per-seed results](capacity_by_seed.csv) · '
                  '[Three-seed summary](capacity_three_seed.csv)'])
    (ANALYSIS / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    groups = load_groups()
    rows = comparisons(groups)
    means = mean_rows(rows)
    assert len(rows) == 39 and len(means) == 13, (len(rows), len(means))
    write_csv(ANALYSIS / 'capacity_by_seed.csv', rows)
    write_csv(ANALYSIS / 'capacity_three_seed.csv', means)
    report(rows, means)
    print(f'analyzed {len(rows)} RL/SPARE pairs in {len(means)} conditions')


if __name__ == '__main__':
    main()
