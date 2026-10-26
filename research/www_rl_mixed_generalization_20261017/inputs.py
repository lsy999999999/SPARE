"""Paired prompt-composition shifts on held-out Conversation windows."""
from __future__ import annotations

from collections import Counter
import hashlib
import random

from research.www_online_rl_20261017 import run as old


OUT = old.ROOT / 'artifacts/www_rl_mixed_generalization_20261017_r1'
SHORT_MAX = 533
MID_MAX = 2120
MIXES = {'A_short': (.40, .50, .10),
         'B_balanced': (.25, .50, .25),
         'C_long': (.10, .50, .40)}


def bucket(prompt):
    return 0 if prompt <= SHORT_MAX else 1 if prompt <= MID_MAX else 2


def samples_for(workload, split='test'):
    if workload == 'original':
        return old.samples(split)
    assert workload in MIXES and split == 'test'
    return [dict(sample, input=str((OUT / 'inputs' / workload /
                                    f'{sample["name"]}.json').relative_to(old.ROOT)))
            for sample in old.samples(split) if sample['source'] == 'conversation']


def prepare():
    source = [sample for sample in old.samples('test')
              if sample['source'] == 'conversation']
    assert len(source) == 50
    pools = [[] for _ in range(3)]
    for sample in source:
        for row in old.base.read(old.ROOT / sample['input']):
            pools[bucket(row['prompt'])].append(row['prompt'])
    assert all(len(pool) >= 100 for pool in pools)
    audit = []
    for sample in source:
        original = old.base.read(old.ROOT / sample['input'])
        assert len(original) == 64
        for workload, mix in MIXES.items():
            seed = int.from_bytes(hashlib.sha256(
                f'{sample["name"]}:{workload}:20261017'.encode()).digest()[:8], 'big')
            rng = random.Random(seed)
            counts = [round(len(original) * mix[0]), round(len(original) * mix[1])]
            counts.append(len(original) - sum(counts))
            assignments = [b for b, count in enumerate(counts) for _ in range(count)]
            rng.shuffle(assignments)
            generated = []
            for row, b in zip(original, assignments):
                choices = [p for p in pools[b] if p + row['output'] <= 8000]
                assert choices, (sample['name'], workload, b, row['output'])
                generated.append(dict(row, prompt=rng.choice(choices)))
            assert Counter(bucket(r['prompt']) for r in generated) == Counter(assignments)
            assert [r['output'] for r in generated] == [r['output'] for r in original]
            path = old.ROOT / samples_for(workload)[source.index(sample)]['input']
            if path.exists():
                assert old.base.read(path) == generated
            else:
                old.base.write(path, generated)
            audit.append(dict(source=sample['name'], workload=workload,
                              input=str(path.relative_to(old.ROOT)),
                              sha256=old.base.digest(path), counts=counts,
                              mean_prompt=sum(r['prompt'] for r in generated) / len(generated),
                              output_sha256=hashlib.sha256(str(
                                  [r['output'] for r in generated]).encode()).hexdigest()))
    old.base.write(OUT / 'input_audit.json', dict(
        short_max=SHORT_MAX, mid_max=MID_MAX, mixes=MIXES,
        windows=50, rows=audit))
    return audit


if __name__ == '__main__':
    print(f'prepared {len(prepare())} paired workload windows')
