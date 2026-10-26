"""Fill the two KV settings absent from the paired round-level replay."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

from research.www_kv_utilization_20261015 import run as prior

OUT = prior.ROOT / 'artifacts/www_kv_degradation_20261016_r1'
CAPACITIES = (1536, 3072)


def run_one(task):
    with patch.object(prior, 'OUT', OUT):
        return prior.run_one(task)


def main():
    base_design = prior.base.read(prior.OUT / 'design.json')
    samples = base_design['samples']
    assert len(samples) == 4
    assert prior.base.digest(prior.base.PREVIOUS / 'timing_profile.json') == base_design['timing_sha256']
    assert prior.base.digest(prior.base.PREVIOUS / 'history.json') == base_design['history_sha256']
    for sample in samples:
        assert prior.base.digest(prior.ROOT / sample['input']) == prior.base.read(prior.base.OUT / 'design.json')['source_inputs'][sample['input']]
    design = dict(base_design, capacities=list(CAPACITIES),
                  base_design_sha256=prior.base.digest(prior.OUT / 'design.json'),
                  source_sha256={**base_design['source_sha256'],
                    'research/www_kv_utilization_20261015/extend.py': prior.base.digest(Path(__file__))})
    if (OUT / 'design.json').exists():
        assert prior.base.read(OUT / 'design.json') == json.loads(json.dumps(design))
    prior.base.write(OUT / 'design.json', design)
    tasks = [(sample, capacity, arm) for sample in samples
             for capacity in CAPACITIES for arm in prior.ARMS]
    with ProcessPoolExecutor(max_workers=8, initializer=prior.initialize) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 8 == 0:
                prior.base.write(OUT / 'status.json', dict(completed=count, total=len(tasks),
                                                         complete=count == len(tasks)))
                print(f'{count}/{len(tasks)}', flush=True)


if __name__ == '__main__':
    main()
