"""Freeze six source windows and the complete, prespecified GPU supplement."""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.www_legacy_baselines_20261006.common import *


def main():
    if (OUT / 'design.json').exists():
        raise RuntimeError('design already frozen')
    previous = ROOT / 'writing/www/experiments/gpu_baselines_4090_20260930_r3'
    # Match the measured baseline dependencies, not possibly edited live files.
    for rel in source_paths():
        source = previous / 'source' / rel
        if source.exists():
            target = OUT / 'source' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    old_design = read(previous / 'design.json')
    inputs = ROOT / 'artifacts/www_evaluation_completion_20261003_r1/inputs'
    cells = []
    for source in ('code', 'conversation'):
        old_cell = next(c for c in old_design['cells'] if f'_{source}_' in c['key'])
        history_target = OUT / 'history' / f'{source}.json'
        history_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(previous / old_cell['history_file'], history_target)
        for w in range(3):
            window = f'{source}_new{w}'
            for shape in ('steady', 'burst_stop'):
                original = inputs / f'{window}_{shape}.json'
                requests = read(original)
                assert len(requests) == 128
                target = OUT / 'inputs' / original.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
                for capacity in (768, 2048, 4096):
                    cells.append(dict(key=f'{window}_{shape}_c{capacity}',
                        window=window, shape=shape, source_input=str(original.relative_to(ROOT)),
                        input=str(target.relative_to(OUT)), input_sha256=digest(target),
                        history_file=str(history_target.relative_to(OUT)),
                        history_sha256=read(history_target)['source_sha256'],
                        history_file_sha256=digest(history_target),
                        capacity=capacity, budget=1024, chunk=256, requests=128))
    # Include every new Python file and the protocol, then freeze the same bytes.
    hashes = source_hashes()
    for rel, source in source_paths().items():
        target = OUT / 'source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
    design = dict(schema=1, frozen=True, frozen_at=time.time(), cells=cells,
        methods=list(ARMS), options=OPTIONS, repeats=4, aa_repeats=2,
        expected_formal_runs=len(cells)*len(ARMS)*4, expected_aa_runs=len(cells)*2,
        max_model_len=8176, max_num_seqs=64, max_num_batched_tokens=2048,
        block_size=16, dtype='float16', gpu_candidates=[0,1,2,3,6,7],
        episode_timeout_s=900, deadline_hours=72, max_group_retries=2,
        ordering='two fixed shuffled/reverse pairs per cell', source_hashes=hashes,
        objective='mean TTFT primary; all service and recovery metrics retained',
        inference_unit='six source windows; configurations are paired correlated observations',
        input_scope='matches 20261003 CPU core inputs; no new claims of unseen GPU validation',
        stopping='fixed matrix; timeout and interference only; no outcome-based stopping')
    write(OUT / 'design.json', design)
    print(json.dumps(dict(cells=len(cells), methods=len(ARMS),
                         formal_runs=design['expected_formal_runs'], output=str(OUT))))


if __name__ == '__main__':
    main()
