"""Freeze new contiguous length traces and enumerate the proposed GPU matrix.

This command does not execute GPU measurements or claim simulator accuracy.
Run before developing/selecting new baseline rules. It never filters lengths.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def save_new(path, payload):
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'Frozen input differs: {path}; use a new output directory')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            stream.write(text)


def extract(source, spec, config, old_manifest):
    path = ROOT / spec['path']
    digest = sha256(path)
    expected = spec.get('sha256', old_manifest['sources'][source]['source_sha256'])
    if digest != expected:
        raise ValueError(f'Source hash mismatch: {source}')
    minutes = {minute: name for name, minute in spec['windows'].items()}
    counts = {name: 0 for name in minutes.values()}
    blocks = {name: [] for name in minutes.values()}
    old_minutes = {bounds[0][:16] for bounds in old_manifest['sources'][source]['declared_windows'].values()}
    if set(minutes) & old_minutes:
        raise ValueError(f'Overlap with previous declared windows: {source}')
    with path.open(newline='') as stream:
        reader = csv.reader(stream)
        if next(reader) != ['TIMESTAMP', 'ContextTokens', 'GeneratedTokens']:
            raise ValueError(f'Unexpected CSV header: {source}')
        for row_index, row in enumerate(reader):
            name = minutes.get(row[0][:16])
            if name is None:
                continue
            counts[name] += 1
            role = 'test' if name.startswith('test') else name
            if len(blocks[name]) == config['requests_per_role'][role]:
                continue
            prompt, output = int(row[1]), int(row[2])
            if prompt <= 0 or output <= 0:
                raise ValueError(f'Invalid predeclared request, not filtered: {source}:{row_index}')
            blocks[name].append(dict(id=f'{source}:raw:{row_index}', source_row=row_index,
                timestamp=row[0], prompt=prompt, output=output))
    result = {}
    for name, rows in blocks.items():
        role = 'test' if name.startswith('test') else name
        if len(rows) != config['requests_per_role'][role]:
            raise ValueError(f'Insufficient requests in {source}/{name}: {len(rows)}')
        if any(b['source_row'] != a['source_row'] + 1 for a, b in zip(rows, rows[1:])):
            raise ValueError(f'Noncontiguous input: {source}/{name}')
        first = datetime.fromisoformat(rows[0]['timestamp'])
        for row in rows:
            row['arrival_ms'] = (datetime.fromisoformat(row['timestamp']) - first).total_seconds() * 1000
        if any(b['arrival_ms'] < a['arrival_ms'] for a, b in zip(rows, rows[1:])):
            raise ValueError(f'Nonmonotone arrival: {source}/{name}')
        span = rows[-1]['arrival_ms']
        if span <= 0:
            raise ValueError('Cannot rescale a zero-span arrival window')
        key = f'{source}_{name}'
        maximum = max(row['prompt'] + row['output'] for row in rows)
        result[key] = dict(source=source, role=role, utc_minute=spec['windows'][name],
            source_sha256=digest, available_rows_in_minute=counts[name], rows=rows,
            summary=dict(requests=len(rows), arrival_span_ms=span,
                prompt_mean=statistics.fmean(row['prompt'] for row in rows),
                output_mean=statistics.fmean(row['output'] for row in rows),
                max_total_tokens=maximum,
                max_single_request_blocks=(maximum + config['block_size'] - 1) // config['block_size'],
                within_model_context=maximum <= config['max_model_len']))
    print(f'{source}: froze {len(result)} windows, source hash verified', flush=True)
    return result


def matrix(config, blocks):
    jobs = []
    def add(stage, block, capacity, arrival, method, repeat, gpu, candidate=None):
        jobs.append(dict(stage=stage, block=block, capacity=capacity, arrival=arrival,
            method=method, repeat=repeat, gpu=gpu, candidate=candidate))
    for block, payload in sorted(blocks.items()):
        if payload['role'] == 'dev':
            for cap in config['tuning_capacity_blocks']:
                for arrival in config['arrival_profiles']:
                    for budget in config['fixed_small_candidates']:
                        add('tune', block, cap, arrival, 'fixed_small', 0,
                            config['tuning_gpu_assignment'][block], budget)
                    for threshold in config['occupancy_threshold_candidates']:
                        add('tune', block, cap, arrival, 'occupancy', 0,
                            config['tuning_gpu_assignment'][block], threshold)
        if payload['role'] != 'test':
            continue
        gpu = config['test_gpu_assignment'][block]
        for cap in config['capacity_blocks']:
            for arrival in config['arrival_profiles']:
                for repeat in range(config['repeats']):
                    for method in config['core_methods']:
                        add('core', block, cap, arrival, method, repeat, gpu)
                    if cap in config['ablation_capacity_blocks']:
                        for method in config['ablation_extra_methods']:
                            add('ablation', block, cap, arrival, method, repeat, gpu)
                    if cap in config['shadow_capacity_blocks']:
                        add('shadow', block, cap, arrival, 'a_shadow', repeat, gpu)
    for index, job in enumerate(jobs):
        summary = blocks[job['block']]['summary']
        job['id'] = f'planned_{index:04d}'
        job['input_feasible'] = (summary['within_model_context'] and
            summary['max_single_request_blocks'] <= job['capacity'] - 1)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/v1_capacity_design_20260923')
    args = parser.parse_args()
    design_hash = sha256(HERE/'experiment.json')
    builder_hash = sha256(Path(__file__))
    config = json.loads((HERE/'experiment.json').read_text())
    old = json.loads((ROOT/'research/state_conditioned_prefill_selection_20260918/data_manifest.json').read_text())
    blocks = {}
    for source, spec in config['sources'].items():
        blocks.update(extract(source, spec, config, old))
    ids = [r['id'] for b in blocks.values() for r in b['rows']]
    if len(ids) != len(set(ids)):
        raise ValueError('Overlap between the new blocks')
    jobs = matrix(config, blocks)
    stages = {stage: sum(j['stage'] == stage for j in jobs)
              for stage in ('tune', 'core', 'ablation', 'shadow')}
    validation_counts = {'gpu_smoke': 27, 'aa': 36}
    if design_hash != sha256(HERE/'experiment.json') or builder_hash != sha256(Path(__file__)):
        raise ValueError('Design or builder changed during input preparation; rerun without editing')
    manifest = dict(status='inputs frozen; GPU runner and validation pending',
        design_sha256=design_hash, builder_sha256=builder_hash,
        previous_manifest_sha256=sha256(ROOT/'research/state_conditioned_prefill_selection_20260918/data_manifest.json'),
        source_window_overlap_with_previous_declared_splits=False,
        global_historical_overlap_audit='pending; only the linked prior manifest was checked',
        config=config, blocks=blocks, planned_counts=stages,
        validation_counts=validation_counts, total_planned=sum(stages.values())+sum(validation_counts.values()),
        infeasible_jobs=[j['id'] for j in jobs if not j['input_feasible']],
        jobs=jobs)
    save_new(args.output/'input_manifest.json', manifest)
    for name, block in blocks.items():
        for profile in config['arrival_profiles']:
            requests = [dict(id=r['id'], prompt=r['prompt'], output=r['output'],
                arrival_ms=r['arrival_ms'] if profile=='original'
                else r['arrival_ms']/block['summary']['arrival_span_ms']*20) for r in block['rows']]
            save_new(args.output/'traces'/f'{name}_{profile}.json', requests)
    print(json.dumps(dict(planned_counts=stages, validation_counts=validation_counts,
        total_planned=manifest['total_planned'], infeasible_jobs=len(manifest['infeasible_jobs']),
        measured_runs=0), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
