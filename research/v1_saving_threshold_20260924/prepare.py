"""Freeze traces, transformations, aliases and complete-cell task definitions."""
import copy
import csv
from datetime import datetime
import json
import math
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.v1_saving_threshold_20260924.common import *


def main():
    target = OUT / 'inputs_manifest.json'
    if target.exists():
        print('Existing immutable inputs:', target)
        return
    assert datetime.fromisoformat('2026-09-24T11:00:00+08:00').timestamp() == DEADLINE
    cases = {}
    def add(name, rows, source, role, factor=None, level=None):
        assert len(rows) == 64
        assert all(r['prompt'] > 0 and r['output'] > 0 for r in rows)
        assert all(b['arrival_ms'] >= a['arrival_ms'] for a,b in zip(rows, rows[1:]))
        key = object_digest(rows)
        cases[name] = dict(source=source, role=role, factor=factor, level=level,
            trace_digest=key, file='traces/' + name + '.json',
            max_total_tokens=max(r['prompt']+r['output'] for r in rows),
            prompt_total=sum(r['prompt'] for r in rows), output_total=sum(r['output'] for r in rows),
            arrival_span_ms=rows[-1]['arrival_ms'])
        write(INPUT / cases[name]['file'], rows)

    old = ROOT / 'artifacts/v1_capacity_design_20260923/traces'
    originals = {}
    for source in ('code', 'conversation'):
        rows = read(old / (source + '_test0_original.json'))
        originals[source] = rows
        for rate in (1, 2):
            changed = [dict(r, arrival_ms=r['arrival_ms']/rate) for r in rows]
            add(f'{source}_rate{rate}', changed, source, 'map', 'arrival_rate', rate)
        for factor in ('density', 'burst', 'prompt', 'output'):
            for level in ('low', 'high'):
                changed = copy.deepcopy(rows)
                if factor == 'density':
                    rate = .5 if level == 'low' else 2
                    for r in changed: r['arrival_ms'] /= rate
                elif factor == 'burst':
                    span = rows[-1]['arrival_ms']
                    for i,r in enumerate(changed):
                        r['arrival_ms'] = span * (i/63 if level == 'low' else (i//16)/3)
                elif factor == 'prompt' and level == 'low':
                    total = sum(r['prompt'] for r in rows)
                    for i,r in enumerate(changed): r['prompt'] = total//64 + int(i < total%64)
                    assert sum(r['prompt'] for r in changed) == total
                elif factor == 'output':
                    for r in changed: r['output'] = max(1, r['output']//2) if level == 'low' else r['output']*2
                add(f'{source}_{factor}_{level}', changed, source, 'factor', factor, level)

    # Prespecified new time anchors, not chosen by model/algorithm outcomes.
    new_specs = {'code': ('data/azure_code_1week.csv', '2024-05-15 10:17'),
                 'conversation': ('data/azure_conv_1week.csv', '2024-05-17 10:17')}
    source_info = {}
    for source,(filename,minute) in new_specs.items():
        raw = ROOT / filename
        chosen = []
        with raw.open(newline='') as f:
            reader = csv.reader(f)
            assert next(reader) == ['TIMESTAMP','ContextTokens','GeneratedTokens']
            for index,row in enumerate(reader):
                if row[0][:16] == minute:
                    chosen.append(dict(id=f'{source}:raw:{index}', prompt=int(row[1]),
                        output=int(row[2]), timestamp=row[0], source_row=index))
                    if len(chosen) == 64: break
        assert len(chosen) == 64, (source, minute)
        first = datetime.fromisoformat(chosen[0]['timestamp'])
        rows = [dict(id=r['id'], prompt=r['prompt'], output=r['output'],
                     arrival_ms=(datetime.fromisoformat(r['timestamp'])-first).total_seconds()*1000)
                for r in chosen]
        add(source+'_new', rows, source, 'heldout')
        source_info[source] = dict(path=filename, sha256=digest(raw), anchor=minute,
                                  rows=chosen, source_row_range=[chosen[0]['source_row'], chosen[-1]['source_row']])
        print('New input extracted:', source, minute, flush=True)

    # Search historical request identities, excluding our new directory and raw CSV.
    patterns = OUT / 'heldout_identity_patterns.txt'
    patterns.write_text('\n'.join('"'+r['id']+'"' for info in source_info.values() for r in info['rows'])+'\n')
    import subprocess
    cmd = ['rg','-l','-F','-f',str(patterns),'--glob','*.json','--glob','*.jsonl',
           '--glob','!v1_saving_factors_20260924_r1/**', 'artifacts','research']
    check = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert check.returncode in (0,1), check.stderr
    matches = [p for p in check.stdout.splitlines() if 'v1_saving_factors_20260924_r1/' not in p]
    assert not matches, ('New identities overlap historical files', matches)
    previous = read(ROOT/'research/state_conditioned_prefill_selection_20260918/data_manifest.json')
    for source,info in source_info.items():
        assert info['sha256'] == previous['sources'][source]['source_sha256']
        assert info['source_row_range'][0] > 20000
        for split in previous['sources'][source]['splits'].values():
            lo,hi = split['source_row_range']
            assert info['source_row_range'][1] < lo or info['source_row_range'][0] > hi

    jobs, aliases, skipped = [], [], []
    seen = {}
    for phase in ('map', 'factor'):
        caps = CAP_ORDER if phase == 'map' else [768,2048,4096]
        for cap in caps:
            for name,case in cases.items():
                if case['role'] != phase: continue
                levels = THRESHOLDS if phase == 'map' else [1,16,'off']
                for n in levels:
                    key = f'{phase}_{name}_c{cap}_n{n}'
                    j = dict(key=key, phase=phase, case=name, capacity=cap, threshold=n,
                             method='threshold', config=dict(BASE,threshold=n))
                    if case['max_total_tokens'] > min(12288, (cap-1)*16):
                        skipped.append(dict(job=j, reason='single_request_capacity_or_context'))
                        continue
                    identity = (case['trace_digest'],cap,str(n))
                    if identity in seen:
                        aliases.append(dict(job=j, canonical=seen[identity]))
                    else:
                        seen[identity] = key
                        jobs.append(j)
    manifest = dict(schema=1, cases=cases, heldout_sources=source_info,
        historical_identity_overlap=[], historical_audit='uncompressed JSON/JSONL identities and prior split row ranges',
        jobs=jobs, aliases=aliases, skipped=skipped,
        source_hashes=sources(), deadline=DEADLINE,
        transformations='controlled_synthetic_separate_from_natural', counts=dict(
            measured_planned=len(jobs), aliases=len(aliases), infeasible=len(skipped)))
    write(target,manifest)
    print(json.dumps(manifest['counts']),flush=True)


if __name__ == '__main__': main()
