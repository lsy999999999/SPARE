"""Recompute published effects from archived request timestamps; never run GPUs."""
import csv
import gzip
import hashlib
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/www_burst_background_gpu_20261009_r2'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-7), (a, b)


def main():
    design = read(OUT / 'design.json')
    manifest = read(OUT / 'manifest.json')
    assert sha(OUT / 'design.json') == manifest['design_sha256']
    rows = list(csv.DictReader((OUT / 'analysis/runs.csv').open()))
    assert len(rows) == design['expected_formal_runs'] == 288
    cells = {c['key']: c for c in design['cells']}
    assert len(cells) == 18
    arms = ('arrival', 'srp', 'fixed256', 'fixed512')
    by_pair = defaultdict(list)
    request_count = round_count = 0
    seen = set()
    for cell in cells:
        group = read(OUT / 'groups' / (cell + '.json'))
        assert group['state'] == 'complete'
        assert read(OUT / group['folder'] / 'aa.json')['passed']
    for row in rows:
        key = (row['cell'], row['arm'], int(row['repeat']))
        assert key not in seen
        seen.add(key)
        group = read(OUT / 'groups' / (row['cell'] + '.json'))
        path = OUT / row['result_path']
        assert path == OUT / group['folder'] / 'runs' / f"{row['arm']}_r{row['repeat']}.json"
        assert sha(path) == row['sha256']
        result = read(path)
        assert result['status'] == 'complete'
        assert (result['cell'], result['arm'], result['repeat']) == key
        requests = result['requests']
        assert len(requests) == result['metrics']['completed'] == 128
        assert len({q['request_id'] for q in requests}) == 128
        for q in requests:
            close(q['first_ms'] - q['arrival_ms'], q['ttft_ms'])
        ttft = st.fmean(q['first_ms'] - q['arrival_ms'] for q in requests)
        close(ttft, result['metrics']['ttft_ms_mean'])
        close(ttft, float(row['ttft_ms_mean']))
        round_path = path.parent.parent / result['rounds_path']
        assert sha(round_path) == result['rounds_sha256']
        with gzip.open(round_path, 'rt') as handle:
            rounds = json.load(handle)
        assert len(rounds) == result['metrics']['rounds']
        request_count += len(requests)
        round_count += len(rounds)
        by_pair[row['cell'], row['arm']].append(dict(
            ttft_ms=ttft, interventions=result['metrics']['interventions'],
            preemptions=result['metrics']['preemptions'],
            schedule_ms=result['metrics']['schedule_ms_mean']))
    assert seen == {(c, a, r) for c in cells for a in arms for r in range(4)}
    means = {k: {m: st.fmean(r[m] for r in v) for m in v[0]} for k, v in by_pair.items()}
    effects = []
    archived = {r['reference']: r for r in read(OUT / 'analysis/summary.json')['relative_to_arrival']}
    for arm in arms[1:]:
        values, windows = [], defaultdict(list)
        for c in cells:
            effect = 100 * (means[c, arm]['ttft_ms'] - means[c, 'arrival']['ttft_ms']) / means[c, arm]['ttft_ms']
            values.append(effect)
            windows[cells[c]['window']].append(effect)
        w = [st.fmean(v) for v in windows.values()]
        assert len(w) == 6
        seed = int(hashlib.sha256(json.dumps(arm, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()[:12], 16)
        rng = random.Random(seed)
        samples = sorted(st.fmean(rng.choice(w) for _ in w) for _ in range(10000))
        item = dict(reference=arm, improvement_pct=st.fmean(values), ci_low=samples[249], ci_high=samples[9749], wins=sum(v > 0 for v in values), losses=sum(v < 0 for v in values))
        for current, original in [('improvement_pct','arrival_mean_ttft_improvement_pct'), ('ci_low','bootstrap_window_95ci_low_pct'), ('ci_high','bootstrap_window_95ci_high_pct'), ('wins','wins'), ('losses','losses')]:
            close(item[current], archived[arm][original])
        effects.append(item)
    diagnosis = []
    for cell in cells:
        if cell.startswith('code'):
            a, b = means[cell, 'arrival'], means[cell, 'srp']
            diagnosis.append(dict(cell=cell, improvement_pct=100*(b['ttft_ms']-a['ttft_ms'])/b['ttft_ms'], spare=a, srp=b))

    # Capacity family only: sustained-arrival runs are a different protocol.
    g4_rows = list(csv.DictReader((ROOT / 'artifacts/www_capacity_load_20261007_r1/analysis/runs.csv').open()))
    g4 = defaultdict(list)
    for r in g4_rows:
        if '_sustained_' not in r['cell'] and r['arm'] in ('arrival','srp'):
            g4[r['cell'],r['arm']].append(r)
    assert len(g4) == 216 and all(len(v) == 4 for v in g4.values())
    capacities = []
    for capacity in [512,768,1024,1536,2048,3072,4096,5120,6144]:
        for source in ('all','code','conversation'):
            chosen = [c for c,a in g4 if a == 'arrival' and int(g4[c,a][0]['capacity']) == capacity and (source == 'all' or c.startswith(source))]
            values = []
            for c in chosen:
                a = st.fmean(float(r['ttft_ms_mean']) for r in g4[c,'arrival'])
                b = st.fmean(float(r['ttft_ms_mean']) for r in g4[c,'srp'])
                values.append(100*(b-a)/b)
            item = dict(capacity=capacity,source=source,cells=len(chosen),improvement_pct=st.fmean(values),wins=sum(v>0 for v in values))
            for arm in ('arrival','srp'):
                rr = [r for c in chosen for r in g4[c,arm]]
                item[arm] = {m:st.fmean(float(r[m]) for r in rr) for m in ('preemptions','schedule_ms_mean','ttft_ms_mean')}
            capacities.append(item)
    result = dict(date='2026-10-10', groups=18, formal_runs=288, request_records=request_count,
                  round_records=round_count, design_and_raw_hashes_verified=True,
                  ttft_recomputed_from_request_timestamps=True, archived_effects_and_ci_match=True,
                  effects=effects, code_diagnosis=diagnosis, g4_from_archived_run_csv=capacities,
                  scope='Accepted formal runs, chosen group A/A results and round files. No GPU rerun. G4 checked from archived runs.csv, not raw request files. Overhead attribution is observational.')
    target = OUT / 'analysis/local_verification_20261010.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('g4_from_archived_run_csv',)},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
