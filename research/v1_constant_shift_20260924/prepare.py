"""Outcome-free constant matching on old development logs, then freeze comparisons."""
import csv
from datetime import datetime
import gzip
import json
import math
from pathlib import Path
import statistics as st
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_constant_shift_20260924.common import *
from research.v1_constant_shift_20260924.policy import formula

PREVIOUS=ROOT/'artifacts/v1_decode_load_formula_20260924_r1'


def main():
    assert not (OUT/'design.json').exists()
    cells=[]
    # Only the previous development (11:17) OLD-formula trajectories are used.
    # Equal weight across cells; pool the two replicates within each cell.
    for cap in (768,1536,2048):
        for source in ('code','conversation'):
            values=[];files=[]
            for repeat in (0,1):
                name=f'dev_{source}_dev_c{cap}_old_r{repeat}.json'
                run=read(PREVIOUS/'runs'/name)
                with gzip.open(PREVIOUS/run['rounds_path'],'rt') as f:rounds=json.load(f)
                for row in rounds:
                    if row['baseline'] is None or len(row['state'])<2 or not row['baseline_growth_blocks']:continue
                    if not any(c['guard'] and (c['saving'] or 0)>0 for c in row['candidate_diagnostics']):continue
                    z=sum(s['blocks'] for s in row['state'] if s['outputs']>0)/(cap-1)
                    values.append(32*(1-z))
                files.append(dict(path=name,rounds_sha256=run['rounds_sha256']))
            assert values
            cells.append(dict(source=source,capacity=cap,mean_addition=st.fmean(values),eligible_rounds=len(values),files=files))
    mean=st.fmean(c['mean_addition'] for c in cells)
    delta=math.floor(mean+.5)
    assert 0<delta<32
    write(OUT/'constant_calibration.json',dict(frozen=True,delta=delta,unrounded_equal_cell_mean=mean,cells=cells,
        selection='Nearest integer of equal-cell mean dynamic addition on old development eligible states; no new TTFT labels.',
        limitation='Matches average threshold addition on old states, not necessarily interventions or compute on new trajectories.'))
    cases={s+'_dev':read(PREVIOUS/f'inputs/{s}_dev.json') for s in ('code','conversation')}
    cases['code_diagnostic']=read(PREVIOUS/'inputs/code_holdout.json')
    dirs=[OLD/'inputs/traces',ROOT/'artifacts/v1_simple_formula_20260924_r1/inputs',PREVIOUS/'inputs']
    seen={r['id'] for directory in dirs for p in directory.glob('*.json') for r in read(p)}
    provenance={}
    for source,minute in dict(code='2024-05-15 13:17',conversation='2024-05-17 13:17').items():
        path=ROOT/f'data/azure_{"conv" if source=="conversation" else "code"}_1week.csv'
        chosen=[]
        with path.open(newline='') as f:
            reader=csv.reader(f);next(reader)
            for i,r in enumerate(reader):
                if r[0][:16]==minute:
                    chosen.append((i,r))
                    if len(chosen)==64:break
        assert len(chosen)==64,(source,minute)
        first=datetime.fromisoformat(chosen[0][1][0])
        rows=[dict(id=f'{source}:raw:{i}',prompt=int(r[1]),output=int(r[2]),
                   arrival_ms=(datetime.fromisoformat(r[0])-first).total_seconds()*1000) for i,r in chosen]
        assert not seen & {r['id'] for r in rows}
        assert all(0<r['prompt'] and 0<r['output'] for r in rows)
        assert max(r['prompt']+r['output'] for r in rows)<min(12288,(768-1)*16)
        cases[source+'_fresh']=rows
        provenance[source]=dict(path=str(path.relative_to(ROOT)),minute=minute,row_range=[chosen[0][0],chosen[-1][0]],source_sha256=digest(path))
    for name,rows in cases.items():write(OUT/f'inputs/{name}.json',rows)
    fresh=[dict(case=s+'_fresh',capacity=c) for c in (768,1536,2048) for s in ('code','conversation')]
    diagnostic=[dict(case='code_diagnostic',capacity=1536)]
    arms=dict(old=formula(0),dynamic=formula(32),constant_matched=formula(0,a=8+delta),constant32=formula(0,a=40))
    write(OUT/'design.json',dict(frozen=True,arms=arms,methods=list(arms),fresh=fresh,diagnostic=diagnostic,
        repeats=2,provenance=provenance,constant_delta=delta,
        trace_hashes={n:object_digest(r) for n,r in cases.items()},
        objective='Compare TTFT, mean/p95 TBT and E2E together; no post-evaluation constant selection or weighted utility.',
        interpretation='Fresh 13:17 windows disjoint from prior inputs; 12:17 code/1536 reused solely as diagnostic replication.',
        expected_formal_runs=56,expected_aa_runs_at_three_gpus=6,
        design_frozen_at=__import__('time').time(),calibration_sha256=digest(OUT/'constant_calibration.json')))
    print('Matched constant delta =',delta,'unrounded =',mean,'formal runs = 56 + A/A')


if __name__=='__main__':main()
