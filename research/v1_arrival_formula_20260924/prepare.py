"""Freeze same-request controlled arrival shapes and two untuned formulas."""
import csv
from datetime import datetime
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_arrival_formula_20260924.common import *
from research.v1_arrival_formula_20260924.policy import formula


def main():
    assert not (OUT/'design.json').exists()
    dirs=[OLD/'inputs/traces']+[ROOT/f'artifacts/{name}_20260924_r1/inputs' for name in
        ('v1_simple_formula','v1_decode_load_formula','v1_constant_shift')]
    seen={r['id'] for directory in dirs for p in directory.glob('*.json') for r in read(p)}
    cases={};provenance={};span=12000.
    for source,minute in dict(code='2024-05-15 14:17',conversation='2024-05-17 14:17').items():
        path=ROOT/f'data/azure_{"conv" if source=="conversation" else "code"}_1week.csv'
        chosen=[]
        with path.open(newline='') as f:
            reader=csv.reader(f);next(reader)
            for i,r in enumerate(reader):
                if r[0][:16]==minute:
                    chosen.append((i,r))
                    if len(chosen)==64:break
        assert len(chosen)==64,(source,minute)
        base=[dict(id=f'{source}:raw:{i}',prompt=int(r[1]),output=int(r[2])) for i,r in chosen]
        assert not seen & {r['id'] for r in base}
        assert all(0<r['prompt'] and 0<r['output'] for r in base)
        assert max(r['prompt']+r['output'] for r in base)<(768-1)*16
        for shape in ('steady','accelerating','burst_stop'):
            rows=[]
            for i,r in enumerate(base):
                u=i/(len(base)-1)
                t=span*u if shape=='steady' else span*(math.sqrt(1+15*u)-1)/3 if shape=='accelerating' else 0.
                rows.append(dict(r,arrival_ms=t))
            cases[source+'_'+shape]=rows
        provenance[source]=dict(path=str(path.relative_to(ROOT)),minute=minute,source_sha256=digest(path),
            row_range=[chosen[0][0],chosen[-1][0]],prompt_total=sum(r['prompt'] for r in base),
            output_total=sum(r['output'] for r in base))
    for name,rows in cases.items():write(OUT/f'inputs/{name}.json',rows)
    cells=[dict(case=source+'_'+shape,capacity=cap) for cap in (768,2048)
           for source in ('code','conversation') for shape in ('steady','accelerating','burst_stop')]
    write(OUT/'design.json',dict(frozen=True,arms=dict(old=formula(0),arrival=formula(1)),methods=['old','arrival'],
        cells=cells,repeats=2,arrival_ewma_s=1.,prediction_horizon_s=1.,span_ms=span,
        shapes=dict(steady='t=T*u',accelerating='t=T*(sqrt(1+15*u)-1)/3; end rate is 4x initial rate',burst_stop='all requests at t=0, then no more arrivals'),
        trace_hashes={n:object_digest(r) for n,r in cases.items()},provenance=provenance,
        expected_formal_runs=48,design_frozen_at=time.time(),
        online_arrivals='Observed when Request enters scheduler.add_request using monotonic elapsed wall time; no trace lookahead.',
        estimator='A(t)=sum(prompt_i/H * exp(-(t-t_i)/H)) over observed arrivals only; H=1 second fixed, no dual-window burst feature.',
        claim='Controlled arrival-shape experiment on fresh request identities, not natural-trace performance or tuned optimal horizon.',
        limitations='Steady request spacing is not constant token rate with heterogeneous prompts. Burst changes both shape and arrival span/density.'))
    print('Frozen 48 GPU comparisons + A/A; 2 sources x 3 shapes x 2 capacities x 2 arms x 2 repeats')


if __name__=='__main__':main()
