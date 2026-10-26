"""Freeze one bounded GPU comparison, with disjoint development and confirmation IDs."""
import csv
from datetime import datetime
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_theory_formula_20260924.common import *
from research.v1_theory_formula_20260924.policy import arms


def main():
    assert not (OUT/'design.json').exists()
    prior=ROOT/'artifacts/v1_constant_shift_20260924_r1'
    cases={s+'_dev':read(prior/f'inputs/{s}_fresh.json') for s in ('code','conversation')}
    dirs=[ROOT/'artifacts'/f'{stem}_20260924_r1'/'inputs' for stem in
          ('v1_simple_formula','v1_decode_load_formula','v1_constant_shift','v1_arrival_formula')]
    dirs.append(OLD/'inputs/traces')
    seen={r['id'] for directory in dirs for p in directory.glob('*.json') for r in read(p)}
    provenance={}
    for source,date in (('code','2024-05-15'),('conversation','2024-05-17')):
        minute=date+' 15:17'
        path=ROOT/f'data/azure_{"conv" if source=="conversation" else "code"}_1week.csv'
        chosen=[]
        with path.open(newline='') as stream:
            reader=csv.reader(stream);next(reader)
            for i,r in enumerate(reader):
                if r[0][:16]==minute:
                    chosen.append((i,r))
                    if len(chosen)==64:break
        assert len(chosen)==64
        first=datetime.fromisoformat(chosen[0][1][0])
        rows=[dict(id=f'{source}:raw:{i}',prompt=int(r[1]),output=int(r[2]),
                   arrival_ms=(datetime.fromisoformat(r[0])-first).total_seconds()*1000) for i,r in chosen]
        assert not seen & {r['id'] for r in rows}
        assert all(r['prompt']>0 and r['output']>0 and r['prompt']+r['output']<(768-1)*16 for r in rows)
        cases[source+'_holdout']=rows
        provenance[source]=dict(minute=minute,path=str(path.relative_to(ROOT)),source_sha256=digest(path),
                                row_range=[chosen[0][0],chosen[-1][0]])
    for name,rows in cases.items():write(OUT/f'inputs/{name}.json',rows)
    methods=['old','decode','constant16','coupled_decode','coupled_exact']
    cells=lambda phase:[dict(case=s+'_'+phase,capacity=c) for c in (768,1536) for s in ('code','conversation')]
    write(OUT/'design.json',dict(frozen=True,arms={n:arms()[n] for n in methods},methods=methods,
        development=cells('dev'),holdout=cells('holdout'),repeats=2,
        trace_hashes={n:object_digest(r) for n,r in cases.items()},provenance=provenance,
        design_frozen_at=time.time(),tie_margin=.005,
        tie_order=['old','constant16','decode','coupled_decode','coupled_exact'],
        selection='Minimum equal-cell mean(mean TTFT arm / mean TTFT old - 1), after averaging two repetitions per cell. Within 0.005 of minimum, use frozen simplicity order.',
        holdout_methods='old,decode,development winner, deduplicated; two repeats; no retuning on confirmation.',
        objective='mean TTFT only; TBT/E2E are recorded, not objectives, constraints or compensation.',
        expected_development=40,expected_aa_range=[4,8],expected_total_range=[60,72],
        gpu_preference=[7,0,6,1,2,3],capacities=[768,1536],
        scope='Known 13:17 windows explicitly promoted to development; new 15:17 identities for confirmation. Same source dates, not cross-day validation.',
        rationale='Lomax-only correction changes just 1.61% of eligible old actions, deferred. Compare D additive, multiplicative interaction, and finite-shortage interaction. Exponential tail deferred.',
        calibration='No parameter search: a=8,b=128,k=32 fixed. Compare structures first; failure does not disprove all parameterizations of a family.'))
    print('Frozen: 40 development + 16-24 confirmation + 4 A/A = 60-68 GPU runs.')


if __name__=='__main__':main()
