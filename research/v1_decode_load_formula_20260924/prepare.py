"""Freeze two positive-signed candidates and entirely new validation minutes."""
import csv
from datetime import datetime
import gzip
from pathlib import Path
import statistics as st
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_decode_load_formula_20260924.common import *
from research.v1_decode_load_formula_20260924.policy import formula,observable
from research.v1_budget_20260922.policy import State

PREVIOUS=ROOT/'artifacts/v1_simple_formula_20260924_r1'


def main():
    assert not (OUT/'design.json').exists()
    diagnostic=[]
    for source in ('code','conversation'):
        for cap in (768,1536,2048,4096):
            p=PREVIOUS/'runs'/f'holdout_{source}_holdout_c{cap}_formula_r0.json'
            r=read(p)
            with gzip.open(PREVIOUS/r['rounds_path'],'rt') as f:rounds=__import__('json').load(f)
            features=[]
            for row in rounds:
                if row['baseline'] is None:continue
                if not any(c['guard'] and (c['saving'] or 0)>0 for c in row['candidate_diagnostics']):continue
                features.append(observable([State(**s) for s in row['state']],row['free_before'],cap,16))
            diagnostic.append(dict(source=source,capacity=cap,eligible_rounds=len(features),
                x_median=st.median(f['work_free_ratio'] for f in features),
                z_median=st.median(f['decoded_kv_fraction'] for f in features),
                source_run=p.name,warning='Old trajectory association, not causal branch labels.'))
    write(OUT/'diagnostic.json',diagnostic)
    cases={s+'_dev':read(PREVIOUS/f'inputs/{s}_holdout.json') for s in ('code','conversation')}
    seen={r['id'] for directory in (OLD/'inputs/traces',PREVIOUS/'inputs') for p in directory.glob('*.json') for r in read(p)}
    provenance={}
    for source,minute in dict(code='2024-05-15 12:17',conversation='2024-05-17 12:17').items():
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
        assert max(r['prompt']+r['output'] for r in rows)<(768-1)*16
        cases[source+'_holdout']=rows
        provenance[source]=dict(path=str(path.relative_to(ROOT)),minute=minute,row_range=[chosen[0][0],chosen[-1][0]],source_sha256=digest(path))
    for level in ('low','high'):
        cases['code_output_'+level]=[dict(r,output=max(1,r['output']//2) if level=='low' else r['output']*2)
            for r in cases['code_holdout']]
    for name,rows in cases.items():write(OUT/f'inputs/{name}.json',rows)
    dev=[dict(case=s+'_dev',capacity=c) for c in (768,1536,2048) for s in ('code','conversation')]
    hold=[dict(case=s+'_holdout',capacity=c) for c in (768,1536,2048,4096) for s in ('code','conversation')]
    hold += [dict(case='code_output_'+level,capacity=2048) for level in ('low','high')]
    write(OUT/'design.json',dict(frozen=True,candidates=dict(load16=formula(16),load32=formula(32)),
        old_formula=formula(0),development=dev,holdout=hold,
        development_methods=['off','old','load16','load32'],holdout_methods=['off','old','n16','formula'],
        provenance=provenance,trace_hashes={n:object_digest(r) for n,r in cases.items()},
        selection='Best mean TTFT/off - 1 among two positive-c candidates; tie within 0.005 chooses smaller c. Old a=8,b=128 frozen.',
        adoption='Provisional adoption only if development improves old by >=0.01 normalized TTFT and worst relative-old regression <=0.05. Still evaluate best extension on holdout even if not adopted.',
        repeats='Two complete development repeats for all arms; single holdout run. No adaptive extra runs.',
        fast_calibration='Only c in {0,16,32}; same coefficients at every capacity; 4096 excluded from calibration.',
        workload_roles='Previous holdout explicitly promoted to development; fresh 12:17 minute holdout has no shared IDs with previous input directories.',
        claim='One GPU/model pair only; no cross-model/hardware coefficient-transfer claim.'))
    print('Frozen:',len(dev),'development cells x 4 arms x 2 repeats; 10 holdout cells x 4 arms')


if __name__=='__main__':main()
