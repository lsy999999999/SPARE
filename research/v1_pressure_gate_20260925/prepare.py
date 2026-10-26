"""Freeze one pressure gate, a no-D control, reused diagnostic and fresh checks."""
import csv,math,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_pressure_gate_20260925.common import *
from research.v1_pressure_gate_20260925.policy import formula

def main():
    assert not (OUT/'design.json').exists()
    prior=ROOT/'artifacts/v1_decode_arrival_confirm_20260925_r1'
    dirs=[OLD/'inputs/traces',prior/'inputs']+[ROOT/f'artifacts/{name}_20260924_r1/inputs' for name in
        ('v1_simple_formula','v1_decode_load_formula','v1_constant_shift','v1_arrival_formula','v1_theory_formula','v1_decode_arrival_formula')]
    seen={r['id'] for directory in dirs for p in directory.glob('*.json') for r in read(p)}
    cases={};provenance={};cells=[]
    for source,shape,date in [('code','accelerating','2024-05-15'),('conversation','steady','2024-05-17')]:
        path=ROOT/f'data/azure_{"conv" if source=="conversation" else "code"}_1week.csv'
        chosen=[];minute=date+' 18:17'
        with path.open(newline='') as f:
            reader=csv.reader(f);next(reader)
            for i,r in enumerate(reader):
                if r[0][:16]==minute:
                    chosen.append((i,r))
                    if len(chosen)==64:break
        assert len(chosen)==64
        fresh=[dict(id=f'{source}:raw:{i}',prompt=int(r[1]),output=int(r[2])) for i,r in chosen]
        assert not seen & {r['id'] for r in fresh}
        assert all(r['prompt']>0 and r['output']>0 and r['prompt']+r['output']<12288 for r in fresh)
        for i,r in enumerate(fresh):
            u=i/63;r['arrival_ms']=12000*u if shape=='steady' else 4000*(math.sqrt(1+15*u)-1)
        reused=read(prior/f'inputs/{source}_{shape}.json')
        for window,rows in [('reused',reused),('fresh',fresh)]:
            case=f'{source}_{shape}_{window}';cases[case]=rows
            for cap in (768,1536):cells.append(dict(case=case,source=source,shape=shape,window=window,capacity=cap))
        provenance[source]=dict(path=str(path.relative_to(ROOT)),source_sha256=digest(path),fresh_minute=minute,
            row_range=[chosen[0][0],chosen[-1][0]],fresh_prompt_total=sum(r['prompt'] for r in fresh),fresh_output_total=sum(r['output'] for r in fresh),
            reused_path=str((prior/f'inputs/{source}_{shape}.json').relative_to(ROOT)))
    for case,rows in cases.items():write(OUT/f'inputs/{case}.json',rows)
    arms=dict(old=formula(0),arrival=formula(1),combined=formula(1,32),
              pressure_decode=formula(1,32,'pressure_decode'),pressure_constant=formula(1,32,'pressure_constant'))
    write(OUT/'design.json',dict(frozen=True,arms=arms,methods=list(arms),cells=cells,repeats=2,
        trace_hashes={case:object_digest(rows) for case,rows in cases.items()},provenance=provenance,
        expected_formal_runs=80,design_frozen_at=time.time(),capacities=[768,1536],max_gpus=4,gpu_preference=[7,0,6,1,2,3],
        arrival_ewma_s=1.,prediction_horizon_s=1.,gate='x/(1+x), x=F/(1+(W+A)/16)',
        objective='Mean TTFT only; gated D must be compared with unmodified D+A, A alone and gated constant.',
        scope='Mechanism-focused two cases (code accelerating, conversation steady), two capacities, reused 17:17 and fresh 18:17 requests; no new coefficient tuning or universal claim.',
        predeclared='Reused and fresh results reported separately; do not select or alter the gate after seeing either GPU result.',
        limitation='Cases intentionally chosen using prior failures and successes; no burst-stop test or broad arrival-shape claim. Fresh minute, not fresh date. Capacity groups may use different cards.'))
    print('Frozen 80 formal comparisons + per-GPU/capacity A/A')

if __name__=='__main__':main()
