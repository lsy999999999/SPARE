"""Freeze denser capacity grid and GPU sustained arrivals before measurement."""
import sys,shutil,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.www_capacity_load_20261007.common import *

def main():
    assert not (OUT/'design.json').exists()
    previous=ROOT/'artifacts/www_evaluation_gpu_completion_20261003_r1'
    for rel in source_paths():
        source=previous/'source'/rel
        if source.exists():
            target=OUT/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    old=read(previous/'design.json');cells=[]
    for cell in old['cells']:
        if cell['capacity']!=2048:continue
        for cap in [512,768,1024,1536,2048,3072,4096,5120,6144]:
            c=dict(cell);c.update(key=f"{c['window']}_{c['shape']}_c{cap}",capacity=cap,family='capacity')
            cells.append(c)
    for source in ['code','conversation']:
        for window in range(3):
            base=next(c for c in old['cells'] if c['window']==f'{source}_new{window}' and c['shape']=='steady' and c['capacity']==2048)
            # Read the frozen 512-request record used in the existing simulator scan.
            path=ROOT/'artifacts/www_evaluation_completion_20261003_r1/inputs'/f'{source}_new{window}_sustained_r2.json'
            if not path.exists():
                candidates=list(path.parent.glob(f'{source}_new{window}*'))
                long=[p for p in candidates if len(read(p))==512]
                assert long, candidates
                path=sorted(long)[0]
            reqs=read(path);assert len(reqs)==512
            for rate in [2,5,10,20]:
                target=OUT/'inputs'/f'{source}_new{window}_sustained_r{rate}.json'
                write(target,[dict(r,arrival_ms=i*1000/rate) for i,r in enumerate(reqs)])
                c=dict(base);c.update(key=f'{source}_new{window}_sustained_r{rate}_c2048',shape='sustained',family='load',arrival_rps=rate,requests=512,input=str(target.relative_to(OUT)),input_sha256=digest(target),source_input=str(path.relative_to(ROOT)))
                cells.append(c)
    for c in cells:
        for field in ['input','history_file']:
            target=OUT/c[field]
            if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(previous/c[field],target)
    hashes=source_hashes()
    for rel,source in source_paths().items():
        target=OUT/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if source.resolve()!=target.resolve():shutil.copyfile(source,target)
    design=dict(schema=1,frozen=True,frozen_at=time.time(),cells=cells,methods=list(ARMS),options=OPTIONS,repeats=4,aa_repeats=2,expected_formal_runs=len(cells)*len(ARMS)*4,expected_aa_runs=len(cells)*2,max_model_len=8176,max_num_seqs=64,max_num_batched_tokens=2048,block_size=16,dtype='float16',gpu_candidates=[1,2,6],episode_timeout_s=1800,deadline_hours=72,max_group_retries=2,ordering='fixed shuffled/reverse pairs; no outcome-based selection',source_hashes=hashes,inference_unit='six source windows',objective='mean TTFT; capacity and load families reported separately',stopping='fixed grid; technical timeout/interference only')
    write(OUT/'design.json',design)
    print(json.dumps(dict(groups=len(cells),formal=design['expected_formal_runs'])))
if __name__=='__main__':main()
