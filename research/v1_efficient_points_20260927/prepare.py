import shutil
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_efficient_points_20260927.common import *
from research.v1_efficient_points_20260927.timing import GRID,CONTEXTS,CHUNKS,REPEATS


def main():
    assert not (OUT/'design.json').exists(),'Never overwrite frozen design'
    cases=[]
    for source in ('code','conversation'):
        for shape in ('steady','burst_stop'):
            case=f'{source}_w0_{shape}'
            original=ROOT/f'artifacts/v1_broad12h_20260925_r1/inputs/{case}.json'
            reqs=read(original)
            assert len(reqs)==64
            write(OUT/f'inputs/{case}.json',reqs)
            cases.append(dict(case=case,source=source,shape=shape,original=str(original.relative_to(ROOT))))
    groups=[dict(key=f'{source}_c{c}',source=source,capacity=c,phase='formal') for c in (768,1536,2048) for source in ('code','conversation')]
    profiles=[dict(key=f'profile_{i}',capacity=2048,phase='profile') for i in range(4)]
    hashes=sources()
    for rel in hashes:
        dest=OUT/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    write(OUT/'design.json',dict(frozen_at=time.time(),cases=cases,groups=groups,profile_groups=profiles,
        arms=ARMS,configs={a:dict(BASE,threshold=spec(a)) for a in ARMS},repeats=2,formal_expected=120,
        grid=GRID,contexts=CONTEXTS,chunks=CHUNKS,profile_repeats=REPEATS,
        profile_per_group=len(GRID)*len(CONTEXTS)*len(CHUNKS)*(REPEATS+1),
        source_hashes=hashes,input_hashes={c['case']:digest(OUT/f"inputs/{c['case']}.json") for c in cases},
        protocol_sha256=digest(HERE/'PROTOCOL_CN.md'),
        claim='directional pilot: separate synthetic timing calibration, previously used real workload windows; not unseen test'))
    print('Frozen: 4 timing groups, 6 formal groups / 120 formal episodes')


if __name__=='__main__':main()
