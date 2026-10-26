"""Record a feasibility-only protocol amendment before launching any GPU labels."""
import shutil,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from research.v1_ratio_closed_loop_20260928.common import *
assert not (OUT/'launcher.json').exists() and not (OUT/'manifest.json').exists()
archive=OUT/'prelaunch_amendment_v0';assert not archive.exists();archive.mkdir()
for name in ('design.json','source','validation.json','simulation','simulation_status.json'):
    path=OUT/name
    if path.exists():shutil.move(str(path),str(archive/name))
d=read(archive/'design.json');old=d['source_hashes'];d.update(source_hashes=sources(),sim_configs=SIM_CONFIGS)
d['amendment']=dict(time=time.time(),reason='Exclude invalid T512/K1024: native scheduler requires K<=T',
    before_gpu_labels=True,previous_design_sha256=digest(archive/'design.json'),
    discarded_simulations=len(list((archive/'simulation').glob('*.json'))),expected_simulations=528)
write(OUT/'design.json',d)
for rel in d['source_hashes']:
    dest=OUT/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
write(OUT/'prelaunch_amendment.json',dict(d['amendment'],changed_sources=[p for p,v in d['source_hashes'].items() if old.get(p)!=v]))
print(d['amendment'])
