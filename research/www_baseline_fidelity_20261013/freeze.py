"""Refresh development snapshot before the first formal launch only."""
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.www_baseline_fidelity_20261013.common import *
if (OUT/'launch.json').exists() or (OUT/'manifest.json').exists():
    raise RuntimeError('cannot change frozen experiment after launch')
design=read(OUT/'design.json')
for rel,source in source_paths().items():
    target=OUT/'source'/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    if source.resolve()!=target.resolve():shutil.copyfile(source,target)
design['source_hashes']=source_hashes()
design['frozen_at']=time.time()
write(OUT/'design.json',design)
print('frozen',len(design['source_hashes']),'source files')
