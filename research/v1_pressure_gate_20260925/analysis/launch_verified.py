"""Launch exactly once after verifying the frozen CPU-validated code and design."""
from pathlib import Path
import subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from research.v1_pressure_gate_20260925.common import OUT,HERE,read,digest,source_hashes
v=read(OUT/'cpu_validation.json')
assert v['passed'] and v['sources']==source_hashes()
assert v['design_sha256']==digest(OUT/'design.json')
assert not (OUT/'launch.json').exists(), 'Already launched; inspect existing pipeline'
print('Frozen source/design verified',flush=True)
subprocess.run([sys.executable,'-B',str(HERE/'analysis/pipeline.py'),'--launch'],check=True)
