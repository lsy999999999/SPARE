"""Frozen proportional chunk-cap evaluation, separate from candidate discovery."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/v1_ratio_closed_loop_20260928_r1'
MODEL='models/Qwen2.5-7B-Instruct'
ARMS=['fixed64','ratio16','ratio4']
CONFIGS=[(512,256),(1024,128),(1024,256),(1024,1024)]
SIM_CONFIGS=[(t,k) for t in (512,1024,2048) for k in (128,256,512,1024) if k<=t]
SHAPES=['burst_stop','steady']

def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def spec(arm):
    from research.v1_broad12h_20260925.policy import formula
    assert arm in ARMS
    return dict(formula(1),candidate_mode=arm)
def sources():
    from research.v1_fixed_order_20260927.common import sources as previous
    paths=set(previous());paths.update(str(p.relative_to(ROOT)) for p in HERE.glob('*.py'))
    paths.update(['research/v1_candidate_audit_20260928/run.py','research/v1_candidate_audit_20260928/common.py',
        'research/v1_candidate_audit_20260928/candidates.py','research/v1_candidate_audit_20260928/audit.py',
        'simulator_core/timing.py',str((HERE/'PROTOCOL_CN.md').relative_to(ROOT))])
    return {p:digest(ROOT/p) for p in sorted(paths)}
def verify():
    d=read(OUT/'design.json');assert d['source_hashes']==sources(),'Frozen source mismatch'
    for case,sha in d['input_hashes'].items():assert digest(OUT/f'inputs/{case}.json')==sha
    return d
