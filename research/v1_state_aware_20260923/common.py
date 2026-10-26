"""Shared immutable design and atomic artifacts (stdlib only)."""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/v1_state_aware_20260923_r1'
INPUT=ROOT/'artifacts/v1_capacity_design_20260923/traces'
OLD=ROOT/'artifacts/v1_capacity_gpu_20260923_r1'
METHODS=('a_frozen','a_any_positive_saving','fixed_small','occupancy',
         'pressure_one','pressure_relative')
BLOCKS=('code_test0','conversation_test0')
CAPS=(768,1536,2048)
CONFIG=dict(budget=1024,chunk=256,concurrency=32)


def read(path): return json.loads(Path(path).read_text())
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def config(method):
    c=dict(CONFIG)
    if method=='fixed_small': c['budget']=512
    if method=='occupancy': c['occupancy_threshold']=.7
    if method.startswith('pressure_'): c.update(pressure='1/4',saving='1/2')
    return c


def state(name,**kwargs):
    write(OUT/(name+'.json'),dict(updated=time.time(),pid=os.getpid(),**kwargs))


def sources():
    files=list(HERE.glob('*.py'))+[HERE/'PROTOCOL_CN.md']
    files += [ROOT/'research/v1_budget_20260922'/n for n in
              ('policy.py','scheduler.py','run.py','test_scheduler.py')]
    files += [ROOT/'research/v1_capacity_ablation_20260923'/n for n in ('policy.py','scheduler.py')]
    return {str(p.relative_to(ROOT)):digest(p) for p in files}


def check_manifest():
    m=read(OUT/'manifest.json')
    assert sources()==m['source_hashes'], 'Source changed: new run directory required'
    for name,sha in m['trace_hashes'].items(): assert digest(INPUT/name)==sha
    return m
