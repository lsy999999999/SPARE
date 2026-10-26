"""Independent-window, preselected mechanism contrasts."""
from research.v1_candidate_audit_20260928.common import read, write, digest, spec, signature, MODEL, CONFIGS, ROOT
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/v1_candidate_transfer_20260928_r1'
REPEATS = 4

def sources():
    from research.v1_candidate_audit_20260928.common import sources as previous
    paths=set(previous())
    paths.update(str(p.relative_to(ROOT)) for p in HERE.glob('*.py'))
    paths.add(str((HERE/'PROTOCOL_CN.md').relative_to(ROOT)))
    return {p:digest(ROOT/p) for p in sorted(paths)}

def verify():
    d=read(OUT/'design.json')
    assert d['source_hashes']==sources(), 'Frozen source mismatch'
    for name,sha in d['input_hashes'].items():
        assert digest(OUT/f'inputs/{name}.json')==sha
    return d
