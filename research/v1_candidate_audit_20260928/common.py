"""Bounded two-dimensional candidate opportunity audit."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/v1_candidate_audit_20260928_r1'
OLD = ROOT / 'artifacts/v1_efficient_points_20260927_r2'
MODEL = 'models/Qwen2.5-7B-Instruct'
CONFIGS = [(1024,256), (512,256), (1024,128)]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(tokens):
    return tuple(sorted(tokens.items()))


def spec():
    from research.v1_broad12h_20260925.policy import formula
    return dict(formula(1), candidate_mode='current')


def sources():
    from research.v1_fixed_order_20260927.common import sources as previous
    paths = set(previous())
    paths.update(str(p.relative_to(ROOT)) for p in HERE.glob('*.py'))
    paths.add(str((HERE/'PROTOCOL_CN.md').relative_to(ROOT)))
    return {p:digest(ROOT/p) for p in sorted(paths)}
