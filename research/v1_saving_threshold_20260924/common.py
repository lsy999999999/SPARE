"""Immutable inputs and artifacts for the factor-first threshold experiment."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/v1_saving_factors_20260924_r1'
INPUT = OUT / 'inputs'
MODEL = 'models/Qwen2.5-7B-Instruct'
BASE = dict(budget=1024, chunk=256, concurrency=32)
CAPS = [512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 7445]
# Cover endpoints before filling intermediate capacities.
CAP_ORDER = [768, 4096, 512, 7445, 2048, 1024, 6144, 1536, 3072]
THRESHOLDS = [1, 4, 16, 32, 64, 'off']
DEADLINE = 1790218800.0  # verified against the ISO deadline by prepare.py


def read(p):
    return json.loads(Path(p).read_text())


def write(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + '.tmp')
    temp.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(p)


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def object_digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sources():
    paths = [HERE / n for n in ('common.py', 'policy.py', 'scheduler.py', 'run.py', 'prepare.py')]
    paths += [ROOT / 'research/v1_budget_20260922' / n for n in ('policy.py', 'scheduler.py', 'run.py')]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}
