"""Bounded mechanism/model/replica pilot; reuse the frozen policy unchanged."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/v1_replica_extension_20260925_r1'
BASE = dict(budget=1024, chunk=256, concurrency=32)
from research.v1_broad12h_20260925.policy import formula

ARMS = {
    'arrival': dict(method='threshold', config=dict(BASE, threshold=formula(1))),
    'old': dict(method='threshold', config=dict(BASE, threshold=formula(0))),
    'srp': dict(method='srp', config=BASE),
    'native': dict(method='native', config=BASE),
    'sarathi_common': dict(method='sarathi_v1', config=BASE),
    'past_future_common': dict(method='past_future_v1', config=BASE),
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sources():
    files = [HERE / n for n in ('common.py', 'prepare.py', 'run.py', 'validate.py', 'supervisor.py')]
    for folder, names in (
        ('v1_arrival_kv_papers_20260925', ['scheduler.py']),
        ('v1_broad12h_20260925', ['policy.py', 'scheduler.py']),
        ('v1_saving_threshold_20260924', ['policy.py']),
        ('v1_budget_20260922', ['policy.py', 'scheduler.py', 'run.py', 'test_scheduler.py']),
        ('v1_paper_comparison_20260922', ['scheduler.py', 'rules.py']),
        ('v1_state_aware_20260923', ['run.py', 'common.py']),
    ):
        files.extend(ROOT / 'research' / folder / n for n in names)
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def requests_for(case, replicas, load):
    original = read(OUT / 'inputs' / (case + '.json'))
    if load == 'fixed_total':
        return original
    assert load == 'per_replica'
    return [dict(r, id=f"{r['id']}:copy{j}") for r in original for j in range(replicas)]


def routes(requests, replicas):
    """Round-robin arrival-order router: no length/output or future-state input."""
    return {r['id']: index % replicas for index, r in enumerate(requests)}
