"""Small frozen pilot for candidate selection and state-derived token budgets."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/v1_completion_order_20260927_r1'
MODEL = 'models/Qwen2.5-7B-Instruct'
BASE = dict(budget=1024, chunk=256, concurrency=32)
ARMS = ['current', 'dynamic', 'nonstall', 'work', 'spt', 'spt_time']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    paths = list(HERE.glob('*.py'))
    for folder, names in [
        ('v1_dynamic_actions_20260927', ['policy.py']),
        ('v1_broad12h_20260925', ['policy.py', 'scheduler.py']),
        ('v1_saving_threshold_20260924', ['policy.py']),
        ('v1_budget_20260922', ['policy.py', 'scheduler.py', 'run.py', 'test_scheduler.py']),
        ('v1_state_aware_20260923', ['run.py', 'common.py']),
    ]:
        paths += [ROOT / 'research' / folder / name for name in names]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def spec(arm):
    from research.v1_broad12h_20260925.policy import formula
    assert arm in ARMS
    return dict(formula(1), candidate_mode=arm, service_model=read(OUT/'service_model.json')['coefficients'])
