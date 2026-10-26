from pathlib import Path
from research.v1_saving_threshold_20260924.common import read, write, digest, object_digest, BASE, MODEL

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/v1_theory_formula_20260924_r1'
OLD = ROOT/'artifacts/v1_saving_factors_20260924_r1'


def source_hashes():
    files = list(HERE.glob('*.py'))
    files += [ROOT/'research'/folder/name for folder, names in (
        ('v1_saving_threshold_20260924', ('policy.py','common.py')),
        ('v1_simple_formula_20260924', ('policy.py',)),
        ('v1_decode_load_formula_20260924', ('policy.py',)),
        ('v1_budget_20260922', ('policy.py', 'scheduler.py', 'run.py', 'test_scheduler.py')),
        ('v1_state_aware_20260923', ('run.py',))) for name in names]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}
