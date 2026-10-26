"""Frozen small capacity/paper-rule pilot; original natural-arrival traces."""
from pathlib import Path
from research.v1_state_aware_20260923.common import ROOT, INPUT, read, write, digest

HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/v1_capacity_paper_pilot_20260923_r1'
HISTORY = ROOT/'artifacts/v1_paper_comparison_multi_20260922_r3/manifest.json'
MODEL = 'models/Qwen2.5-7B-Instruct'
BASE = dict(budget=1024, chunk=256, concurrency=32)
ARMS = {
    'srp': ('srp_matched', BASE),
    'one_block': ('a_any_positive_saving', BASE),
    'fixed512': ('fixed_small', dict(BASE, budget=512)),
    'pressure1_8': ('pressure_one', dict(BASE, pressure='1/8')),
    'sarathi_common': ('sarathi_v1', BASE),
    'past_future_common': ('past_future_v1', BASE),
    'sarathi_frozen': ('sarathi_v1', dict(BASE, chunk=1024)),
    'past_future_frozen': ('past_future_v1', dict(BASE, chunk=1024)),
}
BLOCKS = ('code_test0', 'conversation_test0')
CAPS = (768, 2048, 4096)


def sources():
    files = [HERE/n for n in ('common.py', 'scheduler.py', 'run.py', 'validate.py', 'PROTOCOL_CN.md')]
    for folder, names in (
        ('v1_state_aware_20260923', ('common.py', 'policy.py', 'scheduler.py', 'run.py')),
        ('v1_budget_20260922', ('policy.py', 'scheduler.py', 'run.py', 'test_scheduler.py')),
        ('v1_capacity_ablation_20260923', ('policy.py',)),
        ('v1_paper_comparison_20260922', ('scheduler.py', 'rules.py'))):
        files.extend(ROOT/'research'/folder/n for n in names)
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def history(source):
    h = read(HISTORY)['histories'][source]
    for block in BLOCKS:
        if block.startswith(source):
            assert not set(h['request_ids']) & {r['id'] for r in read(INPUT/(block+'_original.json'))}
    assert len(h['lengths']) == 1000
    return h['lengths']


def prepare(scheduler, source, repeat):
    scheduler.history_context = history(source)
    scheduler.seed_context = 20260922 + repeat
