"""Require KV headroom for one more token after each protected decode."""
from math import ceil

from research.v1_budget_20260922.policy import Plan
from research.www_evaluation_completion_20261003.policy import select as spare_select


def next_decode_blocks(rows, tokens, block_size):
    needed = 0
    for row in rows:
        if not row.decode or tokens.get(row.rid, 0) != 1:
            continue
        after = row.computed + 1
        held = max(row.blocks, ceil(after / block_size))
        needed += max(0, ceil((after + 1) / block_size) - held)
    return needed


def select(rows, budget, chunk, free, total, block_size, concurrency,
           max_model_len, threshold=1, recent=None, options=None):
    if options and options.get('mode', 'spare') in ('fixed', 'occupancy'):
        raise ValueError('two-step decode protection applies only to SPARE candidates')
    out = spare_select(rows, budget, chunk, free, total, block_size,
                       concurrency, max_model_len, threshold, recent, options)
    original = out['selected']
    out['two_step_vetoes'] = 0
    out['two_step_action_changed'] = False
    if out['base'] is None or out['base'].growth == 0 or len(rows) < 2:
        return out
    out.update(budget=budget, chunk=chunk, selected=out['base'], changed=False)
    for candidate in out['candidate_diagnostics']:
        tokens = candidate['tokens']
        if tokens is None or not candidate['guard'] or candidate['saving'] < out['threshold']:
            continue
        headroom = free - candidate['growth']
        needed = next_decode_blocks(rows, tokens, block_size)
        candidate['next_decode_blocks'] = needed
        candidate['two_step_guard'] = headroom >= needed
        if not candidate['two_step_guard']:
            out['two_step_vetoes'] += 1
            continue
        out.update(budget=candidate['budget'], chunk=candidate['chunk'],
                   selected=Plan(tokens, candidate['growth']),
                   changed=tokens != out['base'].tokens)
        break
    out['two_step_action_changed'] = out['selected'].tokens != original.tokens
    return out
