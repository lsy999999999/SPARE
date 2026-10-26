"""Same candidate set and n; isolate progress guard, work and SPT ordering."""
import math
from research.v1_dynamic_actions_20260927.policy import select as dynamic_select, inverse_budget
from research.v1_budget_20260922.policy import Plan


def remaining_work(rows, tokens):
    return [max(0, r.prompt-r.computed-tokens.get(r.rid, 0)) for r in rows if r.outputs == 0]


def completion_work(remaining):
    ordered = sorted(remaining)
    return sum((len(ordered)-j)*work for j, work in enumerate(ordered))


def completion_time(rows, tokens, chunk, model):
    # Measured step time: intercept + token work + attention pairs in millions.
    a, b, c = model
    pending = [r for r in rows if r.outputs == 0]
    q = sum(tokens.values())
    attention = sum(tokens.get(r.rid, 0)*r.computed +
                    tokens.get(r.rid, 0)*(tokens.get(r.rid, 0)+1)/2 for r in rows)
    step = a+b*q+c*attention/1e6
    services = []
    for r in pending:
        computed = min(r.prompt, r.computed+tokens.get(r.rid, 0))
        left = max(0, r.prompt-computed)
        # An ideal serial continuation, using the fixed maximum chunk, not
        # assuming today's reduced chunk remains in force forever.
        service = a*math.ceil(left/chunk)+b*left+c*(computed*left+left*(left+1)/2)/1e6
        services.append(service)
    return len(pending)*step+completion_work(services)


def select(rows, budget, chunk, free, total, block_size, concurrency, max_model_len,
           threshold=1, recent=None):
    if not isinstance(threshold, dict):
        return dynamic_select(rows, budget, chunk, free, total, block_size, concurrency,
                              max_model_len, threshold, recent)
    mode = threshold.get('candidate_mode', 'current') if isinstance(threshold, dict) else 'current'
    forwarded = dict(threshold, candidate_mode='dynamic' if mode not in ('current', 'dynamic') else mode)
    result = dynamic_select(rows, budget, chunk, free, total, block_size, concurrency,
                            max_model_len, forwarded, recent)
    result['features']['candidate_mode'] = mode
    if mode in ('current', 'dynamic') or threshold.get('forced_action') is not None:
        return result
    assert mode in ('nonstall', 'work', 'spt', 'spt_time')
    base, n = result['base'], result['threshold']
    if base is None or n is None or 'candidate_diagnostics' not in result:
        return result
    pending = [r for r in rows if r.outputs == 0]
    base_progress = sum(base.tokens.get(r.rid, 0) for r in pending)
    eligible = []
    rejected = 0
    for index, d in enumerate(result['candidate_diagnostics']):
        if d['tokens'] is None:
            continue
        progress = sum(d['tokens'].get(r.rid, 0) for r in pending)
        guard = not (base_progress > 0 and progress == 0)
        d.update(prefill_progress=progress, progress_guard=guard)
        if not (d['guard'] and d['saving'] >= n and not d['duplicate']):
            continue
        if not guard:
            rejected += 1
            continue
        if mode == 'nonstall':
            key = (d['loss_per_block'], d['loss'], -d['saving'], -sum(d['tokens'].values()), index)
        else:
            rem = remaining_work(rows, d['tokens'])
            score = (sum(rem) if mode == 'work' else completion_work(rem) if mode == 'spt' else
                     completion_time(rows, d['tokens'], chunk, threshold['service_model']))
            d['completion_score'] = score
            # Saving is a gate, then a tie-break; not a divisor in these arms.
            key = (score, -d['saving'], -sum(d['tokens'].values()), index)
        eligible.append((key, d))
    result['features'].update(progress_guard_rejections=rejected,
                              eligible_after_progress_guard=len(eligible), base_prefill_progress=base_progress)
    result.update(budget=budget, chunk=chunk, selected=base, changed=False)
    for k in ('selected_progress_loss', 'selected_loss_per_block'):
        result['features'].pop(k, None)
    if eligible:
        key, d = min(eligible, key=lambda item:item[0])
        result.update(budget=d['budget'], chunk=d['chunk'], selected=Plan(d['tokens'], d['growth']),
                      changed=d['tokens'] != base.tokens)
        result['features'].update(selected_order_score=key[0], selected_progress_loss=d['loss'],
                                  selected_loss_per_block=d['loss_per_block'])
    return result
