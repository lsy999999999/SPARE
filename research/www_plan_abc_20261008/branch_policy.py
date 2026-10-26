"""Single-round action override around the frozen G2 SPARE selector."""
from research.v1_budget_20260922.policy import project
from research.www_evaluation_gpu_completion_20261003.policy import select as spare_select


def select(rows, budget, chunk, free, total, block_size, concurrency,
           max_model_len, threshold=1, recent=None, options=None):
    if not isinstance(threshold, dict) or "forced_action" not in threshold:
        return spare_select(rows, budget, chunk, free, total, block_size,
                            concurrency, max_model_len, threshold, recent, options)
    spec = dict(threshold)
    action = spec.pop("forced_action")
    choice = spare_select(rows, budget, chunk, free, total, block_size,
                          concurrency, max_model_len, spec, recent, options)
    assert choice["base"] is not None
    plan = project(rows, action["budget"], action["chunk"], free,
                   block_size, concurrency, max_model_len)
    assert plan is not None and plan.tokens
    choice.update(budget=action["budget"], chunk=action["chunk"],
                  selected=plan, changed=plan.tokens != choice["base"].tokens)
    return choice
