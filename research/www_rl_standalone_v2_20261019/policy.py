"""Request-aware Double DQN with generic budget and chunk actions."""
from __future__ import annotations

from collections import deque
import math
import random

import numpy as np
import torch
from torch import nn

from research.v1_budget_20260922.policy import project
from research.www_rl_standalone_20261019 import policy as previous


ACTIONS = ((1024, 256), (0, 64)) + tuple(
    (budget, chunk) for budget in (128, 256, 384, 512, 768, 1024)
    for chunk in (64, 256) if (budget, chunk) != (1024, 256))
STATE_SIZE = previous.STATE_SIZE + 14


def state_of(scheduler):
    base = previous.state_of(scheduler)
    waiting = list(scheduler.waiting)
    running = list(scheduler.running)
    pending = [r for r in waiting + running if not r.output_token_ids]
    remaining = [max(0, r.num_prompt_tokens - r.num_computed_tokens)
                 for r in pending]
    count = max(1, len(pending))
    bands = (
        sum(n <= 64 for n in remaining),
        sum(64 < n <= 256 for n in remaining),
        sum(256 < n <= 1024 for n in remaining),
        sum(n > 1024 for n in remaining),
    )
    work = max(1, sum(remaining))
    waiting_work = sum(max(0, r.num_prompt_tokens - r.num_computed_tokens)
                       for r in waiting if not r.output_token_ids)
    running_work = sum(max(0, r.num_prompt_tokens - r.num_computed_tokens)
                       for r in running if not r.output_token_ids)
    now = scheduler.arrival_clock()
    ages = [max(0., now - r.arrival_time) for r in pending]
    contexts = [r.num_computed_tokens for r in running if r.output_token_ids]
    extra = np.array((
        *(n / count for n in bands),
        sum(n <= 64 for n in remaining) / count,
        sum(n <= 256 for n in remaining) / count,
        waiting_work / work,
        running_work / work,
        math.log1p(sum(ages) / count) / math.log1p(20),
        math.log1p(max(ages, default=0.)) / math.log1p(20),
        sum(age >= 1. for age in ages) / count,
        sum(age >= 2. for age in ages) / count,
        (sum(contexts) / len(contexts) if contexts else 0.) / 8176,
        max(contexts, default=0) / 8176,
    ), dtype=np.float32)
    return np.concatenate((base, extra))


def select_action(rows, budget, chunk, free, total, block_size,
                  concurrency, max_model_len, *, options, **kwargs):
    assert options['mode'] == 'fixed'
    chosen_budget, chosen_chunk = ACTIONS[options['standalone_action']]
    if chosen_budget == 0:
        decodes = sum(r.running and r.decode for r in rows)
        chosen_budget = max(1, decodes) if decodes else budget
    chosen_budget = min(chosen_budget, budget)
    chosen_chunk = min(chosen_chunk, chunk)
    args = (free, block_size, concurrency, max_model_len)
    base = project(rows, budget, chunk, *args)
    selected = project(rows, chosen_budget, chosen_chunk, *args)
    return dict(budget=chosen_budget, chunk=chosen_chunk, base=base,
                selected=selected, candidates=1,
                changed=base is not None and
                (selected is None or selected.tokens != base.tokens),
                rejected_guard=0, rejected_saving=0,
                rejected_infeasible=int(selected is None), threshold=None,
                features={'independent_rl_action': options['standalone_action']},
                candidate_diagnostics=[])


def reward_of(round_row):
    step_s = round_row['step_ms'] / 1000.
    ages = round_row['rl_pending_ages_s']
    scheduled = round_row['scheduled']
    skipped_decodes = sum(r['running'] and r['outputs'] > 0 and
                          r['rid'] not in scheduled for r in round_row['state'])
    overdue = sum(age >= 2. for age in ages)
    decode_tail = round_row['rl_decode_count'] * max(0., round_row['step_ms'] - 200.) / 200.
    return -step_s * (len(ages) + 1.5 * overdue + .5 * skipped_decodes
                      + .2 * decode_tail)


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(STATE_SIZE, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, len(ACTIONS)),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, state):
        return self.layers(state)


class DQNPolicy(previous.DQNPolicy):
    state_of = staticmethod(state_of)
    reward_of = staticmethod(reward_of)
    select_action = staticmethod(select_action)

    def __init__(self, *, training=False, seed=1017, weights=None):
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.training = training
        self.rng = random.Random(seed)
        self.network = QNetwork()
        if weights is not None:
            self.network.load_state_dict(weights)
        self.network.train(training)
        self.target = QNetwork() if training else None
        self.optimizer = (torch.optim.Adam(self.network.parameters(), lr=.0003)
                          if training else None)
        if training:
            self.target.load_state_dict(self.network.state_dict())
            self.target.eval()
        self.replay = deque(maxlen=100000) if training else None
        self.steps = 0
        self.updates = 0
        self.action_counts = [0] * len(ACTIONS)
