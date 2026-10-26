"""Capacity-aware DQN that selects among the existing SPARE threshold actions."""
from __future__ import annotations

from collections import deque
import math
import random

import numpy as np
import torch
from torch import nn


ACTIONS = ('spare', 1, 16, 64, 256, 1024)
STATE_SIZE = 8


def state_of(scheduler):
    total = scheduler.kv_cache_config.num_blocks - 1
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    waiting = list(scheduler.waiting)
    running = list(scheduler.running)
    pending = [r for r in waiting + running if not r.output_token_ids]
    decodes = sum(bool(r.output_token_ids) for r in running)
    remaining = [max(0, r.num_prompt_tokens - r.num_computed_tokens)
                 for r in pending]
    return np.array((
        math.log2(max(total, 1)) / 14.,
        free / max(total, 1),
        math.log1p(free) / math.log1p(12032),
        min(len(pending), 32) / 32.,
        min(len(waiting), 32) / 32.,
        min(decodes, 32) / 32.,
        min(sum(remaining), 32768) / 32768.,
        min(max(remaining, default=0), 8192) / 8192.,
    ), dtype=np.float32)


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(STATE_SIZE, 64), nn.ReLU(),
                                    nn.Linear(64, 64), nn.ReLU(),
                                    nn.Linear(64, len(ACTIONS)))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, state):
        return self.layers(state)


class DQNPolicy:
    def __init__(self, *, training=False, seed=1017, weights=None):
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.training = training
        self.rng = random.Random(seed)
        self.network = QNetwork()
        if weights is not None:
            self.network.load_state_dict(weights)
        self.network.eval() if not training else self.network.train()
        self.target = QNetwork() if training else None
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.0005) if training else None
        if training:
            self.target.load_state_dict(self.network.state_dict())
            self.target.eval()
        self.replay = deque(maxlen=100000) if training else None
        self.steps = 0
        self.updates = 0
        self.action_counts = [0] * len(ACTIONS)

    def act(self, state):
        epsilon = max(.05, .25 - .20 * self.steps / 1000000) if self.training else 0.
        if self.training and self.rng.random() < epsilon:
            action = self.rng.randrange(len(ACTIONS))
        else:
            with torch.no_grad():
                values = self.network(torch.from_numpy(state).unsqueeze(0))
            action = int(values.argmax(dim=1).item())
        self.action_counts[action] += 1
        return action

    def observe(self, state, action, reward, next_state, done=False):
        if not self.training:
            return
        self.replay.append((state.copy(), action, float(reward), next_state.copy(), done))
        self.steps += 1
        if len(self.replay) < 1024 or self.steps % 8:
            return
        batch = self.rng.sample(self.replay, 64)
        states = torch.from_numpy(np.stack([row[0] for row in batch]))
        actions = torch.tensor([row[1] for row in batch], dtype=torch.int64)
        rewards = torch.tensor([row[2] for row in batch], dtype=torch.float32)
        next_states = torch.from_numpy(np.stack([row[3] for row in batch]))
        done_mask = torch.tensor([row[4] for row in batch], dtype=torch.float32)
        predicted = self.network(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            best_actions = self.network(next_states).argmax(dim=1, keepdim=True)
            next_values = self.target(next_states).gather(1, best_actions).squeeze(1)
            target = rewards + .98 * (1 - done_mask) * next_values
        loss = nn.functional.smooth_l1_loss(predicted, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 10.)
        self.optimizer.step()
        self.updates += 1
        if self.updates % 500 == 0:
            self.target.load_state_dict(self.network.state_dict())


def reward_of(round_row):
    step_ms = round_row['step_ms']
    waiting = round_row['rl_pending_first_tokens']
    decodes = round_row['rl_decode_count']
    response = waiting * step_ms / 1000.
    decode_tail = .25 * decodes * max(0., step_ms - 200.) / 1000.
    queue_pressure = .1 * max(0, waiting - 8) * step_ms / 1000.
    return -min(response + decode_tail + queue_pressure, 10.)
