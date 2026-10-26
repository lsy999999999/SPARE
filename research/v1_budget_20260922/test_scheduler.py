"""CPU tests use the installed native V1 scheduler and physical block manager."""
import random
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from vllm import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request, RequestStatus

from research.v1_budget_20260922.scheduler import BudgetScheduler
from research.v1_budget_20260922.policy import State, choose, project


def scheduler(method='budget', blocks=96, budget=256, chunk=256, cls=BudgetScheduler):
    config = NS(scheduler_config=NS(max_num_seqs=64, max_num_batched_tokens=2048,
                    max_model_len=12288, policy='fcfs', long_prefill_token_threshold=chunk,
                    chunked_prefill_enabled=True),
                cache_config=NS(num_gpu_blocks=blocks, block_size=16,
                    enable_prefix_caching=False, prefix_caching_hash_algo='builtin'),
                lora_config=None, kv_events_config=None, kv_transfer_config=None,
                speculative_config=None, model_config=None,
                parallel_config=NS(data_parallel_rank=0, pipeline_parallel_size=1))
    kv = KVCacheConfig(blocks, [], [KVCacheGroupSpec(['layer'],
          FullAttentionSpec(16, 1, 16, torch.float16, False))])
    with patch('vllm.v1.core.sched.scheduler.compute_encoder_budget', return_value=(0, 0)):
        s = cls(config, kv, NS(grammar_bitmask=lambda *args: None))
    if cls is BudgetScheduler:
        s.configure(method, budget, chunk, 8)
    else:
        s.max_num_scheduled_tokens = budget
        s.max_num_running_reqs = 8
    return s


def add(s, sizes):
    for i, size in enumerate(sizes):
        s.add_request(Request(str(i), [100]*size, None, None, None,
                             SamplingParams(max_tokens=24), None, None, arrival_time=float(i)))


def advance(s, output, target=24):
    done = []
    for rid in output.num_scheduled_tokens:
        r = s.requests[rid]
        if r.num_computed_tokens == r.num_tokens:
            r.append_output_token_ids(100)
        if len(r.output_token_ids) >= target:
            done.append(rid)
    s.finish_requests(done, RequestStatus.FINISHED_LENGTH_CAPPED)


class NativeTests(unittest.TestCase):
    def test_v1_incremental_allocation(self):
        for chunk in (64, 256):
            s = scheduler('native', chunk=chunk)
            add(s, [1024])
            out = s.schedule()
            self.assertEqual(out.num_scheduled_tokens, {'0': chunk})
            self.assertEqual(len(s.kv_cache_manager.get_block_ids('0')[0]), chunk//16)

    def test_native_differential_including_preemption(self):
        for blocks in (48, 96):
            a, b = scheduler('native', blocks), scheduler(blocks=blocks, cls=Scheduler)
            add(a, [257, 129, 385, 99, 400, 12])
            add(b, [257, 129, 385, 99, 400, 12])
            for _ in range(3000):
                x, y = a.schedule(), b.schedule()
                self.assertEqual(x.num_scheduled_tokens, y.num_scheduled_tokens)
                self.assertEqual(a.kv_cache_manager.block_pool.get_num_free_blocks(),
                                 b.kv_cache_manager.block_pool.get_num_free_blocks())
                advance(a, x)
                advance(b, y)
                if not a.requests:
                    break
            self.assertFalse(a.requests)
            self.assertFalse(b.requests)

    def test_random_closed_loops_and_shadow(self):
        interventions = preemptions = verified = recomputed = 0
        for seed in range(12):
            rng = random.Random(seed)
            sizes = [rng.randrange(17, 650) for _ in range(12)]
            for method in ('budget', 'shadow', 'srp'):
                s = scheduler(method, blocks=(48, 96, 160)[seed % 3])
                add(s, sizes)
                for _ in range(4000):
                    out = s.schedule()  # audits normal plans against native
                    row = s.round_rows[-1]
                    interventions += row['intervened']
                    preemptions += row['preemptions']
                    verified += row['verified']
                    recomputed += row['recomputed_tokens']
                    self.assertGreaterEqual(row['free_after'], 0)
                    self.assertGreaterEqual(row['schedule_ms'], row['decision_ms']+row['native_ms'])
                    if method == 'shadow':
                        self.assertFalse(row['intervened'])
                    advance(s, out)
                    if not s.requests:
                        break
                self.assertFalse(s.requests, (seed, method))
        self.assertGreater(interventions, 0)
        self.assertGreater(preemptions, 0)
        self.assertGreater(recomputed, 0)
        self.assertGreater(verified, 100)

    def test_shadow_executes_srp(self):
        a, b = scheduler('shadow'), scheduler('srp')
        for s in (a, b):
            add(s, [513, 700, 251, 23])
        while a.requests:
            x, y = a.schedule(), b.schedule()
            self.assertEqual(x.num_scheduled_tokens, y.num_scheduled_tokens)
            advance(a, x)
            advance(b, y)
        self.assertFalse(b.requests)

    def test_completion_and_decode_protection(self):
        rows = [State('d', 255, 256, 200, 56, 16, True),
                State('finish', 0, 100, 100, 0, 0, False),
                State('long', 0, 1024, 1024, 0, 0, False)]
        selected = choose(rows, 256, 256, 95, 96, 16, 8, 12288)
        self.assertEqual(selected['selected'].tokens['d'], 1)
        self.assertEqual(selected['selected'].tokens['finish'], 100)
        self.assertIsNone(project([State('x', 512, 513, 512, 1, 32, True)],
                                  256, 256, 0, 16, 8, 12288))


if __name__ == '__main__':
    unittest.main()
