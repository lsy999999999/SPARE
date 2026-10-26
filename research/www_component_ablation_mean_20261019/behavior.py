"""Passive same-stream action/guard audit; no change to the simulated policy."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

from research.www_component_ablation_mean_20261019 import complete, run as pilot
from research.v1_budget_20260922.policy import State, project

ROOT = complete.ROOT
OUT = ROOT / 'artifacts/www_component_behavior_20261020_r1'
ARMS = ('spare', 'srp', 'constant1', 'drop_candidate1', 'drop_candidate2',
        'drop_candidate3', 'drop_candidate4', 'no_finish_guard', 'no_decode_guard')
RATES = (2., 1., 3., 4.)
CANDIDATES = {(256, 256): 1, (512, 256): 2, (1024, 64): 3, (512, 64): 4}
READ, WRITE, DIGEST = complete.READ, complete.WRITE, complete.DIGEST
BASE = pilot.old.prior.base


class Observer:
    def __init__(self, arm):
        self.arm = arm
        self.counts = Counter()
        self.events = []

    def inspect(self, row, at_ms, options):
        if not 20000 <= at_ms < 60000:
            return
        c = self.counts
        c['rounds'] += 1
        c['preemptions'] += row['preemptions']
        c['recomputed_tokens'] += row['recomputed_tokens']
        c['free_blocks_sum'] += row['free_before']
        c['executed_tokens'] += sum(row['scheduled'].values())
        state = row['state']
        c['pending_prefill_tokens_sum'] += sum(max(0, r['prompt']-r['computed'])
                                              for r in state if r['outputs'] == 0)
        baseline = row['baseline']
        if baseline is None:
            c['recovery_rounds'] += 1
            return
        c['feasible_base_rounds'] += 1
        index = {r['rid']: r for r in state}
        finish = {rid: quota for rid, quota in baseline.items()
                  if index[rid]['outputs'] == 0
                  and quota == index[rid]['tokens']-index[rid]['computed']}
        decode = {rid: quota for rid, quota in baseline.items()
                  if index[rid]['outputs'] > 0
                  and index[rid]['tokens']-index[rid]['computed'] == 1}
        for kind, protected in (('finish', finish), ('decode', decode)):
            delays = sum(row['scheduled'].get(rid, 0) < quota
                         for rid, quota in protected.items())
            c[f'{kind}_eligible_rounds'] += bool(protected)
            c[f'{kind}_eligible_instances'] += len(protected)
            c[f'{kind}_conflict_rounds'] += bool(delays)
            c[f'{kind}_conflict_instances'] += delays
        c['intervention_rounds'] += row['scheduled'] != baseline
        c['deferred_tokens'] += sum(max(0, v-row['scheduled'].get(rid, 0))
                                   for rid, v in baseline.items())
        c['deferred_request_rounds'] += sum(row['scheduled'].get(rid, 0) < v
                                           for rid, v in baseline.items())
        c['saved_blocks'] += row['baseline_growth_blocks']-row['actual_growth_blocks']
        if self.arm == 'srp':
            assert row['scheduled'] == baseline
            c['selected_base_rounds'] += 1
            return

        active = []
        finish_guard = options.get('protect_finish', True)
        decode_guard = options.get('protect_decode', True)
        may_select = row['baseline_growth_blocks'] > 0 and len(state) >= 2
        for d in row['candidate_diagnostics']:
            number = CANDIDATES[d['budget'], d['chunk']]
            c[f'candidate{number}_evaluations'] += 1
            tokens = d['tokens']
            if tokens is None:
                c[f'candidate{number}_infeasible'] += 1
                continue
            f_conflict = any(tokens.get(rid, 0) < v for rid, v in finish.items())
            d_conflict = any(tokens.get(rid, 0) < v for rid, v in decode.items())
            guard = not (finish_guard and f_conflict or decode_guard and d_conflict)
            assert guard == d['guard']
            admissible_saving = d['acceptance_saving'] >= row['threshold'] and may_select
            active.append((number, tokens, f_conflict, d_conflict, admissible_saving))
            c[f'candidate{number}_saving_eligible'] += admissible_saving
            c[f'candidate{number}_guard_rejections'] += not guard
        c['finish_potential_rounds'] += any(f and s and not (decode_guard and d)
                                          for _, _, f, d, s in active)
        c['decode_potential_rounds'] += any(d and s and not (finish_guard and f)
                                          for _, _, f, d, s in active)

        def select(f_guard, d_guard):
            for number, tokens, f, d, saving in active:
                if saving and not (f_guard and f or d_guard and d):
                    return number, tokens
            return 0, baseline

        selected, tokens = select(finish_guard, decode_guard)
        assert tokens == row['scheduled'], 'Passive selector disagrees with executed plan'
        c[f'selected_candidate{selected}_rounds' if selected else 'selected_base_rounds'] += 1
        for kind, flags in (('finish', (not finish_guard, decode_guard)),
                            ('decode', (finish_guard, not decode_guard))):
            alternate, alternate_tokens = select(*flags)
            changed = alternate_tokens != tokens
            c[f'{kind}_toggle_changed_rounds'] += changed
            if changed and len(self.events) < 8:
                self.events.append(dict(at_ms=at_ms, kind=kind, selected_candidate=selected,
                                        alternate_candidate=alternate, threshold=row['threshold'],
                                        free_blocks=row['free_before'],
                                        base_growth=row['baseline_growth_blocks'],
                                        selected_tokens=tokens, alternate_tokens=alternate_tokens,
                                        finish_quotas=finish, decode_quotas=decode))
        if finish_guard:
            assert all(row['scheduled'].get(rid, 0) >= v for rid, v in finish.items())
        if decode_guard:
            assert all(row['scheduled'].get(rid, 0) >= v for rid, v in decode.items())


def prepare():
    samples = complete.prepare()
    design = dict(samples=samples, arms=ARMS, rates=RATES, kv_blocks=pilot.KVS,
                  windows_per_source=24, behavioral_interval_ms=[20000, 60000],
                  arrival_measurement_interval_ms=[20000, 60000],
                  latency_follow_to_completion=True, gpu_execution=False,
                  overhead_in_simulated_clock=False,
                  parent_design_sha256=DIGEST(complete.OUT / 'design.json'),
                  policy_sha256=DIGEST(Path(pilot.policy.__file__)),
                  runner_sha256=DIGEST(Path(__file__)),
                  counter_definition='Round-based conflicts relative to feasible baseline; '
                  'potential requires savings and other active guard; toggle changes reselect '
                  'first admissible candidate on identical state; window rates exclude zero denominators')
    normalized = json.loads(json.dumps(design))
    path = OUT / 'design.json'
    if path.exists():
        assert READ(path) == normalized, 'Frozen behavior design changed'
    else:
        WRITE(path, normalized)
    return samples


def replay(sample, kv, arm, rate):
    engine, options = pilot.old.prior.ARMS[arm]
    reqs = BASE.generate(sample, rate)
    seed = int.from_bytes(hashlib.sha256(sample['name'].encode()).digest()[:4], 'big')
    observer = Observer(arm)
    captured = {}

    def make(*args, **kwargs):
        scheduler, clock = BASE.ORIGINAL_MAKE(*args, **kwargs)
        captured['scheduler'] = scheduler
        original_schedule = scheduler.schedule
        before = {}
        if arm == 'srp':
            original_order = scheduler._order

            def order():
                original_order()
                manager = scheduler.kv_cache_manager
                tables = manager.coordinator.single_type_managers[0].req_to_blocks
                rows = [State(r.request_id, r.num_computed_tokens, r.num_tokens,
                              r.num_prompt_tokens, len(r.output_token_ids),
                              len(tables.get(r.request_id, ())), running)
                        for running, queue in ((True, scheduler.running), (False, scheduler.waiting))
                        for r in queue]
                before['rows'] = rows
                before['base'] = project(rows, scheduler.budget, scheduler.chunk,
                                         manager.block_pool.get_num_free_blocks(),
                                         scheduler.block_size, scheduler.max_num_running_reqs,
                                         scheduler.max_model_len)
            scheduler._order = order

        def schedule():
            at_ms = clock[0]
            result = original_schedule()
            row = dict(scheduler.round_rows[-1])
            if arm == 'srp':
                base = before['base']
                row.update(state=[vars(r) for r in before['rows']],
                           baseline=base.tokens if base is not None else None,
                           baseline_growth_blocks=base.growth if base is not None else None,
                           actual_growth_blocks=row['free_before']-row['free_after'])
            observer.inspect(row, at_ms, options)
            return result
        scheduler.schedule = schedule
        return scheduler, clock

    class Clock:
        def estimate(self, batch):
            duration, details = BASE.TIMING.estimate(batch)
            captured['scheduler'].round_rows[-1]['step_ms'] = duration
            return duration, details

    with patch.object(BASE.SIM, 'make_scheduler', make):
        result = BASE.SIM.simulate(reqs, arm=engine, capacity=kv, budget=1024, chunk=256,
                                   policy_options=options, history=BASE.HISTORY[sample['source']],
                                   seed=seed, fitted_timing=Clock(), capture_probe=True,
                                   max_wall_s=180)
    assert result['metrics']['completed'] == len(reqs)
    metrics = BASE.probe(result)
    parity_sources = []
    for archive in (complete.OUT, complete.PREVIOUS, pilot.OUT, pilot.old.OUT):
        path = archive / 'rates' / pilot.old.rate_name(sample['source'], kv, arm, rate)
        if path.exists():
            prior = next((r for r in READ(path)['rows'] if r['name'] == sample['name']), None)
            if prior is not None:
                assert prior['valid'] and prior['trajectory_sha256'] == metrics['trajectory_sha256'], (
                    sample['name'], kv, arm, rate, 'trajectory mismatch')
                assert all(prior[k] == metrics[k] for k in
                           ('mean_ttft_ms', 'p99_tbt_ms', 'backlog_drift', 'preemptions', 'recomputed_tokens'))
                parity_sources.append(str(path.relative_to(ROOT)))
                break
    observed = [r for r in reqs if 20000 <= r['arrival_ms'] < 60000]
    return dict(source=sample['source'], name=sample['name'], day=sample['day'],
                capacity=kv, arm=arm, budget=1024, rate=rate, valid=True, **metrics,
                counters=dict(observer.counts), events=observer.events,
                request_prompt_mean=sum(r['prompt'] for r in observed)/len(observed),
                request_output_mean=sum(r['output'] for r in observed)/len(observed),
                parity_sources=parity_sources)


def group(task):
    source, kv, arm, rate, samples = task
    path = OUT / 'rates' / pilot.old.rate_name(source, kv, arm, rate)
    if path.exists():
        return READ(path)
    checkpoint = OUT / 'checkpoints' / path.name
    rows = READ(checkpoint)['rows'] if checkpoint.exists() else []
    found = {r['name'] for r in rows}
    assert found <= {s['name'] for s in samples} and len(found) == len(rows)
    for sample in samples:
        if sample['name'] not in found:
            rows.append(replay(sample, kv, arm, rate))
            WRITE(checkpoint, dict(rows=rows))
    rows.sort(key=lambda r: r['name'])
    result = dict(rows=rows, summary=complete.confirm.summary(rows, rate))
    WRITE(path, result)
    return result


def smoke():
    samples = prepare()
    pilot.initialize()
    checks = []
    for source, kv, arm, rate in (
            ('code', 1024, 'spare', 2.), ('conversation', 2048, 'srp', 2.),
            ('code', 1024, 'no_finish_guard', 2.),
            ('conversation', 1024, 'drop_candidate4', 2.)):
        sample = next(s for s in samples if s['source'] == source and s['window'] == 2)
        row = replay(sample, kv, arm, rate)
        assert row['parity_sources']
        checks.append(row)
    WRITE(OUT / 'smoke.json', dict(passed=True, checks=checks))
    print('Behavior smoke: four passive replays match archived metrics and trajectories', flush=True)


def run(workers):
    samples = prepare()
    assert READ(OUT / 'smoke.json')['passed']
    tasks = [(source, kv, arm, rate, [s for s in samples if s['source'] == source])
             for rate in RATES for source in pilot.SOURCES for kv in pilot.KVS for arm in ARMS]
    pending = [t for t in tasks if not (OUT / 'rates' /
               pilot.old.rate_name(t[0], t[1], t[2], t[3])).exists()]
    done, errors, start = len(tasks)-len(pending), [], time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=pilot.initialize) as pool:
        futures = {pool.submit(group, t): t[:4] for t in pending}
        for f in as_completed(futures):
            try:
                f.result()
                done += 1
            except Exception as exc:
                errors.append(dict(task=futures[f], error=repr(exc)))
            status = dict(completed=done, total=len(tasks), windows_per_group=24,
                          errors=errors, elapsed_s=round(time.monotonic()-start, 1))
            WRITE(OUT / 'status.json', status)
            if done % 6 == 0 or errors:
                print(status, flush=True)
    if errors:
        raise RuntimeError(errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    smoke() if args.smoke else run(args.workers)
