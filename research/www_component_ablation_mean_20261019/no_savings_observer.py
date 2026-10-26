"""Passive audit for an actually disabled savings condition."""
from collections import Counter

from research.www_component_ablation_mean_20261019.behavior import CANDIDATES


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
            admissible_saving = may_select and (options.get('disable_saving_condition', False)
                                              or d['acceptance_saving'] >= row['threshold'])
            active.append((number, tokens, f_conflict, d_conflict, admissible_saving))
            c[f'candidate{number}_acceptance_eligible'] += admissible_saving
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
        if selected:
            selected_diagnostic = next(d for d in row['candidate_diagnostics']
                                       if CANDIDATES[d['budget'], d['chunk']] == selected)
            saving = selected_diagnostic['saving']
            c['selected_positive_saving_rounds'] += saving > 0
            c['selected_zero_saving_rounds'] += saving == 0
            c['selected_negative_saving_rounds'] += saving < 0
            c['selected_below_dynamic_threshold_rounds'] += saving < row['threshold']
            c['selected_noop_rounds'] += tokens == baseline
            if saving <= 0 and len(self.events) < 8:
                self.events.append(dict(at_ms=at_ms, kind='nonpositive_saving_accepted',
                                        selected_candidate=selected, saving_blocks=saving,
                                        unused_dynamic_threshold=row['threshold'],
                                        selected_tokens=tokens, baseline_tokens=baseline))
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

