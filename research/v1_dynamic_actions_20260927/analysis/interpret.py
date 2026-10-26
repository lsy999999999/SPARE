"""Offline reconstruction only; no GPU use or modification of frozen sources."""
import csv
import gzip
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.v1_dynamic_actions_20260927.common import ROOT, HERE, OUT, read, write, digest, sources
from research.v1_dynamic_actions_20260927.audit import audit
from research.v1_budget_20260922.policy import State, project


def rounds_for(result):
    p = OUT / result['rounds_path']
    assert digest(p) == result['rounds_sha256']
    with gzip.open(p, 'rt') as f:
        return json.load(f)


import json


def features(row, plan, growth):
    details = []
    for r in row['state']:
        if r['outputs'] != 0:
            continue
        rid = r['rid']
        remaining = max(0, r['prompt'] - r['computed'])
        base = row['baseline'].get(rid, 0)
        chosen = plan.get(rid, 0)
        details.append(dict(rid=rid, remaining_before=remaining, base_tokens=base,
                            chosen_tokens=chosen, withheld=max(0, base-chosen),
                            remaining_base=max(0, remaining-base),
                            remaining_after=max(0, remaining-chosen),
                            partial_after=0 < r['computed']+chosen < r['prompt'],
                            newly_partial=r['computed'] == 0 and 0 < chosen < r['prompt']))
    affected = [r for r in details if r['withheld'] > 0]
    loss = sum(max(0., math.log((r['remaining_after']+16)/(r['remaining_base']+16))) for r in details)
    saving = row['baseline_growth_blocks'] - growth
    recompute_withheld = sum(max(0, row['baseline'].get(r['rid'], 0)-plan.get(r['rid'], 0))
                            for r in row['state'] if r['outputs'] > 0)
    summary = dict(pending=len(details), affected=len(affected), withheld_tokens=sum(r['withheld'] for r in affected),
        affected_near_256=sum(0 < r['remaining_base'] <= 256 for r in affected),
        completions=sum(r['remaining_before'] > 0 and r['remaining_after'] == 0 for r in details),
        base_completions=sum(r['remaining_before'] > 0 and r['remaining_base'] == 0 for r in details),
        partial_after=sum(r['partial_after'] for r in details), newly_partial=sum(r['newly_partial'] for r in details),
        remaining_after_total=sum(r['remaining_after'] for r in details),
        min_affected_remaining_base=min((r['remaining_base'] for r in affected), default=None),
        min_affected_remaining_after=min((r['remaining_after'] for r in affected), default=None),
        saving=saving, threshold=row['threshold'], saving_over_n=saving/row['threshold'] if row['threshold'] else None,
        loss=loss, score=loss/saving if saving > 0 else None, recompute_withheld=recompute_withheld)
    return summary, details


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    design = read(OUT/'design.json')
    assert sources() == design['source_hashes']
    assert digest(HERE/'PROTOCOL_CN.md') == design['protocol_sha256']
    for case, sha in design['input_hashes'].items():
        assert digest(OUT/f'inputs/{case}.json') == sha
    audits, branch_features, branch_details, formal = [], [], [], []
    phase_counts = Counter()
    for group in design['groups']:
        folder = OUT/'groups'/group['key']
        anchor = read(folder/'runs/anchor.json')
        anchor_rows = rounds_for(anchor)
        for target in read(folder/'selected_states.json')['states']:
            row = anchor_rows[target['round']]
            assert row['state_fingerprint'] == target['fingerprint']
            for action in target['actions']:
                feat, detail = features(row, action['tokens'], action['growth'])
                key = dict(group=group['key'], target=target['round'], arm=action['arm'], aliases='/'.join(action['aliases']),
                           budget=action['budget'], chunk=action['chunk'], free=row['free_before'])
                branch_features.append(dict(**key, **feat))
                branch_details.append(dict(**key, **feat, requests=detail))
        for p in sorted((folder/'runs').glob('*.json')):
            result = read(p)
            rows = rounds_for(result)
            audited = audit(result, rows)
            phase_counts[result['phase']] += 1
            audits.append(dict(path=str(p.relative_to(OUT)), **audited))
            if result['phase'] == 'branch':
                index = result['target']
                assert all(rows[i]['state_fingerprint'] == anchor_rows[i]['state_fingerprint'] for i in range(index+1))
                assert all(rows[i]['scheduled'] == anchor_rows[i]['scheduled'] for i in range(index))
                assert all(rows[i]['features']['arrival_time_s'] == anchor_rows[i]['features']['arrival_time_s'] for i in range(index+1))
                action = next(a for t in read(folder/'selected_states.json')['states'] if t['round'] == index
                              for a in t['actions'] if a['arm'] == result['arm'])
                assert rows[index]['scheduled'] == action['tokens']
                branch_ms = (rows[index]['probe_decision_wall']-result['start_wall'])*1000
                pending = [r['rid'] for r in rows[index]['state'] if r['outputs'] == 0]
                area = sum(result['clocks'][rid]['first_ms']-branch_ms for rid in pending)
                assert math.isclose(area, result['residual_waiting_area_ms'], rel_tol=1e-12)
            if result['phase'] != 'formal':
                continue
            feats = [features(row, row['scheduled'], row['actual_growth_blocks'])[0] for row in rows if row['intervened']]
            chunks = Counter(row['chunk'] for row in rows if row['intervened'])
            key = dict(group=group['key'], shape=result['shape'], arm=result['arm'], repeat=result['repeat'])
            summary = dict(**key, rounds=len(rows), interventions=len(feats), chunk_counts=dict(chunks),
                decision_ms_total=result['metrics']['decision_ms_total'], schedule_ms_total=result['metrics']['schedule_ms_total'],
                elapsed_ms=result['elapsed_ms'],
                decision_pending_area_per_request_ms=sum(row['decision_ms']*sum(r['outputs']==0 for r in row['state']) for row in rows)/64,
                saving_mean=st.fmean(f['saving'] for f in feats) if feats else None,
                saving_over_n_mean=st.fmean(f['saving_over_n'] for f in feats) if feats else None)
            for name in ['affected', 'withheld_tokens', 'affected_near_256', 'completions', 'base_completions',
                         'partial_after', 'newly_partial', 'recompute_withheld']:
                summary[name+'_sum'] = sum(f[name] for f in feats)
            summary['zero_pending_loss_interventions'] = sum(f['loss'] == 0 for f in feats)
            summary['recompute_withheld_interventions'] = sum(f['recompute_withheld'] > 0 for f in feats)
            no_progress = []
            equivalent_k256 = 0
            streak = max_streak = 0
            for row in rows:
                pending = [r for r in row['state'] if r['outputs'] == 0]
                stalled = bool(row['intervened'] and pending and
                               sum(row['scheduled'].get(r['rid'], 0) for r in pending) == 0 and
                               sum(row['baseline'].get(r['rid'], 0) for r in pending) > 0)
                streak = streak+1 if stalled else 0
                max_streak = max(max_streak, streak)
                if stalled:
                    no_progress.append(row)
                if row['intervened'] and row['chunk'] != 256:
                    plan = project([State(**r) for r in row['state']], row['budget'], 256,
                                   row['free_before'], 16, 32, 8176)
                    equivalent_k256 += plan is not None and plan.tokens == row['scheduled']
            summary.update(no_prefill_progress_interventions=len(no_progress),
                max_no_prefill_progress_streak=max_streak,
                no_prefill_progress_step_ms=sum(r['step_ms'] for r in no_progress),
                no_prefill_progress_waiting_area_per_request_ms=sum(
                    r['step_ms']*sum(s['outputs']==0 for s in r['state']) for r in no_progress)/64,
                non256_chunk_same_plan_at_256=equivalent_k256)
            formal.append(summary)
        print('audited', group['key'], len(audits), flush=True)
    write(OUT/'analysis/local_reaudit.json', dict(passed=True, frozen_hashes_passed=True,
        branch_prefixes_and_areas_passed=True, episodes=len(audits), phase_counts=dict(phase_counts),
        rounds=sum(r['rounds'] for r in audits), audits=audits))
    write(OUT/'analysis/branch_features.json', branch_details)
    csv_write(OUT/'analysis/branch_features.csv', branch_features)
    write(OUT/'analysis/formal_mechanisms.json', formal)
    csv_write(OUT/'analysis/formal_mechanisms.csv', [{**r, 'chunk_counts': json.dumps(r['chunk_counts'])} for r in formal])
    print(dict(passed=True, episodes=len(audits), phases=dict(phase_counts)), flush=True)


if __name__ == '__main__':
    main()
