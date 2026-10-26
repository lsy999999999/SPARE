import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from research.v1_budget_20260922 import run


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.manifest = dict(blocks=run.make_blocks(), aa_max_range_over_mean=.20,
                             claim='test', implementation='test', hardware={'name':'fixture'})

    def test_disjoint_inputs_matrix_and_hidden_lengths(self):
        # make_blocks verifies hashes and rejects overlap with the old matrix.
        self.assertEqual(len(self.manifest['blocks']), 6)
        selected = {regime: {method: run.CONFIGS[2] for method in run.METHODS} for regime in run.REGIMES}
        for phase, count in [('aa', 18), ('tune', 54), ('compare', 180)]:
            jobs = [j for regime in run.REGIMES for j in run.jobs(self.manifest, phase, regime, selected)]
            self.assertEqual(len(jobs), count)
            self.assertEqual(len({j['key'] for j in jobs}), count)
            for j in jobs:
                reqs = run.requests_for(self.manifest['blocks'][j['block']], j['regime'])
                self.assertTrue(all(r['prompt']+r['output']<=12288 for r in reqs))
                self.assertEqual(reqs[0]['arrival_ms'], 0)
                if j['arm']=='srp_matched':
                    self.assertEqual(j['method'], 'srp')
                    self.assertEqual(j['config'], selected[j['regime']]['budget'])

    def test_input_or_failure_or_archive_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(run, 'OUT', Path(directory)):
            paths = []
            for regime in run.REGIMES:
                for j in run.jobs(self.manifest, 'aa', regime):
                    reqs = run.requests_for(self.manifest['blocks'][j['block']], regime)
                    trace = run.OUT/'rounds'/(j['key']+'.json.gz')
                    trace.parent.mkdir(exist_ok=True)
                    with gzip.open(trace, 'wt') as f:
                        json.dump([dict(round=0, free_after=90, total_blocks=96,
                            scheduled={reqs[0]['id']:1}, budget=256, decision_ms=.01,
                            native_ms=.1, schedule_ms=.2, step_ms=1., preemptions=0,
                            recomputed_tokens=0, intervened=False, verified=True, candidates=0)], f)
                    clocks = {q['id']: dict(arrival_ms=q['arrival_ms'], first_ms=q['arrival_ms']+100,
                         finish_ms=q['arrival_ms']+100+q['output']-1, tokens=q['output'],
                         token_times_ms=[q['arrival_ms']+100+i for i in range(q['output'])]) for q in reqs}
                    r = dict(job=j, requests=reqs, status='complete', completed=len(reqs),
                             request_count=len(reqs), rounds_path=str(trace.relative_to(run.OUT)),
                             rounds_sha256=run.digest(trace), clocks=clocks,
                             metrics=dict(mean_ttft_ms=100., round_count=1, preemptions=0,
                                          recomputed_tokens=0, interventions=0, verified_rounds=1, candidates=0))
                    path = run.OUT/'runs'/(j['key']+'.json')
                    run.write(path, r)
                    paths.append(path)
            self.assertEqual(len(run.load_expected(self.manifest, 'aa')), 18)
            run.check_aa(self.manifest)
            self.assertTrue(run.read(run.OUT/'aa_check.json')['passed'])
            original = run.read(paths[0])
            bad = {**original, 'status': 'failed'}
            run.write(paths[0], bad)
            with self.assertRaises(AssertionError):
                run.load_expected(self.manifest, 'aa')
            run.write(paths[0], original)
            altered = run.read(paths[0])
            altered['requests'][0]['prompt'] += 1
            run.write(paths[0], altered)
            with self.assertRaises(AssertionError):
                run.load_expected(self.manifest, 'aa')
            run.write(paths[0], original)
            trace = run.OUT/original['rounds_path']
            trace.write_bytes(b'changed')
            with self.assertRaises(AssertionError):
                run.load_expected(self.manifest, 'aa')

    def test_aa_variation_blocks_tuning(self):
        rows = [dict(job=dict(regime='queue', block='x'), metrics=dict(mean_ttft_ms=v)) for v in (100, 100, 200)]
        with tempfile.TemporaryDirectory() as directory, patch.object(run, 'OUT', Path(directory)), \
                patch.object(run, 'load_expected', return_value=rows):
            with self.assertRaisesRegex(RuntimeError, 'A/A'):
                run.check_aa(self.manifest)
            self.assertFalse(run.read(run.OUT/'aa_check.json')['passed'])

    def test_selection_uses_development_only(self):
        rows = []
        for regime in run.REGIMES:
            for j in run.jobs(self.manifest, 'tune', regime):
                cost = [3, 1, 2][run.CONFIGS.index(j['config'])]
                rows.append(dict(job=j, status='complete', metrics=dict(mean_ttft_ms=cost)))
        with tempfile.TemporaryDirectory() as directory, patch.object(run, 'OUT', Path(directory)), \
                patch.object(run, 'load_expected', return_value=rows) as load:
            selected = run.select(self.manifest)
            load.assert_called_once_with(self.manifest, 'tune')
            self.assertTrue(all(c==run.CONFIGS[1] for r in selected.values() for c in r.values()))

    def test_tuning_timeout_ranks_last_and_preserves_failure(self):
        rows = []
        for regime in run.REGIMES:
            for j in run.jobs(self.manifest, 'tune', regime):
                failed = (regime=='restricted' and j['arm']=='native'
                          and j['config']==run.CONFIGS[1] and j['block']=='dev_conversation_0')
                rows.append(dict(job=j, status='failed' if failed else 'complete',
                    metrics=dict(mean_ttft_ms=None if failed else [3., 1., 2.][run.CONFIGS.index(j['config'])])))
        with tempfile.TemporaryDirectory() as directory, patch.object(run, 'OUT', Path(directory)), \
                patch.object(run, 'load_expected', return_value=rows):
            result = run.select(self.manifest)
            self.assertEqual(result['restricted']['native'],run.CONFIGS[2])
            report=run.read(run.OUT/'tuning_report.json')
            self.assertIsNone(report['scores']['restricted']['native'][1])
            self.assertEqual(len(report['timeouts']),1)
            rows[:] = [dict(r, status='failed', metrics=dict(mean_ttft_ms=None))
                       if r['job']['regime']=='restricted' and r['job']['arm']=='budget' else r for r in rows]
            with self.assertRaisesRegex(RuntimeError, 'No configuration completed'):
                run.select(self.manifest)

    def test_timeout_exits_engine_and_continues_in_fresh_process(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(run,'OUT',Path(directory)), \
                patch.object(run,'initialize',return_value={'hardware': {'gpu': '4'}}), \
                patch.object(run,'import_prior'), patch.object(run,'check_smoke'), \
                patch.object(run,'check_aa'), \
                patch.object(run,'select',side_effect=lambda _: run.write(run.OUT/'selection.json',{})), \
                patch.object(run,'summarize'), patch.object(run,'audit_result'), \
                patch.object(run.subprocess,'check_output',return_value=b'0'):
            run.write(run.OUT/'smoke.json',dict(complete=True))
            target = 'tune_restricted_timeout'
            following = 'tune_restricted_next'
            for phase in ('aa','tune','compare'):
                for regime in run.REGIMES:
                    key = f'{phase}_{regime}_ok'
                    if phase=='tune' and regime=='restricted':
                        continue
                    run.write(run.OUT/'runs'/(key+'.json'),dict(job={'key':key}))
            def fake_jobs(manifest,phase,regime,selection=None):
                key = target if phase=='tune' and regime=='restricted' else f'{phase}_{regime}_ok'
                return [dict(key=key),dict(key=following)] if key==target else [dict(key=key)]
            calls=[]
            def fake_run(cmd,**kwargs):
                calls.append((cmd[-4:],kwargs.get('cwd')))
                if len(calls)==1:
                    run.write(run.OUT/'runs'/(target+'.json'),dict(job={'key':target}))
                    return Mock(returncode=75)
                if len(calls)==2:
                    run.write(run.OUT/'runs'/(following+'.json'),dict(job={'key':following}))
                    return Mock(returncode=0)
                self.fail('Completed tune jobs were attempted again')
            with patch.object(run,'jobs',side_effect=fake_jobs), \
                    patch.object(run.subprocess,'run',side_effect=fake_run):
                run.orchestrate()
            self.assertEqual(len(calls),2)
            self.assertTrue(all(c[0][-2:]==['--regime','restricted'] for c in calls))

    def test_import_audits_timeout_without_touching_old_result(self):
        job = next(j for j in run.jobs(self.manifest, 'tune','restricted')
                   if j['block']=='dev_conversation_0' and j['arm']=='native')
        requests = run.requests_for(self.manifest['blocks'][job['block']], 'restricted')
        clocks = {}
        for i,r in enumerate(requests):
            count = r['output'] if i>=3 else 0
            times = [r['arrival_ms']+1+j for j in range(count)]
            clocks[r['id']] = dict(arrival_ms=r['arrival_ms'], first_ms=r['arrival_ms']+1 if count else None,
                finish_ms=times[-1] if count else None, tokens=count, token_times_ms=times)
        row = dict(round=0, free_after=90, total_blocks=96, scheduled={}, budget=512,
                   decision_ms=.01, native_ms=.1, schedule_ms=.2, step_ms=1.,
                   preemptions=1, recomputed_tokens=32, intervened=False, verified=False, candidates=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, dest = root/'old', root/'new'
            old = dict(self.manifest, schema=1, implementation='V1-budget-light-1',
                hardware=dict(name='fixture'), source_hashes={'research/x/policy.py':'same',
                    'research/x/scheduler.py':'same', 'research/x/run.py':'old'},
                model='fixture', versions={}, model_revision='fixture', configs=run.CONFIGS,
                regimes=run.REGIMES, methods=run.METHODS, arms=run.ARMS,
                repeats=3, restricted_blocks=768, timeout_s=300, claim='test',
                engine=dict(v1=True))
            current = dict(old, schema=2, implementation='V1-budget-light-1/timeout-scoring-2',
                           source_hashes={'research/x/policy.py':'same',
                               'research/x/scheduler.py':'same', 'research/x/run.py':'new'})
            old, current = json.loads(json.dumps(old)), json.loads(json.dumps(current))
            run.write(source/'manifest.json',old)
            run.write(source/'capacity.json',dict(blocks=1000))
            trace=source/'rounds'/(job['key']+'.json.gz')
            trace.parent.mkdir(parents=True,exist_ok=True)
            with gzip.open(trace,'wt') as f:
                json.dump([row],f)
            record=dict(job=job, requests=requests, clocks=clocks, completed=len(requests)-3,
                request_count=len(requests), elapsed_ms=300011., status='failed',
                error='TimeoutError: Predeclared episode deadline exceeded',
                rounds_path=str(trace.relative_to(source)), rounds_sha256=run.digest(trace),
                metrics=dict(completion_rate=(len(requests)-3)/len(requests), mean_ttft_ms=None,
                    p95_ttft_ms=None, round_count=1, preemptions=1, recomputed_tokens=32,
                    interventions=0, verified_rounds=0, candidates=0))
            run.write(source/'runs'/(job['key']+'.json'),record)
            with patch.object(run,'OUT',dest),patch.dict('os.environ',{'V1_BUDGET_IMPORT':str(source)}):
                run.import_prior(current)
                run.import_prior(current)
                self.assertEqual(run.read(dest/'migration.json')['timeout_runs'],[job['key']])
                imported=run.read(dest/'runs'/(job['key']+'.json'))
                run.audit_result(imported,allow_timeout=True)
                self.assertEqual(run.save_episode(None,None,job,requests),imported)
                with self.assertRaises(AssertionError):
                    run.audit_result(imported)
                current['source_hashes']['research/x/scheduler.py']='changed'
                with self.assertRaisesRegex(RuntimeError,'Previous source mismatch'):
                    run.import_prior(current)
            self.assertEqual(run.read(source/'runs'/(job['key']+'.json')),record)

    def test_summary_averages_repeats_inside_blocks(self):
        selected = {regime: {method: run.CONFIGS[1] for method in run.METHODS} for regime in run.REGIMES}
        rows = []
        for regime in run.REGIMES:
            for j in run.jobs(self.manifest, 'compare', regime, selected):
                rows.append(dict(job=j, status='complete',
                                 metrics=dict(mean_ttft_ms=90. if j['arm']=='budget' else 100.)))
        with tempfile.TemporaryDirectory() as directory, patch.object(run, 'OUT', Path(directory)), \
                patch.object(run, 'load_expected', return_value=rows), \
                patch.object(run, 'select', return_value=selected), \
                patch.object(run, 'check_aa'), patch.object(run, 'check_smoke'):
            run.summarize(self.manifest)
            result = run.read(run.OUT/'summary.json')
            comparison = result['comparisons']['queue']['srp']
            self.assertEqual(comparison['mean_gain_ms'], 10.)
            self.assertEqual(comparison['positive_blocks'], 4)
            self.assertEqual(len(comparison['pairs']), 4)
            self.assertEqual(comparison['bootstrap95_ms'], [10., 10.])


if __name__=='__main__':
    unittest.main()
