import unittest

from research.v1_paper_comparison_20260922 import run


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.manifest = dict(blocks={f'{role}_{source}_{i}':dict(role=role)
            for role, count in [('dev',1),('compare',2)]
            for source in ('code','conversation') for i in range(count)})
        self.selection = {regime:{arm:run.CONFIGS[0] for arm in run.ARMS}
                          for regime in run.REGIMES}

    def test_frozen_matrix_counts_and_unique_jobs(self):
        for phase,expected in [('aa',18),('tune',36),('compare',96)]:
            jobs = [j for regime in run.REGIMES
                    for j in run.jobs(self.manifest,phase,regime,self.selection)]
            self.assertEqual(len(jobs), expected)
            self.assertEqual(len({j['key'] for j in jobs}),expected)
        self.assertEqual(len(run.jobs(self.manifest,'smoke','restricted')),2)

    def test_compare_honors_per_arm_selected_config(self):
        self.selection['restricted']['budget'] = run.ref.CONFIGS[1]
        jobs = run.jobs(self.manifest,'compare','restricted',self.selection)
        for job in jobs:
            self.assertEqual(job['config'],self.selection['restricted'][job['arm']])
        self.assertEqual({j['repeat'] for j in jobs},{0,1})

    def test_each_pair_uses_same_requests(self):
        block = dict(role='compare',rows=[dict(request_id=str(i),prompt_tokens=128+i,
                     actual_output_tokens=i+1,original_arrival_ms=10*i) for i in range(64)])
        self.manifest['blocks']['compare_code_0'] = block
        jobs = [j for j in run.jobs(self.manifest,'compare','restricted',self.selection)
                if j['block']=='compare_code_0']
        reference = run.requests(self.manifest,jobs[0])
        self.assertEqual(len(reference),64)
        for job in jobs:
            self.assertEqual(run.requests(self.manifest,job),reference)


if __name__ == '__main__':
    unittest.main()
