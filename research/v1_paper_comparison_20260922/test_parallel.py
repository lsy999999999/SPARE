import unittest
from research.v1_paper_comparison_20260922 import parallel as run


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.m=dict(blocks={f'{role}_{source}_{i}':dict(role=role)
            for role,count in [('dev',1),('compare',2)]
            for source in ('code','conversation') for i in range(count)},
            hardware_by_gpu={str(g):dict(uuid=f'GPU-{g}') for g in run.GPUS})
        self.selection={regime:{arm:run.core.CONFIGS[0] for arm in run.core.ARMS}
                        for regime in run.core.REGIMES}

    def test_all_methods_and_repeats_paired_on_same_gpu(self):
        jobs=[j for regime in run.core.REGIMES for j in run.jobs(self.m,'compare',regime,self.selection)]
        self.assertEqual(len(jobs),96)
        self.assertEqual(len({j['key'] for j in jobs}),96)
        for regime in run.core.REGIMES:
            for block,b in self.m['blocks'].items():
                if b['role']!='compare':continue
                group=[j for j in jobs if j['regime']==regime and j['block']==block]
                self.assertEqual(len(group),8)
                self.assertEqual(len({j['gpu'] for j in group}),1)
                self.assertEqual({j['arm'] for j in group},set(run.core.ARMS))

    def test_checks_cover_each_gpu_and_tuning_not_duplicated(self):
        for phase,expected in [('aa',54),('tune',36)]:
            jobs=[j for regime in run.core.REGIMES for j in run.jobs(self.m,phase,regime)]
            self.assertEqual(len(jobs),expected)
            self.assertEqual({j['gpu'] for j in jobs},set(run.GPUS))
        self.assertEqual(len(run.jobs(self.m,'smoke','restricted')),6)


if __name__=='__main__':unittest.main()
