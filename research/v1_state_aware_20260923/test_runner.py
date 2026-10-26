import unittest
from research.v1_state_aware_20260923.common import BLOCKS,CAPS,METHODS,config
from research.v1_state_aware_20260923.run import jobs


class RunnerTests(unittest.TestCase):
    def test_paired_matrix_unique_and_reverse_order(self):
        cells=[dict(key=f'{b}_k{c}',block=b,capacity=c,gpu=(2,6,7)[i%3])
               for i,(c,b) in enumerate((c,b) for c in CAPS for b in BLOCKS)]
        m=dict(cells=cells,configs={k:config(k) for k in METHODS},
               hardware={str(g):dict(uuid=str(g)) for g in (2,6,7)})
        all_jobs=[]
        for cell in cells:
            js=jobs(m,cell); self.assertEqual(len(js),12)
            self.assertEqual([j['method'] for j in js[:6]],list(reversed([j['method'] for j in js[6:]])))
            self.assertEqual(len({j['gpu_uuid'] for j in js}),1)
            all_jobs+=js
        self.assertEqual(len({j['key'] for j in all_jobs}),72)
        for g in (2,6,7):
            js=jobs(m,dict(key=f'validation_g{g}',gpu=g,capacity=768,block=None))
            self.assertEqual(len(js),9)
            self.assertEqual(sum(j['phase']=='aa' for j in js),3)


if __name__=='__main__': unittest.main()
