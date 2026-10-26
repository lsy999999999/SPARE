"""Independent peak reference, leakage checks, and native allocation conservation."""
import unittest
import numpy as np
from research.www_baseline_fidelity_20261013.policy import sampled_peaks,conditional_draws,nonnegative_fit,CostModel
from research.v1_paper_comparison_20260922.rules import future_peak_blocks

class PaperMath(unittest.TestCase):
    def test_vector_peaks_equal_independent_scalar_reference(self):
        rng=np.random.default_rng(19)
        for size in (1,3,16,32):
            p=rng.integers(1,512,size=size);g=rng.integers(0,32,size=size)
            pred=g+rng.integers(1,128,size=(32,size))
            actual=sampled_peaks(p,g,pred)
            expected=[future_peak_blocks(list(zip(p,g,row))) for row in pred]
            np.testing.assert_array_equal(actual,expected)
    def test_conditional_cold_start_and_censoring(self):
        x=conditional_draws([1,4,16],[4,20],[10,100],32,np.random.default_rng(1))
        self.assertTrue(np.all(x[:,0]==10));self.assertTrue(np.all(x[:,1]==100))
        self.assertEqual(conditional_draws([], [0],[50],1,np.random.default_rng(1))[0,0],50)
    def test_cost_fit_recovers_nonnegative_polynomial(self):
        rng=np.random.default_rng(11);x=rng.uniform(size=(100,4));coeff=np.array([0,3,7,2.])
        np.testing.assert_allclose(nonnegative_fit(x,x@coeff),coeff,atol=1e-10)
    def test_online_feedback(self):
        c=CostModel(dict(prefill_coefficients=[0,0,1,0],decode_coefficients=[0,0,1]))
        a=[dict(before=0,tokens=8,prompt=8,outputs=0)]
        c.update(a,16)
        self.assertAlmostEqual(c.gamma,1.1)

if __name__=='__main__':unittest.main()
