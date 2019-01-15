import sys, os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest
import numpy as np

import implicitmodules

class NumpyUnitTestCase(unittest.TestCase):
    M = int(10)
    N = int(6)
    D = int(3)
    E = int(3)
    
    x = np.random.rand(M, D)
    a = np.random.rand(M, E)
    f = np.random.rand(M, 1)
    y = np.random.rand(N, D)
    b = np.random.rand(N, E)
    g = np.random.rand(N, 1)
    sigma = np.array([0.4])

    def test_generic_syntax_sum(self):
        # compare output
        self.assertTrue(np.allclose(1, 1.00000001, atol=1e-6))
        
        
if __name__ == '__main__':
    unittest.main()