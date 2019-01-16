import sys, os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 1)

import unittest
import numpy as np


class NumpyUnitTestCase(unittest.TestCase):
    sig = 0.3
    
    ############################################################
    def test_my_nker_order_3(self):
        from src.kernels import my_nker
        x = np.array([1., 0.5])
        xp = x + 0.0001 * np.random.normal(0, 1, (2))
        Dk = my_nker(xp, 2, self.sig) - my_nker(x, 2, self.sig)
        Dx = xp - x
        dk = my_nker(x, 3, self.sig)
        self.assertTrue(np.allclose(Dk, np.dot(dk, Dx), atol=1e-6))
    
    ############################################################
    def test_my_VsToV_1(self):
        from src.field_structures import my_VsToV
        x, p = np.random.normal(0, 1, (13, 2)), np.random.normal(0, 1, (13, 2))
        Par = {'0': [(x, p)], 'p': [], 'm': [], 'sig': self.sig}
        
        z = np.random.normal(0, 1, (20, 2))
        zp = z + 0.000001 * np.random.normal(0, 1, z.shape)
        Dz = zp - z
        
        for j in range(2):
            with self.subTest(j=j):
                Djv = np.asarray(my_VsToV(Par, zp, j)) - np.asarray(my_VsToV(Par, z, j))
                djv = my_VsToV(Par, z, j + 1)
                k = 10
                self.assertTrue(np.allclose(Djv[k], np.dot(djv[k], Dz[k]), atol=1e-6))
    
    ############################################################
    def test_my_VsToV_2(self):
        from src.field_structures import my_VsToV
        x, p = np.random.normal(0, 1, (13, 2)), np.random.normal(0, 1, (13, 2, 2))
        Par = {'0': [], 'p': [(x, p)], 'm': [], 'sig': self.sig}
        z = np.random.normal(0, 1, (20, 2))
        zp = z + 0.00001 * np.random.normal(0, 1, z.shape)
        Dz = zp - z
        
        for j in range(2):
            with self.subTest(j=j):
                Djv = np.asarray(my_VsToV(Par, zp, j)) - np.asarray(my_VsToV(Par, z, j))
                djv = my_VsToV(Par, z, j + 1)
                k = 10
                self.assertTrue(np.allclose(Djv[k], np.dot(djv[k], Dz[k]), atol=1e-6))


if __name__ == '__main__':
    unittest.main()
