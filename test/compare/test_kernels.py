import sys
sys.path.append("../")

import unittest

import numpy as np
import torch

from implicitmodules.numpy.Kernels.ScalarGaussian import my_vker, my_K, my_xmy
from implicitmodules.torch.kernels import gauss_kernel, compute_sks, rel_differences

torch.set_default_tensor_type(torch.DoubleTensor)


class TestCompareKernels(unittest.TestCase):
    def test_rel_differences(self):
        n = 100
        m = 50
        x = np.random.rand(n, 2)
        y = np.random.rand(m, 2)
        X = torch.tensor(x)
        Y = torch.tensor(y)

        self.assertTrue(np.allclose(my_xmy(x, y), rel_differences(X, Y)))
        self.assertTrue(np.allclose(my_xmy(x, x), rel_differences(X, X)))
        self.assertTrue(np.allclose(my_xmy(y, y), rel_differences(Y, Y)))

    def test_gaussian_kernel(self):
        sigma = 4.
        n = 100
        x = np.random.rand(n, 2)
        X = torch.tensor(x)

        self.assertTrue(np.allclose(gauss_kernel(X, 0, 1.).numpy(), my_vker(x, 0, 1.)))
        self.assertTrue(np.allclose(gauss_kernel(X, 1, 1.).numpy(), my_vker(x, 1, 1.)))
        self.assertTrue(np.allclose(gauss_kernel(X, 2, 1.).numpy(), my_vker(x, 2, 1.)))
        self.assertTrue(np.allclose(gauss_kernel(X, 3, 1.).numpy(), my_vker(x, 3, 1.)))
        
        self.assertTrue(np.allclose(gauss_kernel(X, 0, sigma).numpy(), my_vker(x, 0, sigma)))
        self.assertTrue(np.allclose(gauss_kernel(X, 1, sigma).numpy(), my_vker(x, 1, sigma)))
        self.assertTrue(np.allclose(gauss_kernel(X, 2, sigma).numpy(), my_vker(x, 2, sigma)))
        self.assertTrue(np.allclose(gauss_kernel(X, 3, sigma).numpy(), my_vker(x, 3, sigma)))


    def test_sks(self):
        sigma = 6.
        n = 100
        x = np.random.rand(n, 2)
        X = torch.tensor(x)

        self.assertTrue(np.allclose(compute_sks(X, sigma, 0).numpy(), my_K(x, x, sigma, 0)))
        self.assertTrue(np.allclose(compute_sks(X, sigma, 1).numpy(), my_K(x, x, sigma, 1)))

