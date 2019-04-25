import sys
sys.path.append("../../")

import unittest

import numpy as np
import torch

from implicitmodules.numpy.Kernels.ScalarGaussian import my_vker, my_K, my_xmy
from implicitmodules.torch.kernels import gauss_kernel, compute_sks, distances


class TestCompareKernels(unittest.TestCase):
    def setUp(self):
        pass

    def test_gaussian_kernel(self):
        sigma = 4.
        n = 100
        x = np.random.rand(n, 2)
        X = torch.tensor(x).float()

        self.assertTrue(np.allclose(gauss_kernel(X, 0, sigma).numpy(), my_vker(x, 0, sigma)))
        self.assertTrue(np.allclose(gauss_kernel(X, 1, sigma).numpy(), my_vker(x, 1, sigma)))
        self.assertTrue(np.allclose(gauss_kernel(X, 2, sigma).numpy(), my_vker(x, 2, sigma)))
        self.assertTrue(np.allclose(gauss_kernel(X, 3, sigma).numpy(), my_vker(x, 3, sigma)))

    def test_sks(self):
        sigma = 2.
        n = 100
        x = np.random.rand(n, 2)
        X = torch.tensor(x).float()

        self.assertTrue(np.allclose(compute_sks(X, sigma, 0).numpy(), my_K(x, sigma, 0)))
        self.assertTrue(np.allclose(compute_sks(X, sigma, 1).numpy(), my_K(x, sigma, 1)))

