import sys
sys.path.append("../../../")

import unittest

import torch

import implicitmodules.torch as dm

class TestKernels(unittest.TestCase):
    def setUp(self):
        pass

    def test_scal(self):
        m = 10
        x = torch.rand(m, 2)
        y = torch.rand(m, 2)

        self.assertIsInstance(dm.kernels.scal(x, y), torch.Tensor)
        self.assertTrue(torch.allclose(
            dm.kernels.scal(x, y), torch.dot(x.view(-1), y.view(-1))))

    def test_distancematrix(self):
        m = 10
        x = torch.rand(m, 2)
        y = torch.rand(m, 2)

        dist_matrix = dm.kernels.distances(x, y)
        self.assertIsInstance(dist_matrix, torch.Tensor)
        self.assertEqual(dist_matrix.shape, torch.Size([m, m]))
        self.assertTrue(torch.allclose(dist_matrix[1, 2], torch.dist(x[1], y[2])))

    def test_sqdistancematrix(self):
        m = 10
        x = torch.rand(m, 2)
        y = torch.rand(m, 2)

        sqdist_matrix = dm.kernels.sqdistances(x, y)
        self.assertIsInstance(sqdist_matrix, torch.Tensor)
        self.assertEqual(sqdist_matrix.shape, torch.Size([m, m]))
        self.assertTrue(torch.allclose(sqdist_matrix[1, 2], torch.dist(x[1], y[2])**2))

    def test_kernelxxmatrix(self):
        m = 10
        x = torch.rand(m, 2)

        kxx_matrix = dm.kernels.K_xx(x, 1.)
        self.assertIsInstance(kxx_matrix, torch.Tensor)
        self.assertEqual(kxx_matrix.shape, torch.Size([m, m]))

    def test_kernelxymatrix(self):
        m = 10
        n = 5
        x = torch.rand(m, 2)
        y = torch.rand(n, 2)

        kxy_matrix = dm.kernels.K_xy(x, y, 1.)
        self.assertIsInstance(kxy_matrix, torch.Tensor)
        self.assertEqual(kxy_matrix.shape, torch.Size([m, n]))
