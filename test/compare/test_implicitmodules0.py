import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest

import numpy as np
import torch

from implicitmodules.numpy.DeformationModules.ElasticOrder0 import ElasticOrder0
from implicitmodules.torch.DeformationModules.ElasticOrder0 import ImplicitModule0

torch.set_default_tensor_type(torch.DoubleTensor)

class TestCompareImplicitModules0(unittest.TestCase):
    def setUp(self):
        self.sigma = 3.
        self.N = 100
        self.q = 10.*np.random.rand(self.N, 2)
        self.p = 10.*np.random.rand(self.N, 2)
        self.controls = np.random.rand(self.N, 2)

        self.implicit0_torch = ImplicitModule0(2, self.N, self.sigma, 0.001, gd=torch.tensor(self.p), cotan=torch.tensor(self.q))

        self.implicit0_numpy = ElasticOrder0(self.sigma, self.N, 2, 1., 0.001)
        self.implicit0_numpy.GD.fill_cot_from_param((self.q, self.p))
        self.implicit0_numpy.update()

    def test_geodesic_controls(self):
        self.implicit0_torch.compute_geodesic_control(self.implicit0_torch.manifold)
        self.implicit0_numpy.GeodesicControls_curr(self.implicit0_numpy.GD)

        np.allclose(self.implicit0_torch.controls.detach().numpy().reshape(-1, 2), self.implicit0_numpy.Cont)

    def test_cost(self):
        self.implicit0_torch.fill_controls(torch.tensor(self.controls))
        self.implicit0_numpy.fill_Cont(self.controls)
        self.implicit0_numpy.update()

        cost_torch = self.implicit0_torch.cost()
        self.implicit0_numpy.Cost_curr()
        cost_numpy = self.implicit0_numpy.cost

        np.allclose(cost_torch.detach().numpy(), cost_numpy)

    def test_apply(self):
        self.implicit0_torch.fill_controls(torch.tensor(self.controls))
        self.implicit0_numpy.fill_Cont(self.controls)
        self.implicit0_numpy.update()

        nb_pts = 100
        points = np.random.rand(nb_pts, 2)
        speed_torch = self.implicit0_torch(torch.tensor(points))
        speed_numpy = self.implicit0_numpy.field_generator_curr()(points, 0)

        np.allclose(speed_torch.detach().numpy(), speed_numpy)


if __name__ == '__main__':
    unittest.main()
