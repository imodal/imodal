import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)
import unittest

import numpy as np
import torch

from implicitmodules.numpy.DeformationModules.ElasticOrder1 import ElasticOrder1
from implicitmodules.torch.DeformationModules.ElasticOrder1 import ImplicitModule1

torch.set_default_tensor_type(torch.DoubleTensor)

class TestCompareImplicitModules1(unittest.TestCase):
    def setUp(self):
        self.sigma = 1.
        self.N = 4
        self.q = 5.*np.random.rand(self.N, 2)
        self.R = 5.*np.random.rand(self.N, 2, 2)
        self.p = 5.*np.random.rand(self.N, 2)
        self.p_R = 5.*np.random.rand(self.N, 2, 2)
        self.dim_controls = 1
        self.controls = 5.*np.random.rand(self.dim_controls)
        self.C = 5.*np.random.rand(self.N, 2, self.dim_controls)

        self.implicit1_torch = ImplicitModule1(2, self.N, self.sigma, torch.tensor(self.C), 0.001, gd=(torch.tensor(self.q), torch.tensor(self.R)), cotan=(torch.tensor(self.p), torch.tensor(self.p_R)))

        self.implicit1_numpy = ElasticOrder1(self.sigma, self.N, 2, 1., self.C, 0.001)
        self.implicit1_numpy.GD.fill_cot_from_param(((self.q, self.R), (self.p, self.p_R)))
        self.implicit1_numpy.update()

    def test_geodesic_controls(self):
        self.implicit1_torch.compute_geodesic_control(self.implicit1_torch.manifold)
        self.implicit1_numpy.GeodesicControls_curr(self.implicit1_numpy.GD)

        self.assertTrue(np.allclose(self.implicit1_torch.controls.detach().numpy(), self.implicit1_numpy.Cont))

    def test_cost(self):
        self.implicit1_torch.fill_controls(torch.tensor(self.controls))
        self.implicit1_numpy.fill_Cont(self.controls)
        self.implicit1_numpy.update()

        cost_torch = self.implicit1_torch.cost()
        self.implicit1_numpy.Cost_curr()
        cost_numpy = self.implicit1_numpy.cost

        self.assertTrue(np.allclose(cost_torch.detach().numpy(), cost_numpy))

    def test_apply(self):
        nb_pts = 5
        self.implicit1_torch.fill_controls(torch.tensor(self.controls))
        self.implicit1_numpy.fill_Cont(self.controls)
        self.implicit1_numpy.update()

        points = np.random.rand(nb_pts, 2)
        speed_torch = self.implicit1_torch(torch.tensor(points))
        speed_numpy = self.implicit1_numpy.field_generator_curr()(points, 0)

        self.assertTrue(np.allclose(speed_torch.numpy(), speed_numpy))


if __name__ == '__main__':
    unittest.main()
