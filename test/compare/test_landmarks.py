import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest

import numpy as np
import torch

from implicitmodules.numpy.Manifolds.Landmark import Landmark
from implicitmodules.torch.Manifolds.Landmark import Landmarks
from implicitmodules.numpy.StructuredFields import StructuredField_0 as n_StructuredField_0
from implicitmodules.torch.StructuredFields import StructuredField_0 as t_StructuredField_0

torch.set_default_tensor_type(torch.DoubleTensor)

class TestCompareLandmarks(unittest.TestCase):
    def setUp(self):
        self.dim = 2
        self.nb_pts = 100
        self.gd = np.random.rand(self.nb_pts, self.dim)
        self.cotan = np.random.rand(self.nb_pts, self.dim)

        self.torch_landmarks = Landmarks(self.dim, self.nb_pts, gd=torch.tensor(self.gd), cotan=torch.tensor(self.cotan))
        self.numpy_landmarks = Landmark(self.nb_pts, self.dim)
        self.numpy_landmarks.fill_cot_from_param((self.gd, self.cotan))

    def test_inner_prod_field(self):
        sigma = 0.2
        nb_pts_support = 50

        support = np.random.rand(nb_pts_support, self.dim)
        moments = np.random.rand(nb_pts_support, self.dim)
        torch_field = t_StructuredField_0(torch.tensor(support), torch.tensor(moments), sigma)
        numpy_field = n_StructuredField_0(support, moments, sigma)

        torch_prod = self.torch_landmarks.inner_prod_field(torch_field)
        numpy_prod = self.numpy_landmarks.inner_prod_v(numpy_field)

        self.assertTrue(np.allclose(torch_prod.detach().numpy(), numpy_prod))


if __name__ == '__main__':
    unittest.main()
