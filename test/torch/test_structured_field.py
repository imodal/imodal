import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest

import torch
import matplotlib.pyplot as plt

import implicitmodules.torch as im

torch.set_default_tensor_type(torch.DoubleTensor)


class TestStructuredField0(unittest.TestCase):
    def setUp(self):
        self.sigma = 2.5
        self.N = 10
        self.M = 5
        self.points = torch.rand(self.M, 2)
        self.moments = torch.rand(self.M, 2)
        self.support = torch.rand(self.M, 2)

    def test_call(self):
        sfield_torch = im.StructuredFields.StructuredField_0(self.support, self.moments, self.sigma, backend='torch')
        result_torch = sfield_torch(self.points, k=0)

        sfield_keops = im.StructuredFields.StructuredField_0(self.support, self.moments, self.sigma, backend='keops')
        result_keops = sfield_keops(self.points, k=0)

        self.assertTrue(torch.allclose(result_torch, result_keops))

        sfield_torch = im.StructuredFields.StructuredField_0(self.support, self.moments, self.sigma, backend='torch')
        result_torch = sfield_torch(self.points, k=1)

        sfield_keops = im.StructuredFields.StructuredField_0(self.support, self.moments, self.sigma, backend='keops')
        result_keops = sfield_keops(self.points, k=1)

        self.assertTrue(torch.allclose(result_torch, result_keops))


class TestStructuredFieldP(unittest.TestCase):
    def setUp(self):
        self.sigma = 2.5
        self.N = 10
        self.M = 5
        self.points = torch.rand(self.M, 2)
        self.moments = torch.rand(self.M, 2, 2)
        self.support = torch.rand(self.M, 2)

    def test_call(self):
        sfield_torch = im.StructuredFields.StructuredField_p(self.support, self.moments, self.sigma, backend='torch')
        result_torch = sfield_torch(self.points, k=0)

        sfield_keops = im.StructuredFields.StructuredField_p(self.support, self.moments, self.sigma, backend='keops')
        result_keops = sfield_keops(self.points, k=0)

        self.assertTrue(torch.allclose(result_torch, result_keops))

        sfield_torch = im.StructuredFields.StructuredField_p(self.support, self.moments, self.sigma, backend='torch')
        result_torch = sfield_torch(self.points, k=1)

        sfield_keops = im.StructuredFields.StructuredField_p(self.support, self.moments, self.sigma, backend='keops')
        result_keops = sfield_keops(self.points, k=1)

        self.assertTrue(torch.allclose(result_torch, result_keops))


class TestStructuredFieldM(unittest.TestCase):
    def setUp(self):
        self.sigma = 2.5
        self.N = 10
        self.M = 5
        self.points = torch.rand(self.M, 2)
        self.moments = torch.rand(self.M, 2, 2)
        self.support = torch.rand(self.M, 2)

    def test_call(self):
        sfield_torch = im.StructuredFields.StructuredField_m(self.support, self.moments, self.sigma, backend='torch')
        result_torch = sfield_torch(self.points, k=0)

        sfield_keops = im.StructuredFields.StructuredField_m(self.support, self.moments, self.sigma, backend='keops')
        result_keops = sfield_keops(self.points, k=0)

        self.assertTrue(torch.allclose(result_torch, result_keops))

        sfield_torch = im.StructuredFields.StructuredField_m(self.support, self.moments, self.sigma, backend='torch')
        result_torch = sfield_torch(self.points, k=1)

        sfield_keops = im.StructuredFields.StructuredField_m(self.support, self.moments, self.sigma, backend='keops')
        result_keops = sfield_keops(self.points, k=1)

        self.assertTrue(torch.allclose(result_torch, result_keops))


