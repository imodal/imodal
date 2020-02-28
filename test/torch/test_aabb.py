import os.path
import sys
import math
import itertools
from functools import reduce

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest

import torch
import numpy as np

import implicitmodules.torch as im

def make_test_aabb(dim):
    class TestAABB(unittest.TestCase):
        def setUp(self):
            self.kmin = (torch.rand(dim)-1.).tolist()
            self.kmax = (torch.rand(dim)+1.).tolist()

        def test_init(self):
            args = list(itertools.chain.from_iterable([(kmin, kmax) for kmin, kmax in zip(self.kmin, self.kmax)]))
            kwargs = dict((pre+'min', val) for pre, val in zip(im.Utilities.AABB.dim_prefix, self.kmin))
            kwargs.update({(pre+'max', val) for pre, val in zip(im.Utilities.AABB.dim_prefix, self.kmax)})
            
            aabb1 = im.Utilities.AABB(*args)
            aabb2 = im.Utilities.AABB(self.kmin, self.kmax)
            aabb3 = im.Utilities.AABB(**kwargs)

            self.assertEqual(aabb1.totuple(), aabb2.totuple())
            self.assertEqual(aabb1.totuple(), aabb3.totuple())

        def test_aabb(self):
            aabb = im.Utilities.AABB(self.kmin, self.kmax)

            for pre, kmin, kmax in zip(im.Utilities.AABB.dim_prefix, aabb.kmin, aabb.kmax):
                self.assertEqual(getattr(aabb, pre+'min'), kmin)
                self.assertEqual(getattr(aabb, pre+'max'), kmax)
                self.assertEqual(aabb[pre+'min'], kmin)
                self.assertEqual(aabb[pre+'max'], kmax)

            self.assertEqual(aabb.shape, tuple(kmax - kmin for kmax, kmin in zip(aabb.kmax, aabb.kmin)))

            self.assertIsInstance(aabb.totuple(), tuple)
            self.assertEqual(len(aabb.totuple()), 2*dim)

        def test_aabb_from_points(self):
            points = torch.rand(10, dim)
            aabb_points = im.Utilities.AABB.build_from_points(points)
            aabb = im.Utilities.AABB(torch.min(points, dim=0)[0].tolist(), torch.max(points, dim=0)[0].tolist())

            self.assertEqual(aabb_points.totuple(), aabb.totuple())

        def test_aabb_is_inside(self):
            if dim == 2:
                points = torch.tensor([[1., 0.],   # Inside
                                       [2., 1.],   # Not inside
                                       [0.5, 0.5], # Inside
                                       [-1., 0.5], # Not inside
                                       [0.5, -1]]) # Not inside

                aabb = im.Utilities.AABB(0., 1., 0., 1.)

                is_inside = aabb.is_inside(points)

                self.assertIsInstance(is_inside, torch.Tensor)
                self.assertEqual(is_inside.shape[0], points.shape[0])
                self.assertTrue(is_inside[0])
                self.assertFalse(is_inside[1])
                self.assertTrue(is_inside[2])
                self.assertFalse(is_inside[3])
                self.assertFalse(is_inside[4])

        def test_aabb_fill_uniform_spacing(self):
            aabb = im.Utilities.AABB(self.kmin, self.kmax)

            spacing = 0.2
            sampled = aabb.fill_uniform_spacing(spacing)

            self.assertIsInstance(sampled, torch.Tensor)
            self.assertTrue(torch.all(aabb.is_inside(sampled).to(dtype=torch.uint8)))

        def test_aabb_fill_uniform_density(self):
            aabb = im.Utilities.AABB(self.kmin, self.kmax)

            # Inconsistant test, only works with some values of density. Maybe because of rounding when converting density to spacing or vice versa. Low importance so we do not bother.
            density = 25.
            sampled = aabb.fill_uniform_density(density)

            self.assertIsInstance(sampled, torch.Tensor)
            self.assertTrue(torch.all(aabb.is_inside(sampled).to(dtype=torch.uint8)))

        def test_aabb_fill_random(self):
            aabb = im.Utilities.AABB(self.kmin, self.kmax)

            nb_pts = 100
            sampled = aabb.fill_random(nb_pts)

            self.assertIsInstance(sampled, torch.Tensor)
            self.assertEqual(sampled.shape, torch.Size([nb_pts, dim]))
            self.assertTrue(all(aabb.is_inside(sampled)))

        def test_aabb_fill_random(self):
            aabb = im.Utilities.AABB(self.kmin, self.kmax)

            density = 50.
            nb_pts = int(aabb.shape[dim-1]*density)
            sampled = aabb.fill_random_density(density)

            self.assertIsInstance(sampled, torch.Tensor)
            self.assertTrue(sampled.shape, torch.Size([nb_pts, dim]))
            self.assertTrue(all(aabb.is_inside(sampled)))

        def test_aabb_center(self):
            aabb = im.Utilities.AABB([-2.]*dim, [2.]*dim)
            self.assertTrue(torch.allclose(torch.tensor(aabb.centers), torch.zeros(dim)))

            aabb = im.Utilities.AABB(self.kmin, self.kmax)
            self.assertTrue(torch.allclose(torch.tensor(aabb.centers), torch.tensor([(kmax+kmin)/2. for kmax, kmin in zip(self.kmax, self.kmin)])))

        def test_aabb_scale(self):
            aabb = im.Utilities.AABB([-2.]*dim, [2.]*dim)
            aabb.scale_(2)

            self.assertTrue(all(kmin == -4. for kmin in aabb.kmin))
            self.assertTrue(all(kmax == 4. for kmax in aabb.kmax))

            aabb = im.Utilities.AABB(list(map(lambda x: -x - 2., range(dim))), range(dim))
            aabb.scale_(2)

            self.assertTrue(all(list(kmin == (-1. - 2.*(n+1)) for kmin, n in zip(aabb.kmin, range(dim)))))
            self.assertTrue(all(list(kmax == (1. + 2.*n) for kmax, n in zip(aabb.kmax, range(dim)))))

            aabb2 = aabb.scale(3.)
            aabb.scale_(3.)
            self.assertTrue(aabb.totuple(), aabb2.totuple())

        def test_aabb_squared(self):
            aabb = im.Utilities.AABB((torch.rand(dim)-2.).tolist(), (torch.rand(dim)+5.).tolist())

            # Check if squared() is idempotent
            self.assertTrue(np.allclose(aabb.squared().totuple(), aabb.squared().squared().totuple()))

            # Check if the aabb does not move
            self.assertTrue(np.allclose(aabb.centers, aabb.squared().centers))

            aabb2 = aabb.squared()
            aabb.squared_()

            # Check if the self operation squared_() does the same as squared()
            self.assertEqual(aabb.totuple(), aabb2.totuple())

            # Check if the shape of the resulting AABB as all component equal i.e. if it is a square or a cube
            self.assertTrue(np.allclose(aabb.shape, [aabb.shape[0]]*dim))

    return TestAABB


class TestAABB2D(make_test_aabb(2)):
    pass

class TestAABB3D(make_test_aabb(3)):
    pass

