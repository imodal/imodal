import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest
import math

import torch

import implicitmodules.torch as im


class TestUsefulFunctions(unittest.TestCase):
    def test_aabb(self):
        points = torch.rand(10, 2)
        aabb = im.usefulfunctions.AABB.build_from_points(points)

        self.assertEqual(aabb.xmin, torch.min(points[:, 0]))
        self.assertEqual(aabb.ymin, torch.min(points[:, 1]))
        self.assertEqual(aabb.xmax, torch.max(points[:, 0]))
        self.assertEqual(aabb.ymax, torch.max(points[:, 1]))

        self.assertEqual(aabb.width, torch.max(points[:, 0]) - torch.min(points[:, 0]))
        self.assertEqual(aabb.height, torch.max(points[:, 1]) - torch.min(points[:, 1]))

        self.assertIsInstance(aabb.get_list(), list)
        self.assertEqual(len(aabb.get_list()), 4)
        
        aabb_list = aabb.get_list()
        self.assertEqual(aabb_list[0], aabb.xmin)
        self.assertEqual(aabb_list[1], aabb.xmax)
        self.assertEqual(aabb_list[2], aabb.ymin)
        self.assertEqual(aabb_list[3], aabb.ymax)
        
        self.assertEqual(aabb[0], aabb.xmin)
        self.assertEqual(aabb[1], aabb.xmax)
        self.assertEqual(aabb[2], aabb.ymin)
        self.assertEqual(aabb[3], aabb.ymax)

    def test_aabb_is_inside(self):
        points = torch.tensor([[1., 0.],   # Inside
                               [2., 1.],   # Not inside
                               [0.5, 0.5], # Inside
                               [-1., 0.5], # Not inside
                               [0.5, -1]]) # Not inside

        aabb = im.usefulfunctions.AABB(0., 1., 0., 1.)
        
        is_inside = aabb.is_inside(points)

        self.assertIsInstance(is_inside, torch.Tensor)
        self.assertEqual(is_inside.shape[0], points.shape[0])
        self.assertTrue(is_inside[0])
        self.assertFalse(is_inside[1])
        self.assertTrue(is_inside[2])
        self.assertFalse(is_inside[3])
        self.assertFalse(is_inside[4])

    def test_aabb_sample_random_point(self):
        points = torch.randn(4, 2)
        aabb = im.usefulfunctions.AABB.build_from_points(points)

        sampled = aabb.sample_random_point(100)

        self.assertIsInstance(sampled, torch.Tensor)
        self.assertTrue(sampled.shape, torch.Size([100, 2]))
        self.assertTrue(torch.all(aabb.is_inside(sampled)))

    def test_flatten_tensor_list(self):
        a = torch.rand(4)
        b = torch.rand(4)
        c = torch.rand(4)
        d = torch.rand(4)
        e = torch.rand(4)

        l = [a, b, [c, [d, e]]]

        l_out = im.usefulfunctions.flatten_tensor_list(l)

        self.assertEqual(len(l_out), 5)
        self.assertTrue(torch.all(torch.eq(l_out[0], a)))
        self.assertTrue(torch.all(torch.eq(l_out[1], b)))
        self.assertTrue(torch.all(torch.eq(l_out[2], c)))
        self.assertTrue(torch.all(torch.eq(l_out[3], d)))
        self.assertTrue(torch.all(torch.eq(l_out[4], e)))

    def test_gridandvec(self):
        m = 10
        n = 8
        u, v = torch.meshgrid(torch.tensor(range(0, m)), torch.tensor(range(0, n)))

        vec = im.usefulfunctions.grid2vec(u, v)
        self.assertIsInstance(vec, torch.Tensor)
        self.assertEqual(vec.shape, torch.Size([m*n, 2]))

        u_out, v_out = im.usefulfunctions.vec2grid(vec, u.shape[0], v.shape[1])
        self.assertIsInstance(u_out, torch.Tensor)
        self.assertIsInstance(v_out, torch.Tensor)
        self.assertTrue(torch.all(torch.eq(u, u_out)))
        self.assertTrue(torch.all(torch.eq(v, v_out)))

    def test_indices2coords(self):
        m, n = 8, 4
        u, v = torch.meshgrid(torch.tensor(range(0, m)), torch.tensor(range(0, n)))
        indices = im.usefulfunctions.grid2vec(u, v)

        coords = im.usefulfunctions.indices2coords(indices, torch.Size([m, n]))

        self.assertIsInstance(coords, torch.Tensor)
        self.assertEqual(coords.shape, indices.shape)

    def test_blocks_to_2d(self):
        M = torch.arange(4).view(4, 1, 1)
        M_ = torch.tensor([[0, 1], [2, 3]])
        self.assertTrue(torch.all(torch.eq(im.usefulfunctions.blocks_to_2d(M), M_)))

        M = torch.arange(16).view(4, 2, 2)
        M_ = torch.tensor([[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]])
        self.assertTrue(torch.all(torch.eq(im.usefulfunctions.blocks_to_2d(M), M_)))

    def test_blocks_to_2d_fast(self):
        M = torch.arange(4).view(4, 1, 1)
        M_ = torch.tensor([[0, 1], [2, 3]])
        self.assertTrue(torch.all(torch.eq(im.usefulfunctions.blocks_to_2d_fast(M), M_)))

        M = torch.arange(16).view(4, 2, 2)
        M_ = torch.tensor([[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]])
        self.assertTrue(torch.all(torch.eq(im.usefulfunctions.blocks_to_2d_fast(M), M_)))

    def test_close_shape(self):
        nb_pts = 10
        dim = 2
        points = torch.rand(nb_pts, dim)
        closed = im.usefulfunctions.close_shape(points)

        self.assertTrue(closed.shape, torch.Size([nb_pts + 1, dim]))
        self.assertTrue(torch.all(torch.eq(closed[0, :], closed[-1, :])))

    def test_point_side(self):

        pts = torch.tensor([[1., 1.],
                            [0., 0.],
                            [-1., -1]])
        origin = torch.tensor([0.5, 0.])
        vec = torch.tensor([1., 3.])

        self.assertEqual(im.usefulfunctions.point_side(origin, vec, pts[0]), -1) 
        self.assertEqual(im.usefulfunctions.point_side(origin, vec, pts[1]), 1)
        self.assertEqual(im.usefulfunctions.point_side(origin, vec, pts[2]), 1)

    def test_is_inside_shape(self):
        nb_pts_shape = 500
        t = torch.linspace(0, 2*math.pi, nb_pts_shape)
        shape = torch.zeros(nb_pts_shape, 2)

        shape[:, 0] = torch.cos(t)
        shape[:, 1] = torch.sin(t)

        nb_pts = 10
        pts = torch.rand(nb_pts, 2)*2. - 1.

        inside = torch.norm(pts, dim=1) <= 1.
        inside_func = im.usefulfunctions.is_inside_shape(shape, pts)

        self.assertTrue(torch.all(torch.eq(inside, inside_func)))

