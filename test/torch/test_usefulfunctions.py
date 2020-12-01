import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest

import torch

import implicitmodules.torch as im


class TestUsefulFunctions(unittest.TestCase):
    def test_gridandvec2d(self):
        m = 10
        n = 8
        u, v = torch.meshgrid(torch.arange(m).float(), torch.arange(n).float())

        vec = im.Utilities.usefulfunctions.grid2vec(u, v)
        self.assertIsInstance(vec, torch.Tensor)
        self.assertEqual(vec.shape, torch.Size([m*n, 2]))

        u_out, v_out = im.Utilities.usefulfunctions.vec2grid(vec, u.shape[0], v.shape[1])

        self.assertIsInstance(u_out, torch.Tensor)
        self.assertIsInstance(v_out, torch.Tensor)
        self.assertTrue(torch.all(torch.eq(u, u_out)))
        self.assertTrue(torch.all(torch.eq(v, v_out)))

    def test_gridandvec3d(self):
        m = 10
        n = 8
        l = 12
        u, v, w = torch.meshgrid(torch.arange(m).float(), torch.arange(n).float(), torch.arange(l).float())

        vec = im.Utilities.usefulfunctions.grid2vec(u, v, w)
        self.assertIsInstance(vec, torch.Tensor)
        self.assertEqual(vec.shape, torch.Size([m*n*l, 3]))

        u_out, v_out, w_out = im.Utilities.usefulfunctions.vec2grid(vec, u.shape[0], v.shape[1], w.shape[2])

        self.assertIsInstance(u_out, torch.Tensor)
        self.assertIsInstance(v_out, torch.Tensor)
        self.assertIsInstance(w_out, torch.Tensor)
        self.assertTrue(torch.all(torch.eq(u, u_out)))
        self.assertTrue(torch.all(torch.eq(v, v_out)))
        self.assertTrue(torch.all(torch.eq(w, w_out)))


    def test_indices2coords(self):
        m, n = 8, 4
        u, v = torch.meshgrid(torch.tensor(range(0, m)), torch.tensor(range(0, n)))
        indices = im.Utilities.usefulfunctions.grid2vec(u, v)

        coords = im.Utilities.usefulfunctions.indices2coords(indices, torch.Size([m, n]))

        self.assertIsInstance(coords, torch.Tensor)
        self.assertEqual(coords.shape, indices.shape)

