import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest
from collections import Iterable

import torch
from torch.autograd import gradcheck

import implicitmodules.torch as im

torch.set_default_tensor_type(torch.DoubleTensor)

class TestStiefel(unittest.TestCase):
    def setUp(self):
        self.nb_pts = 10
        self.dim = 2
        self.gd_pts = torch.rand(self.nb_pts, self.dim).view(-1)
        self.gd_mat = torch.rand(self.nb_pts, self.dim, self.dim).view(-1)
        self.tan_pts = torch.rand(self.nb_pts, self.dim).view(-1)
        self.tan_mat = torch.rand(self.nb_pts, self.dim, self.dim).view(-1)
        self.cotan_pts = torch.rand(self.nb_pts, self.dim).view(-1)
        self.cotan_mat = torch.rand(self.nb_pts, self.dim, self.dim).view(-1)

        self.gd = (self.gd_pts, self.gd_mat)
        self.tan = (self.tan_pts, self.tan_mat)
        self.cotan = (self.cotan_pts, self.cotan_mat)

    def test_constructor(self):
        stiefel = im.Manifolds.Stiefel(self.dim, self.nb_pts, gd=self.gd, tan=self.tan, cotan=self.cotan)

        self.assertEqual(stiefel.nb_pts, self.nb_pts)
        self.assertEqual(stiefel.dim, self.dim)
        self.assertEqual(stiefel.numel_gd, self.nb_pts * (self.dim + self.dim * self.dim))
        self.assertEqual(stiefel.len_gd, 2)
        self.assertEqual(stiefel.dim_gd, (self.nb_pts * self.dim, self.nb_pts * self.dim * self.dim))

        self.assertTrue(torch.all(torch.eq(stiefel.gd[0], self.gd[0])))
        self.assertTrue(torch.all(torch.eq(stiefel.gd[1], self.gd[1])))
        self.assertTrue(torch.all(torch.eq(stiefel.tan[0], self.tan[0])))
        self.assertTrue(torch.all(torch.eq(stiefel.tan[1], self.tan[1])))
        self.assertTrue(torch.all(torch.eq(stiefel.cotan[0], self.cotan[0])))
        self.assertTrue(torch.all(torch.eq(stiefel.cotan[1], self.cotan[1])))
        
        self.assertIsInstance(stiefel.unroll_gd(), Iterable)
        self.assertIsInstance(stiefel.unroll_tan(), Iterable)
        self.assertIsInstance(stiefel.unroll_cotan(), Iterable)

        l_gd = stiefel.unroll_gd()
        l_tan = stiefel.unroll_tan()
        l_cotan = stiefel.unroll_cotan()

        self.assertTrue(torch.all(torch.eq(l_gd[0], self.gd[0])))
        self.assertTrue(torch.all(torch.eq(l_gd[1], self.gd[1])))
        self.assertTrue(torch.all(torch.eq(l_tan[0], self.tan[0])))
        self.assertTrue(torch.all(torch.eq(l_tan[1], self.tan[1])))
        self.assertTrue(torch.all(torch.eq(l_cotan[0], self.cotan[0])))
        self.assertTrue(torch.all(torch.eq(l_cotan[1], self.cotan[1])))

        l_rolled_gd = stiefel.roll_gd(l_gd)
        l_rolled_tan = stiefel.roll_tan(l_tan)
        l_rolled_cotan = stiefel.roll_cotan(l_cotan)

        self.assertTrue(torch.all(torch.eq(l_rolled_gd[0], self.gd[0])))
        self.assertTrue(torch.all(torch.eq(l_rolled_gd[1], self.gd[1])))
        self.assertTrue(torch.all(torch.eq(l_rolled_tan[0], self.tan[0])))
        self.assertTrue(torch.all(torch.eq(l_rolled_tan[1], self.tan[1])))
        self.assertTrue(torch.all(torch.eq(l_rolled_cotan[0], self.cotan[0])))
        self.assertTrue(torch.all(torch.eq(l_rolled_cotan[1], self.cotan[1])))

    def test_fill(self):
        stiefel = im.Manifolds.Stiefel(self.dim, self.nb_pts)

        stiefel.fill_gd(self.gd, copy=True)
        stiefel.fill_tan(self.tan, copy=True)
        stiefel.fill_cotan(self.cotan, copy=True)

        self.assertTrue(torch.all(torch.eq(stiefel.gd[0], self.gd[0])))
        self.assertTrue(torch.all(torch.eq(stiefel.gd[1], self.gd[1])))
        self.assertTrue(torch.all(torch.eq(stiefel.tan[0], self.tan[0])))
        self.assertTrue(torch.all(torch.eq(stiefel.tan[1], self.tan[1])))
        self.assertTrue(torch.all(torch.eq(stiefel.cotan[0], self.cotan[0])))
        self.assertTrue(torch.all(torch.eq(stiefel.cotan[1], self.cotan[1])))

    def test_assign(self):
        stiefel = im.Manifolds.Stiefel(self.dim, self.nb_pts)

        stiefel.gd = self.gd
        stiefel.tan = self.tan
        stiefel.cotan = self.cotan

        self.assertTrue(torch.all(torch.eq(stiefel.gd[0], self.gd[0])))
        self.assertTrue(torch.all(torch.eq(stiefel.gd[1], self.gd[1])))
        self.assertTrue(torch.all(torch.eq(stiefel.tan[0], self.tan[0])))
        self.assertTrue(torch.all(torch.eq(stiefel.tan[1], self.tan[1])))
        self.assertTrue(torch.all(torch.eq(stiefel.cotan[0], self.cotan[0])))
        self.assertTrue(torch.all(torch.eq(stiefel.cotan[1], self.cotan[1])))

    def test_muladd(self):
        stiefel = im.Manifolds.Stiefel(self.dim, self.nb_pts, gd=self.gd, tan=self.tan, cotan=self.cotan)

        scale = 1.5
        d_gd = (torch.rand(self.nb_pts, self.dim).view(-1),
                torch.rand(self.nb_pts, self.dim, self.dim).view(-1))
        d_tan = (torch.rand(self.nb_pts, self.dim).view(-1),
                 torch.rand(self.nb_pts, self.dim, self.dim).view(-1))
        d_cotan = (torch.rand(self.nb_pts, self.dim).view(-1),
                   torch.rand(self.nb_pts, self.dim, self.dim).view(-1))

        stiefel.muladd_gd(d_gd, scale)
        stiefel.muladd_tan(d_tan, scale)
        stiefel.muladd_cotan(d_cotan, scale)

        self.assertTrue(torch.all(torch.eq(stiefel.gd[0], self.gd[0] + scale * d_gd[0])))
        self.assertTrue(torch.all(torch.eq(stiefel.gd[1], self.gd[1] + scale * d_gd[1])))
        self.assertTrue(torch.all(torch.eq(stiefel.tan[0], self.tan[0] + scale * d_tan[0])))
        self.assertTrue(torch.all(torch.eq(stiefel.tan[1], self.tan[1] + scale * d_tan[1])))
        self.assertTrue(torch.all(torch.eq(stiefel.cotan[0], self.cotan[0] + scale * d_cotan[0])))
        self.assertTrue(torch.all(torch.eq(stiefel.cotan[1], self.cotan[1] + scale * d_cotan[1])))

    def test_action(self):
        stiefel = im.Manifolds.Stiefel(self.dim, self.nb_pts, gd=self.gd, tan=self.tan, cotan=self.cotan)

        nb_pts_mod = 15
        landmarks_mod = im.Manifolds.Landmarks(2, nb_pts_mod, gd=torch.rand(nb_pts_mod, 2).view(-1))
        trans = im.DeformationModules.Translations_Torch(landmarks_mod, 1.5)
        trans.fill_controls(torch.rand_like(landmarks_mod.gd))

        man = stiefel.infinitesimal_action(trans.field_generator())
    
        self.assertIsInstance(man, im.Manifolds.Stiefel)

    def test_inner_prod_field(self):
        stiefel = im.Manifolds.Stiefel(self.dim, self.nb_pts, gd=self.gd, tan=self.tan, cotan=self.cotan)

        nb_pts_mod = 15
        landmarks_mod = im.Manifolds.Landmarks(2, nb_pts_mod, gd=torch.rand(nb_pts_mod, 2).view(-1))
        trans = im.DeformationModules.Translations_Torch(landmarks_mod, 1.5)
        trans.fill_controls(torch.rand_like(landmarks_mod.gd))

        inner_prod = stiefel.inner_prod_field(trans.field_generator())

        self.assertIsInstance(inner_prod, torch.Tensor)
        self.assertEqual(inner_prod.shape, torch.Size([]))

    def test_gradcheck_fill(self):
        def fill_gd(gd_pts, gd_mat):
            stiefel.fill_gd((gd_pts, gd_mat))
            return stiefel.gd[0], stiefel.gd[1]
 
        def fill_tan(tan_pts, tan_mat):
            stiefel.fill_tan((tan_pts, tan_mat))
            return stiefel.tan[0], stiefel.tan[1]

        def fill_cotan(cotan_pts, cotan_mat):
            stiefel.fill_cotan((cotan_pts, cotan_mat))
            return stiefel.cotan[0], stiefel.cotan[1]

        self.gd_pts.requires_grad_()
        self.gd_mat.requires_grad_()
        self.tan_pts.requires_grad_()
        self.tan_mat.requires_grad_()
        self.cotan_pts.requires_grad_()
        self.cotan_mat.requires_grad_()

        stiefel = im.Manifolds.Stiefel(2, self.nb_pts)

        self.assertTrue(gradcheck(fill_gd, (self.gd_pts, self.gd_mat), raise_exception=False))
        self.assertTrue(gradcheck(fill_tan, (self.tan_pts, self.tan_mat), raise_exception=False))
        self.assertTrue(gradcheck(fill_cotan, (self.cotan_pts, self.cotan_mat), raise_exception=False))

    def test_gradcheck_muladd(self):
        def muladd_gd(gd_pts, gd_mat):
            stiefel.fill_gd(self.gd)
            stiefel.muladd_gd((gd_pts, gd_mat), scale)
            return stiefel.gd[0], stiefel.gd[1]

        def muladd_tan(tan_pts, tan_mat):
            stiefel.fill_tan(self.tan)
            stiefel.muladd_tan((tan_pts, tan_mat), scale)
            return stiefel.tan[0], stiefel.cotan[1]

        def muladd_cotan(cotan_pts, cotan_mat):
            stiefel.fill_cotan(self.cotan)
            stiefel.muladd_cotan((cotan_pts, cotan_mat), scale)
            return stiefel.cotan[0], stiefel.cotan[1]

        stiefel = im.Manifolds.Stiefel(self.dim, self.nb_pts)

        self.gd[0].requires_grad_()
        self.gd[1].requires_grad_()
        self.tan[0].requires_grad_()
        self.tan[1].requires_grad_()
        self.cotan[0].requires_grad_()
        self.cotan[1].requires_grad_()

        scale = 2.

        gd_mul = (torch.rand_like(self.gd[0], requires_grad=True),
                  torch.rand_like(self.gd[1], requires_grad=True))
        tan_mul = (torch.rand_like(self.tan[0], requires_grad=True),
                   torch.rand_like(self.tan[1], requires_grad=True))
        cotan_mul = (torch.rand_like(self.cotan[0], requires_grad=True),
                     torch.rand_like(self.cotan[1], requires_grad=True))

        self.assertTrue(gradcheck(muladd_gd, gd_mul, raise_exception=False))
        self.assertTrue(gradcheck(muladd_tan, tan_mul, raise_exception=False))
        self.assertTrue(gradcheck(muladd_cotan, cotan_mul, raise_exception=False))

    def test_gradcheck_action(self):
        def action(gd_pts, gd_mat, controls):
            stiefel.fill_gd((gd_pts, gd_mat))
            module = im.DeformationModules.ImplicitModule1_Torch(stiefel, C, 1., 0.01)
            module.fill_controls(controls)
            man = stiefel.infinitesimal_action(module.field_generator())
            return man.gd[0], man.gd[1], man.tan[0], man.tan[1], man.cotan[0], man.cotan[1]

        self.gd_pts.requires_grad_()
        self.gd_mat.requires_grad_()

        controls = torch.rand(1, requires_grad=True)
        stiefel = im.Manifolds.Stiefel(2, self.nb_pts)
        C = torch.rand(self.nb_pts, 2, 1)

        self.assertTrue(gradcheck(action, (self.gd_pts, self.gd_mat, controls), raise_exception=False))

    # def test_gradcheck_inner_prod_field(self):
    #     def inner_prod_field(gd, controls):
    #         landmarks = im.manifold.Landmarks(2, self.nb_pts, gd=self.gd)
    #         landmarks.fill_gd(gd)
    #         module = im.deformationmodules.Translations(landmarks, 2.)
    #         module.fill_controls(controls)
    #         return landmarks.inner_prod_field(module.field_generator())

    #     self.gd[0].requires_grad_()
    #     self.gd[1].requires_grad_()
    #     controls = torch.rand_like(self.gd[0], requires_grad=True)

    #     self.assertTrue(gradcheck(inner_prod_field, (self.gd, controls), raise_exception=False))


if __name__ == '__main__':
    unittest.main()
