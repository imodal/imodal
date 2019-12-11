import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest
from collections import Iterable

import torch
from torch.autograd import gradcheck

import implicitmodules.torch as im

torch.set_default_tensor_type(torch.DoubleTensor)

def make_test_stiefel(dim, backend):
    class TestStiefel(unittest.TestCase):
        def setUp(self):
            self.nb_pts = 10
            self.gd_pts = torch.rand(self.nb_pts, dim)
            self.gd_mat = torch.rand(self.nb_pts, dim, dim)
            self.tan_pts = torch.rand(self.nb_pts, dim)
            self.tan_mat = torch.rand(self.nb_pts, dim, dim)
            self.cotan_pts = torch.rand(self.nb_pts, dim)
            self.cotan_mat = torch.rand(self.nb_pts, dim, dim)

            self.gd = (self.gd_pts, self.gd_mat)
            self.tan = (self.tan_pts, self.tan_mat)
            self.cotan = (self.cotan_pts, self.cotan_mat)

        def test_constructor(self):
            stiefel = im.Manifolds.Stiefel(dim, self.nb_pts, gd=self.gd, tan=self.tan, cotan=self.cotan)

            self.assertEqual(stiefel.nb_pts, self.nb_pts)
            self.assertEqual(stiefel.dim, dim)
            self.assertEqual(sum(stiefel.numel_gd), self.nb_pts * (dim + dim * dim))
            self.assertEqual(stiefel.len_gd, 2)

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
            stiefel = im.Manifolds.Stiefel(dim, self.nb_pts)

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
            stiefel = im.Manifolds.Stiefel(dim, self.nb_pts)

            stiefel.gd = self.gd
            stiefel.tan = self.tan
            stiefel.cotan = self.cotan

            self.assertTrue(torch.all(torch.eq(stiefel.gd[0], self.gd[0])))
            self.assertTrue(torch.all(torch.eq(stiefel.gd[1], self.gd[1])))
            self.assertTrue(torch.all(torch.eq(stiefel.tan[0], self.tan[0])))
            self.assertTrue(torch.all(torch.eq(stiefel.tan[1], self.tan[1])))
            self.assertTrue(torch.all(torch.eq(stiefel.cotan[0], self.cotan[0])))
            self.assertTrue(torch.all(torch.eq(stiefel.cotan[1], self.cotan[1])))

        def test_add(self):
            stiefel = im.Manifolds.Stiefel(dim, self.nb_pts, gd=self.gd, tan=self.tan, cotan=self.cotan)

            d_gd = (torch.rand(self.nb_pts, dim),
                    torch.rand(self.nb_pts, dim, dim))
            d_tan = (torch.rand(self.nb_pts, dim),
                     torch.rand(self.nb_pts, dim, dim))
            d_cotan = (torch.rand(self.nb_pts, dim),
                       torch.rand(self.nb_pts, dim, dim))

            stiefel.add_gd(d_gd)
            stiefel.add_tan(d_tan)
            stiefel.add_cotan(d_cotan)

            self.assertTrue(torch.all(torch.eq(stiefel.gd[0], self.gd[0] + d_gd[0])))
            self.assertTrue(torch.all(torch.eq(stiefel.gd[1], self.gd[1] + d_gd[1])))
            self.assertTrue(torch.all(torch.eq(stiefel.tan[0], self.tan[0] + d_tan[0])))
            self.assertTrue(torch.all(torch.eq(stiefel.tan[1], self.tan[1] + d_tan[1])))
            self.assertTrue(torch.all(torch.eq(stiefel.cotan[0], self.cotan[0] + d_cotan[0])))
            self.assertTrue(torch.all(torch.eq(stiefel.cotan[1], self.cotan[1] + d_cotan[1])))

        def test_action(self):
            stiefel = im.Manifolds.Stiefel(dim, self.nb_pts, gd=self.gd, tan=self.tan, cotan=self.cotan)

            nb_pts_mod = 15
            trans = im.DeformationModules.Translations(dim, nb_pts_mod, 0.2, gd=torch.randn(nb_pts_mod, dim), backend=backend)
            trans.fill_controls(torch.randn_like(trans.manifold.gd))

            man = stiefel.infinitesimal_action(trans.field_generator())

            self.assertIsInstance(man, im.Manifolds.Stiefel)

        def test_inner_prod_field(self):
            stiefel = im.Manifolds.Stiefel(dim, self.nb_pts, gd=self.gd, tan=self.tan, cotan=self.cotan)

            nb_pts_mod = 15
            trans = im.DeformationModules.Translations(dim, nb_pts_mod, 0.2, gd=torch.randn(nb_pts_mod, dim), backend=backend)
            trans.fill_controls(torch.rand_like(trans.manifold.gd))

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

            stiefel = im.Manifolds.Stiefel(dim, self.nb_pts)

            self.assertTrue(gradcheck(fill_gd, (self.gd_pts, self.gd_mat), raise_exception=False))
            self.assertTrue(gradcheck(fill_tan, (self.tan_pts, self.tan_mat), raise_exception=False))
            self.assertTrue(gradcheck(fill_cotan, (self.cotan_pts, self.cotan_mat), raise_exception=False))

        def test_gradcheck_add(self):
            def add_gd(gd_pts, gd_mat):
                stiefel.fill_gd(self.gd)
                stiefel.add_gd((gd_pts, gd_mat))
                return stiefel.gd[0], stiefel.gd[1]

            def add_tan(tan_pts, tan_mat):
                stiefel.fill_tan(self.tan)
                stiefel.add_tan((tan_pts, tan_mat))
                return stiefel.tan[0], stiefel.tan[1]

            def add_cotan(cotan_pts, cotan_mat):
                stiefel.fill_cotan(self.cotan)
                stiefel.add_cotan((cotan_pts, cotan_mat))
                return stiefel.cotan[0], stiefel.cotan[1]

            stiefel = im.Manifolds.Stiefel(dim, self.nb_pts)

            self.gd[0].requires_grad_()
            self.gd[1].requires_grad_()
            self.tan[0].requires_grad_()
            self.tan[1].requires_grad_()
            self.cotan[0].requires_grad_()
            self.cotan[1].requires_grad_()

            gd_mul = (torch.rand_like(self.gd[0], requires_grad=True),
                      torch.rand_like(self.gd[1], requires_grad=True))
            tan_mul = (torch.rand_like(self.tan[0], requires_grad=True),
                       torch.rand_like(self.tan[1], requires_grad=True))
            cotan_mul = (torch.rand_like(self.cotan[0], requires_grad=True),
                         torch.rand_like(self.cotan[1], requires_grad=True))

            self.assertTrue(gradcheck(add_gd, gd_mul, raise_exception=False))
            self.assertTrue(gradcheck(add_tan, tan_mul, raise_exception=False))
            self.assertTrue(gradcheck(add_cotan, cotan_mul, raise_exception=False))

        def test_gradcheck_action(self):
            def action(gd_pts, gd_mat, controls):
                module = im.DeformationModules.ImplicitModule1(dim, self.nb_pts, 0.001, C, nu=0.01, gd=(gd_pts, gd_mat), backend=backend)
                module.fill_controls(controls)
                stiefel = im.Manifolds.Stiefel(dim, self.nb_pts, gd=(gd_pts, gd_mat))
                man = stiefel.infinitesimal_action(module.field_generator())
                return man.gd[0], man.gd[1], man.tan[0], man.tan[1]

            self.gd_pts.requires_grad_()
            self.gd_mat.requires_grad_()

            controls = 0.1*torch.randn(1, requires_grad=True)
            C = torch.randn(self.nb_pts, dim, 1)

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

    return TestStiefel


class TestStiefel2D_Torch(make_test_stiefel(2, 'torch')):
    pass


class TestStiefel2D_KeOps(make_test_stiefel(2, 'keops')):
    pass


class TestStiefel3D_Torch(make_test_stiefel(3, 'torch')):
    pass


class TestStiefel3D_KeOps(make_test_stiefel(3, 'keops')):
    pass

if __name__ == '__main__':
    unittest.main()
