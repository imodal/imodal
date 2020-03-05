import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest
from collections import Iterable

import torch
from torch.autograd import gradcheck

import implicitmodules.torch as im

torch.set_default_tensor_type(torch.DoubleTensor)

def make_test_landmarksdirection(dim, transport, backend):
    class TestLandmarksDirection(unittest.TestCase):
        def setUp(self):
            self.nb_pts = 10
            self.gd_pts = torch.rand(self.nb_pts, dim)
            self.gd_dir = torch.rand(self.nb_pts, dim)
            self.tan_pts = torch.rand(self.nb_pts, dim)
            self.tan_dir = torch.rand(self.nb_pts, dim)
            self.cotan_pts = torch.rand(self.nb_pts, dim)
            self.cotan_dir = torch.rand(self.nb_pts, dim)

            self.gd = (self.gd_pts, self.gd_dir)
            self.tan = (self.tan_pts, self.tan_dir)
            self.cotan = (self.cotan_pts, self.cotan_dir)

        def test_constructor(self):
            man = im.Manifolds.LandmarksDirection(dim, self.nb_pts, transport, gd=self.gd, tan=self.tan, cotan=self.cotan)

            self.assertEqual(man.nb_pts, self.nb_pts)
            self.assertEqual(man.dim, dim)
            self.assertEqual(sum(man.numel_gd), self.nb_pts * (dim + dim))
            self.assertEqual(man.len_gd, 2)

            self.assertTrue(torch.all(torch.eq(man.gd[0], self.gd[0])))
            self.assertTrue(torch.all(torch.eq(man.gd[1], self.gd[1])))
            self.assertTrue(torch.all(torch.eq(man.tan[0], self.tan[0])))
            self.assertTrue(torch.all(torch.eq(man.tan[1], self.tan[1])))
            self.assertTrue(torch.all(torch.eq(man.cotan[0], self.cotan[0])))
            self.assertTrue(torch.all(torch.eq(man.cotan[1], self.cotan[1])))

            self.assertIsInstance(man.unroll_gd(), Iterable)
            self.assertIsInstance(man.unroll_tan(), Iterable)
            self.assertIsInstance(man.unroll_cotan(), Iterable)

            l_gd = man.unroll_gd()
            l_tan = man.unroll_tan()
            l_cotan = man.unroll_cotan()

            self.assertTrue(torch.all(torch.eq(l_gd[0], self.gd[0])))
            self.assertTrue(torch.all(torch.eq(l_gd[1], self.gd[1])))
            self.assertTrue(torch.all(torch.eq(l_tan[0], self.tan[0])))
            self.assertTrue(torch.all(torch.eq(l_tan[1], self.tan[1])))
            self.assertTrue(torch.all(torch.eq(l_cotan[0], self.cotan[0])))
            self.assertTrue(torch.all(torch.eq(l_cotan[1], self.cotan[1])))

            l_rolled_gd = man.roll_gd(l_gd)
            l_rolled_tan = man.roll_tan(l_tan)
            l_rolled_cotan = man.roll_cotan(l_cotan)

            self.assertTrue(torch.all(torch.eq(l_rolled_gd[0], self.gd[0])))
            self.assertTrue(torch.all(torch.eq(l_rolled_gd[1], self.gd[1])))
            self.assertTrue(torch.all(torch.eq(l_rolled_tan[0], self.tan[0])))
            self.assertTrue(torch.all(torch.eq(l_rolled_tan[1], self.tan[1])))
            self.assertTrue(torch.all(torch.eq(l_rolled_cotan[0], self.cotan[0])))
            self.assertTrue(torch.all(torch.eq(l_rolled_cotan[1], self.cotan[1])))

        def test_clone(self):
            man = im.Manifolds.LandmarksDirection(dim, self.nb_pts, transport, gd=self.gd, tan=self.tan, cotan=self.cotan)
            man2 = man.clone()

            self.assertNotEqual(id(man), id(man2))
            self.assertNotEqual(id(man.gd[0]), id(man2.gd[0]))
            self.assertNotEqual(id(man.gd[1]), id(man2.gd[1]))
            self.assertNotEqual(id(man.tan[0]), id(man2.tan[0]))
            self.assertNotEqual(id(man.tan[1]), id(man2.tan[1]))
            self.assertNotEqual(id(man.cotan[0]), id(man2.cotan[0]))
            self.assertNotEqual(id(man.cotan[1]), id(man2.cotan[1]))

            self.assertEqual(man.transport, man2.transport)
            self.assertTrue(torch.all(torch.eq(man.gd[0], self.gd[0])))
            self.assertTrue(torch.all(torch.eq(man.gd[1], self.gd[1])))
            self.assertTrue(torch.all(torch.eq(man.tan[0], self.tan[0])))
            self.assertTrue(torch.all(torch.eq(man.tan[1], self.tan[1])))
            self.assertTrue(torch.all(torch.eq(man.cotan[0], self.cotan[0])))
            self.assertTrue(torch.all(torch.eq(man.cotan[1], self.cotan[1])))


        def test_fill(self):
            man = im.Manifolds.LandmarksDirection(dim, self.nb_pts, transport)

            man.fill_gd(self.gd, copy=True)
            man.fill_tan(self.tan, copy=True)
            man.fill_cotan(self.cotan, copy=True)

            self.assertTrue(torch.all(torch.eq(man.gd[0], self.gd[0])))
            self.assertTrue(torch.all(torch.eq(man.gd[1], self.gd[1])))
            self.assertTrue(torch.all(torch.eq(man.tan[0], self.tan[0])))
            self.assertTrue(torch.all(torch.eq(man.tan[1], self.tan[1])))
            self.assertTrue(torch.all(torch.eq(man.cotan[0], self.cotan[0])))
            self.assertTrue(torch.all(torch.eq(man.cotan[1], self.cotan[1])))

            self.assertNotEqual(id(man.gd[0]), id(self.gd[0]))
            self.assertNotEqual(id(man.gd[1]), id(self.gd[1]))
            self.assertNotEqual(id(man.tan[0]), id(self.tan[0]))
            self.assertNotEqual(id(man.tan[1]), id(self.tan[1]))
            self.assertNotEqual(id(man.cotan[0]), id(self.cotan[0]))
            self.assertNotEqual(id(man.cotan[1]), id(self.cotan[1]))

        def test_assign(self):
            man = im.Manifolds.LandmarksDirection(dim, self.nb_pts, transport)

            man.gd = self.gd
            man.tan = self.tan
            man.cotan = self.cotan

            self.assertTrue(torch.all(torch.eq(man.gd[0], self.gd[0])))
            self.assertTrue(torch.all(torch.eq(man.gd[1], self.gd[1])))
            self.assertTrue(torch.all(torch.eq(man.tan[0], self.tan[0])))
            self.assertTrue(torch.all(torch.eq(man.tan[1], self.tan[1])))
            self.assertTrue(torch.all(torch.eq(man.cotan[0], self.cotan[0])))
            self.assertTrue(torch.all(torch.eq(man.cotan[1], self.cotan[1])))

            self.assertEqual(id(man.gd[0]), id(self.gd[0]))
            self.assertEqual(id(man.gd[1]), id(self.gd[1]))
            self.assertEqual(id(man.tan[0]), id(self.tan[0]))
            self.assertEqual(id(man.tan[1]), id(self.tan[1]))
            self.assertEqual(id(man.cotan[0]), id(self.cotan[0]))
            self.assertEqual(id(man.cotan[1]), id(self.cotan[1]))

        def test_add(self):
            man = im.Manifolds.LandmarksDirection(dim, self.nb_pts, transport, gd=self.gd, tan=self.tan, cotan=self.cotan)

            d_gd = (torch.randn(self.nb_pts, dim),
                    torch.randn(self.nb_pts, dim))
            d_tan = (torch.randn(self.nb_pts, dim),
                     torch.randn(self.nb_pts, dim))
            d_cotan = (torch.randn(self.nb_pts, dim),
                       torch.randn(self.nb_pts, dim))

            man.add_gd(d_gd)
            man.add_tan(d_tan)
            man.add_cotan(d_cotan)

            self.assertTrue(torch.all(torch.eq(man.gd[0], self.gd[0] + d_gd[0])))
            self.assertTrue(torch.all(torch.eq(man.gd[1], self.gd[1] + d_gd[1])))
            self.assertTrue(torch.all(torch.eq(man.tan[0], self.tan[0] + d_tan[0])))
            self.assertTrue(torch.all(torch.eq(man.tan[1], self.tan[1] + d_tan[1])))
            self.assertTrue(torch.all(torch.eq(man.cotan[0], self.cotan[0] + d_cotan[0])))
            self.assertTrue(torch.all(torch.eq(man.cotan[1], self.cotan[1] + d_cotan[1])))

        def test_action(self):
            landmarksdirection = im.Manifolds.LandmarksDirection(dim, self.nb_pts, transport, gd=self.gd, tan=self.tan, cotan=self.cotan)

            nb_pts_mod = 15
            trans = im.DeformationModules.Translations(dim, nb_pts_mod, 0.2, gd=torch.randn(nb_pts_mod, dim), backend=backend)
            trans.fill_controls(torch.randn_like(trans.manifold.gd))

            man = landmarksdirection.infinitesimal_action(trans.field_generator())

            self.assertIsInstance(man, im.Manifolds.LandmarksDirection)

        def test_inner_prod_field(self):
            landmarkdirection = im.Manifolds.LandmarksDirection(dim, self.nb_pts, transport, gd=self.gd, tan=self.tan, cotan=self.cotan)

            nb_pts_mod = 15
            trans = im.DeformationModules.Translations(dim, nb_pts_mod, 0.2, gd=torch.randn(nb_pts_mod, dim), backend=backend)
            trans.fill_controls(torch.rand_like(trans.manifold.gd))

            inner_prod = landmarkdirection.inner_prod_field(trans.field_generator())

            self.assertIsInstance(inner_prod, torch.Tensor)
            self.assertEqual(inner_prod.shape, torch.Size([]))

        def test_gradcheck_fill(self):
            def fill_gd(gd_pts, gd_mat):
                man.fill_gd((gd_pts, gd_mat))
                return man.gd[0], man.gd[1]

            def fill_tan(tan_pts, tan_mat):
                man.fill_tan((tan_pts, tan_mat))
                return man.tan[0], man.tan[1]

            def fill_cotan(cotan_pts, cotan_mat):
                man.fill_cotan((cotan_pts, cotan_mat))
                return man.cotan[0], man.cotan[1]

            self.gd_pts.requires_grad_()
            self.gd_dir.requires_grad_()
            self.tan_pts.requires_grad_()
            self.tan_dir.requires_grad_()
            self.cotan_pts.requires_grad_()
            self.cotan_dir.requires_grad_()

            man = im.Manifolds.LandmarksDirection(dim, self.nb_pts, transport)

            self.assertTrue(gradcheck(fill_gd, (self.gd_pts, self.gd_dir), raise_exception=False))
            self.assertTrue(gradcheck(fill_tan, (self.tan_pts, self.tan_dir), raise_exception=False))
            self.assertTrue(gradcheck(fill_cotan, (self.cotan_pts, self.cotan_dir), raise_exception=False))

        def test_gradcheck_add(self):
            def add_gd(gd_pts, gd_dir):
                man.fill_gd(self.gd)
                man.add_gd((gd_pts, gd_dir))
                return man.gd[0], man.gd[1]

            def add_tan(tan_pts, tan_dir):
                man.fill_tan(self.tan)
                man.add_tan((tan_pts, tan_dir))
                return man.tan[0], man.tan[1]

            def add_cotan(cotan_pts, cotan_dir):
                man.fill_cotan(self.cotan)
                man.add_cotan((cotan_pts, cotan_dir))
                return man.cotan[0], man.cotan[1]

            man = im.Manifolds.LandmarksDirection(dim, self.nb_pts, transport)

            self.gd[0].requires_grad_()
            self.gd[1].requires_grad_()
            self.tan[0].requires_grad_()
            self.tan[1].requires_grad_()
            self.cotan[0].requires_grad_()
            self.cotan[1].requires_grad_()

            d_gd = (torch.randn_like(self.gd[0], requires_grad=True),
                    torch.randn_like(self.gd[1], requires_grad=True))
            d_tan = (torch.randn_like(self.tan[0], requires_grad=True),
                     torch.randn_like(self.tan[1], requires_grad=True))
            d_cotan = (torch.randn_like(self.cotan[0], requires_grad=True),
                       torch.randn_like(self.cotan[1], requires_grad=True))

            self.assertTrue(gradcheck(add_gd, d_gd, raise_exception=False))
            self.assertTrue(gradcheck(add_tan, d_tan, raise_exception=False))
            self.assertTrue(gradcheck(add_cotan, d_cotan, raise_exception=False))

        # def test_gradcheck_action(self):
        #     def action(gd_pts, gd_dir, controls):
        #         module = im.DeformationModules.ImplicitModule1(dim, self.nb_pts, 0.001, C, nu=0.01, gd=(gd_pts, gd_dir), backend=backend)
        #         module.fill_controls(controls)
        #         landmarksdirection = im.Manifolds.LandmarksDirection(dim, self.nb_pts, transport, gd=(gd_pts, gd_dir))
        #         man = landmarksdirection.infinitesimal_action(module.field_generator())
        #         return man.gd[0], man.gd[1], man.tan[0], man.tan[1]

        #     self.gd_pts.requires_grad_()
        #     self.gd_dir.requires_grad_()

        #     gd_pts = torch.randn(self.nb_pts, dim, requires_grad=True)
        #     gd_mat = torch.randn(self.nb_pts, dim, dim, requires_grad=True)

        #     controls = 0.1*torch.randn(1, requires_grad=True)
        #     C = torch.randn(self.nb_pts, dim, 1)

        #     self.assertTrue(gradcheck(action, (gd_pts, gd_mat, controls), raise_exception=False))

    return TestLandmarksDirection


class TestLandmarksDirection2D_Vector_Torch(make_test_landmarksdirection(2, 'vector', 'torch')):
    pass


class TestLandmarksDirection2D_Surface_Torch(make_test_landmarksdirection(2, 'orthogonal', 'torch')):
    pass


class TestLandmarksDirection3D_Vector_Torch(make_test_landmarksdirection(3, 'vector', 'torch')):
    pass


class TestLandmarksDirection3D_Surface_Torch(make_test_landmarksdirection(3, 'orthogonal', 'torch')):
    pass


# class TestLandmarksDirection2D_Vector_KeOps(make_test_landmarksdirection(2, 'vector', 'keops')):
#     pass


# class TestLandmarksDirection2D_Surface_KeOps(make_test_landmarksdirection(2, 'surface', 'keops')):
#     pass


# class TestLandmarksDirection3D_Vector_KeOps(make_test_landmarksdirection(3, 'vector', 'keops')):
#     pass


# class TestLandmarksDirection3D_Surface_KeOps(make_test_landmarksdirection(3, 'surface', 'keops')):
#     pass


if __name__ == '__main__':
    unittest.main()

