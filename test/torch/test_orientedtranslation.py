import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest

import torch

import implicitmodules.torch as im

torch.set_default_tensor_type(torch.DoubleTensor)


def make_test_orientedtranslations(dim, transport, backend):
    class TestOrientedTranslations(unittest.TestCase):
        def setUp(self):
            self.nb_pts = 10
            self.sigma = 0.001
            self.gd_pos = torch.randn(self.nb_pts, dim, requires_grad=True)
            self.gd_dir = torch.randn(self.nb_pts, dim, requires_grad=True)
            self.mom_pos = torch.randn(self.nb_pts, dim, requires_grad=True)
            self.mom_dir = torch.randn(self.nb_pts, dim, requires_grad=True)
            self.gd = (self.gd_pos, self.gd_dir)
            self.mom = (self.mom_pos, self.mom_dir)
            self.controls = torch.randn(self.nb_pts, requires_grad=True)

            self.oriented = im.DeformationModules.OrientedTranslations(dim, self.nb_pts, self.sigma, transport=transport, gd=self.gd, cotan=self.mom, backend=backend)

        def test_call(self):
            points = torch.rand(100, dim)

            result = self.oriented(points)

            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(result.shape, points.shape)

            self.oriented.fill_controls(torch.zeros_like(self.controls))
            result = self.oriented(points)
        
            self.assertEqual(torch.all(torch.eq(result, torch.zeros_like(result))), True)

        def test_field_generator(self):
            self.assertIsInstance(self.oriented.field_generator(), im.StructuredFields.StructuredField_0)

        def test_cost(self):
            cost = self.oriented.cost()

            self.assertIsInstance(cost, torch.Tensor)
            self.assertEqual(cost.shape, torch.tensor(0.).shape)

            self.oriented.manifold.fill_gd_zeros()
            cost = self.oriented.cost()

            self.assertEqual(cost, torch.tensor([0.]))

        def test_compute_geodesic_control(self):
            self.oriented.compute_geodesic_control(self.oriented.manifold)

            self.assertIsInstance(self.oriented.controls, torch.Tensor)
            self.assertEqual(self.oriented.controls.shape, torch.Size([self.nb_pts]))

        def test_gradcheck_call(self):
            def call(gd_pos, gd_dir, controls, points):
                self.oriented.fill_controls(controls)
                self.oriented.manifold.gd = (gd_pos, gd_dir)

                return self.oriented(points)

            points = torch.randn(10, dim, requires_grad=True)
            self.gd_pos.requires_grad_()
            self.gd_dir.requires_grad_()
            self.controls.requires_grad_()

            self.assertTrue(torch.autograd.gradcheck(call, (self.gd_pos, self.gd_dir, self.controls, points), raise_exception=False))

        def test_gradcheck_cost(self):
            def cost(gd_pos, gd_dir, controls):
                self.oriented.fill_controls(controls)
                self.oriented.manifold.gd = (gd_pos, gd_dir)

                return self.oriented.cost()

            self.gd_pos.requires_grad_()
            self.gd_dir.requires_grad_()
            self.controls.requires_grad_()

            self.assertTrue(torch.autograd.gradcheck(cost, (self.gd_pos, self.gd_dir, self.controls), raise_exception=False))

        def test_gradcheck_compute_geodesic_control(self):
            def compute_geodesic_control(gd_pos, gd_dir, mom_pos, mom_dir):
                self.oriented.manifold.gd = (gd_pos, gd_dir)
                self.oriented.manifold.cotan = (mom_pos, mom_dir)

                self.oriented.compute_geodesic_control(self.oriented.manifold)

                return self.oriented.controls

            self.gd_pos.requires_grad_()
            self.gd_dir.requires_grad_()
            self.mom_pos.requires_grad_()
            self.mom_dir.requires_grad_()

            self.assertTrue(torch.autograd.gradcheck(compute_geodesic_control, (self.gd_pos, self.gd_dir, self.mom_pos, self.mom_dir), raise_exception=False))

        def test_hamiltonian_control_grad_zero(self):
            self.oriented.fill_controls_zero()
            h = im.HamiltonianDynamic.Hamiltonian([self.oriented])
            h.geodesic_controls()

            [d_controls] = torch.autograd.grad(h(), [self.oriented.controls])

            self.assertTrue(torch.allclose(d_controls, torch.zeros_like(d_controls)))

    return TestOrientedTranslations


class TestOrientedTranslations2D_Vector_Torch(make_test_orientedtranslations(2, 'vector', 'torch')):
    pass


class TestOrientedTranslations2D_Surface_Torch(make_test_orientedtranslations(2, 'orthogonal', 'torch')):
    pass


class TestOrientedTranslations3D_Vector_Torch(make_test_orientedtranslations(3, 'vector', 'torch')):
    pass


class TestOrientedTranslations3D_Surface_Torch(make_test_orientedtranslations(3, 'orthogonal', 'torch')):
    pass


# class TestOrientedTranslations2D_Vector_Keops(make_test_orientedtranslations(2, 'vector', 'keops')):
#     pass


# class TestOrientedTranslations2D_Surface_Keops(make_test_orientedtranslations(2, 'surface', 'keops')):
#     pass


# class TestOrientedTranslations3D_Vector_Keops(make_test_orientedtranslations(3, 'vector', 'keops')):
#     pass


# class TestOrientedTranslations3D_Surface_Keops(make_test_orientedtranslations(3, 'surface', 'keops')):
#     pass

