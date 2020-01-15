import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest

import torch

import implicitmodules.torch as im

torch.set_default_tensor_type(torch.DoubleTensor)


def make_test_localconstrainedtranslations(dim, module, backend):
    class TestLocalConstrainedTranslations(unittest.TestCase):
        def setUp(self):
            self.nb_pts = 1
            self.sigma = 0.2
            self.gd = torch.randn(self.nb_pts, dim, requires_grad=True)
            self.mom = torch.randn(self.nb_pts, dim, requires_grad=True)
            self.controls = torch.randn(self.nb_pts, requires_grad=True)

            self.constrained = module(dim, self.nb_pts, self.sigma, gd=self.gd, cotan=self.mom, backend=backend)

        def test_call(self):
            points = torch.rand(100, dim)

            result = self.constrained(points)

            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(result.shape, points.shape)

            self.constrained.fill_controls(torch.zeros_like(self.controls))
            result = self.constrained(points)
        
            self.assertEqual(torch.all(torch.eq(result, torch.zeros_like(result))), True)

        def test_field_generator(self):
            self.assertIsInstance(self.constrained.field_generator(), im.StructuredFields.StructuredField_0)

        def test_cost(self):
            cost = self.constrained.cost()

            self.assertIsInstance(cost, torch.Tensor)
            self.assertEqual(cost.shape, torch.tensor(0.).shape)

            self.constrained.manifold.fill_gd_zeros()
            cost = self.constrained.cost()

            self.assertEqual(cost, torch.tensor([0.]))

        def test_compute_geodesic_control(self):
            self.constrained.compute_geodesic_control(self.constrained.manifold)

            self.assertIsInstance(self.constrained.controls, torch.Tensor)
            self.assertEqual(self.constrained.controls.shape, torch.Size([]))

        def test_gradcheck_call(self):
            def call(gd, controls, points):
                self.constrained.fill_controls(controls)
                self.constrained.manifold.gd = gd

                return self.constrained(points)

            points = torch.randn(10, dim, requires_grad=True)
            self.gd.requires_grad_()
            self.controls.requires_grad_()

            self.assertTrue(torch.autograd.gradcheck(call, (self.gd, self.controls, points), raise_exception=False))

        def test_gradcheck_cost(self):
            def cost(gd, controls):
                self.constrained.fill_controls(controls)
                self.constrained.manifold.gd = gd

                return self.constrained.cost()

            self.gd.requires_grad_()
            self.controls.requires_grad_()

            self.assertTrue(torch.autograd.gradcheck(cost, (self.gd, self.controls), raise_exception=False))

        def test_gradcheck_compute_geodesic_control(self):
            def compute_geodesic_control(gd, mom):
                self.constrained.manifold.gd = gd
                self.constrained.manifold.cotan = mom

                self.constrained.compute_geodesic_control(self.constrained.manifold)

                return self.constrained.controls

            self.gd.requires_grad_()
            self.mom.requires_grad_()

            self.assertTrue(torch.autograd.gradcheck(compute_geodesic_control, (self.gd, self.mom), raise_exception=False))

        def test_hamiltonian_control_grad_zero(self):
            self.constrained.fill_controls_zero()
            h = im.HamiltonianDynamic.Hamiltonian([self.constrained])
            h.geodesic_controls()

            [d_controls] = torch.autograd.grad(h(), [self.constrained.controls])

            self.assertTrue(torch.allclose(d_controls, torch.zeros_like(d_controls)))

        def test_gradcheck_hamiltonian(self):

            def ham(gd, mom):
                self.constrained.manifold.gd = gd
                self.constrained.manifold.cotan = mom
                self.constrained.compute_geodesic_control(self.constrained.manifold)
                h = im.HamiltonianDynamic.Hamiltonian([self.constrained])

                return h()

            self.gd.requires_grad_()
            self.mom.requires_grad_()

            self.assertTrue(torch.autograd.gradcheck(ham, (self.gd, self.mom)))

        def test_hamiltonian_speed(self):
            self.gd.requires_grad_()
            self.mom.requires_grad_()

            self.constrained.manifold.fill_gd(self.gd)
            self.constrained.manifold.fill_cotan(self.mom)
            self.constrained.compute_geodesic_control(self.constrained.manifold)

            h = im.HamiltonianDynamic.Hamiltonian([self.constrained])

            speed_h = torch.autograd.grad(h(), (self.mom,))[0]

            speed_field = self.constrained(self.gd.detach())

            self.assertTrue(torch.allclose(speed_h, speed_field))


    return TestLocalConstrainedTranslations


class TestLocalConstrainedTranslations2D_Scaling_Torch(make_test_localconstrainedtranslations(2, im.DeformationModules.LocalScaling, 'torch')):
    pass


class TestLocalConstrainedTranslations2D_Rotation_Torch(make_test_localconstrainedtranslations(2, im.DeformationModules.LocalRotation, 'torch')):
    pass


# class TestLocalConstrainedTranslations2D_Scaling_Keops(make_test_localconstrainedtranslations(2, im.DeformationModules.LocalScaling, 'keops')):
#     pass


# class TestLocalConstrainedTranslations2D_Rotation_Keops(make_test_localconstrainedtranslations(2, im.DeformationModules.LocalRotation, 'keops')):
#     pass


