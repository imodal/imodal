import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest

import torch

import implicitmodules.torch as im

torch.set_default_tensor_type(torch.DoubleTensor)

def make_test_linear_deformation(dim):
    class TestLinearDeformation(unittest.TestCase):
        def setUp(self):
            self.gd = torch.randn(1, dim)
            self.tan = torch.randn(1, dim)
            self.mom = torch.randn(1, dim)
            self.A = torch.randn(dim, dim) + torch.eye(dim)
            self.controls = torch.randn([])

            self.linear = im.DeformationModules.LinearDeformation.build(self.A, gd=self.gd, tan=self.tan, cotan=self.mom)

        def test_field_generator(self):
            self.assertIsInstance(self.linear.field_generator(), im.StructuredFields.StructuredField_Affine)

        def test_cost(self):
            cost = self.linear.cost()

            self.assertIsInstance(cost, torch.Tensor)
            self.assertEqual(cost.shape, torch.Size([]))

            self.linear.manifold.fill_gd(torch.zeros_like(self.gd))
            self.linear.manifold.fill_gd(torch.zeros_like(self.mom))
            cost = self.linear.cost()

            self.assertEqual(cost, torch.tensor([0.]))

        def test_compute_geodesic_control(self):
            self.linear.compute_geodesic_control(self.linear.manifold)

            self.assertIsInstance(self.linear.controls, torch.Tensor)
            self.assertEqual(self.linear.controls.shape, torch.Size([]))

        def test_gradcheck_call(self):
            def call(gd, controls, points):
                self.linear.fill_controls(controls)
                self.linear.manifold.fill_gd(gd)

                return self.linear(points)

            points = torch.rand(10, dim, requires_grad=True)
            self.gd.requires_grad_()
            self.controls.requires_grad_()

            self.assertTrue(torch.autograd.gradcheck(call, (self.gd, self.controls, points), raise_exception=True))

        def test_gradcheck_cost(self):
            def cost(gd, controls):
                self.linear.fill_controls(controls)
                self.linear.manifold.fill_gd(gd)

                return self.linear.cost()

            self.gd.requires_grad_()
            self.controls.requires_grad_()

            self.assertTrue(torch.autograd.gradcheck(cost, (self.gd, self.controls), raise_exception=False))

        def test_gradcheck_compute_geodesic_control(self):
            def compute_geodesic_control(gd, mom):
                self.linear.manifold.gd = gd
                self.linear.manifold.cotan = mom

                self.linear.compute_geodesic_control(self.linear.manifold)

                return self.linear.controls

            self.gd.requires_grad_()
            self.mom.requires_grad_()

            self.assertTrue(torch.autograd.gradcheck(compute_geodesic_control, (100.*self.gd, self.mom), raise_exception=False))

        def test_hamiltonian_control_grad_zero(self):
            self.linear.manifold.gd.requires_grad_()
            self.linear.manifold.cotan.requires_grad_()
            self.linear.fill_controls(torch.zeros_like(self.linear.controls, requires_grad=True))
            h = im.HamiltonianDynamic.Hamiltonian([self.linear])
            h.geodesic_controls()

            [d_controls] = torch.autograd.grad(h(), [self.linear.controls])

            self.assertTrue(torch.allclose(d_controls, torch.zeros_like(d_controls)))

        def test_hamiltonian_speed(self):
            self.gd.requires_grad_()
            self.mom.requires_grad_()

            self.linear.manifold.fill_gd(self.gd)
            self.linear.manifold.fill_cotan(self.mom)
            self.linear.compute_geodesic_control(self.linear.manifold)

            h = im.HamiltonianDynamic.Hamiltonian([self.linear])

            speed_h = torch.autograd.grad(h(), (self.mom,))[0]

            speed_field = self.linear(self.gd.detach())

            self.assertTrue(torch.allclose(speed_h, speed_field))

    return TestLinearDeformation


class TestLinearDeformation2D(make_test_linear_deformation(2)):
    pass

