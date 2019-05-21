import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest

import torch

import implicitmodules.torch as im

torch.set_default_tensor_type(torch.DoubleTensor)

class TestImplicitModule0(unittest.TestCase):
    def setUp(self):
        self.nu = 0.01
        self.nb_pts = 10
        self.dim = 2
        self.sigma = 0.5
        self.gd = torch.rand(self.nb_pts, self.dim).view(-1)
        self.mom = torch.rand(self.nb_pts, self.dim).view(-1)
        self.controls = torch.rand(self.nb_pts, self.dim).view(-1)
        self.landmarks = im.Manifolds.Landmarks(self.dim, self.nb_pts, gd=self.gd, cotan=self.mom)
        self.implicit = im.DeformationModules.ImplicitModule0(self.landmarks, self.sigma, self.nu)

    def test_call(self):
        points = torch.rand(100, self.dim)

        result = self.implicit(points)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, points.shape)

        self.implicit.fill_controls(torch.zeros_like(self.implicit.controls))
        result = self.implicit(points)
        
        self.assertEqual(torch.all(torch.eq(result, torch.zeros_like(result))), True)

    def test_field_generator(self):
        self.assertIsInstance(self.implicit.field_generator(), im.StructuredFields.Abstract.StructuredField)

    def test_cost(self):
        cost = self.implicit.cost()

        self.assertIsInstance(cost, torch.Tensor)
        self.assertEqual(cost.shape, torch.tensor(0.).shape)

        self.implicit.manifold.fill_gd(torch.zeros_like(self.gd))
        self.implicit.manifold.fill_gd(torch.zeros_like(self.mom))
        cost = self.implicit.cost()

        self.assertEqual(cost, torch.tensor([0.]))

    def test_compute_geodesic_control(self):
        self.implicit.compute_geodesic_control(self.implicit.manifold)

        self.assertIsInstance(self.implicit.controls, torch.Tensor)
        self.assertEqual(self.implicit.controls.shape, self.gd.shape)

    def test_gradcheck_call(self):
        def call(gd, controls, points):
            self.implicit.fill_controls(controls)
            self.implicit.manifold.fill_gd(gd)

            return self.implicit(points)
            
        points = torch.rand(10, self.dim, requires_grad=True)
        self.gd.requires_grad_()
        self.controls.requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(call, (self.gd, self.controls, points), raise_exception=False))

    def test_gradcheck_cost(self):
        def cost(gd, controls):
            self.implicit.fill_controls(controls)
            self.implicit.manifold.fill_gd(gd)

            return self.implicit.cost()

        self.gd.requires_grad_()
        self.controls.requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(cost, (self.gd, self.controls), raise_exception=False))

    def test_gradcheck_compute_geodesic_control(self):
        def compute_geodesic_control(gd, mom):
            self.implicit.manifold.gd = gd
            self.implicit.manifold.cotan = mom

            self.implicit.compute_geodesic_control(self.implicit.manifold)

            return self.implicit.controls

        self.gd.requires_grad_()
        self.mom.requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(compute_geodesic_control, (100.*self.gd, self.mom), raise_exception=False))

    def test_hamiltonian_control_grad_zero(self):
        self.implicit.fill_controls(torch.zeros_like(self.implicit.controls, requires_grad=True))
        h = im.HamiltonianDynamic.Hamiltonian([self.implicit])
        h.geodesic_controls()

        [d_controls] = torch.autograd.grad(h(), [self.implicit.controls])

        self.assertTrue(torch.allclose(d_controls, torch.zeros_like(d_controls)))



class TestImplicitModule1(unittest.TestCase):
    def setUp(self):
        self.dim = 2
        self.nb_pts = 7
        self.gd = (torch.rand(self.nb_pts, 2).view(-1), torch.rand(self.nb_pts, 2, 2).view(-1))
        self.tan = (torch.rand(self.nb_pts, 2).view(-1), torch.rand(self.nb_pts, 2, 2).view(-1))
        self.cotan = (torch.rand(self.nb_pts, 2).view(-1), torch.rand(self.nb_pts, 2, 2).view(-1))
        self.stiefel = im.Manifolds.Stiefel(self.dim, self.nb_pts, gd=self.gd, tan=self.tan, cotan=self.cotan)
        self.dim_controls = 2
        self.controls = torch.rand(self.dim_controls)
        self.C = torch.rand(self.nb_pts, self.dim, self.dim_controls)
        self.nu = 1e-3
        self.sigma = 1.0

        self.implicit = im.DeformationModules.ImplicitModule1(self.stiefel, self.C, self.sigma, self.nu)
        self.implicit.fill_controls(self.controls)

    def test_call(self):
        points = torch.rand(10, 2)
        speed = self.implicit(points)

        self.assertIsInstance(speed, torch.Tensor)
        self.assertEqual(speed.shape, points.shape)

    def test_field_generator(self):
        self.assertIsInstance(self.implicit.field_generator(), im.StructuredFields.StructuredField_p)

    def test_cost(self):
        cost = self.implicit.cost()

        self.assertIsInstance(cost, torch.Tensor)
        self.assertEqual(cost.shape, torch.Size([]))

    def test_compute_geodesic_control(self):
        self.implicit.compute_geodesic_control(self.implicit.manifold)

        self.assertIsInstance(self.implicit.controls, torch.Tensor)
        self.assertEqual(self.implicit.controls.shape, self.controls.shape)

    def test_gradcheck_call(self):
        def call(gd_pts, gd_mat, controls, points):
            self.implicit.manifold.fill_gd((gd_pts, gd_mat))
            self.implicit.fill_controls(controls)

            return self.implicit(points)

        self.gd[0].requires_grad_()
        self.gd[1].requires_grad_()
        self.controls.requires_grad_()
        points = torch.rand(100, self.dim, requires_grad=True)

        self.assertTrue(torch.autograd.gradcheck(call, (self.gd[0], self.gd[1], self.controls, points), raise_exception=False))

    def test_gradcheck_cost(self):
        def cost(gd_pts, gd_mat, controls):
            self.implicit.manifold.fill_gd((gd_pts, gd_mat))
            self.implicit.fill_controls(controls)

            return self.implicit.cost()

        self.gd[0].requires_grad_()
        self.gd[1].requires_grad_()
        self.controls.requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(cost, (self.gd[0], self.gd[1], self.controls), raise_exception=False))

    # TODO: Gradcheck
    def test_hamiltonian_control_grad_zero(self):
        self.implicit.fill_controls(torch.zeros_like(self.implicit.controls, requires_grad=True))
        h = im.HamiltonianDynamic.Hamiltonian([self.implicit])
        h.geodesic_controls()
        h.module[0].fill_controls(h.module[0].controls.requires_grad_())

        [d_controls] = torch.autograd.grad(h(), [self.implicit.controls])

        self.assertTrue(torch.allclose(d_controls, torch.zeros_like(d_controls)))


    # TODO: make compute_geodesic_control() differentiable wrt gd and cotan
    def test_gradcheck_compute_geodesic_control(self):
        def compute_geodesic_control(gd_pts, gd_mat, mom_pts, mom_mat):
            self.implicit.manifold.fill_gd((gd_pts, gd_mat))
            self.implicit.manifold.fill_cotan((mom_pts, mom_mat))
            self.implicit.compute_geodesic_control(self.implicit.manifold)

            return self.implicit.controls

        self.gd[0].requires_grad_()
        self.gd[1].requires_grad_()
        self.cotan[0].requires_grad_()
        self.cotan[1].requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(compute_geodesic_control, (self.gd[0], self.gd[1], self.cotan[0], self.cotan[1]), raise_exception=False))


if __name__ == '__main__':
    unittest.main()
