import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest

import torch

import implicitmodules.torch as im

torch.set_default_tensor_type(torch.DoubleTensor)

class TestTranslations2D(unittest.TestCase):
    def setUp(self):
        self.nb_pts = 10
        self.dim = 2
        self.sigma = 0.5
        self.gd = torch.rand(self.nb_pts, self.dim).view(-1)
        self.mom = torch.rand(self.nb_pts, self.dim).view(-1)
        self.controls = torch.rand(self.nb_pts, self.dim).view(-1)
        self.landmarks = im.deformationmodules.Landmarks(self.dim, self.nb_pts, gd=self.gd, cotan=self.mom)
        self.trans = im.deformationmodules.Translations(self.landmarks, self.sigma)

    def test_call(self):
        points = torch.rand(100, self.dim)

        result = self.trans(points)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, points.shape)

        self.trans.fill_controls(torch.zeros_like(self.trans.controls))
        result = self.trans(points)
        
        self.assertEqual(torch.all(torch.eq(result, torch.zeros_like(result))), True)

    def test_field_generator(self):
        self.assertIsInstance(self.trans.field_generator(), im.structuredfield.StructuredField_0)

    def test_cost(self):
        cost = self.trans.cost()

        self.assertIsInstance(cost, torch.Tensor)
        self.assertEqual(cost.shape, torch.tensor(0.).shape)

        self.trans.manifold.fill_gd(torch.zeros_like(self.gd))
        self.trans.manifold.fill_gd(torch.zeros_like(self.mom))
        cost = self.trans.cost()

        self.assertEqual(cost, torch.tensor([0.]))

    def test_compute_geodesic_control(self):
        self.trans.compute_geodesic_control(self.trans.manifold)

        self.assertIsInstance(self.trans.controls, torch.Tensor)
        self.assertEqual(self.trans.controls.shape, self.gd.shape)

    def test_gradcheck_call(self):
        def call(gd, controls, points):
            self.trans.fill_controls(controls)
            self.trans.manifold.fill_gd(gd)

            return self.trans(points)
            
        points = torch.rand(10, self.dim, requires_grad=True)
        self.gd.requires_grad_()
        self.controls.requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(call, (self.gd, self.controls, points), raise_exception=False))

    def test_gradcheck_cost(self):
        def cost(gd, controls):
            self.trans.fill_controls(controls)
            self.trans.manifold.fill_gd(gd)

            return self.trans.cost()

        self.gd.requires_grad_()
        self.controls.requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(cost, (self.gd, self.controls), raise_exception=False))

    def test_gradcheck_compute_geodesic_control(self):
        def compute_geodesic_control(gd, mom):
            self.trans.manifold.gd = gd
            self.trans.manifold.cotan = mom

            self.trans.compute_geodesic_control(self.trans.manifold)

            return self.trans.controls

        self.gd.requires_grad_()
        self.mom.requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(compute_geodesic_control, (100.*self.gd, self.mom), raise_exception=False))

    def test_hamiltonian_control_grad_zero(self):
        self.trans.fill_controls(torch.zeros_like(self.trans.controls, requires_grad=True))
        h = im.hamiltonian.Hamiltonian([self.trans])
        h.geodesic_controls()

        [d_controls] = torch.autograd.grad(h(), [self.trans.controls])

        self.assertTrue(torch.allclose(d_controls, torch.zeros_like(d_controls)))


class TestSilentPoints2D(unittest.TestCase):
    def setUp(self):
        self.nb_pts = 10
        self.dim = 2
        self.gd = torch.rand(self.nb_pts, self.dim).view(-1)
        self.mom = torch.rand(self.nb_pts, self.dim).view(-1)
        self.controls = torch.rand(self.nb_pts, self.dim).view(-1)
        self.landmarks = im.manifold.Landmarks(self.dim, self.nb_pts, gd=self.gd, cotan=self.mom)
        self.silent_points = im.deformationmodules.SilentPoints(self.landmarks)
        self.silent_points.fill_controls(self.controls)

    def test_call(self):
        points = torch.rand(100, self.dim)

        result = self.silent_points(points)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, points.shape)
        self.assertEqual(torch.all(torch.eq(result, torch.zeros_like(points))), True)

    def test_field_generator(self):
        self.assertIsInstance(self.silent_points.field_generator(), im.structuredfield.StructuredField_Null)

    def test_cost(self):
        cost = self.silent_points.cost()

        self.assertIsInstance(cost, torch.Tensor)
        self.assertEqual(cost.shape, torch.tensor(0.).shape)
        self.assertEqual(cost, torch.tensor([0.]))

    def test_compute_geodesic_control(self):
        self.silent_points.compute_geodesic_control(self.silent_points.manifold)
        
        self.assertIsInstance(self.silent_points.controls, torch.Tensor)
        self.assertEqual(self.silent_points.controls.shape, torch.tensor([]).shape)

    def test_gradcheck_call(self):
        def call(gd, controls, points):
            self.silent_points.fill_controls(controls)
            self.silent_points.manifold.fill_gd(gd)

            return self.silent_points(points)
            
        points = torch.rand(10, self.dim, requires_grad=True)
        self.gd.requires_grad_()
        self.controls.requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(call, (self.gd, self.controls, points), raise_exception=False))

    def test_gradcheck_cost(self):
        def cost(gd, controls):
            self.silent_points.fill_controls(controls)
            self.silent_points.manifold.fill_gd(gd)

            return self.silent_points.cost()

        self.gd.requires_grad_()
        self.controls.requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(cost, (self.gd, self.controls), raise_exception=False))

    def test_gradcheck_compute_geodesic_control(self):
        def compute_geodesic_control(gd, mom):
            self.silent_points.manifold.gd = gd
            self.silent_points.manifold.cotan = mom

            self.silent_points.compute_geodesic_control(self.silent_points.manifold)

            return self.silent_points.controls

        self.gd.requires_grad_()
        self.mom.requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(compute_geodesic_control, (self.gd, self.mom), raise_exception=False))


class CompoundTest2D(unittest.TestCase):
    def setUp(self):
        self.dim = 2
        self.sigma = 0.5
        self.nb_pts_trans = 5
        self.nb_pts_silent = 12
        self.nb_pts = self.nb_pts_silent + self.nb_pts_trans
        self.gd_trans = torch.rand(self.nb_pts_trans, self.dim).view(-1)
        self.mom_trans = torch.rand(self.nb_pts_trans, self.dim).view(-1)
        self.gd_silent = torch.rand(self.nb_pts_silent, self.dim).view(-1)
        self.mom_silent = torch.rand(self.nb_pts_silent, self.dim).view(-1)
        self.landmarks_trans = im.manifold.Landmarks(self.dim, self.nb_pts_trans, gd=self.gd_trans,
                                                     cotan=self.mom_trans)
        self.landmarks_silent = im.manifold.Landmarks(self.dim, self.nb_pts_silent, gd=self.gd_silent,
                                                      cotan=self.mom_silent)
        self.trans = im.deformationmodules.Translations(self.landmarks_trans, self.sigma)
        self.silent = im.deformationmodules.SilentPoints(self.landmarks_silent)
        self.compound = im.deformationmodules.CompoundModule([self.silent, self.trans])
        self.controls_trans = torch.rand_like(self.gd_trans)
        self.controls = [None, self.controls_trans]
        self.compound.fill_controls(self.controls)

    def test_compound(self):
        self.assertEqual(self.compound.module_list, [self.silent, self.trans])
        self.assertEqual(self.compound.dim_controls, 2*self.nb_pts_trans)

        self.assertEqual(self.compound.manifold.nb_pts, self.nb_pts)

    def test_call(self):
        points = torch.rand(100, self.dim)
        self.compound.fill_controls(self.controls)

        result = self.compound(points)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, points.shape)

        self.compound.fill_controls([None, torch.zeros_like(self.controls_trans)])
        result = self.compound(points)

        self.assertEqual(torch.all(torch.eq(result, torch.zeros_like(points))), True)

    def test_field_generator(self):
        self.assertIsInstance(self.compound[0].field_generator(), im.structuredfield.StructuredField_Null)
        self.assertIsInstance(self.compound[1].field_generator(), im.structuredfield.StructuredField_0)

    def test_cost(self):
        cost = self.compound.cost()

        self.assertIsInstance(cost, torch.Tensor)
        self.assertEqual(cost.shape, torch.tensor(0.).shape)

    def test_compute_geodesic_control(self):
        self.compound.compute_geodesic_control(self.compound.manifold)

        self.assertIsInstance(self.compound.controls, list)
        self.assertIsInstance(self.compound.controls[0], torch.Tensor)
        self.assertIsInstance(self.compound.controls[1], torch.Tensor)
        
    def test_gradcheck_call(self):
        def call(gd_silent, gd_trans, controls_trans, points):
            self.compound.fill_controls([None, controls_trans])
            self.compound.manifold.fill_gd([gd_silent, gd_trans])

            return self.compound(points)

        self.gd_silent.requires_grad_()
        self.gd_trans.requires_grad_()
        self.controls_trans.requires_grad_()
        points = torch.rand(100, self.dim, requires_grad=True)

        self.assertTrue(torch.autograd.gradcheck(call, (self.gd_silent, self.gd_trans, self.controls_trans, points), raise_exception=False))

    def test_gradcheck_cost(self):
        def cost(gd_silent, gd_trans, controls_trans):
            self.compound.fill_controls([None, controls_trans])
            self.compound.manifold.fill_gd([gd_silent, gd_trans])

            return self.compound.cost()

        self.gd_silent.requires_grad_()
        self.gd_trans.requires_grad_()
        self.controls_trans.requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(cost, (self.gd_silent, self.gd_trans, self.controls_trans), raise_exception=False))

    def test_gradcheck_compute_geodesic_control(self):
        def compute_geodesic_control(gd_silent, gd_trans, mom_silent, mom_trans):
            self.compound.manifold.fill_gd([gd_silent, gd_trans])
            self.compound.manifold.fill_cotan([mom_silent, mom_trans])
            self.compound.compute_geodesic_control(self.compound.manifold)

            return self.compound.controls

        self.gd_silent.requires_grad_()
        self.gd_trans.requires_grad_()
        self.mom_silent.requires_grad_()
        self.mom_trans.requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(compute_geodesic_control, (self.gd_silent, self.gd_trans, self.mom_silent, self.mom_trans), raise_exception=False))

    def test_hamiltonian_control_grad_zero(self):
        self.compound.fill_controls([torch.tensor([]), torch.zeros_like(self.compound[1].controls, requires_grad=True)])
        h = im.hamiltonian.Hamiltonian(self.compound)
        h.geodesic_controls()

        [d_controls_silent, d_controls_trans] = torch.autograd.grad(h(), self.compound.controls, allow_unused=True)

        self.assertTrue(torch.allclose(d_controls_trans, torch.zeros_like(d_controls_trans)))


if __name__ == '__main__':
    unittest.main()
