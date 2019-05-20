import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest

import torch

import implicitmodules.torch as im

torch.set_default_tensor_type(torch.DoubleTensor)

class TestHamiltonian(unittest.TestCase):
    def setUp(self):
        self.nb_pts = 10
        self.sigma = 0.5
        self.gd = 100. * torch.rand(self.nb_pts, 2).view(-1)
        self.mom = 100. * torch.rand_like(self.gd).view(-1)
        self.landmarks = im.Manifolds.Landmarks(2, self.nb_pts, gd=self.gd, cotan=self.mom)
        self.controls = 100.*torch.rand_like(self.gd)

        self.trans = im.DeformationModules.Translations(self.landmarks, self.sigma)
        self.trans.fill_controls(self.controls)

        self.h = im.HamiltonianDynamic.Hamiltonian([self.trans])

    def test_good_init(self):
        self.assertIsInstance(self.h.module, im.DeformationModules.Abstract.DeformationModule)

    def test_apply_mom(self):
        self.assertIsInstance(self.h.apply_mom(), torch.Tensor)
        self.assertEqual(self.h.apply_mom().shape, torch.Size([]))

    def test_call(self):
        self.assertIsInstance(self.h(), torch.Tensor)
        self.assertEqual(self.h().shape, torch.Size([]))

    def test_geodesic_controls(self):
        self.h.geodesic_controls()
        self.assertIsInstance(self.h.module.controls, list)
        self.assertIsInstance(self.h.module[0].controls, torch.Tensor)
        self.assertTrue(self.h.module.controls[0].shape, self.controls)

    def test_gradcheck_call(self):
        def call(gd, mom, controls):
            self.h.module.manifold.fill_gd([gd])
            self.h.module.manifold.fill_cotan([mom])
            self.h.module.fill_controls([controls])

            return self.h()

        self.gd.requires_grad_()
        self.mom.requires_grad_()
        self.controls.requires_grad_()
        
        self.assertTrue(torch.autograd.gradcheck(call, (self.gd, self.mom, self.controls), raise_exception=False))

    def test_gradcheck_apply_mom(self):
        def apply_mom(gd, mom, controls):

            self.h.module.manifold.fill_gd([gd])
            self.h.module.manifold.fill_cotan([mom])
            self.h.module.fill_controls([controls])

            return self.h.apply_mom()

        self.gd.requires_grad_()
        self.mom.requires_grad_()
        self.controls.requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(apply_mom, (self.gd, self.mom, self.controls), raise_exception=False))

    def test_gradcheck_geodesic_controls(self):
        def geodesic_controls(gd, mom):
            self.h.module.manifold.fill_gd([gd])
            self.h.module.manifold.fill_cotan([mom])

            self.h.geodesic_controls()

            return self.h.module.controls

        self.gd.requires_grad_()
        self.mom.requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(geodesic_controls, (self.gd, self.mom),
                                                 raise_exception=False))


# This constitutes more as an integration test than an unit test, but using Hamiltonian with
# compound modules needs some attentions
class TestHamiltonianCompound(unittest.TestCase):
    def setUp(self):
        self.nb_pts_trans = 10
        self.nb_pts_silent = 15
        self.sigma = 0.5

        self.gd_trans = 100. * torch.rand(self.nb_pts_trans, 2).view(-1)
        self.mom_trans = 100. * torch.rand_like(self.gd_trans).view(-1)
        self.gd_silent = 100. * torch.rand(self.nb_pts_silent, 2).view(-1)
        self.mom_silent = 100. * torch.rand_like(self.gd_silent).view(-1)
        self.gd = [self.gd_trans, self.gd_silent]
        self.mom = [self.mom_trans, self.mom_silent]

        self.landmarks_trans = im.Manifolds.Landmarks(2, self.nb_pts_trans, gd=self.gd_trans, cotan=self.mom_trans)
        self.landmarks_silent = im.Manifolds.Landmarks(2, self.nb_pts_silent, gd=self.gd_silent, cotan=self.mom_silent)
        self.controls_trans = 100. * torch.rand_like(self.gd_trans)
        self.controls_silent = torch.tensor([])
        self.controls = [self.controls_trans, self.controls_silent]

        self.trans = im.DeformationModules.Translations(self.landmarks_trans, self.sigma)
        self.trans.fill_controls(self.controls[0])
        self.silent = im.DeformationModules.SilentLandmarks(self.landmarks_silent)

        self.h = im.HamiltonianDynamic.Hamiltonian([self.trans, self.silent])

    def test_good_init(self):
        self.assertIsInstance(self.h.module, im.DeformationModules.Abstract.DeformationModule)

    def test_apply_mom(self):
        self.assertIsInstance(self.h.apply_mom(), torch.Tensor)
        self.assertEqual(self.h.apply_mom().shape, torch.Size([]))

    def test_call(self):
        self.assertIsInstance(self.h(), torch.Tensor)
        self.assertEqual(self.h().shape, torch.Size([]))

    def test_geodesic_controls(self):
        self.gd_trans.requires_grad_()
        self.mom_trans.requires_grad_()
        self.h.geodesic_controls()
        self.assertIsInstance(self.h.module.controls, list)
        self.assertIsInstance(self.h.module.controls[0], torch.Tensor)
        self.assertIsInstance(self.h.module.controls[1], torch.Tensor)
        self.assertTrue(self.h.module.controls[0].shape, self.controls_trans.shape)
        self.assertTrue(self.h.module.controls[1].shape, self.controls_silent.shape)

    def test_gradcheck_call(self):
        def call(gd_trans, gd_silent, mom_trans, mom_silent, controls_trans, controls_silent):
            self.h.module.manifold.fill_gd([gd_trans, gd_silent])
            self.h.module.manifold.fill_cotan([mom_trans, mom_silent])
            self.h.module.fill_controls([controls_trans, controls_silent])

            return self.h()

        self.gd_trans.requires_grad_()
        self.gd_silent.requires_grad_()
        self.mom_trans.requires_grad_()
        self.mom_silent.requires_grad_()
        self.controls_trans.requires_grad_()
        self.controls_silent.requires_grad_()
        
        self.assertTrue(torch.autograd.gradcheck(call, (self.gd_trans, self.gd_silent, self.mom_trans, self.mom_silent, self.controls_trans, self.controls_silent), raise_exception=False))

    def test_gradcheck_apply_mom(self):
        def apply_mom(gd_trans, gd_silent, mom_trans, mom_silent, controls_trans, controls_silent):
            self.h.module.manifold.fill_gd([gd_trans, gd_silent])
            self.h.module.manifold.fill_cotan([mom_trans, mom_silent])
            self.h.module.fill_controls([controls_trans, controls_silent])

            return self.h.apply_mom()

        self.gd_trans.requires_grad_()
        self.gd_silent.requires_grad_()
        self.mom_trans.requires_grad_()
        self.mom_silent.requires_grad_()
        self.controls_trans.requires_grad_()
        self.controls_silent.requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(apply_mom, (self.gd_trans, self.gd_silent, self.mom_trans, self.mom_silent, self.controls_trans, self.controls_silent), raise_exception=False))

    def test_gradcheck_geodesic_controls(self):
        def geodesic_controls(gd_trans, gd_silent, mom_trans, mom_silent):
            self.h.module.manifold.fill_gd([gd_trans, gd_silent])
            self.h.module.manifold.fill_cotan([mom_trans, mom_silent])

            self.h.geodesic_controls()

            return self.h.module.controls

        self.gd_trans.requires_grad_()
        self.gd_silent.requires_grad_()
        self.mom_trans.requires_grad_()
        self.mom_silent.requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(geodesic_controls, (self.gd_trans, self.gd_silent, self.mom_trans, self.mom_silent), raise_exception=False))


if __name__ == '__main__':
    unittest.main()
