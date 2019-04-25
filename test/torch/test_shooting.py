import sys
sys.path.append("../../../")

import unittest

import torch

import implicitmodules.torch as dm

torch.set_default_tensor_type(torch.DoubleTensor)


class TestShooting(unittest.TestCase):
    def setUp(self):
        self.it = 2
        self.m = 4
        self.gd = torch.rand(self.m, 2, requires_grad=True).view(-1)
        self.mom = torch.rand(self.m, 2, requires_grad=True).view(-1)
        self.landmarks = dm.manifold.Landmarks(2, self.m, gd=self.gd, cotan=self.mom)
        self.trans = dm.deformationmodules.Translations(self.landmarks, 0.5)
        self.h = dm.hamiltonian.Hamiltonian([self.trans])
        self.method = "rk4"

    def test_shooting(self):
        intermediates = dm.shooting.shoot(self.h, self.it, self.method)

        self.assertIsInstance(self.h.module.manifold.gd, list)
        self.assertIsInstance(self.h.module.manifold.gd[0], torch.Tensor)
        self.assertIsInstance(self.h.module.manifold.cotan, list)
        self.assertIsInstance(self.h.module.manifold.cotan[0], torch.Tensor)

        self.assertEqual(self.h.module.manifold.gd[0].shape, self.gd.shape)
        self.assertEqual(self.h.module.manifold.cotan[0].shape, self.mom.shape)

        #self.assertEqual(len(intermediates), self.it)

    def test_shooting_zero(self):
        mom = torch.zeros_like(self.mom, requires_grad=True)
        self.h.module.manifold.fill_cotan([mom])
        dm.shooting.shoot(self.h, self.it, self.method)

        self.assertTrue(torch.allclose(self.h.module.manifold.gd[0], self.gd))
        self.assertTrue(torch.allclose(self.h.module.manifold.cotan[0], mom))

    def test_shooting_rand(self):
        dm.shooting.shoot(self.h, self.it, self.method)

        self.assertFalse(torch.allclose(self.h.module.manifold.gd[0], self.gd[0]))
        self.assertFalse(torch.allclose(self.h.module.manifold.cotan[0], self.mom[0]))

    # def test_shooting_precision(self):
    #     dm.shooting.shoot(self.h, it=2000)
    #     gd_torchdiffeq = self.h.module.manifold.gd[0]
    #     mom_torchdiffeq = self.h.module.manifold.cotan[0]

    #     self.h.module.manifold.fill_gd([self.gd])
    #     self.h.module.manifold.fill_cotan([self.mom])
    #     dm.shooting.shoot_euler(self.h, it=2000)
    #     print(gd_torchdiffeq)
    #     print(self.h.module.manifold.gd[0])

    #     self.assertTrue(torch.allclose(gd_torchdiffeq, self.h.module.manifold.gd[0], rtol=0.5))
    #     self.assertTrue(torch.allclose(mom_torchdiffeq, self.h.module.manifold.cotan[0], rtol=0.5))

    def test_gradcheck_shoot(self):
        def shoot(gd, mom):
            self.h.module.manifold.fill_gd([gd])
            self.h.module.manifold.fill_cotan([mom])

            dm.shooting.shoot(self.h, self.it, self.method)

            return self.h.module.manifold.gd[0], self.h.module.manifold.cotan[0]

        self.gd.requires_grad_()
        self.mom.requires_grad_()

        # We multiply GD by 400. as it seems gradcheck is very sensitive to
        # badly conditioned problems
        # TODO: be sure it is because of that
        self.assertTrue(torch.autograd.gradcheck(shoot, (100.0*self.gd, self.mom), raise_exception=True))

class TestShootingEuler(unittest.TestCase):
    def setUp(self):
        self.it = 2
        self.m = 4
        self.gd = torch.rand(self.m, 2, requires_grad=True).view(-1)
        self.mom = torch.rand(self.m, 2, requires_grad=True).view(-1)
        self.landmarks = dm.manifold.Landmarks(2, self.m, gd=self.gd, cotan=self.mom)
        self.trans = dm.deformationmodules.Translations(self.landmarks, 0.5)
        self.h = dm.hamiltonian.Hamiltonian([self.trans])
        self.method = "torch_euler"

    def test_shooting(self):
        intermediates = dm.shooting.shoot(self.h, self.it, self.method)

        self.assertIsInstance(self.h.module.manifold.gd, list)
        self.assertIsInstance(self.h.module.manifold.gd[0], torch.Tensor)
        self.assertIsInstance(self.h.module.manifold.cotan, list)
        self.assertIsInstance(self.h.module.manifold.cotan[0], torch.Tensor)

        self.assertEqual(self.h.module.manifold.gd[0].shape, self.gd.shape)
        self.assertEqual(self.h.module.manifold.cotan[0].shape, self.mom.shape)

        self.assertEqual(len(intermediates), self.it)

    def test_shooting_zero(self):
        mom = torch.zeros_like(self.mom, requires_grad=True)
        self.h.module.manifold.fill_cotan([mom])
        dm.shooting.shoot(self.h, self.it, self.method)

        self.assertTrue(torch.allclose(self.h.module.manifold.gd[0], self.gd))
        self.assertTrue(torch.allclose(self.h.module.manifold.cotan[0], mom))

    def test_shooting_rand(self):
        dm.shooting.shoot(self.h, self.it, self.method)

        self.assertFalse(torch.allclose(self.h.module.manifold.gd[0], self.gd[0]))
        self.assertFalse(torch.allclose(self.h.module.manifold.cotan[0], self.mom[0]))

    def test_gradcheck_shoot(self):
        def shoot(gd, mom):
            self.h.module.manifold.fill_gd([gd])
            self.h.module.manifold.fill_cotan([mom])

            dm.shooting.shoot(self.h, self.it, self.method)

            return self.h.module.manifold.gd[0], self.h.module.manifold.cotan[0]

        self.gd.requires_grad_()
        self.mom.requires_grad_()

        # We multiply GD by 400. as it seems gradcheck is very sensitive to
        # badly conditioned problems
        # TODO: be sure it is because of that
        self.assertTrue(torch.autograd.gradcheck(shoot, (100.0*self.gd, self.mom), raise_exception=True))

