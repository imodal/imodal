import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest

import torch

import implicitmodules.torch as im

torch.set_default_tensor_type(torch.DoubleTensor)

def make_test_shooting(dim):
    class TestShooting(unittest.TestCase):
        # The different ODE solving methods we want to use and test
        methods = ["torch_euler", "euler", "midpoint", "rk4"]
        methods_it = [2, 2, 2, 2]

        def setUp(self):
            self.nb_pts_silent = 2
            self.nb_pts_trans = 2

            self.gd_silent = torch.randn(self.nb_pts_silent, dim, requires_grad=True)
            self.cotan_silent = torch.randn(self.nb_pts_silent, dim, requires_grad=True)
            self.gd_trans = torch.randn(self.nb_pts_trans, dim, requires_grad=True)
            self.cotan_trans = torch.randn(self.nb_pts_trans, dim, requires_grad=True)

        def _build_h(self, fill_cotan=True):
            self.silent = im.DeformationModules.SilentLandmarks.build_from_points(self.gd_silent)
            if fill_cotan:
                self.silent.manifold.fill_cotan(self.cotan_silent.view(-1))

            self.trans = im.DeformationModules.ImplicitModule0.build_from_points(dim, self.nb_pts_trans, 0.5, 0.01, gd=self.gd_trans.view(-1))
            if fill_cotan:
                self.trans.manifold.fill_cotan(self.cotan_trans.view(-1))

            return im.HamiltonianDynamic.Hamiltonian([self.silent, self.trans])        

        def test_shooting(self):
            for method, it in zip(self.methods, self.methods_it):
                with self.subTest(method=method, it=it):
                    states, controls = im.HamiltonianDynamic.shoot(self._build_h(), it, method, intermediates=True)

                    self.assertEqual(len(states), it + 1)

                    states2, controls2 = im.HamiltonianDynamic.shoot(self._build_h(), it, method, controls=controls, intermediates=True)
                    for control, control2 in zip(controls, controls2):
                        self.assertFalse(control[0].requires_grad)
                        self.assertFalse(control[1].requires_grad)
                        self.assertFalse(control2[0].requires_grad)
                        self.assertFalse(control2[1].requires_grad)
                        self.assertTrue(torch.allclose(control[0], control2[0]))
                        self.assertTrue(torch.allclose(control[1], control2[1]))

                    for state, state2 in zip(states, states2):
                        self.assertFalse(state.gd[0].requires_grad)
                        self.assertFalse(state.gd[1].requires_grad)
                        self.assertFalse(state.cotan[0].requires_grad)
                        self.assertFalse(state.cotan[1].requires_grad)
                        self.assertFalse(state2.gd[0].requires_grad)
                        self.assertFalse(state2.gd[1].requires_grad)
                        self.assertFalse(state2.cotan[0].requires_grad)
                        self.assertFalse(state2.cotan[1].requires_grad)
                        self.assertTrue(torch.allclose(state.gd[0], state2.gd[0]))
                        self.assertTrue(torch.allclose(state.gd[1], state2.gd[1]))
                        self.assertTrue(torch.allclose(state.cotan[0], state2.cotan[0]))
                        self.assertTrue(torch.allclose(state.cotan[1], state2.cotan[1]))

        def test_shooting_zero(self):
            for method, it in zip(self.methods, self.methods_it):
                with self.subTest(method=method, it=it):
                    states, controls = im.HamiltonianDynamic.shoot(self._build_h(fill_cotan=False), it, method, intermediates=True)

                    for state in states:
                        self.assertTrue(torch.allclose(state.gd[0], self.gd_silent.view(-1)))
                        self.assertTrue(torch.allclose(state.gd[1], self.gd_trans.view(-1)))
                        self.assertTrue(torch.allclose(state.cotan[0], torch.zeros_like(state.cotan[0].view(-1))))
                        self.assertTrue(torch.allclose(state.cotan[1], torch.zeros_like(state.cotan[1]).view(-1)))

                    for control in controls:
                        self.assertTrue(torch.allclose(control[0], torch.zeros_like(control[0])))
                        self.assertTrue(torch.allclose(control[1], torch.zeros_like(control[1])))

        # Compares torch_euler and euler methods.
        def test_shooting_euler(self):
            it = 2
            states, controls = im.HamiltonianDynamic.shoot(self._build_h(), it, "torch_euler", intermediates=True)
            states2, controls2 = im.HamiltonianDynamic.shoot(self._build_h(), it, "euler", intermediates=True)

            for state, state2 in zip(states, states2):
                self.assertTrue(torch.allclose(state.gd[0], state2.gd[0]))
                self.assertTrue(torch.allclose(state.gd[1], state2.gd[1]))
                self.assertTrue(torch.allclose(state.cotan[0], state2.cotan[0]))
                self.assertTrue(torch.allclose(state.cotan[1], state2.cotan[1]))

            for control, control2 in zip(controls, controls2):
                self.assertTrue(torch.allclose(control[0], control2[0]))
                self.assertTrue(torch.allclose(control[1], control2[1]))


        def test_gradcheck_shoot(self):
            for method, it in zip(self.methods, self.methods_it):
                with self.subTest(method=method, it=it):
                    def shoot(gd_silent, gd_trans, cotan_silent, cotan_trans):
                        h = self._build_h()
                        h.module.manifold.fill_gd([gd_silent.view(-1), gd_trans.view(-1)])
                        h.module.manifold.fill_cotan([cotan_silent.view(-1), cotan_trans.view(-1)])

                        im.HamiltonianDynamic.shoot(h, it, method)

                        return h.module.manifold.gd[0], h.module.manifold.gd[1], h.module.manifold.cotan[0], h.module.manifold.cotan[1]

                    self.assertTrue(torch.autograd.gradcheck(shoot, (self.gd_silent, self.gd_trans, self.cotan_silent, self.cotan_trans), raise_exception=True))

        @unittest.expectedFailure
        def test_gradcheck_shoot_controls(self):
            for method, it in zip(self.methods, self.methods_it):
                with self.subTest(method=method, it=it):
                    def shoot(gd_silent, gd_trans, cotan_silent, cotan_trans):
                        h = self._build_h()
                        h.module.manifold.fill_gd([gd_silent.view(-1).detach().requires_grad_(),
                                                   gd_trans.view(-1).detach().requires_grad_()])
                        h.module.manifold.fill_cotan([cotan_silent.view(-1).detach().requires_grad_(),
                                                      cotan_trans.view(-1).detach().requires_grad_()])

                        _, controls = im.HamiltonianDynamic.shoot(h, it, method, intermediates=True)

                        h = self._build_h()
                        h.module.manifold.fill_gd([gd_silent.view(-1), gd_trans.view(-1)])
                        h.module.manifold.fill_cotan([cotan_silent.view(-1), cotan_trans.view(-1)])

                        im.HamiltonianDynamic.shoot(h, it, method, controls=controls)

                        return h.module.manifold.gd[0], h.module.manifold.gd[1], h.module.manifold.cotan[0], h.module.manifold.cotan[1]

                    self.assertTrue(torch.autograd.gradcheck(shoot, (self.gd_silent, self.gd_trans, self.cotan_silent, self.cotan_trans), raise_exception=True))

    return TestShooting


class TestShooting2D(make_test_shooting(2)):
    pass


class TestShooting3D(make_test_shooting(3)):
    pass


if __name__ == '__main__':
    unittest.main()

