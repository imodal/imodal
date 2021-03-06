import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest

import torch

import implicitmodules.torch as im

torch.set_default_tensor_type(torch.DoubleTensor)

def make_test_shooting(dim, backend):
    class TestShooting(unittest.TestCase):
        # The different ODE solving solvers we want to use and test
        solvers = ['torch_euler', 'euler', 'midpoint', 'rk4']
        solvers_it = [2, 2, 2, 2]

        def setUp(self):
            self.sigma = 0.5
            self.nu = 0.001

            self.nb_pts_silent = 2
            self.nb_pts_trans = 2

            self.gd_silent = torch.randn(self.nb_pts_silent, dim, requires_grad=True)
            self.cotan_silent = torch.randn(self.nb_pts_silent, dim, requires_grad=True)
            self.gd_trans = torch.randn(self.nb_pts_trans, dim, requires_grad=True)
            self.cotan_trans = torch.randn(self.nb_pts_trans, dim, requires_grad=True)

        def _build_h(self, fill_cotan=True):
            self.silent = im.DeformationModules.SilentLandmarks(dim, self.nb_pts_silent, gd=self.gd_silent)
            if fill_cotan:
                self.silent.manifold.fill_cotan(self.cotan_silent)
            else:
                self.silent.manifold.fill_cotan(torch.zeros_like(self.cotan_silent, requires_grad=True))

            self.trans = im.DeformationModules.ImplicitModule0(dim, self.nb_pts_trans, self.sigma, nu=self.nu, gd=self.gd_trans, backend=backend)
            if fill_cotan:
                self.trans.manifold.fill_cotan(self.cotan_trans)
            else:
                self.trans.manifold.fill_cotan(torch.zeros_like(self.cotan_trans, requires_grad=True))

            return im.HamiltonianDynamic.Hamiltonian([self.silent, self.trans])        

        def test_shooting(self):
            for solver, it in zip(self.solvers, self.solvers_it):
                with self.subTest(solver=solver, it=it):
                    intermediates = {}
                    im.HamiltonianDynamic.shoot(self._build_h(), solver, it, intermediates=intermediates)

                    states = intermediates['states']
                    controls = intermediates['controls']

                    self.assertEqual(len(states), it + 1)

                    intermediates = {}
                    im.HamiltonianDynamic.shoot(self._build_h(), solver, it, controls=controls, intermediates=intermediates)

                    states2 = intermediates['states']
                    controls2 = intermediates['controls']

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
            for solver, it in zip(self.solvers, self.solvers_it):
                with self.subTest(solver=solver, it=it):
                    intermediates = {}
                    im.HamiltonianDynamic.shoot(self._build_h(fill_cotan=False), solver, it, intermediates=intermediates)
                    states = intermediates['states']
                    controls = intermediates['controls']

                    for state in states:
                        self.assertTrue(torch.allclose(state.gd[0], self.gd_silent))
                        self.assertTrue(torch.allclose(state.gd[1], self.gd_trans))
                        self.assertTrue(torch.allclose(state.cotan[0], torch.zeros_like(state.cotan[0])))
                        self.assertTrue(torch.allclose(state.cotan[1], torch.zeros_like(state.cotan[1])))

                    for control in controls:
                        self.assertTrue(torch.allclose(control[0], torch.zeros_like(control[0])))
                        self.assertTrue(torch.allclose(control[1], torch.zeros_like(control[1])))

        # Compares torch_euler and euler solvers.
        def test_shooting_euler(self):
            it = 2
            intermediates = {}
            im.HamiltonianDynamic.shoot(self._build_h(), 'torch_euler', it, intermediates=intermediates)
            states = intermediates['states']
            controls = intermediates['controls']

            intermediates = {}
            im.HamiltonianDynamic.shoot(self._build_h(), 'euler', it, intermediates=intermediates)
            states2 = intermediates['states']
            controls2 = intermediates['controls']

            for state, state2 in zip(states, states2):
                self.assertTrue(torch.allclose(state.gd[0], state2.gd[0]))
                self.assertTrue(torch.allclose(state.gd[1], state2.gd[1]))
                self.assertTrue(torch.allclose(state.cotan[0], state2.cotan[0]))
                self.assertTrue(torch.allclose(state.cotan[1], state2.cotan[1]))

            for control, control2 in zip(controls, controls2):
                self.assertTrue(torch.allclose(control[0], control2[0]))
                self.assertTrue(torch.allclose(control[1], control2[1]))


        def test_gradcheck_shoot(self):
            for solver, it in zip(self.solvers, self.solvers_it):
                with self.subTest(solver=solver, it=it):
                    def shoot(gd_silent, gd_trans, cotan_silent, cotan_trans):
                        h = self._build_h()
                        h.module.manifold.fill_gd([gd_silent, gd_trans])
                        h.module.manifold.fill_cotan([cotan_silent, cotan_trans])

                        im.HamiltonianDynamic.shoot(h, solver, it)

                        return h.module.manifold.gd[0], h.module.manifold.gd[1], h.module.manifold.cotan[0], h.module.manifold.cotan[1]

                    self.assertTrue(torch.autograd.gradcheck(shoot, (self.gd_silent, self.gd_trans, self.cotan_silent, self.cotan_trans), raise_exception=True))

        @unittest.expectedFailure
        def test_gradcheck_shoot_controls(self):
            for solver, it in zip(self.solvers, self.solvers_it):
                with self.subTest(solver=solver, it=it):
                    def shoot(gd_silent, gd_trans, cotan_silent, cotan_trans):
                        h = self._build_h()
                        h.module.manifold.fill_gd([gd_silent.detach().requires_grad_(),
                                                   gd_trans.detach().requires_grad_()])
                        h.module.manifold.fill_cotan([cotan_silent.detach().requires_grad_(),
                                                      cotan_trans.detach().requires_grad_()])

                        _, controls = im.HamiltonianDynamic.shoot(h, solver, it, intermediates=True)

                        h = self._build_h()
                        h.module.manifold.fill_gd([gd_silent, gd_trans])
                        h.module.manifold.fill_cotan([cotan_silent, cotan_trans])

                        im.HamiltonianDynamic.shoot(h, it, solver, controls=controls)

                        return h.module.manifold.gd[0], h.module.manifold.gd[1], h.module.manifold.cotan[0], h.module.manifold.cotan[1]

                    self.assertTrue(torch.autograd.gradcheck(shoot, (self.gd_silent, self.gd_trans, self.cotan_silent, self.cotan_trans), raise_exception=True))

    return TestShooting


class TestShooting2D_Torch(make_test_shooting(2, 'torch')):
    pass


# class TestShooting3D_Torch(make_test_shooting(3, 'torch')):
#     pass


# class TestShooting2D_KeOps(make_test_shooting(2, 'keops')):
#     pass


# class TestShooting3D_KeOps(make_test_shooting(3, 'keops')):
#     pass

if __name__ == '__main__':
    unittest.main()

