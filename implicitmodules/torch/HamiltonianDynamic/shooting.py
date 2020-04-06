import time

import torch
from torch.autograd import grad
from torchdiffeq import odeint as odeint

from implicitmodules.torch.HamiltonianDynamic import Hamiltonian
from implicitmodules.torch.Manifolds import CompoundManifold


def shoot(h, it, method, controls=None, intermediates=False):
    """ Shoot the hamiltonian system.
    integrate ODE, associe a gd et mom initiaux la trajectoire lien article 
    minimisation energie definit par le modele
    obtient trajectoire minimisante
    

    Parameters
    ----------
    h : HamiltonianDynamic.Hamiltonian
        The hamiltonian system that will be shot.
    it : int
        The number of iterations the solver will do.
    method : str
        Numerical scheme that will be used to integrate the system.

        Supported solvers are :

        * 'torch_euler' : Euler scheme

        The following solvers uses torchdiffea :

        * 'euler' : Euler scheme
        * 'midpoint' : RK2 scheme
        * 'rk4' : RK$ scheme

    controls : iterable, default=None
        Optional iterable of tensors representing the controls at each step that will be filled to the deformation module.

        **controls** has to be of length **it**. Each element `i` of **controls** has to be an iterable of size **len(h.module.modules)** each element `j` representing the controls given to the module `j` of **h.module**.
    intermediates : boolean, default=False
        If true, outputs intermediate states of the system. Intermediate states are represented as two list 
    """
    if method == "torch_euler":
        return shoot_euler(h, it, controls=controls, intermediates=intermediates)
    else:
        return shoot_torchdiffeq(h, it, method, controls=controls, intermediates=intermediates)


def shoot_euler(h, it, controls=None, intermediates=False):
    step = 1. / it

    if intermediates:
        intermediate_states = [h.module.manifold.clone(requires_grad=False)]
        intermediate_controls = []

    for i in range(it):
        if controls is not None:
            h.module.fill_controls(controls[i])
        else:
            h.geodesic_controls()

        l = [*h.module.manifold.unroll_gd(), *h.module.manifold.unroll_cotan()]

        delta = list(grad(h(), l, create_graph=True, allow_unused=True))

        # Nulls are replaced by zero tensors
        for i in range(len(delta)):
            if delta[i] is None:
                delta[i] = torch.zeros_like(l[i])

        # First extract gradients and multiply them by the step
        d_gd = list(map(lambda x: -step*x, delta[:int(len(delta)/2)]))
        d_mom = list(map(lambda x: step*x, delta[int(len(delta)/2):]))

        # Roll them back
        rolled_d_gd = h.module.manifold.roll_gd(d_gd)
        rolled_d_mom = h.module.manifold.roll_cotan(d_mom)

        # Add them
        h.module.manifold.add_gd(rolled_d_mom)
        h.module.manifold.add_cotan(rolled_d_gd)

        if intermediates:
            intermediate_states.append(h.module.manifold.clone(requires_grad=False))
            intermediate_controls.append(list(map(lambda x: x.detach().clone(), h.module.controls)))

    if intermediates:
        return intermediate_states, intermediate_controls


def shoot_torchdiffeq(h, it, method='euler', controls=None, intermediates=False):
    # Wrapper class used by TorchDiffEq
    # Returns (\partial H \over \partial p, -\partial H \over \partial q)
    class TorchDiffEqHamiltonianGrad():
        def __init__(self, h, intermediates=False, controls=None):
            self.h = h
            self.intermediates = intermediates
            self.controls = controls
            self.out_controls = []
            self.it = 0

        def __call__(self, t, x):
            with torch.enable_grad():
                gd, mom = [], []
                index = 0

                for m in self.h.module:
                    for i in range(m.manifold.len_gd):
                        gd.append(x[0][index:index+m.manifold.numel_gd[i]].view(m.manifold.shape_gd[i]).requires_grad_())
                        mom.append(x[1][index:index+m.manifold.numel_gd[i]].view(m.manifold.shape_gd[i]).requires_grad_())
                        index = index + m.manifold.numel_gd[i]

                self.h.module.manifold.fill_gd(self.h.module.manifold.roll_gd(gd))
                self.h.module.manifold.fill_cotan(self.h.module.manifold.roll_cotan(mom))

                # If controls are provided, use them, else we compute the geodesic controls.
                if self.controls is not None:
                    self.h.module.fill_controls(self.controls[self.it])
                else:
                    self.h.geodesic_controls()

                if self.intermediates:
                    self.out_controls.append(list(map(lambda x: x.detach().clone(), self.h.module.controls)))
                delta = grad(h(),
                             [*self.h.module.manifold.unroll_gd(),
                              *self.h.module.manifold.unroll_cotan()],
                             create_graph=True, allow_unused=True)

                gd_out = delta[:int(len(delta)/2)]
                mom_out = delta[int(len(delta)/2):]

                self.it = self.it + 1

                return torch.cat(list(map(lambda x: x.flatten(), [*mom_out, *list(map(lambda x: -x, gd_out))])), dim=0).view(2, -1)

    steps = it + 1
    if intermediates:
        intermediate_controls = []

    init_manifold = h.module.manifold.clone()
    gradH = TorchDiffEqHamiltonianGrad(h, intermediates, controls)

    x_0 = torch.cat(list(map(lambda x: x.view(-1), [*gradH.h.module.manifold.unroll_gd(), *gradH.h.module.manifold.unroll_cotan()])), dim=0).view(2, -1)
    x_1 = odeint(gradH, x_0, torch.linspace(0., 1., steps), method=method)

    gd, mom = [], []
    index = 0
    for m in h.module:
        for i in range(m.manifold.len_gd):
            gd.append(x_1[-1, 0, index:index+m.manifold.numel_gd[i]].view(m.manifold.shape_gd[i]))
            mom.append(x_1[-1, 1, index:index+m.manifold.numel_gd[i]].view(m.manifold.shape_gd[i]))
            index = index + m.manifold.numel_gd[i]

    h.module.manifold.fill_gd(h.module.manifold.roll_gd(gd))
    h.module.manifold.fill_cotan(h.module.manifold.roll_cotan(mom))

    if intermediates:
        # TODO: very very dirty, change this
        intermediate_states = []
        for i in range(0, steps):
            gd, mom = [], []
            index = 0
            for m in h.module:
                for j in range(m.manifold.len_gd):
                    gd.append(x_1[i, 0, index:index+m.manifold.numel_gd[j]].detach().view(m.manifold.shape_gd[j]))
                    mom.append(x_1[i, 1, index:index+m.manifold.numel_gd[j]].detach().view(m.manifold.shape_gd[j]))
                    index = index + m.manifold.numel_gd[j]

            intermediate_states.append(init_manifold.clone())

            intermediate_states[-1].fill_gd(intermediate_states[-1].roll_gd(gd))
            intermediate_states[-1].fill_cotan(intermediate_states[-1].roll_cotan(mom))

        return intermediate_states, gradH.out_controls

