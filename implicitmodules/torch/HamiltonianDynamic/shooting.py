import time

import torch
from torch.autograd import grad
from torchdiffeq import odeint as odeint

from implicitmodules.torch.HamiltonianDynamic import Hamiltonian
from implicitmodules.torch.Manifolds import CompoundManifold


def shoot(h, it, method, controls=None, intermediates=False):
    """
    Add documentation for this function.
    """
    if method == "torch_euler":
        return shoot_euler(h, it, controls=controls, intermediates=intermediates)
    else:
        return shoot_torchdiffeq(h, it, method, controls=controls, intermediates=intermediates)


def shoot_euler(h, it, controls=None, intermediates=False):
    """
    Add documentation for this function.
    """
    step = 1. / it

    if intermediates:
        intermediate_states = [h.module.manifold.copy(requires_grad=False)]
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

        d_gd = h.module.manifold.roll_gd(delta[:int(len(delta)/2)])
        d_mom = h.module.manifold.roll_cotan(delta[int(len(delta)/2):])
        h.module.manifold.muladd_gd(d_mom, step)
        h.module.manifold.muladd_cotan(d_gd, -step)

        if intermediates:
            intermediate_states.append(h.module.manifold.copy(requires_grad=False))
            intermediate_controls.append(list(map(lambda x: x.detach().clone(), h.module.controls)))

    if intermediates:
        return intermediate_states, intermediate_controls


def shoot_torchdiffeq(h, it, method='euler', controls=None, intermediates=False):
    """
    Add documentation for this function.
    """
    # Wrapper class used by TorchDiffEq
    # Returns (\partial H \over \partial p, -\partial H \over \partial q)
    class TorchDiffEqHamiltonianGrad(Hamiltonian, torch.nn.Module):
        def __init__(self, module):
            self.intermediates = False
            self.in_controls = None
            self.out_controls = []
            self.it = 0
            super().__init__(module)

        def __call__(self, t, x):
            with torch.enable_grad():
                gd, mom = [], []
                index = 0

                for m in self.module:
                    for i in range(m.manifold.len_gd):
                        gd.append(x[0][index:index+m.manifold.numel_gd[i]].view(m.manifold.shape_gd[i]).requires_grad_())
                        mom.append(x[1][index:index+m.manifold.numel_gd[i]].view(m.manifold.shape_gd[i]).requires_grad_())
                        index = index + m.manifold.numel_gd[i]

                self.module.manifold.fill_gd(self.module.manifold.roll_gd(gd))
                self.module.manifold.fill_cotan(self.module.manifold.roll_cotan(mom))

                # If controls are provided, use them, else we compute the geodesic controls.
                if self.in_controls is not None:
                    self.module.fill_controls(self.in_controls[self.it])
                else:
                    self.geodesic_controls()

                if self.intermediates:
                    self.out_controls.append(list(map(lambda x: x.detach().clone(), self.module.controls)))
                delta = grad(super().__call__(),
                             [*self.module.manifold.unroll_gd(),
                              *self.module.manifold.unroll_cotan()],
                             create_graph=True, allow_unused=True)

                gd_out = delta[:int(len(delta)/2)]
                mom_out = delta[int(len(delta)/2):]

                self.it = self.it + 1

                return torch.cat(list(map(lambda x: x.view(-1), [*mom_out, *list(map(lambda x: -x, gd_out))])), dim=0).view(2, -1)
    steps = it + 1
    if intermediates:
        intermediate_controls = []

    init_manifold = h.module.manifold.copy()
    H = TorchDiffEqHamiltonianGrad.from_hamiltonian(h)
    H.intermediates = intermediates
    H.in_controls = controls

    x_0 = torch.cat(list(map(lambda x: x.view(-1), [*h.module.manifold.unroll_gd(), *h.module.manifold.unroll_cotan()])), dim=0).view(2, -1)
    x_1 = odeint(H, x_0, torch.linspace(0., 1., steps), method=method)

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
                    gd.append(x_1[i, 0, index:index+m.manifold.numel_gd[j]].detach().view(-1, m.dim))
                    mom.append(x_1[i, 1, index:index+m.manifold.numel_gd[j]].detach().view(-1, m.dim))
                    index = index + m.manifold.numel_gd[j]

            intermediate_states.append(init_manifold.copy())

            intermediate_states[-1].fill_gd(intermediate_states[-1].roll_gd(gd))
            intermediate_states[-1].fill_cotan(intermediate_states[-1].roll_cotan(mom))

        return intermediate_states, H.out_controls

