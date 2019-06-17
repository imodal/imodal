import torch
from torch.autograd import grad
from torchdiffeq import odeint as odeint

from implicitmodules.torch.HamiltonianDynamic import Hamiltonian


def shoot(h, it, method):
    if method == "torch_euler":
        return shoot_euler(h, it)
    else:
        return shoot_torchdiffeq(h, it, method)


# TODO: merge shoot_euler() and shoot_euler_controls() into one function to lower maintenance efforts.
def shoot_euler(h, it):
    step = 1. / it

    intermediate_states = [h.module.manifold.copy()]
    intermediate_controls = []
    for i in range(it):
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
        intermediate_states.append(h.module.manifold.copy())
        intermediate_controls.append(h.module.controls)

    return intermediate_states, intermediate_controls


def shoot_euler_controls(h, controls, it):
    assert len(controls) == it
    step = 1. / it

    intermediate_states = [h.module.manifold.copy()]
    for i in range(it):
        h.module.fill_controls(controls[i])
        l = [*h.module.manifold.unroll_gd(), *h.module.manifold.unroll_cotan()]

        delta = list(grad(h(), l, create_graph=True))

        # Nulls are replaced by zero tensors
        for i in range(len(delta)):
            if delta[i] is None:
                delta[i] = torch.zeros_like(l[i])

        d_gd = h.module.manifold.roll_gd(delta[:int(len(delta)/2)])
        d_mom = h.module.manifold.roll_cotan(delta[int(len(delta)/2):])
        h.module.manifold.muladd_gd(d_mom, step)
        h.module.manifold.muladd_cotan(d_gd, -step)
        intermediate_states.append(h.module.manifold.copy())

    return intermediate_states


# No more maintained until we find out why backward is slow
def shoot_torchdiffeq(h, it, method='rk4'):
    # Wrapper class used by TorchDiffEq
    # Returns (\partial H \over \partial p, -\partial H \over \partial q)
    class TorchDiffEqHamiltonianGrad(Hamiltonian, torch.nn.Module):
        def __init__(self, module):
            super().__init__(module)

        def __call__(self, t, x):
            with torch.enable_grad():
                gd, mom = [], []
                index = 0

                for m in self.module:
                    for i in range(m.manifold.len_gd):
                        gd.append(x[0][index:index+m.manifold.dim_gd[i]].requires_grad_())
                        mom.append(x[1][index:index+m.manifold.dim_gd[i]].requires_grad_())
                        index = index + m.manifold.dim_gd[i]

                self.module.manifold.fill_gd(self.module.manifold.roll_gd(gd))
                self.module.manifold.fill_cotan(self.module.manifold.roll_cotan(mom))

                self.geodesic_controls()
                delta = grad(super().__call__(),
                             [*self.module.manifold.unroll_gd(),
                              *self.module.manifold.unroll_cotan()],
                             create_graph=True, allow_unused=True)

                gd_out = delta[:int(len(delta)/2)]
                mom_out = delta[int(len(delta)/2):]

                return torch.cat(list(map(lambda x: x.view(-1), [*mom_out, *list(map(lambda x: -x, gd_out))])), dim=0).view(2, -1)

    steps = it + 1
    intermediate = []
    init_manifold = h.module.manifold.copy()

    x_0 = torch.cat(list(map(lambda x: x.view(-1), [*h.module.manifold.unroll_gd(), *h.module.manifold.unroll_cotan()])), dim=0).view(2, -1)
    x_1 = odeint(TorchDiffEqHamiltonianGrad.from_hamiltonian(h), x_0, torch.linspace(0., 1., steps), method=method)

    gd, mom = [], []
    index = 0
    for m in h.module:
        for i in range(m.manifold.len_gd):
            gd.append(x_1[-1, 0, index:index+m.manifold.dim_gd[i]])
            mom.append(x_1[-1, 1, index:index+m.manifold.dim_gd[i]])
            index = index + m.manifold.dim_gd[i]

    h.module.manifold.fill_gd(h.module.manifold.roll_gd(gd))
    h.module.manifold.fill_cotan(h.module.manifold.roll_cotan(mom))

    # TODO: very very dirty, change this
    for i in range(0, steps):
        gd, mom = [], []
        index = 0
        for m in h.module:
            for j in range(m.manifold.len_gd):
                gd.append(x_1[i, 0, index:index+m.manifold.dim_gd[j]])
                mom.append(x_1[i, 1, index:index+m.manifold.dim_gd[j]])
                index = index + m.manifold.dim_gd[j]

        intermediate.append(init_manifold.copy())

        intermediate[-1].fill_gd(intermediate[-1].roll_gd(gd))
        intermediate[-1].fill_cotan(intermediate[-1].roll_cotan(mom))

    return intermediate

