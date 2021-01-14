import torch
from torch.autograd import grad
from torchdiffeq import odeint as odeint
from torchdiffeq._impl.odeint import SOLVERS as torchdiffeq_solvers


def shoot(h, solver, it, controls=None, intermediates=None, t1=1.):
    """ Shoot the hamiltonian system.
    integrate ODE, associe a gd et mom initiaux la trajectoire lien article.
    minimisation energie definit par le modele.
    obtient trajectoire minimisante.

    Parameters
    ----------
    h : HamiltonianDynamic.Hamiltonian
        The hamiltonian system that will be shot.
    it : int
        The number of iterations the solver will do.
    solver : str
        Numerical scheme that will be used to integrate the system.

        Supported solvers are :

        * 'torch_euler' : Euler scheme

        The following solvers uses torchdiffeq :

        * 'euler' : Euler scheme
        * 'midpoint' : RK2 scheme
        * 'rk4' : RK$ scheme

    controls : iterable, default=None
        Optional iterable of tensors representing the controls at each step that will be filled to the deformation module.

        **controls** has to be of length **it**. Each element `i` of **controls** has to be an iterable of size **len(h.module.modules)** each element `j` representing the controls given to the module `j` of **h.module**.
    intermediates : dict, default=None
        Dictionnary that will be filled with intermediate states and controls.
    """
    assert isinstance(intermediates, dict) or intermediates is None

    if solver == "torch_euler":
        _shoot_euler(h, solver, it, controls=controls, intermediates=intermediates)
    elif solver in torchdiffeq_solvers:
        _shoot_torchdiffeq(h, solver, it, controls=controls, intermediates=intermediates, t1=t1)
    else:
        raise NotImplementedError("shoot(): {solver} solver not implemented!".format(solver=solver))


def _shoot_euler(h, solver, it, controls, intermediates):
    step = 1. / it

    if intermediates is not None:
        intermediates['states'] = [h.module.manifold.clone(requires_grad=False)]
        intermediates['controls'] = []

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

        # Extract gradients and multiply them by the step
        d_gd = list(map(lambda x: -step*x, delta[:int(len(delta)/2)]))
        d_mom = list(map(lambda x: step*x, delta[int(len(delta)/2):]))

        # Roll them back
        rolled_d_gd = h.module.manifold.roll_gd(d_gd)
        rolled_d_mom = h.module.manifold.roll_cotan(d_mom)

        # Add them
        h.module.manifold.add_gd(rolled_d_mom)
        h.module.manifold.add_cotan(rolled_d_gd)

        if intermediates is not None:
            intermediates['states'].append(h.module.manifold.clone(requires_grad=False))
            intermediates['controls'].append(list(map(lambda x: x.detach().clone(), h.module.controls)))


def _shoot_torchdiffeq(h, solver, it, controls, intermediates, t1=1.):
    # Wrapper class used by TorchDiffEq
    # Returns (\partial H \over \partial p, -\partial H \over \partial q)
    class TorchDiffEqHamiltonianGrad(torch.nn.Module):
        def __init__(self, h, intermediates=None, controls=None):
            self.h = h
            self.intermediates = intermediates
            self.controls = controls
            self.it = 0

        def __call__(self, t, x):
            with torch.enable_grad():
                # Fill manifold out of the flattened state vector
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

                if self.intermediates is not None:
                    self.intermediates['controls'].append(list(map(lambda x: x.detach().clone(), self.h.module.controls)))

                l = [*self.h.module.manifold.unroll_gd(), *self.h.module.manifold.unroll_cotan()]

                delta = list(grad(h(), l, create_graph=True, allow_unused=True))

                # Nulls are replaced by zero tensors
                for i in range(len(delta)):
                    if delta[i] is None:
                        delta[i] = torch.zeros_like(l[i])

                gd_out = delta[:int(len(delta)/2)]
                mom_out = delta[int(len(delta)/2):]

                self.it = self.it + 1

                return torch.cat(list(map(lambda x: x.flatten(), [*mom_out, *list(map(lambda x: -x, gd_out))])), dim=0).view(2, -1)

    steps = it + 1
    if intermediates is not None:
        intermediates['controls'] = []

    init_manifold = h.module.manifold.clone()
    gradH = TorchDiffEqHamiltonianGrad(h, intermediates, controls)

    x_0 = torch.cat(list(map(lambda x: x.flatten(), [*gradH.h.module.manifold.unroll_gd(), *gradH.h.module.manifold.unroll_cotan()])), dim=0).view(2, -1)
    x_1 = odeint(gradH, x_0, torch.linspace(0., t1, steps), method=solver)

    # Retrieve shot manifold out of the flattened state vector
    gd, mom = [], []
    index = 0
    for m in h.module:
        for i in range(m.manifold.len_gd):
            gd.append(x_1[-1, 0, index:index+m.manifold.numel_gd[i]].view(m.manifold.shape_gd[i]))
            mom.append(x_1[-1, 1, index:index+m.manifold.numel_gd[i]].view(m.manifold.shape_gd[i]))
            index = index + m.manifold.numel_gd[i]

    h.module.manifold.fill_gd(h.module.manifold.roll_gd(gd))
    h.module.manifold.fill_cotan(h.module.manifold.roll_cotan(mom))

    if intermediates is not None:
        intermediates['states'] = []
        for i in range(0, steps):
            gd, mom = [], []
            index = 0
            for m in h.module:
                for j in range(m.manifold.len_gd):
                    gd.append(x_1[i, 0, index:index+m.manifold.numel_gd[j]].detach().view(m.manifold.shape_gd[j]))
                    mom.append(x_1[i, 1, index:index+m.manifold.numel_gd[j]].detach().view(m.manifold.shape_gd[j]))
                    index = index + m.manifold.numel_gd[j]

            state = init_manifold.clone()
            state.fill_gd(state.roll_gd(gd))
            state.fill_cotan(state.roll_cotan(mom))
            intermediates['states'].append(state)

