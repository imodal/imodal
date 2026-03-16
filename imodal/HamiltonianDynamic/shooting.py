import torch
import math
from torch.autograd import grad
from torchdiffeq import odeint as odeint
from torchdiffeq._impl.odeint import SOLVERS as torchdiffeq_solvers


def shoot(h, solver, it, costs, controls=None, intermediates=None, t1=1.):
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
        _shoot_euler(h, solver, it, controls=controls,
                     intermediates=intermediates)
    elif solver in torchdiffeq_solvers:
        _shoot_torchdiffeq(h, solver, it, costs, controls=controls,
                           intermediates=intermediates, t1=t1)
    else:
        raise NotImplementedError(
            "shoot(): {solver} solver not implemented!".format(solver=solver))


def _shoot_euler(h, solver, it, controls, intermediates):
    step = 1. / it

    if intermediates is not None:
        intermediates['states'] = [
            h.module.manifold.clone(requires_grad=False)]
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
            intermediates['states'].append(
                h.module.manifold.clone(requires_grad=False))
            intermediates['controls'].append(
                list(map(lambda x: x.detach().clone(), h.module.controls)))


def _shoot_torchdiffeq(h, solver, it, costs, controls, intermediates, t1=1.):
    # Wrapper class used by TorchDiffEq
    # Returns (\partial H \over \partial p, -\partial H \over \partial q)
    class TorchDiffEqHamiltonianGrad(torch.nn.Module):
        def __init__(self, h, costs, intermediates=None, controls=None):
            super(TorchDiffEqHamiltonianGrad, self).__init__()
            self.h = h
            self.intermediates = intermediates
            self.controls = controls
            self.it = 0
            self.true_it = 0
            self.costs = costs

            if controls is None:
                self.forward = self.forward_moments
            else:
                self.forward = self.forward_control

        def t_to_it(self, t):
            return min(math.floor(t / (t1 / it)), it - 1)

        def forward_moments(self, t, x):
            # store previous iteration
            prev_it = self.true_it

            # compute from t, t1, and it the true iteration number (to be used with rk and midpoint)
            self.true_it = self.t_to_it(t)

            # print(f't={t}, true_it={self.true_it}, prev_it={prev_it}, self.it={self.it}')
            
            # If we are at the first iteration or if we are at a new iteration, we add the cost to the list of costs
            if prev_it != self.true_it or t == 0:
                ADD_COST = True
            else:
                ADD_COST = False
                
            with torch.enable_grad():
                # Fill manifold out of the flattened state vector
                gd, mom = [], []
                index = 0
                for m in self.h.module:
                    for i in range(m.manifold.len_gd):
                        gd.append(x[0][index:index+m.manifold.numel_gd[i]
                                       ].view(m.manifold.shape_gd[i]).requires_grad_())
                        mom.append(x[1][index:index+m.manifold.numel_gd[i]
                                        ].view(m.manifold.shape_gd[i]).requires_grad_())
                        index = index + m.manifold.numel_gd[i]

                self.h.module.manifold.fill_gd(
                    self.h.module.manifold.roll_gd(gd))
                self.h.module.manifold.fill_cotan(
                    self.h.module.manifold.roll_cotan(mom))

                self.h.geodesic_controls()

                if self.intermediates is not None:
                    self.intermediates['controls'].append(
                        list(map(lambda x: x.clone(), self.h.module.controls)))

                l = [*self.h.module.manifold.unroll_gd(), *
                     self.h.module.manifold.unroll_cotan()]

                delta = list(
                    grad(h(), l, create_graph=True, allow_unused=True))

                if ADD_COST:
                    self.costs['deformation'].append(
                        self.h.module.cost())

                # Nulls are replaced by zero tensors
                for i in range(len(delta)):
                    if delta[i] is None:
                        delta[i] = torch.zeros_like(l[i])

                gd_out = delta[:int(len(delta)/2)]
                mom_out = delta[int(len(delta)/2):]

                gd_out_neg = list(map(lambda x: -x, gd_out))

                self.it = self.it + 1

                return torch.cat(list(map(lambda x: x.flatten(), [*mom_out, *gd_out_neg])), dim=0).view(2, -1)

        def forward_control(self, t, x):
            # store previous iteration
            prev_it = self.true_it

            # compute from t, t1, and it the true iteration number (to be used with rk and midpoint)
            self.true_it = self.t_to_it(t)

            # print(f't={t}, true_it={self.true_it}, prev_it={prev_it}, self.it={self.it}')
            
            # If we are at the first iteration or if we are at a new iteration, we add the cost to the list of costs
            if prev_it != self.true_it or t == 0:
                ADD_COST = True
            else:
                ADD_COST = False
                
            with torch.enable_grad():
                # Fill manifold out of the flattened state vector
                gd = []
                index = 0
                for m in self.h.module:
                    for i in range(m.manifold.len_gd):
                        gd.append(x[0][index:index+m.manifold.numel_gd[i]
                                       ].view(m.manifold.shape_gd[i]).requires_grad_())
                        index = index + m.manifold.numel_gd[i]
      
                self.h.module.manifold.fill_gd(
                    self.h.module.manifold.roll_gd(gd[:]))
       
                if len(self.controls) == it:
                    # control for each step 
                    self.h.module.fill_controls(self.controls[self.true_it])
                elif len(self.controls) > it:
                    # control for each step + for intermediate steps (rk4, midpoint)
                    self.h.module.fill_controls(self.controls[self.it])
                else:
                    # constant control across steps
                    self.h.module.fill_controls(self.controls[0])

                # Q: Do we store only the controls 
                # at the true iteration ?
                # or at each iteration (including intermediate steps)?
                if self.intermediates is not None:
                    self.intermediates['controls'].append(
                        list(map(lambda x: x.clone(), self.h.module.controls)))
                    
                if ADD_COST:
                    self.costs['deformation'].append(
                        self.h.module.cost())

                # for gd__ in gd:
                #     print('gd__', gd__.shape)
                # mom_out = [self.h.module(gd__) for gd__ in gd[:-1]]
                # mom_out.append(torch.zeros_like(gd[-1]))

                field = self.h.module.field_generator()
                man = self.h.module.manifold.infinitesimal_action(field)
                mom_out = man.unroll_tan()   

                # mom_out = []
                # for m in self.h.module:
                #     field = m.field_generator()
                #     man = m.manifold.infinitesimal_action(field)
                #     mom_out.extend(man.unroll_tan())

                # print('len(gd)', len(gd))
                # print('len(mom_out)', len(mom_out))
                # for i in range(len(gd)):
                #     print(f'gd[{i}].shape={gd[i].shape}, mom_out[{i}].shape={mom_out[i].shape}')
                #     print(f'mom_out[{i}] min={mom_out[i].min()}, max={mom_out[i].max()}')

                self.it = self.it + 1

                return torch.cat(list(map(lambda x: x.flatten(), mom_out)), dim=0).view(1, -1)

    steps = it + 1
    if intermediates is not None:
        intermediates['controls'] = []

    init_manifold = h.module.manifold.clone()
    costs['deformation'] = []
    gradH = TorchDiffEqHamiltonianGrad(h, costs, intermediates, controls)

    # gd + cotan
    x_0 = torch.cat(list(map(lambda x: x.flatten(), [
                    *gradH.h.module.manifold.unroll_gd(), *gradH.h.module.manifold.unroll_cotan()])), dim=0).view(2, -1)
    
    # gd
    if controls is not None:
        x_0 = x_0[0].view(1, -1)
    
    x_1 = odeint(gradH, x_0, torch.linspace(0., t1, steps), method=solver)

    # costs['deformation'].append(h.module.cost())

    # Retrieve shot manifold out of the flattened state vector
    gd, mom = [], []
    index = 0
    for m in h.module:
        for i in range(m.manifold.len_gd):
            gd.append(x_1[-1, 0, index:index+m.manifold.numel_gd[i]
                          ].view(m.manifold.shape_gd[i]))
            if controls is None:
                mom.append(
                    x_1[-1, 1, index:index+m.manifold.numel_gd[i]].view(m.manifold.shape_gd[i]))
            index = index + m.manifold.numel_gd[i]

    h.module.manifold.fill_gd(h.module.manifold.roll_gd(gd))

    if controls is None:
        h.module.manifold.fill_cotan(h.module.manifold.roll_cotan(mom))

    if intermediates is not None:
        intermediates['states'] = []
        for i in range(0, steps):
            gd, mom = [], []
            index = 0
            for m in h.module:
                for j in range(m.manifold.len_gd):
                    gd.append(x_1[i, 0, index:index+m.manifold.numel_gd[j]
                                  ].view(m.manifold.shape_gd[j]))
                    if controls is None:
                        mom.append(x_1[i, 1, index:index+m.manifold.numel_gd[j]
                                    ].view(m.manifold.shape_gd[j]))
                    index = index + m.manifold.numel_gd[j]

            state = init_manifold.clone()
            state.fill_gd(state.roll_gd(gd))
            if controls is None:
                state.fill_cotan(state.roll_cotan(mom))
            intermediates['states'].append(state)