import copy
import time
from collections import Iterable

import torch
import torch.optim
from .LBFGS import FullBatchLBFGS

from implicitmodules.torch.DeformationModules import CompoundModule, SilentLandmarks
from implicitmodules.torch.HamiltonianDynamic import Hamiltonian, shoot
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.Utilities.sampling import sample_from_greyscale, deformed_intensities
from implicitmodules.torch.Utilities.usefulfunctions import grid2vec, vec2grid


class Model():
    def __init__(self, attachement):
        self.__attachement = attachement

    @property
    def attachement(self):
        return self.__attachement

    def compute(self):
        raise NotImplementedError

    def fit(self, target, step_length=1., l=1., max_iter=100, options={}):
        optim = FullBatchLBFGS(self.parameters, lr=step_length, history_size=500, line_search='Wolfe')

        last_costs = {}
        def closure():
            optim.zero_grad()

            # Call precompute callback if available
            if self.precompute_callback is not None:
                self.precompute_callback(self.modules, self.parameters)

            # Shooting + loss computation
            deformation_cost, attach_cost = self.compute(target)
            cost = l*attach_cost + deformation_cost

            # Save for printing purpose
            last_costs['deformation_cost'] = deformation_cost.detach()
            last_costs['attach_cost'] = l*attach_cost.detach()
            last_costs['cost'] = cost.detach()

            return cost

        loss = closure()
        loss.backward()

        print("Initial state")
        print("Attach cost = %.3f" % (loss.detach().numpy()))

        closure_count = 0
        start = time.time()
        costs = []
        for i in range(max_iter):
            # Computing step
            step_options = {'closure': closure, 'current_loss': loss, 'ls_debug': True}
            step_options.update(options)
            loss, _, step_length, _, F_eval, G_eval, desc_dir, fail = optim.step(step_options)

            # Retrieving costs
            costs.append((last_costs['deformation_cost'], last_costs['attach_cost'], last_costs['cost']))

            print("="*80)
            print("Iteration: %d \n Total energy = %f \n Attach cost = %f \n Deformation cost = %f \n Step length = %.12f \n Closure evaluations = %d" % (i + 1, last_costs['cost'], last_costs['attach_cost'], last_costs['deformation_cost'], step_length, F_eval))
            closure_count += F_eval

            if fail or not desc_dir:
                break

        print("="*80)
        print("End of the optimisation process.")
        print("Final energy =", last_costs['cost'])
        print("Closure evaluations =", closure_count)
        print("Time elapsed =", time.time() - start)

        return costs


class ModelCompound(Model):
    def __init__(self, modules, attachement, fit_moments, precompute_callback, other_parameters):
        super().__init__(attachement)
        self.__modules = modules
        self.__precompute_callback = precompute_callback
        self.__fit_moments = fit_moments

        self.__init_manifold = CompoundModule(self.__modules).manifold.copy()
        # We copy each parameters
        self.__init_other_parameters = []
        for p in other_parameters:
            #self.__init_other_parameters.append(p.detach().clone().requires_grad_())
            self.__init_other_parameters.append(p)

        # Called to update the parameter list sent to the optimizer
        self.compute_parameters()

    @property
    def modules(self):
        return self.__modules

    @property
    def precompute_callback(self):
        return self.__precompute_callback

    @property
    def fit_moments(self):
        return self.__fit_moments

    @property
    def init_manifold(self):
        return self.__init_manifold

    @property
    def init_parameters(self):
        return self.__init_parameters

    @property
    def parameters(self):
        return self.__parameters

    def compute_parameters(self):
        """
        Fill the parameter list that will be optimized by the optimizer. 
        We first fill in the moments of each modules and then adds other model parameters.
        """
        self.__parameters = []

        if self.__fit_moments:
            for i in range(len(self.__modules)):
                self.__parameters.extend(self.__init_manifold[i].unroll_cotan())

        self.__parameters.extend(self.__init_other_parameters)

    def compute_deformation_grid(self, grid_origin, grid_size, grid_resolution, it=20, intermediate=False):
        x, y = torch.meshgrid([
            torch.linspace(grid_origin[0], grid_origin[0]+grid_size[0], grid_resolution[0]),
            torch.linspace(grid_origin[1], grid_origin[1]+grid_size[1], grid_resolution[1])])

        gridpos = grid2vec(x, y)

        grid_landmarks = Landmarks(2, gridpos.shape[0], gd=gridpos.view(-1))
        grid_silent = SilentLandmarks(grid_landmarks)
        compound = CompoundModule(self.modules)
        compound.manifold.fill(self.init_manifold)

        intermediate_states= shoot(Hamiltonian([grid_silent, *compound]), 10, 'euler')

        return [vec2grid(inter[0].gd.detach().view(-1, 2), grid_resolution[0], grid_resolution[1]) for inter in intermediate_states]


class ModelCompoundWithPointsRegistration(ModelCompound):
    """
    TODO: add documentation
    """
    def __init__(self, source, modules, attachement, fit_moments=True, precompute_callback=None, other_parameters=[]):
        assert isinstance(source, Iterable) and not isinstance(source, torch.Tensor)
        
        # We first determinate the number of sources
        self.source_count = len(source)

        # We now create the corresponding silent landmarks and save the alpha values
        self.weights = []
        for i in range(self.source_count):
            if isinstance(source[i], tuple):
                # Weights are provided
                self.weights.insert(i, source[i][1])
                modules.insert(i, SilentLandmarks(Landmarks(2, source[i][0].shape[0], gd=source[i][0].view(-1).requires_grad_())))
            elif isinstance(source[i], torch.Tensor):
                # No weights provided
                self.weights.insert(i, None)
                modules.insert(i, SilentLandmarks(Landmarks(2, source[i].shape[0], gd=source[i].view(-1).requires_grad_())))

        super().__init__(modules, attachement, fit_moments, precompute_callback, other_parameters)

    def compute(self, target):
        """ Does shooting. Outputs compute deformation and attach cost. """
        # We first create and fill the compound module we will shoot
        compound = CompoundModule(self.modules)
        compound.manifold.fill(self.init_manifold)

        # Shooting
        # TODO: Iteraction count and method should be provided by the user.
        shoot(Hamiltonian(compound), 30, 'euler')
        deformation_cost = compound.cost()

        # We compute the attach cost for each source/target couple
        attach_costs = []
        for i in range(self.source_count):
            if self.weights[i] is not None:
                attach_costs.append(self.attachement[i]((compound[i].manifold.gd.view(-1, 2), self.weights[i]), target[i]))
            else:
                attach_costs.append(self.attachement[i]((compound[i].manifold.gd.view(-1, 2), None), (target[i], None)))
        #print(attach_costs)
        return deformation_cost, sum(attach_costs)


