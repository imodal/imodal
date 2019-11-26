import copy
import time
from collections import Iterable

import torch

from implicitmodules.torch.DeformationModules import CompoundModule, SilentLandmarks
from implicitmodules.torch.HamiltonianDynamic import Hamiltonian, shoot
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.Utilities.sampling import sample_from_greyscale, deformed_intensities
from implicitmodules.torch.Utilities.usefulfunctions import grid2vec, vec2grid


class Model:
    def __init__(self, modules, attachments, fit_moments, fit_gd, lam, precompute_callback, other_parameters):
        self.__modules = modules
        self.__attachments = attachments
        self.__precompute_callback = precompute_callback
        self.__fit_moments = fit_moments
        self.__fit_gd = fit_gd
        self.__lam = lam

        if other_parameters is None:
            other_parameters = []

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
    def attachments(self):
        return self.__attachments

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

    @property
    def lam(self):
        return self.__lam

    def gradcheck(self):
        def energy(*param):
            parameters = list(param)
            init_manifold = []
            init_other_parameters = []

            if self.__fit_moments:
                init_manifold = parameters[:self.__init_manifold.len_gd]
                self.__init_other_parameters = parameters[len(init_manifold):]
                self.__init_manifold.fill_cotan(self.__init_manifold.roll_cotan(init_manifold))
            else:
                self.__init_other_parameters = parameters

            self.compute_parameters()

            if self.__precompute_callback:
                self.__precompute_callback(self.__modules, self.__parameters)

            deformation_cost, attach_cost = self.compute()
            cost = deformation_cost + attach_cost
            return cost

        return torch.autograd.gradcheck(energy, self.__parameters, raise_exception=False)

    def compute_parameters(self):
        """
        Fill the parameter list that will be optimized by the optimizer. 
        We first fill in the moments of each modules and then adds other model parameters.
        """
        self.__parameters = []

        if self.__fit_moments:
            for i in range(len(self.__modules)):
                self.__parameters.extend(self.__init_manifold[i].unroll_cotan())

        if self.__fit_gd is not None:
            for i in range(len(self.__modules)):
                if self.__fit_gd[i]:
                    self.__parameters.extend(self.__init_manifold[i].unroll_gd())

        self.__parameters.extend(self.__init_other_parameters)

    def compute_deformation_grid(self, grid_origin, grid_size, grid_resolution, it=10, method="euler", intermediate=False):
        x, y = torch.meshgrid([
            torch.linspace(grid_origin[0], grid_origin[0]+grid_size[0], grid_resolution[0]),
            torch.linspace(grid_origin[1], grid_origin[1]+grid_size[1], grid_resolution[1])])

        gridpos = grid2vec(x, y)

        grid_landmarks = Landmarks(2, gridpos.shape[0], gd=gridpos.view(-1))
        grid_silent = SilentLandmarks(grid_landmarks)
        compound = CompoundModule(self.modules)
        compound.manifold.fill(self.init_manifold)

        intermediate_states, _ = shoot(Hamiltonian([grid_silent, *compound]), it, method, intermediates=True)

        return [vec2grid(inter[0].gd.detach().view(-1, 2), grid_resolution[0], grid_resolution[1]) for inter in intermediate_states]


class ModelPointsRegistration(Model):
    """
    TODO: add documentation
    """
    def __init__(self, source, modules, attachments, lam=1., fit_gd=None, fit_moments=True, precompute_callback=None, other_parameters=None):
        assert isinstance(source, Iterable) and not isinstance(source, torch.Tensor)

        if other_parameters is None:
            other_parameters = []

        # We first determinate the number of sources
        self.source_count = len(source)

        # We now create the corresponding silent landmarks and save the alpha values
        self.weights = []
        self.__source_dim = []
        for i in range(self.source_count):
            if isinstance(source[i], tuple):
                # Weights are provided
                self.weights.insert(i, source[i][1])
                modules.insert(i, SilentLandmarks(Landmarks(source[i].shape[1], source[i][0].shape[0], gd=source[i][0].view(-1).requires_grad_())))
                self.__source_dim.append(source[i].shape[1])
            elif isinstance(source[i], torch.Tensor):
                # No weights provided
                self.weights.insert(i, None)
                modules.insert(i, SilentLandmarks(Landmarks(source[i].shape[1], source[i].shape[0], gd=source[i].view(-1).requires_grad_())))
                self.__source_dim.append(source[i].shape[1])

        super().__init__(modules, attachments, fit_moments, fit_gd, lam, precompute_callback, other_parameters)

    def compute(self, it=10, method='euler'):
        """ Does shooting. Outputs compute deformation and attach cost. """
        # Call precompute callback if available
        # TODO: maybe do this in Model and not ModelPointsRegistration ?
        pc_cost = None
        if self.precompute_callback is not None:
            pc_cost = self.precompute_callback(self.init_manifold, self.modules, self.parameters)

        # We first create and fill the compound module we will shoot
        compound = CompoundModule(self.modules)
        compound.manifold.fill(self.init_manifold)

        # Shooting
        shoot(Hamiltonian(compound), it, method)
        deformation_cost = compound.cost()

        # We compute the attach cost for each source/target couple
        attach_costs = []
        for i in range(self.source_count):
            if self.weights[i] is not None:
                attach_costs.append(self.attachments[i]((compound[i].manifold.gd.view(-1, self.__source_dim[i]), self.weights[i])))
            else:
                attach_costs.append(self.attachments[i](compound[i].manifold.gd.view(-1, self.__source_dim[i])))

        attach_cost = self.lam*sum(attach_costs)
        c = deformation_cost + attach_cost
        if pc_cost is not None:
            c = c + pc_cost

        cost = c.detach()
        c.backward()
        del c

        return cost, deformation_cost.detach(), attach_cost.detach()

