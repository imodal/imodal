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
    def __init__(self, modules, attachments, fit_gd, lam, precompute_callback, other_parameters):
        self.__modules = modules
        self.__attachments = attachments
        self.__precompute_callback = precompute_callback
        self.__fit_gd = fit_gd
        self.__lam = lam

        if other_parameters is None:
            other_parameters = []

        [module.manifold.fill_cotan_zeros(requires_grad=True) for module in self.__modules]

        self.__init_manifold = CompoundModule(self.__modules).manifold.clone(requires_grad=True)
        self.__init_other_parameters = other_parameters

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
    def fit_gd(self):
        return self.__fit_gd

    @property
    def init_manifold(self):
        return self.__init_manifold

    @property
    def init_parameters(self):
        return self.__init_parameters

    @property
    def init_other_parameters(self):
        return self.__init_other_parameters

    @property
    def parameters(self):
        return self.__parameters

    @property
    def lam(self):
        return self.__lam

    @property
    def attachments(self):
        return self.__attachments

    # def gradcheck(self, target, l):
    #     def energy(*param):
    #         parameters = list(param)
    #         init_manifold = []
    #         init_other_parameters = []

    #         init_manifold = parameters[:self.__init_manifold.len_gd]
    #         self.__init_other_parameters = parameters[len(init_manifold):]
    #         self.__init_manifold.fill_cotan(self.__init_manifold.roll_cotan(init_manifold))

    #         self.compute_parameters()

    #         if self.__precompute_callback:
    #             self.__precompute_callback(self.__modules, self.__parameters)

    #         deformation_cost, attach_cost = self.compute()
    #         cost = deformation_cost + attach_cost
    #         return cost

    #     return torch.autograd.gradcheck(energy, self.__parameters, raise_exception=False)

    def compute_parameters(self):
        """
        Fill the parameter list that will be given to the optimizer. 
        """
        self.__parameters = {}

        # Initial moments
        self.__parameters['cotan'] = self.__init_manifold.unroll_cotan()

        # Geometrical descriptors if specified
        if self.__fit_gd and any(self.__fit_gd):
            list_gd = []
            for fit_gd, init_manifold in zip(self.__fit_gd, self.__init_manifold):
                if fit_gd:
                    list_gd.extend(init_manifold.unroll_gd())

            self.__parameters['gd'] = list_gd

        # Other parameters
        self.__parameters.update(self.__init_other_parameters)

    # def compute_deformation_grid(self, aabb, resolution, method, it, intermediates=False):
    #     x, y = torch.meshgrid([
    #         torch.linspace(grid_origin[0], grid_origin[0]+grid_size[0], grid_resolution[0]),
    #         torch.linspace(grid_origin[1], grid_origin[1]+grid_size[1], grid_resolution[1])])

    #     gridpos = grid2vec(x, y)

    #     grid_silent = SilentLandmarks(2, gridpos.shape[0], gd=gridpos.requires_grad_())
    #     grid_silent.manifold.fill_cotan_zeros()
    #     compound = CompoundModule(copy.copy(self.modules))
    #     compound.manifold.fill(self.init_manifold.clone())

    #     if intermediates:
    #         intermediate_states, _ = shoot(Hamiltonian([grid_silent, *compound]), it, method, intermediates=intermediates)

    #         return [vec2grid(inter[0].gd.detach(), grid_resolution[0], grid_resolution[1]) for inter in intermediate_states]
    #     else:
    #         shoot(Hamiltonian([grid_silent, *compound]), it, method)

    #         return vec2grid(grid_silent.manifold.gd.detach(), grid_resolution[0], grid_resolution[1])

    def evaluate(self, targets, method, it, compute_backward=True, ext_cost=None):
        """ Evaluate the model and output its cost.
        
        Parameters
        ----------
        targets : torch.Tensor or list of torch.Tensor
            Targets we try to approach.
        method : str
            Integration method to use.
        it : int
            Number of iterations for the integration.
        compute_backward : bool
            If True, computes the gradient of the output cost tensor.
        ext_cost : torch.Tensor, default=None
            Scalar tensor representing an external cost we want to add to the final cost.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the cost of the model.
        """
        # Call precompute callback if available
        pc_cost = None
        if self.precompute_callback is not None:
            pc_cost = self.precompute_callback(self.init_manifold, self.modules, self.parameters)

        # Compute the deformed source
        deformed_sources, deformation_cost = self.compute_deformed(method, it, deformation_cost=True)

        # Compute the attach cost for each source/target couple
        attach_costs = []
        for deformed_source, target, attachment in zip(deformed_sources, targets, self.attachments):
            if isinstance(deformed_source, torch.Tensor):
                attach_costs.append(attachment(deformed_source, target))
            else:
                attach_costs.append(attachment((deformed_source, source[1]), target))

        # Compute attach cost and final cost
        attach_cost = self.lam*sum(attach_costs)
        cost = deformation_cost + attach_cost

        # If the precompute callback outputed a cost, add it
        if pc_cost is not None:
            cost = cost + pc_cost

        # If an external cost is defined, add it
        if ext_cost is not None:
            cost = cost + ext_cost

        # If we need to compute the backward of the cost, do it.
        if compute_backward and cost.requires_grad:
            cost.backward()

        return cost.detach().item(), deformation_cost.detach().item(), attach_cost.detach().item()

    """Compute the deformed source.

    Parameters
    ----------
    method : str
        Integration method to use for the shooting.
    it : int
        Number of iterations the integration method will do.
    def_cost : bool
        If True, also output the deformation cost.

    Returns
    -------
    list
        List of deformed sources.
    """
    def compute_deformed(self, method, it, def_cost=False):
        raise NotImplementedError()



