import copy
import time
from collections import Iterable

import torch

from implicitmodules.torch.DeformationModules import CompoundModule, SilentLandmarks
from implicitmodules.torch.HamiltonianDynamic import Hamiltonian, shoot
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.Utilities.sampling import sample_from_greyscale, deformed_intensities
from implicitmodules.torch.Utilities.usefulfunctions import grid2vec, vec2grid


class BaseModel:
    def __init__(self):
        pass

    def evaluate(self, target, solver, it):
        raise NotImplementedError()

    def compute_deformed(self, solver, it, costs=None, intermediates=None):
        raise NotImplementedError()


class Model(BaseModel):
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

        # Update the parameter dict
        self._compute_parameters()

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

    def _compute_parameters(self):
        """
        Fill the parameter dictionary that will be given to the optimizer. 
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

    def evaluate(self, target, solver, it, costs=None):
        """ Evaluate the model and output its cost.

        Parameters
        ----------
        targets : torch.Tensor or list of torch.Tensor
            Targets we try to approach.
        solver : str
            Solver to use for the shooting.
        it : int
            Number of iterations for the integration.

        Returns
        -------
        dict
            Dictionnary of (string, float) pairs, representing the costs.
        """
        if costs is None:
            costs = {}

        # Call precompute callback if available
        precompute_cost = None
        if self.precompute_callback is not None:
            precompute_cost = self.precompute_callback(self.init_manifold, self.modules, self.parameters)

        if costs is not None and precompute_cost is not None:
            costs['precompute'] = precompute_cost

        deformed = self.compute_deformed(solver, it, costs=costs)
        costs['attach'] = self.lam*self._compute_attachment_cost(deformed, target)

        total_cost = sum(costs.values())

        if total_cost.requires_grad:
            total_cost.backward()

        # Return costs as a dictionary of floats
        costs['total'] = total_cost
        return dict([(key, costs[key].item()) for key in costs])

    def _compute_attachment_cost(self, deformed, deformation_costs=None):
        raise NotImplementedError

    def compute_deformed(self, solver, it, costs=None, intermediates=None):
        """ Compute the deformed source.

        Parameters
        ----------
        solver : str
            Solver to use for the shooting.
        it : int
            Number of iterations the integration method will do.
        costs : dict, default=None
            If provided, will be filled with the costs associated to the deformation.

        Returns
        -------
        list
            List of deformed sources.
        """
        raise NotImplementedError()



