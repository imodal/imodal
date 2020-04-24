import copy
import time
from collections import Iterable

import torch

from implicitmodules.torch.Models import Model
from implicitmodules.torch.DeformationModules import CompoundModule, SilentLandmarks
from implicitmodules.torch.HamiltonianDynamic import Hamiltonian, shoot
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.Utilities.sampling import sample_from_greyscale, deformed_intensities
from implicitmodules.torch.Utilities.usefulfunctions import grid2vec, vec2grid


class ModelPointsRegistration(Model):
    """
    TODO: add documentation
    """
    def __init__(self, source, modules, attachments, lam=1., fit_gd=None, precompute_callback=None, other_parameters=None):
        assert isinstance(source, Iterable) and not isinstance(source, torch.Tensor)

        if other_parameters is None:
            other_parameters = []

        # We keep a copy of the sources
        self.__sources = copy.deepcopy(source)

        # We now create the corresponding silent modules
        model_modules = []
        self.__weights = []
        for source in self.__sources:
            # Some weights provided
            if isinstance(source, tuple) and len(source) == 2:
                model_modules.append(SilentLandmarks(source[0].shape[1], source[0].shape[0], gd=source[0].clone().requires_grad_(), cotan=torch.zeros_like(source[0], requires_grad=True, device=source[0].device, dtype=source[0].dtype)))
                self.__weights.append(source[1])

            # No weights provided
            elif isinstance(source, torch.Tensor):
                model_modules.append(SilentLandmarks(source.shape[1], source.shape[0], gd=source.clone().requires_grad_(), cotan=torch.zeros_like(source, requires_grad=True, device=source.device, dtype=source.dtype)))
                self.__weights.append(None)

            else:
                raise RuntimeError("ModelPointsRegistration.__init__(): source type {source_type} not implemented or of wrong length!".format(source_type=source.__class__.__name__))

        model_modules.extend(modules)

        super().__init__(model_modules, attachments, fit_gd, lam, precompute_callback, other_parameters)

    def _compute_attachment_cost(self, deformed, target):
        # Compute the attach cost for each source/target couple
        attach_costs = []
        for deformed_source, target, attachment in zip(deformed, target, self.attachments):
            if isinstance(deformed_source, torch.Tensor):
                attach_costs.append(attachment(deformed_source, target))
            else:
                attach_costs.append(attachment((deformed_source, source[1]), target))

        return sum(attach_costs)

    def compute_deformed(self, solver, it, costs=None, intermediates=None):
        assert isinstance(costs, dict) or costs is None
        assert isinstance(intermediates, dict) or intermediates is None

        # Create and fill the compound module
        compound = CompoundModule(self.modules)
        compound.manifold.fill_gd([manifold.gd for manifold in self.init_manifold])
        compound.manifold.fill_cotan([manifold.cotan for manifold in self.init_manifold])

        # Compute the deformation cost if needed
        if costs is not None:
            compound.compute_geodesic_control(compound.manifold)
            costs['deformation'] = compound.cost()

        # Shoot the dynamical system
        shoot(Hamiltonian(compound), solver, it, intermediates=intermediates)

        # Retrieve and return deformed sources
        deformed_sources = []
        for source, silent in zip(self.__sources, compound):
            if isinstance(source, torch.Tensor):
                deformed_sources.append(silent.manifold.gd)
            else:
                deformed_sources.append((silent.manifold.gd, source[1]))

        return deformed_sources

