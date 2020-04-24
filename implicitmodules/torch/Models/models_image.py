import copy

import torch

from implicitmodules.torch.Models import Model
from implicitmodules.torch.Utilities import sample_from_greyscale, deformed_intensities
from implicitmodules.torch.DeformationModules import SilentLandmarks, CompoundModule
from implicitmodules.torch.HamiltonianDynamic import shoot, Hamiltonian


class ModelImageRegistration(Model):
    def __init__(self, source_image, modules, attachment, lam=1., fit_gd=None, precompute_callback=None, other_parameters=None):
        if other_parameters is None:
            other_parameters = []

        self.__image_resolution = source_image.shape
        source_pos, source_weight = sample_from_greyscale(source_image, 0., centered=False, normalise_weights=False, normalise_position=False)

        model_modules = []
        model_modules.append(SilentLandmarks(source_pos.shape[1], source_pos.shape[0], gd=source_pos.clone().requires_grad_(), cotan=torch.zeros_like(source_pos, requires_grad=True)))
        model_modules.extend(modules)

        self.__weights = source_weight

        if fit_gd:
            fit_gd = [False, *fit_gd]
        
        super().__init__(model_modules, [attachment], fit_gd, lam, precompute_callback, other_parameters)

    def _compute_attachment_cost(self, deformed, target):
        return self.attachments[0](deformed, target)

    def compute_deformed(self, solver, it, costs=None, intermediates=None):
        assert isinstance(costs, dict) or costs is None
        assert isinstance(intermediates, list) or intermediates is None

        if intermediates:
            raise NotImplementedError()

        # First, forward step shooting only the deformation modules
        compound = CompoundModule(self.modules[1:])
        compound.manifold.fill_gd([manifold.gd for manifold in self.init_manifold[1:]])
        compound.manifold.fill_cotan([manifold.cotan for manifold in self.init_manifold[1:]])

        # Forward shooting
        shoot(Hamiltonian(compound), solver, it)

        # Prepare for reverse shooting
        compound.manifold.negate_cotan()
        silent = self.modules[0]
        silent.manifold.fill_gd(self.init_manifold[0].gd)
        silent.manifold.fill_cotan(self.init_manifold[0].cotan)
        compound = CompoundModule([silent, *compound.modules])

        # Then, backward shooting in order to get the final deformed image
        shoot(Hamiltonian(compound), solver, it)

        deformed_image = deformed_intensities(compound[0].manifold.gd, self.__weights.view(self.__image_resolution))

        if costs is not None:
            costs['deformation'] = compound.cost()

        return deformed_image

