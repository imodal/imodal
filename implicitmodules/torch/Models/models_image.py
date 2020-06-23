import copy

import torch

from implicitmodules.torch.Models import Model
from implicitmodules.torch.Utilities import sample_from_greyscale, deformed_intensities, AABB
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

        compound = CompoundModule(self.modules)
        compound.manifold.fill_gd([manifold.gd for manifold in self.init_manifold])
        compound.manifold.fill_cotan([manifold.cotan for manifold in self.init_manifold])

        # # Forward shooting
        shoot(Hamiltonian(compound), solver, it)

        # Prepare for reverse shooting
        compound.manifold.negate_cotan()
        pixels = AABB(0., self.__image_resolution[0], 0., self.__image_resolution[1]).fill_count(self.__image_resolution)
        silent = SilentLandmarks(2, pixels.shape[0], gd=pixels)
        compound = CompoundModule([silent, *compound.modules])

        shoot(Hamiltonian(compound), solver, it)

        # intermediates = {}
        # shoot(Hamiltonian(compound), solver, it, intermediates=intermediates)

        # # Prepare for reverse shooting
        # pixels = AABB(0., self.__image_resolution[0], 0., self.__image_resolution[1]).fill_count(self.__image_resolution)
        # silent = SilentLandmarks(2, pixels.shape[0], gd=pixels.requires_grad_())
        # compound = CompoundModule([silent, *compound.modules])

        # reverse_controls = intermediates['controls'][::-1]
        # [controls.insert(0, torch.tensor([])) for controls in reverse_controls]
        # [[control.mul_(-1.) for control in controls] for controls in reverse_controls]

        # # Then, backward shooting in order to get the final deformed image
        # shoot(Hamiltonian(compound), solver, it, controls=reverse_controls)

        if costs is not None:
            costs['deformation'] = compound.cost()

        # TODO remove this horrible t().flip(0) hack
        return deformed_intensities(silent.manifold.gd, self.__weights.view(self.__image_resolution)).t().flip(0)

