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

        super().__init__(model_modules, attachment, fit_gd, lam, precompute_callback, other_parameters)

    def compute(self, target, it=10, method='torch_euler', compute_backward=True, ext_cost=None):
        pc_cost = None
        if self.precompute_callback is not None:
            pc_cost = self.precompute_callback(self.init_manifold, self.modules, self.parameters)

        # First, forward step shooting only the deformation modules
        compound = CompoundModule(self.modules[1:])
        compound.manifold.fill_gd([manifold.gd for manifold in self.init_manifold[1:]])
        compound.manifold.fill_cotan([manifold.cotan for manifold in self.init_manifold[1:]])

        # Compute deformation cost before shooting
        compound.compute_geodesic_control(compound.manifold)
        deformation_cost = compound.cost()

        # Forward shooting
        shoot(Hamiltonian(compound), it, method)

        # Prepare for reverse shooting
        compound.manifold.negate_cotan()
        silent = self.modules[0]
        silent.manifold.fill_cotan(self.init_manifold[0].cotan)
        compound = CompoundModule([silent, *compound.modules])

        # Then, backward shooting in order to get the final deformed image
        shoot(Hamiltonian(compound), it, method)

        self.__deformed_image = deformed_intensities(compound[0].manifold.gd, self.__weights.view(self.__image_resolution)).clone()

        # import matplotlib.pyplot as plt
        # plt.imshow(self.__deformed_image)
        # plt.show()

        # Compute attach cost
        attach_cost = self.lam * self.attachments(self.__deformed_image, target)
        cost = deformation_cost + attach_cost

        if pc_cost is not None:
            cost = cost + pc_cost

        if ext_cost is not None:
            cost = cost + ext_cost

        if compute_backward and cost.requires_grad:
            cost.backward()

        return cost.detach().item(), deformation_cost.detach().item(), attach_cost.detach().item()

