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

    def gradcheck(self, target, l):
        def energy(*param):
            parameters = list(param)
            init_manifold = []
            init_other_parameters = []

            init_manifold = parameters[:self.__init_manifold.len_gd]
            self.__init_other_parameters = parameters[len(init_manifold):]
            self.__init_manifold.fill_cotan(self.__init_manifold.roll_cotan(init_manifold))

            self.compute_parameters()

            if self.__precompute_callback:
                self.__precompute_callback(self.__modules, self.__parameters)

            deformation_cost, attach_cost = self.compute()
            cost = deformation_cost + attach_cost
            return cost

        return torch.autograd.gradcheck(energy, self.__parameters, raise_exception=False)

    def compute_parameters(self):
        """
        Fill the parameter list that will be given to the optimizer. 
        """
        self.__parameters = []

        # Initial moments
        self.__parameters.extend(self.__init_manifold.unroll_cotan())

        # Geometrical descriptors if specified
        # TODO: Pythonise this ?
        if self.__fit_gd is not None:
            for i in range(len(self.__modules)):
                if self.__fit_gd[i]:
                    self.__parameters.extend(self.__init_manifold[i].unroll_gd())

        # Other parameters
        self.__parameters.extend(self.__init_other_parameters)

    def compute_deformation_grid(self, grid_origin, grid_size, grid_resolution, it=10, method="euler", intermediates=False):
        x, y = torch.meshgrid([
            torch.linspace(grid_origin[0], grid_origin[0]+grid_size[0], grid_resolution[0]),
            torch.linspace(grid_origin[1], grid_origin[1]+grid_size[1], grid_resolution[1])])

        gridpos = grid2vec(x, y)

        grid_silent = SilentLandmarks(2, gridpos.shape[0], gd=gridpos.requires_grad_())
        compound = CompoundModule(copy.copy(self.modules))
        compound.manifold.fill(self.init_manifold.clone())

        if intermediates:
            intermediate_states, _ = shoot(Hamiltonian([grid_silent, *compound]), it, method, intermediates=intermediates)

            return [vec2grid(inter[0].gd.detach(), grid_resolution[0], grid_resolution[1]) for inter in intermediate_states]
        else:
            shoot(Hamiltonian([grid_silent, *compound]), it, method)

            return vec2grid(grid_silent.manifold.gd.detach(), grid_resolution[0], grid_resolution[1])


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

    def compute(self, targets, it=10, method='euler', compute_backward=True, ext_cost=None):
        """ Does shooting. Outputs compute deformation and attach cost. """
        # Call precompute callback if available
        # TODO: maybe do this in Model and not ModelPointsRegistration ?
        pc_cost = None
        if self.precompute_callback is not None:
            pc_cost = self.precompute_callback(self.init_manifold, self.modules, self.parameters)

        # We first create and fill the compound module we will shoot
        compound = CompoundModule(self.modules)
        compound.manifold.fill_gd([manifold.gd for manifold in self.init_manifold])
        compound.manifold.fill_cotan([manifold.cotan for manifold in self.init_manifold])

        # Compute the deformation cost (constant)
        compound.compute_geodesic_control(compound.manifold)
        deformation_cost = compound.cost()

        # Shooting
        shoot(Hamiltonian(compound), it, method)

        # We compute the attach cost for each source/target couple
        attach_costs = []
        for source, target, silent, attachment in zip(self.__sources, targets, compound, self.attachments):
            if isinstance(source, torch.Tensor):
                attach_costs.append(attachment(silent.manifold.gd, target))
            else:
                attach_costs.append(attachment((silent.manifold.gd, source[1]), target))

        attach_cost = self.lam*sum(attach_costs)
        cost = deformation_cost + attach_cost

        if pc_cost is not None:
            cost = cost + pc_cost

        if ext_cost is not None:
            cost = cost + ext_cost

        if compute_backward:
            cost.backward()

            return cost.detach().item(), deformation_cost.detach().item(), attach_cost.detach().item()

