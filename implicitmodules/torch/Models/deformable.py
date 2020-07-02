import copy

import torch

from implicitmodules.torch.HamiltonianDynamic import Hamiltonian, shoot
from implicitmodules.torch.DeformationModules import SilentBase, CompoundModule, SilentLandmarks
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.Utilities import grid2vec, vec2grid, deformed_intensities, AABB, load_greyscale_image


class DeformableBase:
    def __init__(self, module):
        self.__module = module

    @property
    def geometry(self):
        raise NotImplementedError()

    @property
    def module(self):
        return self.__module

    @property
    def _has_backward(self):
        raise NotImplementedError()

    def _backward_module(self):
        raise NotImplementedError()

    def compute_deformed(self, modules, solver, it, costs=None, intermediates=None):
        raise NotImplementedError()

    def _to_deformed(self):
        raise NotImplementedError()


class Deformable(DeformableBase):
    def __init__(self, manifold, module_label=None):
        super().__init__(SilentBase(manifold, module_label))

    def compute_deformed(self, modules, solver, it, costs=None, intermediates=None):
        raise NotImplementedError()


class DeformablePoints(Deformable):
    def __init__(self, points):
        super().__init__(Landmarks(points.shape[1], points.shape[0], gd=points))

    @classmethod
    def load_from_file(cls, filename):
        pass

    @classmethod
    def load_from_pickle(cls, filename):
        pass

    @classmethod
    def load_from_csv(cls, filename):
        pass

    @property
    def geometry(self):
        return (self.module.manifold.gd,)

    @property
    def _has_backward(self):
        return False

    def compute_deformed(self, modules, solver, it, costs=None, intermediates=None):
        assert isinstance(costs, dict) or costs is None
        assert isinstance(intermediates, dict) or intermediates is None

        compound = CompoundModule([self.module, *modules])

        # Compute the deformation cost if needed
        if costs is not None:
            compound.compute_geodesic_control(compound.manifold)
            costs['deformation'] = compound.cost()

        # Shoot the dynamical system
        shoot(Hamiltonian(compound), solver, it, intermediates=intermediates)

        return self._to_deformed(self.module.manifold.gd)

    def _to_deformed(self, gd):
        return (gd,)

# class DeformablePolylines(DeformablePoints):
#     def __init__(self, points, connections):
#         self.__connections = connections
#         super().__init__(points)

#     @classmethod
#     def load_from_file(cls, filename):
#         pass

#     @property
#     def geometry(self):
#         return (self.module.manifold.gd.detach(), self.__connections)


# class DeformableMesh(DeformablePoints):
#     def __init__(self, points, triangles):
#         self.__triangles = triangles
#         super().__init__(points)

#     @classmethod
#     def load_from_file(cls, filename):
#         pass

#     @property
#     def geometry(self):
#         return self.module.manifold.gd.detach(), self.__triangles


class DeformableImage(Deformable):
    def __init__(self, bitmap, extent=None):
        assert isinstance(extent, AABB) or extent is None or isinstance(extent, str)

        self.__shape = bitmap.shape

        if extent is None:
            extent = AABB(0, 1., 0., 1.)
        elif isinstance(extent, str) and extent == 'match':
            extent = AABB(0., self.__shape[0], 0., self.__shape[1])

        self.__extent = extent

        pixel_points = self.__extent.fill_count(self.__shape)

        self.__bitmap = bitmap
        super().__init__(Landmarks(2, pixel_points.shape[0], gd=pixel_points))

    @classmethod
    def load_from_file(cls, filename, origin='lower', device=None):
        return cls(load_greyscale_image(filename, origin=origin, device=device))

    @property
    def geometry(self):
        return (self.to_bitmap(),)

    @property
    def shape(self):
        return self.__shape

    @property
    def extent(self):
        return self.__extent

    def to_points(self):
        return self.module.manifold.gd

    def to_bitmap(self):
        return self.__bitmap

    @property
    def _has_backward(self):
        return True

    def _backward_module(self):
        pixel_grid = self.__extent.fill_count(self.__shape)
        return SilentLandmarks(2, pixel_grid.shape[0], gd=pixel_grid.requires_grad_())

    def compute_deformed(self, modules, solver, it, costs=None, intermediates=None):
        assert isinstance(costs, dict) or costs is None
        assert isinstance(intermediates, dict) or intermediates is None

        # Forward shooting
        compound_modules = [self.module, *modules]
        compound = CompoundModule(compound_modules)

        shoot(Hamiltonian(compound), solver, it, intermediates=intermediates)

        # Prepare for reverse shooting
        compound.manifold.negate_cotan()

        # pixel_grid = self.__extent.fill_count(self.__shape)
        # silent_pixel_grid = SilentLandmarks(2, pixel_grid.shape[0], gd=pixel_grid.requires_grad_())
        silent_pixel_grid = self._backward_module()
        
        # Reverse shooting with the newly constructed pixel grid module
        compound = CompoundModule([silent_pixel_grid, *compound.modules])

        shoot(Hamiltonian(compound), solver, it)

        if costs is not None:
            costs['deformation'] = compound.cost()

        return self._to_deformed(silent_pixel_grid.manifold.gd)

    def _to_deformed(self, gd):
        return (deformed_intensities(gd, self.__bitmap, self.__extent), )


class DeformableCompound(DeformableBase):
    def __init__(self, deformables):
        self.__deformables = deformables

        super().__init__()

    def compute_deformed(self, modules, solver, it, costs=None, intermediates=None):
        assert isinstance(costs, dict) or costs is None
        assert isinstance(intermediates, dict) or intermediates is None

        silent_modules = [module.module for module in self.__deformables]
        compound = CompoundModule([*silent_modules, modules])

        shoot(Hamiltonian(compound), solver, it, intermediates=intermediates)

        backward_silent_modules = [deformable._backward_module() for deformable in self.__deformables if deformable._has_backward]

        if backward_silent_modules is not None:
            backward_modules = [deformable.module for deformable in self.__deformables if deformable._has_backward]

            backward_compound = CompoundModule([*backward_silent_modules, *backward_modules, *modules])

            backward_compound.manifold.negate_cotan()

            shoot(Hamiltonian(compound), solver, it)

        # For now, we need to compute the deformation cost after each shooting (and not before any shooting) for computation tree reasons
        if costs is not None:
            costs['deformation'] = backward_compound.cost()

        # Ugly way to compute the list of deformed objects
        deformed = []
        for deformable in self.__deformables:
            if deformable._has_backward:
                deformable.append(deformable._to_deformable(backward_silent_modules.pop(0)))
            else:
                deformable.append(deformable._to_deformable(deformable.module.manifold.gd))

        return deformed
