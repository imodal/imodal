import copy

import torch

from implicitmodules.torch.HamiltonianDynamic import Hamiltonian, shoot
from implicitmodules.torch.DeformationModules import SilentBase, CompoundModule, SilentLandmarks
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.Utilities import grid2vec, vec2grid, deformed_intensities, AABB, load_greyscale_image


class Deformable:
    def __init__(self, manifold, module_label=None):
        self.__silent_module = SilentBase(manifold, module_label)

    @property
    def geometry(self):
        raise NotImplementedError()

    @property
    def silent_module(self):
        return self.__silent_module

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
        return (self.silent_module.manifold.gd,)

    def compute_deformed(self, modules, solver, it, costs=None, intermediates=None):
        assert isinstance(costs, dict) or costs is None
        assert isinstance(intermediates, dict) or intermediates is None

        compound = CompoundModule([self.silent_module, *modules])

        # Compute the deformation cost if needed
        if costs is not None:
            compound.compute_geodesic_control(compound.manifold)
            costs['deformation'] = compound.cost()

        # Shoot the dynamical system
        shoot(Hamiltonian(compound), solver, it, intermediates=intermediates)

        return compound[0].manifold.gd


# class DeformablePolylines(DeformablePoints):
#     def __init__(self, points, connections):
#         self.__connections = connections
#         super().__init__(points)

#     @classmethod
#     def load_from_file(cls, filename):
#         pass

#     @property
#     def geometry(self):
#         return (self.silent_module.manifold.gd.detach(), self.__connections)


# class DeformableMesh(DeformablePoints):
#     def __init__(self, points, triangles):
#         self.__triangles = triangles
#         super().__init__(points)

#     @classmethod
#     def load_from_file(cls, filename):
#         pass

#     @property
#     def geometry(self):
#         return self.silent_module.manifold.gd.detach(), self.__triangles


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
        return self.silent_module.manifold.gd

    def to_bitmap(self):
        return self.__bitmap

    def compute_deformed(self, modules, solver, it, costs=None, intermediates=None):
        assert isinstance(costs, dict) or costs is None
        assert isinstance(intermediates, dict) or intermediates is None


        # Forward shooting
        compound_modules = [self.silent_module, *modules]
        compound = CompoundModule(compound_modules)

        shoot(Hamiltonian(compound), solver, it, intermediates=intermediates)

        # Prepare for reverse shooting
        compound.manifold.negate_cotan()

        pixel_grid = self.__extent.fill_count(self.__shape)
        # pixel_grid = AABB(0., self.__shape[0], 0, self.__shape[1]).fill_count(self.__shape)
        silent_pixel_grid = SilentLandmarks(2, pixel_grid.shape[0], gd=pixel_grid.requires_grad_())

        # Reverse shooting with the newly constructed pixel grid module
        compound = CompoundModule([silent_pixel_grid, *compound.modules])

        shoot(Hamiltonian(compound), solver, it)

        if costs is not None:
            costs['deformation'] = compound.cost()

        return deformed_intensities(silent_pixel_grid.manifold.gd, self.__bitmap, self.__extent)

