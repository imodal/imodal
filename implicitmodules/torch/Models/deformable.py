from pathlib import Path

import torch
from numpy import loadtxt

from implicitmodules.torch.HamiltonianDynamic import Hamiltonian, shoot
from implicitmodules.torch.DeformationModules import SilentBase, CompoundModule, SilentLandmarks
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.Utilities import deformed_intensities, AABB, load_greyscale_image, pixels2points


class Deformable:
    def __init__(self, manifold, module_label=None):
        self.__silent_module = SilentBase(manifold, module_label)

    @property
    def silent_module(self):
        return self.__silent_module

    @property
    def geometry(self):
        raise NotImplementedError()

    @property
    def _has_backward(self):
        raise NotImplementedError()

    def _backward_module(self):
        raise NotImplementedError()

    def compute_deformed(self, modules, solver, it, costs=None, intermediates=None):
        raise NotImplementedError()

    def _to_deformed(self):
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
    def load_from_csv(cls, filename, **kwargs):
        points = loadtxt(filename, **kwargs)
        return cls(torch.tensor(points))

    @property
    def geometry(self):
        return (self.silent_module.manifold.gd,)

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


class DeformableMesh(DeformablePoints):
    def __init__(self, points, triangles):
        self.__triangles = triangles
        super().__init__(points)

    @classmethod
    def load_from_file(cls, filename):
        pass

    @property
    def triangles(self):
        return self.__triangles

    @property
    def geometry(self):
        return (self.silent_module.manifold.gd, self.__triangles)

    def _to_deformed(self, gd):
        return (gd, self.__triangles)


class DeformableImage(Deformable):
    def __init__(self, bitmap, output='bitmap', extent=None):
        assert isinstance(extent, AABB) or extent is None or isinstance(extent, str)
        assert output == 'bitmap' or output == 'points'

        self.__shape = bitmap.shape
        self.__output = output

        if extent is None:
            extent = AABB(0, 1., 0., 1.)
        elif isinstance(extent, str) and extent == 'match':
            extent = AABB(0., self.__shape[1]-1, 0., self.__shape[0]-1)

        self.__extent = extent

        pixel_points = pixels2points(self.__extent.fill_count(self.__shape), self.__shape, self.__extent)

        self.__bitmap = bitmap
        super().__init__(Landmarks(2, pixel_points.shape[0], gd=pixel_points))

    @classmethod
    def load_from_file(cls, filename, origin='lower', device=None):
        return cls(load_greyscale_image(filename, origin=origin, device=device))

    @property
    def geometry(self):
        if self.__output == 'bitmap':
            return (self.bitmap,)
        elif self.__output == 'points':
            return (self.silent_module.manifold.gd, self.__bitmap.flatten()/torch.sum(self.__bitmap))
        else:
            raise ValueError()

    @property
    def shape(self):
        return self.__shape

    @property
    def extent(self):
        return self.__extent

    @property
    def points(self):
        return self.silent_module.manifold.gd

    @property
    def bitmap(self):
        return self.__bitmap

    @property
    def _has_backward(self):
        return True

    def __set_output(self):
        return self.__output

    def __get_output(self, output):
        self.__output = output

    output = property(__set_output, __get_output)

    def _backward_module(self):
        pixel_grid = pixels2points(self.__extent.fill_count(self.__shape), self.__shape, self.__extent)
        return SilentLandmarks(2, pixel_grid.shape[0], gd=pixel_grid)

    def compute_deformed(self, modules, solver, it, costs=None, intermediates=None):
        assert isinstance(costs, dict) or costs is None
        assert isinstance(intermediates, dict) or intermediates is None

        # Forward shooting
        compound_modules = [self.module, *modules]
        compound = CompoundModule(compound_modules)

        shoot(Hamiltonian(compound), solver, it, intermediates=intermediates)

        # Prepare for reverse shooting
        compound.manifold.negate_cotan()

        silent_pixel_grid = self._backward_module()

        # Reverse shooting with the newly constructed pixel grid module
        compound = CompoundModule([silent_pixel_grid, *compound.modules])

        shoot(Hamiltonian(compound), solver, it)

        if costs is not None:
            costs['deformation'] = compound.cost()

        return self._to_deformed(silent_pixel_grid.manifold.gd)

    def _to_deformed(self, gd):
        if self.__output == 'bitmap':
            return (deformed_intensities(gd, self.__bitmap, self.__extent), )
        elif self.__output == 'points':
            deformed_bitmap = deformed_intensities(gd, self.__bitmap, self.__extent)
            return (gd, deformed_bitmap.flatten()/torch.sum(deformed_bitmap))
        else:
            raise ValueError()


def deformables_compute_deformed(deformables, modules, solver, it, costs=None, intermediates=None):
    assert isinstance(costs, dict) or costs is None
    assert isinstance(intermediates, dict) or intermediates is None

    # Regroup silent modules of each deformable and build a compound module
    silent_modules = [deformable.silent_module for deformable in deformables]
    compound = CompoundModule([*silent_modules, *modules])

    # Forward shooting
    shoot(Hamiltonian(compound), solver, it, intermediates=intermediates)

    # Regroup silent modules of each deformable thats need to shoot backward
    backward_silent_modules = [deformable.silent_module for deformable in deformables if deformable._has_backward]

    if backward_silent_modules is not None:
        # Backward shooting is needed

        # Build/assemble the modules that will be shot backward
        backward_modules = [deformable._backward_module() for deformable in deformables if deformable._has_backward]
        compound = CompoundModule([*backward_silent_modules, *backward_modules, *modules])

        # Reverse the moments for backward shooting
        compound.manifold.negate_cotan()

        # Backward shooting
        shoot(Hamiltonian(compound), solver, it)

    # For now, we need to compute the deformation cost after each shooting (and not before any shooting) for computation tree reasons
    if costs is not None:
        costs['deformation'] = compound.cost()

    # Ugly way to compute the list of deformed objects. Not intended to stay!
    deformed = []
    for deformable in deformables:
        if deformable._has_backward:
            deformed.append(deformable._to_deformed(backward_modules.pop(0).manifold.gd))
        else:
            deformed.append(deformable._to_deformed(deformable.silent_module.manifold.gd))

    return deformed

