import torch

from implicitmodules.torch.HamiltonianDynamic import Hamiltonian, shoot
from implicitmodules.torch.DeformationModules import SilentBase, CompoundModule
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.Utilities import grid2vec, vec2grid

class Deformable:
    def __init__(self, manifold, module_label=None):
        print(manifold)
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

    @property
    def geometry(self):
        return (self.silent_module.manifold.gd,)

    def compute_deformed(self, modules, solver, it, costs=None, intermediates=None):
        assert isinstance(costs, dict) or costs is None
        assert isinstance(intermediates, dict) or intermediates is None

        # Create and fill the compound module
        compound = CompoundModule([self.silent_module, *modules])
        # compound.manifold.fill_gd([manifold.gd for manifold in self.init_manifold])
        # compound.manifold.fill_cotan([manifold.cotan for manifold in self.init_manifold])

        # Compute the deformation cost if needed
        if costs is not None:
            compound.compute_geodesic_control(compound.manifold)
            costs['deformation'] = compound.cost()

        # Shoot the dynamical system
        shoot(Hamiltonian(compound), solver, it, intermediates=intermediates)

        return compound[0].manifold.gd


class DeformablePolylines(DeformablePoints):
    def __init__(self, points, connections):
        self.__connections = connections
        super().__init__(points)

    @classmethod
    def load_from_file(cls, filename):
        pass

    @property
    def geometry(self):
        return (self.silent_module.manifold.gd.detach(), self.__connections)


class DeformableMesh(DeformablePoints):
    def __init__(self, points, triangles):
        self.__triangles = triangles
        super().__init__(points)

    @classmethod
    def load_from_file(cls, filename):
        pass

    @property
    def geometry(self):
        return self.silent_module.manifold.gd.detach(), self.__triangles


class DeformableImage(Deformable):
    def __init__(self, bitmap):
        self.__shape = bitmap.shape
        grid = torch.meshgrid([torch.linspace(0., self.__shape[0], self.__shape[0]), torch.linspace(0., self.__shape[1], self.__shape[1])])
        points = grid2vec(grid)

    @classmethod
    def load_from_file(cls, filename):
        pass

    @property
    def geometry(self):
        return self.to_bitmap()

    def to_points(self):
        pass

    def to_bitmap(self):
        pass

    @property
    def shape(self):
        return self.__shape

    def compute_deformed(self, modules, solver, it, costs=None, intermediates=None):
        assert isinstance(costs, dict) or costs is None
        assert isinstance(intermediates, list) or intermediates is None

        if intermediates:
            raise NotImplementedError()

        # First, forward step shooting only the deformation modules
        compound = CompoundModule(modules)
        # compound.manifold.fill_gd([manifold.gd for manifold in self.init_manifold[1:]])
        # compound.manifold.fill_cotan([manifold.cotan for manifold in self.init_manifold[1:]])

        # Forward shooting

        if costs is not None:
            costs['deformation'] = compound.cost()

        shoot(Hamiltonian(compound), solver, it)

        # Prepare for reverse shooting
        compound.manifold.negate_cotan()
        # silent = self.silent_module()
        # silent.manifold.fill_gd(self.init_manifold[0].gd)
        # silent.manifold.fill_cotan(self.init_manifold[0].cotan)
        compound = CompoundModule([self.silent_module, *compound.modules])

        # Then, backward shooting in order to get the final deformed image
        shoot(Hamiltonian(compound), solver, it)

        return deformed_intensities(compound[0].manifold.gd, self.__weights.view(self.__image_resolution))

