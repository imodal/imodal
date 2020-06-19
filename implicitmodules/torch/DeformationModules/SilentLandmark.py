import torch

from implicitmodules.torch.DeformationModules.Abstract import DeformationModule
from implicitmodules.torch.Manifolds.Abstract import BaseManifold
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.StructuredFields import StructuredField_Null
from implicitmodules.torch.Utilities import generate_mesh_grid, grid2vec, vec2grid

class SilentBase(DeformationModule):
    """Module handling silent points."""

    def __init__(self, manifold, label):
        assert isinstance(manifold, BaseManifold)
        super().__init__(label)
        self.__manifold = manifold

    def __str__(self):
        outstr = "Silent module\n"
        if self.label:
            outstr += "  Label=" + self.label + "\n"
        outstr += "  Manifold type=" + self.__manifold.__class__.__name__
        outstr += "  Nb pts=" + str(self.__manifold.nb_pts)
        return outstr

    @classmethod
    def build(cls, dim, nb_pts, manifold, gd=None, tan=None, cotan=None, label=None, **kwargs):
        return cls(manifold(dim, nb_pts, **kwargs, gd=gd, tan=tan, cotan=cotan), label)

    def to_(self, *args, **kwargs):
        self.__manifold.to_(*args, **kwargs)

    @property
    def device(self):
        return self.__manifold.device

    @property
    def dim(self):
        return self.__manifold.dim

    @property
    def manifold(self):
        return self.__manifold

    def __get_controls(self):
        return torch.tensor([], requires_grad=True)

    def fill_controls(self, controls):
        pass

    controls = property(__get_controls, fill_controls)

    def fill_controls_zero(self):
        pass

    # For now, for unit test to pass, we need to set requires_grad=True.
    # But is it realy necessary ?
    def __call__(self, points, k=0):
        return torch.zeros_like(points, requires_grad=True, device=self.device)

    # For now, for unit test to pass, we need to set requires_grad=True.
    # But is it realy necessary ?
    def cost(self):
        return torch.tensor(0., requires_grad=True, device=self.device)

    def compute_geodesic_control(self, man):
        pass

    def field_generator(self):
        return StructuredField_Null(self.__manifold.dim, device=self.device)

    def adjoint(self, manifold):
        return StructuredField_Null(self.__manifold.dim, device=self.device)


Silent = SilentBase.build


def SilentLandmarks(dim, nb_pts, gd=None, tan=None, cotan=None, label=None):
    return SilentBase.build(dim, nb_pts, Landmarks, gd=gd, tan=tan, cotan=cotan, label=label)


class DeformationGrid(SilentBase):
    """
    Helper class to manipulate deformation grids as deformation modules.
    Built on top of silent module
    """

    def __init__(self, aabb, resolution, label=None):
        self.__aabb = aabb
        self.__resolution = resolution

        grid = generate_mesh_grid(aabb, resolution)
        points_grid = grid2vec(*grid)

        manifold = Landmarks(aabb.dim, points_grid.shape[0], gd=points_grid)

        super().__init__(manifold, label)

    @property
    def aabb(self):
        return self.__aabb

    @property
    def resolution(self):
        return self.__resolution

    def togrid(self):
        return vec2grid(self.manifold.gd.detach(), *self.__resolution)


