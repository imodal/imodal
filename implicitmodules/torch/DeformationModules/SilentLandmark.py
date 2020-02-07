import torch

from implicitmodules.torch.DeformationModules.Abstract import DeformationModule
from implicitmodules.torch.Manifolds.Abstract import BaseManifold
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.StructuredFields import StructuredField_Null


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
        outstr += "  Nb pts=" + self.__manifold.nb_pts
        return outstr

    @classmethod
    def build(cls, dim, nb_pts, manifold, gd=None, tan=None, cotan=None, label=None, **kwargs):
        return cls(manifold(dim, nb_pts, **kwargs, gd=gd, tan=tan, cotan=cotan), label)

    def to_(self, device):
        self.__manifold.to_(device)

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

    # Experimental: requires_grad=True
    # Disable if necessary
    def __call__(self, points):
        """Applies the generated vector field on given points."""
        return torch.zeros_like(points, requires_grad=points.requires_grad, device=self.device)

    # Experimental: requires_grad=True
    # Disable if necessary
    def cost(self):
        """Returns the cost."""
        return torch.tensor(0., requires_grad=True, device=self.device)

    def compute_geodesic_control(self, man):
        """Computes geodesic control from StructuredField vs. For SilentLandmarks, does nothing."""
        pass

    def field_generator(self):
        return StructuredField_Null(self.__manifold.dim, device=self.device)

    def adjoint(self, manifold):
        return StructuredField_Null(self.__manifold.dim, device=self.device)


# Silent module for any kind of manifold
# Silent = SilentBase


# # Give SilentLandmarks the same interface than other deformations modules
# SilentLandmarks = SilentBase.build

Silent = SilentBase.build

def SilentLandmarks(dim, nb_pts, gd=None, tan=None, cotan=None, label=None):
    return SilentBase.build(dim, nb_pts, Landmarks, gd=gd, tan=tan, cotan=cotan, label=label)


