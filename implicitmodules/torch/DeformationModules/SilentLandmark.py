import torch

from implicitmodules.torch.DeformationModules.Abstract import DeformationModule
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.StructuredFields import StructuredField_Null


class SilentLandmarks(DeformationModule):
    """Module handling silent points."""

    def __init__(self, manifold):
        assert isinstance(manifold, Landmarks)
        super().__init__()
        self.__manifold = manifold

    @classmethod
    def build_from_points(cls, pts):
        """Builds the Translations deformation module from tensors."""
        return cls(Landmarks(pts.shape[1], pts.shape[0], gd=pts.view(-1)))

    def to(self, device):
        self.__manifold.to(device)

    @property
    def device(self):
        return self.__manifold.device

    @property
    def dim_controls(self):
        return 0

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
        return torch.zeros_like(points, requires_grad=True, device=self.device)

    # Experimental: requires_grad=True
    # Disable if necessary
    def cost(self):
        """Returns the cost."""
        return torch.tensor(0., requires_grad=True, device=self.device)

    def compute_geodesic_control(self, man):
        """Computes geodesic control from StructuredField vs. For SilentLandmarks, does nothing."""
        pass

    def field_generator(self):
        return StructuredField_Null(device=self.device)

    def adjoint(self, manifold):
        return StructuredField_Null(device=self.device)

