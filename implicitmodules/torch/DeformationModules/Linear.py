import torch

from implicitmodules.torch.StructuredFields import StructuredField_Affine
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.DeformationModules.Abstract import DeformationModule


class LinearDeformation(DeformationModule):
    """Global translation module."""
    def __init__(self, manifold, A, coeff=1., label=None):
        super().__init__(label)
        self.__controls = torch.tensor(0., dtype=A.dtype)
        self.__coeff = coeff
        self.__manifold = manifold

        self.__A = A

    @property
    def A(self):
        return self.__A

    @classmethod
    def build(cls, A, coeff=1., gd=None, tan=None, cotan=None, label=None):
        return cls(Landmarks(A.shape[0], 1, gd=gd, tan=tan, cotan=cotan), A, coeff, label)

    @property
    def dim_controls(self):
        return self.__controls.shape[0]

    @property
    def coeff(self):
        return self.__coeff

    @property
    def manifold(self):
        return self.__manifold

    def __get_controls(self):
        return self.__controls

    def fill_controls(self, controls):
        assert controls.shape == torch.Size([])

        self.__controls = controls
    
    def __get_coeff(self):
        return self.__coeff

    def __set_coeff(self, coeff):
        self.__coeff = coeff

    controls = property(__get_controls, fill_controls)
    coeff = property(__get_coeff, __set_coeff)

    def fill_controls_zero(self):
        self.fill_controls(torch.zeros_like(self.__controls))

    def __call__(self, points, k=0):
        """Applies the generated vector field on given points."""
        return self.field_generator()(points, k=k)

    def cost(self):
        """Returns the cost."""
        return 0.5*self.__coeff*self.__controls**2

    def compute_geodesic_control(self, man):
        """Computes geodesic control from StructuredField vs."""
        vs = StructuredField_Affine(self.__A, self.__manifold.gd.flatten(), torch.zeros_like(self.__manifold.gd.flatten()))
        self.__controls = man.inner_prod_field(vs)/self.__coeff

    def field_generator(self):
        return StructuredField_Affine(self.__controls*self.__A, self.__manifold.gd.flatten(), torch.zeros_like(self.__manifold.gd.flatten()))
