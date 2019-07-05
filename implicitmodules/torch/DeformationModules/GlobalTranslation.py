import torch

from implicitmodules.torch.StructuredFields import ConstantField
from implicitmodules.torch.Manifolds import EmptyManifold
from implicitmodules.torch.DeformationModules.Abstract import DeformationModule


class GlobalTranslation(DeformationModule):
    """Global translation module."""
    def __init__(self, dim, coeff=1.):
        self.__controls = torch.zeros(dim)
        self.__coeff = coeff

    @property
    def dim_controls(self):
        return self.__controls.shape[0]

    @property
    def coeff(self):
        return self.__coeff

    @property
    def manifold(self):
        return EmptyManifold()

    def __get_controls(self):
        return self.__controls

    def fill_controls(self, controls):
        self.__controls = controls.clone()

    controls = property(__get_controls, fill_controls)

    def fill_controls_zero(self):
        self.fill_controls(torch.zeros_like(self.__controls))

    def __call__(self, points):
        """Applies the generated vector field on given points."""
        return self.field_generator()(points)

    def cost(self):
        """Returns the cost."""
        return 0.5 * self.__coeff * torch.dot(self.__controls, self.__controls)

    def compute_geodesic_control(self, man):
        """Computes geodesic control from StructuredField vs."""
        geodesic_controls = torch.zeros_like(self.__controls)
        for i in range(self.__controls.shape[0]):
            cont_i = torch.zeros_like(self.__controls)
            cont_i[i] = 1.
            v_i = ConstantField(cont_i)
            geodesic_controls[i] = man.inner_prod_field(v_i) / self.__coeff

        self.__controls = geodesic_controls

    def field_generator(self):
        return ConstantField(self.__controls)
