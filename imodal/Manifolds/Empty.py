import torch

from imodal.Manifolds.Abstract import BaseManifold
from imodal.StructuredFields import StructuredField_Null


class EmptyManifold(BaseManifold):
    def __init__(self, dim, device=None, dtype=torch.float):
        super().__init__(device, dtype)
        self.__dim = dim

    def clone(self, requires_grad=True):
        return EmptyManifold(self.__dim, device=self.device, dtype=self.dtype)

    def _to_device(self, device):
        pass

    def _to_dtype(self, dtype):
        pass

    @property
    def nb_pts(self):
        return 0

    @property
    def dim(self):
        return self.__dim

    @property
    def numel_gd(self):
        return (0,)

    @property
    def shape_gd(self):
        return(torch.Size([]),)

    @property
    def len_gd(self):
        return 0

    @property
    def dim_gd(self):
        return ()

    def gd_requires_grad_(self, requires_grad=True):
        pass

    def tan_requires_grad_(self, requires_grad=True):
        pass

    def cotan_requires_grad_(self, requires_grad=True):
        pass

    def unroll_gd(self):
        return []

    def unroll_tan(self):
        return []

    def unroll_cotan(self):
        return []

    def roll_gd(self, l):
        pass

    def roll_tan(self, l):
        pass

    def roll_cotan(self, l):
        pass

    def __get_gd(self):
        return torch.tensor([], device=self.device, dtype=self.dtype)

    def __get_tan(self):
        return torch.tensor([], device=self.device, dtype=self.dtype)
 
    def __get_cotan(self):
        return torch.tensor([], device=self.device, dtype=self.dtype)

    def fill_gd(self, gd, copy=False, requires_grad=True):
        pass

    def fill_tan(self, tan, copy=False, requires_grad=True):
        pass

    def fill_cotan(self, cotan, copy=False, requires_grad=True):
        pass

    def fill_gd_zeros(self, requires_grad=True):
        pass

    def fill_tan_zeros(self, requires_grad=True):
        pass

    def fill_cotan_zeros(self, requires_grad=True):
        pass

    def fill_gd_randn(self, requires_grad=True):
        pass

    def fill_tan_randn(self, requires_grad=True):
        pass

    def fill_cotan_randn(self, requires_grad=True):
        pass

    gd = property(__get_gd, fill_gd)
    tan = property(__get_tan, fill_tan)
    cotan = property(__get_cotan, fill_cotan)

    def add_gd(self, gd):
        pass

    def add_tan(self, tan):
        pass

    def add_cotan(self, cotan):
        pass

    def negate_gd(self):
        pass

    def negate_tan(self):
        pass

    def negate_cotan(self):
        pass

    def inner_prod_field(self, field):
        return 0.

    def action(self, field) :
        """Applies the vector field generated by the module on the landmark."""
        return EmptyManifold(self.__dim, self.device, self.dtype)

    def cot_to_vs(self, sigma, backend=None):
        return StructuredField_Null(self.__dim, device=self.device)

