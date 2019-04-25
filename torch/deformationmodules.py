import copy
from collections import Iterable

import torch
import numpy as np

from .structuredfield import StructuredField_Null, StructuredField_0, CompoundStructuredField
from .kernels import gauss_kernel, K_xx, K_xy, compute_sks
from .manifold import Landmarks, CompoundManifold


class DeformationModule:
    """Abstract module."""
    def __init__(self):
        super().__init__()

    def copy(self):
        return copy.copy(self)

    def __call__(self, gd, controls, points):
        """Applies the generated vector field on given points."""
        raise NotImplementedError

    def cost(self, gd, controls):
        """Returns the cost."""
        raise NotImplementedError


class Translations(DeformationModule):
    """Module generating sum of translations."""
    def __init__(self, manifold, sigma):
        assert isinstance(manifold, Landmarks)
        super().__init__()
        self.__manifold = manifold
        self.__sigma = sigma
        self.__dim_controls = self.__manifold.dim*self.__manifold.nb_pts
        self.__controls = torch.zeros(self.__dim_controls, requires_grad=True)

    @classmethod
    def build_and_fill(cls, dim, nb_pts, sigma, gd=None, tan=None, cotan=None):
        """Builds the Translations deformation module from tensors."""
        return cls(Landmarks(dim, nb_pts, gd=gd, tan=tan, cotan=cotan), sigma)

    @property
    def manifold(self):
        return self.__manifold

    @property
    def sigma(self):
        return self.__sigma

    @property
    def dim_controls(self):
        return self.__dim_controls

    def __get_controls(self):
        return self.__controls

    def fill_controls(self, controls):
        self.__controls = controls

    controls = property(__get_controls, fill_controls)

    def fill_controls_zero(self):
        self.__controls = torch.zeros(self.__dim_controls)

    def __call__(self, points):
        """Applies the generated vector field on given points."""
        K_q = K_xy(points, self.__manifold.gd.view(-1, self.__manifold.dim), self.__sigma)
        return torch.mm(K_q, self.__controls.view(-1, self.__manifold.dim))

    def cost(self):
        """Returns the cost."""
        K_q = K_xx(self.manifold.gd.view(-1, self.__manifold.dim), self.__sigma)
        m = torch.mm(K_q, self.__controls.view(-1, self.__manifold.dim))
        return 0.5*torch.dot(m.view(-1), self.__controls.view(-1))

    def compute_geodesic_control(self, man):
        """Computes geodesic control from StructuredField vs."""
        vs = self.adjoint(man)
        K_q = K_xx(self.manifold.gd.view(-1, self.__manifold.dim), self.__sigma)
        controls, _ = torch.gesv(vs(self.manifold.gd.view(-1, self.manifold.dim)), K_q)
        self.__controls = controls.contiguous().view(-1)

    def field_generator(self):
        return StructuredField_0(self.__manifold.gd.view(-1, self.__manifold.dim),
                                 self.__controls.view(-1, self.__manifold.dim), self.__sigma)

    def adjoint(self, manifold):
        return manifold.cot_to_vs(self.__sigma)


class SilentPoints(DeformationModule):
    """Module handling silent points."""
    def __init__(self, manifold):
        assert isinstance(manifold, Landmarks)
        super().__init__()
        self.__manifold = manifold

    @classmethod
    def build_from_points(cls, pts):
        """Builds the Translations deformation module from tensors."""
        return cls(Landmarks(pts.shape[1], pts.shape[0], gd=pts.view(-1)))

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

    def __call__(self, points):
        """Applies the generated vector field on given points."""
        return torch.zeros_like(points)

    def cost(self):
        """Returns the cost."""
        return torch.tensor(0.)

    def compute_geodesic_control(self, man):
        """Computes geodesic control from StructuredField vs. For SilentPoints, does nothing."""
        pass

    def field_generator(self):
        return StructuredField_Null()
    
    def adjoint(self, manifold):
        return StructuredField_Null()


class CompoundModule(DeformationModule, Iterable):
    """Combination of modules."""
    def __init__(self, module_list):
        assert isinstance(module_list, Iterable)
        super().__init__()
        self.__module_list = [*module_list]

    @property
    def module_list(self):
        return self.__module_list

    def __getitem__(self, index):
        return self.__module_list[index]

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current >= len(self.__module_list):
            raise StopIteration
        else:
            self.current = self.current + 1
            return self.__module_list[self.current - 1]

    @property
    def nb_module(self):
        return len(self.__module_list)

    @property
    def dim_controls(self):
        return sum([mod.dim_controls for mod in self.__module_list])

    def __get_controls(self):
        return [m.controls for m in self.__module_list]

    def fill_controls(self, controls):
        assert len(controls) == self.nb_module
        for i in range(self.nb_module):
            self.__module_list[i].fill_controls(controls[i])

    controls = property(__get_controls, fill_controls)

    def fill_controls_zero(self):
        for m in self.__module_list:
            m.fill_controls_zero()

    @property
    def manifold(self):
        return CompoundManifold([m.manifold for m in self.__module_list])

    def __call__(self, points) :
        """Applies the generated vector field on given points."""
        app_list = []
        for m in self.__module_list:
            app_list.append(m(points))

        return sum(app_list).view(-1, self.manifold.dim)

    def cost(self):
        """Returns the cost."""
        cost_list = []
        for m in self.__module_list:
            cost_list.append(m.cost())

        return sum(cost_list)

    def compute_geodesic_control(self, man):
        """Computes geodesic control from \delta \in H^\ast."""
        for i in range(self.nb_module):
            self.__module_list[i].compute_geodesic_control(man)

    def field_generator(self):
        return CompoundStructuredField([m.field_generator() for m in self.__module_list])

