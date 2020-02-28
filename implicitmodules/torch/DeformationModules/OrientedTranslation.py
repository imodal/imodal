import torch
import numpy as np

from pykeops.torch import Genred, KernelSolve

from implicitmodules.torch.DeformationModules.Abstract import DeformationModule, create_deformation_module_with_backends
from implicitmodules.torch.Kernels.kernels import K_xy, K_xx
from implicitmodules.torch.Manifolds import LandmarksDirection
from implicitmodules.torch.StructuredFields import StructuredField_0


class OrientedTranslationsBase(DeformationModule):
    """Module generating sum of translations."""
    
    def __init__(self, manifold, sigma, coeff, label):
        assert isinstance(manifold, LandmarksDirection)
        super().__init__(label)
        self.__manifold = manifold
        self.__sigma = sigma
        self.__coeff = coeff
        self.__controls = torch.zeros(self.__manifold.nb_pts, device=manifold.device, dtype=manifold.dtype)

    def __str__(self):
        outstr = "Oriented translation\n"
        if self.label:
            outstr += "  Label=" + self.label + "\n"
        outstr += "  Sigma=" + str(self.__sigma) + "\n"
        outstr += "  Coeff=" + str(self.__coeff) + "\n"
        outstr += "  Nb pts=" + str(self.__manifold.nb_pts)
        return outstr

    @classmethod
    def build(cls, dim, nb_pts, sigma, transport='vector', coeff=1., gd=None, tan=None, cotan=None, label=None):
        """Builds the Translations deformation module from tensors."""
        return cls(LandmarksDirection(dim, nb_pts, transport, gd=gd, tan=tan, cotan=cotan), sigma, coeff, label)

    def to_(self, *args, **kwargs):
        self.__manifold.to_(*args, **kwargs)
        self.__controls = self.__controls.to(*args, **kwargs)

    @property
    def device(self):
        return self.__manifold.device

    @property
    def manifold(self):
        return self.__manifold

    @property
    def dim(self):
        return self.__manifold.dim

    @property
    def sigma(self):
        return self.__sigma

    @property
    def coeff(self):
        return self.__coeff

    def __get_controls(self):
        return self.__controls

    def fill_controls(self, controls):
        self.__controls = controls

    controls = property(__get_controls, fill_controls)

    def fill_controls_zero(self):
        self.__controls = torch.zeros(self.__manifold.nb_pts, device=self.__manifold.device, dtype=self.__manifold.dtype)

    def __call__(self, points, k=0):
        """Applies the generated vector field on given points."""
        return self.field_generator()(points, k)

    def cost(self):
        raise NotImplementedError

    def compute_geodesic_control(self, man):
        raise NotImplementedError

    def field_generator(self):
        return StructuredField_0(self.__manifold.gd[0], self.__controls.unsqueeze(1).repeat(1, self.dim)*self.__manifold.gd[1], self.__sigma, device=self.device, backend=self.backend)

    def adjoint(self, manifold):
        return manifold.cot_to_vs(self.__sigma, backend=self.backend)


class OrientedTranslations_Torch(OrientedTranslationsBase):
    def __init__(self, manifold, sigma, coeff, label):
        super().__init__(manifold, sigma, coeff, label)

    @property
    def backend(self):
        return 'torch'

    def cost(self):
        K_q = K_xx(self.manifold.gd[0], self.sigma)

        m = torch.mm(K_q, self.controls.unsqueeze(1).repeat(1, self.dim)*self.manifold.gd[1])
        return 0.5 * self.coeff * torch.dot(m.flatten(), (self.controls.unsqueeze(1).repeat(1, self.dim)*self.manifold.gd[1]).flatten())

    def compute_geodesic_control(self, man):
        """Computes geodesic control from StructuredField vs."""
        vs = self.adjoint(man)
        Z = K_xx(self.manifold.gd[0], self.sigma) * torch.mm(self.manifold.gd[1], self.manifold.gd[1].T)
        controls, _ = torch.solve(torch.einsum('ni, ni->n', vs(self.manifold.gd[0]), self.manifold.gd[1]).unsqueeze(1), Z)

        self.controls = controls.flatten().contiguous()/self.coeff


class OrientedTranslations_KeOps(OrientedTranslationsBase):
    def __init__(self, manifold, sigma, coeff, label):
        super().__init__(manifold, sigma, coeff, label)

        self.__keops_dtype = str(manifold.gd.dtype).split(".")[1]
        self.__keops_backend = 'CPU'
        if str(self.device) != 'cpu':
            self.__keops_backend = 'GPU'

    @property
    def backend(self):
        return 'keops'

    def cost(self):
        raise NotImplementedError()

    def compute_geodesic_control(self, man):
        raise NotImplementedError()

OrientedTranslations = create_deformation_module_with_backends(OrientedTranslations_Torch.build, OrientedTranslations_Torch.build)

