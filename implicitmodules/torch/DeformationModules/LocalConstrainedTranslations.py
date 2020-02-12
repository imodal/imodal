import torch
import math

from pykeops.torch import Genred, KernelSolve

from implicitmodules.torch.DeformationModules.Abstract import DeformationModule, create_deformation_module_with_backends
from implicitmodules.torch.Kernels.kernels import K_xy, K_xx
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.StructuredFields import StructuredField_0

class LocalConstrainedTranslationsBase(DeformationModule):
    """Module generating sum of translations."""
    
    def __init__(self, manifold, sigma, descstr, f_support, f_vectors, coeff, label):
        assert isinstance(manifold, Landmarks)
        super().__init__(label)
        self.__manifold = manifold
        self.__sigma = sigma
        self.__descstr = descstr
        self.__controls = torch.zeros(1).view([])
        self.__coeff = coeff

        self._f_support = f_support
        self._f_vectors = f_vectors

    def __str__(self):
        outstr = "Local constrained translation module\n"
        if self.label:
            outstr += "  Label=" + self.label + "\n"
        outstr += "  Type=" + self.descstr + "\n"
        outstr += "  Sigma=" + str(self.__sigma) + "\n"
        outstr += "  Coeff=" + str(self.__coeff)
        return outstr

    @classmethod
    def build(cls, dim, nb_pts, sigma, descstr, f_support, f_vectors, coeff=1., gd=None, tan=None, cotan=None, label=None):
        """Builds the Translations deformation module from tensors."""
        return cls(Landmarks(dim, nb_pts, gd=gd, tan=tan, cotan=cotan), sigma, descstr, f_support, f_vectors, coeff, label)

    def to_(self, *args, **kwargs):
        self.__manifold.to_(*args, **kwargs)
        self.__controls = self.__controls.to(*args, **kwargs)

    @property
    def descstr(self):
        """Description string. Used by __str__()."""
        return self.__descstr

    @property
    def coeff(self):
        return self.__coeff

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

    def __get_controls(self):
        return self.__controls

    def fill_controls(self, controls):
        self.__controls = controls

    controls = property(__get_controls, fill_controls)

    def fill_controls_zero(self):
        self.__controls = torch.zeros(1, requires_grad=True)

    def __call__(self, points, k=0):
        """Applies the generated vector field on given points."""
        return self.field_generator()(points, k)

    def cost(self):
        raise NotImplementedError()

    def compute_geodesic_control(self, man):
        raise NotImplementedError()

    def field_generator(self):
        support = self._f_support(self.__manifold.gd)
        vectors = self._f_vectors(self.__manifold.gd)
        
        return StructuredField_0(support, self.__controls*vectors, self.__sigma, device=self.device, backend=self.backend)

    def adjoint(self, manifold):
        return manifold.cot_to_vs(self.__sigma, backend=self.backend)


class LocalConstrainedTranslations_Torch(LocalConstrainedTranslationsBase):
    def __init__(self, manifold, sigma, descstr, f_support, f_vectors, coeff, label):
        super().__init__(manifold, sigma, descstr, f_support, f_vectors, coeff, label)

    @property
    def backend(self):
        return 'torch'

    def cost(self):
        support = self._f_support(self.manifold.gd)
        vectors = self._f_vectors(self.manifold.gd)

        K_q = K_xx(support, self.sigma)
        m = torch.mm(K_q, vectors)

        return 0.5 * self.coeff * torch.dot(m.flatten(), vectors.flatten()).view([]) * self.controls * self.controls

    def compute_geodesic_control(self, man):
        support = self._f_support(self.manifold.gd)
        vectors = self._f_vectors(self.manifold.gd)

        # vector field for control = 1
        v = StructuredField_0(support, vectors, self.sigma, device=self.device, backend='torch')

        K_q = K_xx(support, self.sigma)
        m = torch.mm(K_q, vectors)
        co = self.coeff * torch.dot(m.flatten(), vectors.flatten())

        self.controls = man.inner_prod_field(v)/co


class LocalConstrainedTranslations_KeOps(LocalConstrainedTranslationsBase):
    def __init__(self, manifold, sigma, descstr, f_support, f_vectors, coeff, label):
        super().__init__(manifold, sigma, descstr, f_support, f_vectors, coeff, label)

    @property
    def backend(self):
        return 'keops'

    def cost(self):
        raise NotImplementedError()

    def compute_geodesic_control(self, man):
        raise NotImplementedError()


LocalConstrainedTranslations = create_deformation_module_with_backends(LocalConstrainedTranslations_Torch.build, LocalConstrainedTranslations_Torch.build)


def LocalScaling(dim, sigma, coeff=1., gd=None, tan=None, cotan=None, label=None, backend=None):
    def f_vectors(gd):
        return torch.tensor([[math.cos(2.*math.pi/3.*i), math.sin(2.*math.pi/3.*i)] for i in range(3)], device=gd.device, dtype=gd.dtype)

    def f_support(gd):
        return gd.repeat(3, 1) + sigma/3. * f_vectors(gd)

    return LocalConstrainedTranslations(dim, 1, sigma, "Local scaling", f_support, f_vectors, coeff=coeff, gd=gd, tan=tan, cotan=cotan, label=label, backend=backend)


def LocalRotation(dim, sigma, coeff=1., gd=None, tan=None, cotan=None, label=None, backend=None):
    def f_vectors(gd):
        return torch.tensor([[-math.sin(2.*math.pi/3.*i), math.cos(2.*math.pi/3.*i)] for i in range(3)], device=gd.device, dtype=gd.dtype)

    def f_support(gd):
        return gd.repeat(3, 1) + sigma/3. * torch.tensor([[math.cos(2.*math.pi/3.*i), math.sin(2.*math.pi/3.*i)] for i in range(3)], device=gd.device, dtype=gd.dtype)

    return LocalConstrainedTranslations(dim, 1, sigma, "Local rotation", f_support, f_vectors, coeff=coeff, gd=gd, tan=tan, cotan=cotan, label=label, backend=backend)

