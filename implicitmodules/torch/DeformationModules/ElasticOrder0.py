import torch

from implicitmodules.torch.DeformationModules.Abstract import DeformationModule
from implicitmodules.torch.Kernels.kernels import K_xx
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.StructuredFields import StructuredField_0
from implicitmodules.torch.Utilities import get_compute_backend


class ImplicitModule0(DeformationModule):
    """Module generating sum of translations."""

    def __init__(self, manifold, sigma, nu, coeff=1., backend=None):
        assert isinstance(manifold, Landmarks)
        super().__init__()
        self.__manifold = manifold
        self.__sigma = sigma
        self.__nu = nu
        self.__coeff = coeff
        self.__dim_controls = self.__manifold.dim * self.__manifold.nb_pts
        self.__controls = torch.zeros(self.__dim_controls, device=manifold.device)

        if backend is None:
            backend = get_compute_backend()

        if backend == 'torch':
            self.__backend = 'torch'
            self.__cost = self.__cost_torch
        elif backend == 'keops':
            self.__backend = 'keops'
            self.__cost = self.__cost_keops
        else:
            raise RuntimeError("ImplicitModule0.__init__(): unrecognised compute backend " + backend)

    @classmethod
    def build_from_points(cls, dim, nb_pts, sigma, nu, coeff=1., gd=None, tan=None, cotan=None):
        """Builds the Translations deformation module from tensors."""
        return cls(Landmarks(dim, nb_pts, gd=gd, tan=tan, cotan=cotan), sigma, nu, coeff)

    def to(self, device):
        self.__manifold.to(device)
        self.__controls.to(device)

    @property
    def device(self):
        return self.__manifold.device

    @property
    def manifold(self):
        return self.__manifold

    @property
    def sigma(self):
        return self.__sigma

    @property
    def nu(self):
        return self.__nu

    @property
    def dim_controls(self):
        return self.__dim_controls

    def __get_controls(self):
        return self.__controls

    def fill_controls(self, controls):
        self.__controls = controls

    def __get_coeff(self):
        return self.__coeff

    def __set_coeff(self, coeff):
        self.__coeff = coeff

    coeff = property(__get_coeff, __set_coeff)
    controls = property(__get_controls, fill_controls)

    def fill_controls_zero(self):
        self.__controls = torch.zeros(self.__dim_controls)

    def __call__(self, points, k=0):
        """Applies the generated vector field on given points."""
        return self.field_generator()(points, k)

    def cost(self):
        """Returns the cost."""
        return self.__cost()

    def __cost_torch(self):
        K_q = K_xx(self.manifold.gd.view(-1, self.__manifold.dim), self.__sigma) + self.__nu * torch.eye(self.__manifold.nb_pts, device=self.device)
        m = torch.mm(K_q , self.__controls.view(-1, self.__manifold.dim))
        return 0.5 * self.__coeff * torch.dot(m.view(-1), self.__controls.view(-1))

    def __cost_keops(self):
        raise NotImplementedError()

    def compute_geodesic_control(self, man):
        """Computes geodesic control from \delta \in H^\ast."""
        vs = self.adjoint(man)
        K_q = K_xx(self.manifold.gd.view(-1, self.__manifold.dim), self.__sigma) + self.__nu * torch.eye(self.__manifold.nb_pts, device=self.device)

        controls, _ = torch.solve(vs(self.manifold.gd.view(-1, self.manifold.dim)), K_q)
        self.__controls = controls.contiguous().view(-1) / self.__coeff

    def field_generator(self):
        return StructuredField_0(self.__manifold.gd.view(-1, self.__manifold.dim),
                                 self.__controls.view(-1, self.__manifold.dim), self.__sigma, device=self.device, backend=self.__backend)

    def adjoint(self, manifold):
        return manifold.cot_to_vs(self.__sigma)

