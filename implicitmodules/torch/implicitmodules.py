import torch

from implicitmodules.torch.Manifolds import Stiefel, Landmarks
from implicitmodules.torch.structuredfield import StructuredField_0, StructuredField_p
from .deformationmodules import DeformationModule
from .kernels import K_xx, K_xy, compute_sks, eta


class ImplicitModule0(DeformationModule):
    """Module generating sum of translations."""
    def __init__(self, manifold, sigma, nu, coeff=1.):
        assert isinstance(manifold, Landmarks)
        super().__init__()
        self.__manifold = manifold
        self.__sigma = sigma
        self.__nu = nu
        self.__coeff = coeff
        self.__dim_controls = self.__manifold.dim*self.__manifold.nb_pts
        self.__controls = torch.zeros(self.__dim_controls)

    @classmethod
    def build_and_fill(cls, dim, nb_pts, sigma, nu, coeff=1., gd=None, tan=None, cotan=None):
        """Builds the Translations deformation module from tensors."""
        return cls(Landmarks(dim, nb_pts, gd=gd, tan=tan, cotan=cotan), sigma, nu, coeff)

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
    def coeff(self):
        return self.__coeff

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
        K_q = K_xy(points, self.manifold.gd.view(-1, self.__manifold.dim), self.__sigma)
        return torch.mm(K_q, self.__controls.view(-1, self.__manifold.dim))

    def cost(self):
        """Returns the cost."""
        K_q = K_xx(self.manifold.gd.view(-1, self.__manifold.dim), self.__sigma)
        m = torch.mm(K_q + self.__nu*torch.eye(self.__manifold.nb_pts), self.__controls.view(-1, self.__manifold.dim))
        return 0.5 * self.__coeff * torch.dot(m.view(-1), self.__controls.view(-1))

    def compute_geodesic_control(self, man):
        """Computes geodesic control from \delta \in H^\ast."""
        vs = self.adjoint(man)
        K_q = K_xx(self.manifold.gd.view(-1, self.__manifold.dim), self.__sigma) + self.__nu*torch.eye(self.__manifold.nb_pts)
        controls, _ = torch.gesv(vs(self.manifold.gd.view(-1, self.manifold.dim)), K_q)
        self.__controls = controls.contiguous().view(-1) / self.__coeff

    def field_generator(self):
        return StructuredField_0(self.__manifold.gd.view(-1, self.__manifold.dim),
                                 self.__controls.view(-1, self.__manifold.dim), self.__sigma)

    def adjoint(self, manifold):
        return manifold.cot_to_vs(self.__sigma)


class ImplicitModule1(DeformationModule):
    """Module generating sum of translations."""
    def __init__(self, manifold, C, sigma, nu, coeff=1.):
        assert isinstance(manifold, Stiefel)
        super().__init__()
        self.__manifold = manifold
        self.__C = C.clone()
        self.__sigma = sigma
        self.__nu = nu
        self.__coeff = coeff
        self.__dim_controls = C.shape[2]
        self.__controls = torch.zeros(self.__dim_controls)

    @classmethod
    def build_and_fill(cls, dim, nb_pts, C, sigma, nu, gd=None, tan=None, cotan=None, coeff=1.):
        """Builds the Translations deformation module from tensors."""
        return cls(Stiefel(dim, nb_pts, gd=gd, tan=tan, cotan=cotan), C, sigma, nu, coeff)

    @property
    def manifold(self):
        return self.__manifold

    @property
    def coeff(self):
        return self.__coeff

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
        self.__compute_sks()
        self.__compute_moments()

    controls = property(__get_controls, fill_controls)

    def fill_controls_zero(self):
        self.fill_controls(torch.zeros(self.__dim_controls))

    def __call__(self, points, k=0):
        """Applies the generated vector field on given points."""
        return self.field_generator()(points, k)

    def cost(self):
        """Returns the cost."""
        return 0.5 * self.coeff * torch.dot(self.__aqh.view(-1), self.__lambdas.view(-1))

    def compute_geodesic_control(self, man):
        """Computes geodesic control from vs structuredfield."""
        vs = self.adjoint(man)
        d_vx = vs(self.__manifold.gd[0].view(-1, self.__manifold.dim), k=1)

        S = 0.5 * (d_vx + torch.transpose(d_vx, 1, 2))
        S = torch.tensordot(S, eta(), dims=2)

        self.__compute_sks()

        tlambdas, _ = torch.gesv(S.view(-1, 1), self.__coeff * self.__sks)

        (aq, aqkiaq) = self.__compute_aqkiaq()
        c, _ = torch.gesv(torch.mm(aq.t(), tlambdas), aqkiaq)
        self.__controls = c.reshape(-1)
        self.__compute_moments()

    def field_generator(self):
        return StructuredField_p(self.__manifold.gd[0].view(-1, self.__manifold.dim),
                                 self.__moments, self.__sigma)

    def adjoint(self, manifold):
        return manifold.cot_to_vs(self.__sigma)

    def __compute_aqh(self, h):
        R = self.__manifold.gd[1].view(-1, 2, 2)

        return torch.einsum('nli, nik, k, nui, niv, lvt->nt', R, self.__C, h, torch.eye(self.__manifold.dim).repeat(self.__manifold.nb_pts, 1, 1), torch.transpose(R, 1, 2), eta())

    def __compute_sks(self):
        self.__sks = compute_sks(self.__manifold.gd[0].view(-1, self.__manifold.dim), self.sigma, 1) + self.__nu * torch.eye(3 * self.__manifold.nb_pts)

    def __compute_moments(self):
        self.__aqh = self.__compute_aqh(self.__controls)
        lambdas, _ = torch.gesv(self.__aqh.view(-1, 1), self.__sks)
        self.__lambdas = lambdas.contiguous()
        self.__moments = torch.tensordot(self.__lambdas.view(-1, 3), torch.transpose(eta(), 0, 2), dims=1)

    def __compute_aqkiaq(self):
        lambdas = torch.zeros(self.__dim_controls, 3 * self.__manifold.nb_pts)
        aq = torch.zeros(3 * self.__manifold.nb_pts, self.__dim_controls)
        for i in range(self.__dim_controls):
            h = torch.zeros(self.__dim_controls)
            h[i] = 1.
            aqi = self.__compute_aqh(h).view(-1)
            aq[:, i] = aqi
            l, _ = torch.gesv(aqi.view(-1, 1), self.__sks)
            lambdas[i, :] = l.view(-1)

        return (aq, torch.mm(lambdas, aq))

