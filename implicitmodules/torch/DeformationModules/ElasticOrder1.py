import torch

from implicitmodules.torch.DeformationModules.Abstract import DeformationModule
from implicitmodules.torch.Kernels.SKS import eta, compute_sks
from implicitmodules.torch.Manifolds import Stiefel
from implicitmodules.torch.StructuredFields import StructuredField_p


class ImplicitModule1(DeformationModule):
    """Module generating sum of translations."""
    
    def __init__(self, manifold, C, sigma, nu, coeff=1.):
        assert isinstance(manifold, Stiefel)
        super().__init__()
        self.__manifold = manifold
        self.__C = C
        self.__sigma = sigma
        self.__nu = nu
        self.__coeff = coeff
        self.__dim_controls = C.shape[2]
        self.__controls = torch.zeros(self.__dim_controls, device=self.__manifold.device)

    @classmethod
    def build_and_fill(cls, dim, nb_pts, C, sigma, nu, gd=None, tan=None, cotan=None, coeff=1.):
        """Builds the Translations deformation module from tensors."""
        return cls(Stiefel(dim, nb_pts, gd=gd, tan=tan, cotan=cotan), C, sigma, nu, coeff)

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
    def C(self):
        return self.__C

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
        self.fill_controls(torch.zeros(self.__dim_controls, device=self.device))

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
        S = torch.tensordot(S, eta(device=self.device), dims=2)

        self.__compute_sks()

        tlambdas, _ = torch.solve(S.view(-1, 1), self.__coeff * self.__sks)

        (aq, aqkiaq) = self.__compute_aqkiaq()
        c, _ = torch.solve(torch.mm(aq.t(), tlambdas), aqkiaq)
        self.__controls = c.reshape(-1)
        self.__compute_moments()

    def field_generator(self):
        return StructuredField_p(self.__manifold.gd[0].view(-1, self.__manifold.dim),
                                 self.__moments, self.__sigma)

    def adjoint(self, manifold):
        return manifold.cot_to_vs(self.__sigma)

    def __compute_aqh(self, h):
        R = self.__manifold.gd[1].view(-1, 2, 2)

        return torch.einsum('nli, nik, k, nui, niv, lvt->nt', R, self.__C, h, torch.eye(self.__manifold.dim, device=self.device).repeat(self.__manifold.nb_pts, 1, 1), torch.transpose(R, 1, 2), eta(device=self.device))

    def __compute_sks(self):
        self.__sks = compute_sks(self.__manifold.gd[0].view(-1, self.__manifold.dim), self.sigma, 1) + self.__nu * torch.eye(3 * self.__manifold.nb_pts, device=self.device)

    def __compute_moments(self):
        self.__aqh = self.__compute_aqh(self.__controls)
        lambdas, _ = torch.solve(self.__aqh.view(-1, 1), self.__sks)
        self.__lambdas = lambdas.contiguous()
        self.__moments = torch.tensordot(self.__lambdas.view(-1, 3), torch.transpose(eta(device=self.device), 0, 2), dims=1)

    def __compute_aqkiaq(self):
        lambdas = torch.zeros(self.__dim_controls, 3 * self.__manifold.nb_pts, device=self.device)
        aq = torch.zeros(3 * self.__manifold.nb_pts, self.__dim_controls, device=self.device)
        for i in range(self.__dim_controls):
            h = torch.zeros(self.__dim_controls, device=self.device)
            h[i] = 1.
            aqi = self.__compute_aqh(h).view(-1)
            aq[:, i] = aqi
            l, _ = torch.solve(aqi.view(-1, 1), self.__sks)
            lambdas[i, :] = l.view(-1)

        return (aq, torch.mm(lambdas, aq))

