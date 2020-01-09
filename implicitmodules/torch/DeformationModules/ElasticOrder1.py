import torch
import math

from pykeops.torch import Genred, KernelSolve

from implicitmodules.torch.DeformationModules.Abstract import DeformationModule, create_deformation_module_with_backends
from implicitmodules.torch.Kernels.SKS import eta, compute_sks, A
from implicitmodules.torch.Manifolds import Stiefel
from implicitmodules.torch.StructuredFields import StructuredField_p


class ImplicitModule1Base(DeformationModule):
    """Module generating sum of translations."""

    def __init__(self, manifold, sigma, C, nu, coeff, label):
        assert isinstance(manifold, Stiefel)
        super().__init__(label)
        self.__manifold = manifold
        self.__C = C
        self.__sigma = sigma
        self.__nu = nu
        self.__coeff = coeff
        self.__dim_controls = C.shape[2]        
        self.__sym_dim = int(self.manifold.dim * (self.manifold.dim + 1) / 2)
        self.__controls = torch.zeros(self.__dim_controls, device=self.__manifold.device)

    @classmethod
    def build(cls, dim, nb_pts, sigma, C, nu=0., coeff=1., gd=None, tan=None, cotan=None, label=None):
        """Builds the Translations deformation module from tensors."""
        return cls(Stiefel(dim, nb_pts, gd=gd, tan=tan, cotan=cotan), sigma, C, nu, coeff, label)

    @property
    def dim(self):
        return self.__manifold.dim

    def to(self, device):
        self.__manifold = self.__manifold.to(device)
        self.__controls = self.__controls.to(device)

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
    def sigma(self):
        return self.__sigma

    @property
    def nu(self):
        return self.__nu

    @property
    def sym_dim(self):
        return self.__sym_dim

    @property
    def dim_controls(self):
        return self.__dim_controls

    def __get_controls(self):
        return self.__controls

    def fill_controls(self, controls):
        self.__controls = controls
        self.compute_moments()

    def __get_coeff(self):
        return self.__coeff

    def __set_coeff(self, coeff):
        self.__coeff = coeff

    controls = property(__get_controls, fill_controls)
    coeff = property(__get_coeff, __set_coeff)

    def fill_controls_zero(self):
        self.fill_controls(torch.zeros(self.__dim_controls, device=self.device, requires_grad=True))

    def __call__(self, points, k=0):
        """Applies the generated vector field on given points."""
        return self.field_generator()(points, k)

    def cost(self):
        raise NotImplementedError()

    def compute_geodesic_control(self, man):
        """Computes geodesic control from vs structuredfield."""
        raise NotImplementedError()

    def compute_moments(self):
        raise NotImplementedError()

    def field_generator(self):
        return StructuredField_p(self.__manifold.gd[0],
                                 self.moments, self.__sigma, backend=self.backend)

    def adjoint(self, manifold):
        return manifold.cot_to_vs(self.__sigma, backend=self.backend)


class ImplicitModule1_Torch(ImplicitModule1Base):
    def __init__(self, manifold, sigma, C, nu, coeff, label):
        super().__init__(manifold, sigma, C, nu, coeff, label)

    @property
    def backend(self):
        return 'torch'

    def cost(self):
        return 0.5 * self.coeff * torch.dot(self.__aqh.view(-1), self.__lambdas.view(-1))

    def compute_geodesic_control(self, man):
        vs = self.adjoint(man)
        d_vx = vs(self.manifold.gd[0], k=1)

        S = 0.5 * (d_vx + torch.transpose(d_vx, 1, 2))
        S = torch.tensordot(S, eta(self.manifold.dim, device=self.device), dims=2)

        self.__compute_sks()

        tlambdas, _ = torch.solve(S.view(-1, 1), self.coeff * self.__sks)

        (aq, aqkiaq) = self.__compute_aqkiaq()
        c, _ = torch.solve(torch.mm(aq.t(), tlambdas), aqkiaq)
        self.controls = c.reshape(-1)
        self.__compute_moments()

    def compute_moments(self):
        self.__compute_sks()
        self.__compute_moments()

    def __compute_aqh(self, h):
        R = self.manifold.gd[1].view(-1, self.manifold.dim, self.manifold.dim)

        return torch.einsum('nli, nik, k, nui, niv, lvt->nt', R, self.C, h, torch.eye(self.manifold.dim, device=self.device).repeat(self.manifold.nb_pts, 1, 1), torch.transpose(R, 1, 2), eta(self.dim, device=self.device))

    def __compute_sks(self):
        self.__sks = compute_sks(self.manifold.gd[0].view(-1, self.manifold.dim), self.sigma, 1) + self.nu * torch.eye(self.sym_dim * self.manifold.nb_pts, device=self.device)

    def __compute_moments(self):
        self.__aqh = self.__compute_aqh(self.controls)
        lambdas, _ = torch.solve(self.__aqh.view(-1, 1), self.__sks)
        self.__lambdas = lambdas.contiguous()
        self.moments = torch.tensordot(self.__lambdas.view(-1, self.sym_dim), torch.transpose(eta(self.manifold.dim, device=self.device), 0, 2), dims=1)

    def __compute_aqkiaq(self):
        lambdas = torch.zeros(self.dim_controls, self.sym_dim * self.manifold.nb_pts, device=self.device)
        aq = torch.zeros(self.sym_dim * self.manifold.nb_pts, self.dim_controls, device=self.device)
        for i in range(self.dim_controls):
            h = torch.zeros(self.dim_controls, device=self.device)
            h[i] = 1.
            aqi = self.__compute_aqh(h).flatten()
            aq[:, i] = aqi
            l, _ = torch.solve(aqi.view(-1, 1), self.__sks)
            lambdas[i, :] = l.flatten()

        return (aq, torch.mm(lambdas, aq))


class ImplicitModule1_KeOps(ImplicitModule1Base):
    def __init__(self, manifold, sigma, C, nu, coeff, label):
        super().__init__(manifold, sigma, C, nu, coeff, label)

        self.__keops_dtype = str(manifold.gd[0].dtype).split(".")[1]
        self.__keops_backend = 'CPU'
        if str(self.device) != 'cpu':
            self.__keops_backend = 'GPU'

        self.__keops_invsigmasq = torch.tensor([1./sigma/sigma], dtype=manifold.dtype, device=self.device)
        self.__keops_eye = torch.eye(self.dim, device=self.device, dtype=manifold.dtype).flatten()

        self.__keops_A = A(self.dim, device=self.device, dtype=manifold.dtype).flatten()

        formula_solve_sks = "TensorDot(TensorDot((-S*Exp(-S*SqNorm2(x_i - y_j)*IntInv(2))*(S*TensorDot(x_i - y_j, x_i - y_j, Ind({dim}), Ind({dim}), Ind(), Ind()) - eye)), A, Ind({dim}, {dim}), Ind({dim}, {dim}, {symdim}, {symdim}), Ind(0, 1), Ind(0, 1)), X, Ind({symdim}, {symdim}), Ind({symdim}), Ind(0), Ind(0))".format(dim=self.dim, symdim=self.sym_dim)

        alias_solve_sks = ["x_i=Vi({dim})".format(dim=self.dim),
                           "y_j=Vj({dim})".format(dim=self.dim),
                           "X=Vj({symdim})".format(symdim=self.sym_dim),
                           "eye=Pm({dimsq})".format(dimsq=self.dim*self.dim),
                           "S=Pm(1)",
                           "A=Pm({dima})".format(dima=self.__keops_A.numel())]

        self.solve_sks = KernelSolve(formula_solve_sks, alias_solve_sks, "X", axis=1, dtype=self.__keops_dtype)

    @property
    def backend(self):
        return 'keops'

    def cost(self):
        return 0.5 * self.coeff * torch.dot(self.__aqh.view(-1), self.__lambdas.view(-1))

    def compute_geodesic_control(self, man):
        vs = self.adjoint(man)
        d_vx = vs(self.manifold.gd[0].view(-1, self.manifold.dim), k=1)

        S = 0.5 * (d_vx + torch.transpose(d_vx, 1, 2))
        S = torch.tensordot(S, eta(self.manifold.dim, device=self.device), dims=2)

        tlambdas = self.solve_sks(self.manifold.gd[0].reshape(-1, self.dim), self.manifold.gd[0].reshape(-1, self.dim), self.coeff * S, self.__keops_eye, self.__keops_invsigmasq, self.__keops_A, backend=self.__keops_backend, alpha=self.nu)

        (aq, aqkiaq) = self.__compute_aqkiaq()

        c, _ = torch.solve(torch.mm(aq.t(), tlambdas.view(-1, 1)), aqkiaq)

        self.controls = c.flatten()
        self.__compute_moments()

    def compute_moments(self):
        self.__compute_moments()

    def __compute_aqh(self, h):
        R = self.manifold.gd[1]

        return torch.einsum('nli, nik, k, nui, niv, lvt->nt', R, self.C, h, torch.eye(self.manifold.dim, device=self.device).repeat(self.manifold.nb_pts, 1, 1), torch.transpose(R, 1, 2), eta(self.manifold.dim, device=self.device))

    def __compute_moments(self):
        self.__aqh = self.__compute_aqh(self.controls)
        self.__lambdas = self.solve_sks(self.manifold.gd[0].reshape(-1, self.dim), self.manifold.gd[0].reshape(-1, self.dim), self.__aqh, self.__keops_eye, self.__keops_invsigmasq, self.__keops_A, backend=self.__keops_backend, alpha=self.nu)
        self.moments = torch.tensordot(self.__lambdas.view(-1, self.sym_dim), torch.transpose(eta(self.manifold.dim, device=self.device), 0, 2), dims=1)

    def __compute_aqkiaq(self):
        lambdas = torch.zeros(self.dim_controls, self.sym_dim * self.manifold.nb_pts, device=self.device)
        aq = torch.zeros(self.sym_dim * self.manifold.nb_pts, self.dim_controls, device=self.device)
        for i in range(self.dim_controls):
            h = torch.zeros(self.dim_controls, device=self.device)
            h[i] = 1.
            aqi = self.__compute_aqh(h).flatten()
            aq[:, i] = aqi

            lambdas[i, :] = self.solve_sks(self.manifold.gd[0], self.manifold.gd[0], aqi.view(-1, self.sym_dim), self.__keops_eye, self.__keops_invsigmasq, self.__keops_A, backend=self.__keops_backend, alpha=self.nu).view(-1)

        return (aq, torch.mm(lambdas, aq))


ImplicitModule1 = create_deformation_module_with_backends(ImplicitModule1_Torch.build, ImplicitModule1_KeOps.build)

