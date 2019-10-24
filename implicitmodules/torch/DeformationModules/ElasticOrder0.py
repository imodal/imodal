import torch

from pykeops.torch import Genred, KernelSolve

from implicitmodules.torch.DeformationModules.Abstract import DeformationModule, register_deformation_module_builder
from implicitmodules.torch.Kernels.kernels import K_xx
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.StructuredFields import StructuredField_0
from implicitmodules.torch.Utilities import get_compute_backend


class ImplicitModule0(DeformationModule):
    """Module generating sum of translations."""

    def __init__(self, manifold, sigma, nu, coeff=1.):
        assert isinstance(manifold, Landmarks)
        super().__init__()
        self.__manifold = manifold
        self.__sigma = sigma
        self.__nu = nu
        self.__coeff = coeff
        self.__dim_controls = self.__manifold.dim * self.__manifold.nb_pts
        self.__controls = torch.zeros(self.__dim_controls, device=manifold.device)

    # TODO: remove deprecated method name
    @classmethod
    def build_from_points(cls, dim, nb_pts, sigma, nu=0., coeff=1., gd=None, tan=None, cotan=None):
        """Builds the Translations deformation module from tensors."""
        return cls(Landmarks(dim, nb_pts, gd=gd, tan=tan, cotan=cotan), sigma, nu, coeff)

    @classmethod
    def build(cls, dim, nb_pts, sigma, nu=0., coeff=1., gd=None, tan=None, cotan=None):
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
        raise NotImplementedError

    def compute_geodesic_control(self, man):
        """Computes geodesic control from \delta \in H^\ast."""
        raise NotImplementedError

    def field_generator(self):
        return StructuredField_0(self.__manifold.gd.view(-1, self.__manifold.dim), self.__controls.view(-1, self.__manifold.dim), self.__sigma, device=self.device, backend=self.backend)

    def adjoint(self, manifold):
        return manifold.cot_to_vs(self.__sigma, backend=self.backend)


class ImplicitModule0_Torch(ImplicitModule0):
    def __init__(self, manifold, sigma, nu, coeff=1.):
        super().__init__(manifold, sigma, nu, coeff=coeff)

    @property
    def backend(self):
        return 'torch'

    def cost(self):
        K_q = K_xx(self.manifold.gd.view(-1, self.manifold.dim), self.sigma) + self.nu * torch.eye(self.manifold.nb_pts, device=self.device)

        m = torch.mm(K_q , self.controls.view(-1, self.manifold.dim))
        return 0.5 * self.coeff * torch.dot(m.view(-1), self.controls.view(-1))

    def compute_geodesic_control(self, man):
        vs = self.adjoint(man)
        K_q = K_xx(self.manifold.gd.view(-1, self.manifold.dim), self.sigma) + self.nu * torch.eye(self.manifold.nb_pts, device=self.device)

        controls, _ = torch.solve(vs(self.manifold.gd.view(-1, self.manifold.dim)), K_q)
        self.controls = controls.contiguous().view(-1) / self.coeff

class ImplicitModule0_KeOps(ImplicitModule0):
    def __init__(self, manifold, sigma, nu, coeff=1.):
        super().__init__(manifold, sigma, nu, coeff=1.)

        self.__keops_dtype = str(manifold.gd.dtype).split(".")[1]
        self.__keops_backend = 'CPU'
        if str(self.device) != 'cpu':
            self.__keops_backend = 'GPU'

        self.__keops_invsigmasq = torch.tensor([1/sigma**2], dtype=manifold.gd.dtype, device=manifold.gd.device)

        self.dim = manifold.dim
        formula_cost = "(Exp(-S*SqNorm2(x - y)/IntCst(2))*px | py)"
        alias_cost = ["x=Vi("+str(self.dim)+")", "y=Vj("+str(self.dim)+")", "px=Vi(" + str(self.dim)+")", "py=Vj("+str(self.dim)+")", "S=Pm(1)"]
        self.reduction_cost = Genred(formula_cost, alias_cost, reduction_op='Sum', axis=0, dtype=self.__keops_dtype)

        formula_cgc = "Exp(-S*SqNorm2(x - y)/IntCst(2))*X"
        alias_cgc = ["x=Vi("+str(self.dim)+")", "y=Vj("+str(self.dim)+")", "X=Vj("+str(self.dim) + ")", "S=Pm(1)"]
        self.solve_cgc = KernelSolve(formula_cgc, alias_cgc, "X", axis=1, dtype=self.__keops_dtype)

    @property
    def backend(self):
        return 'keops'

    def cost(self):
        return 0.5 * self.coeff * (1. * self.reduction_cost(self.manifold.gd.reshape(-1, self.dim), self.manifold.gd.reshape(-1, self.dim), self.controls.reshape(-1, self.dim), self.controls.reshape(-1, self.dim), self.__keops_invsigmasq, backend=self.__keops_backend)).sum() + (self.nu*self.controls**2).sum()

    def compute_geodesic_control(self, man):
        vs = self.adjoint(man)(self.manifold.gd.view(-1, self.dim))
        self.fill_controls(self.solve_cgc(self.manifold.gd.reshape(-1, self.dim), self.manifold.gd.reshape(-1, self.dim), vs.reshape(-1, self.dim), self.__keops_invsigmasq, backend=self.__keops_backend, alpha=self.nu).flatten()/self.coeff)


register_deformation_module_builder('implicit_order_0', {'torch': ImplicitModule0_Torch.build, 'keops': ImplicitModule0_KeOps.build})

