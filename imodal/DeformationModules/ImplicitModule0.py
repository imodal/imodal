import torch

from pykeops.torch import Genred, KernelSolve

from imodal.DeformationModules.Abstract import DeformationModule, create_deformation_module_with_backends
from imodal.Kernels.kernels import K_xx
from imodal.Manifolds import Landmarks
from imodal.StructuredFields import StructuredField_0


class ImplicitModule0Base(DeformationModule):
    """Implicit module of order 0. Effectively identical to a local translation module with the added benefit of better numerical behaviour thanks to the `nu` parameters (explain)."""

    def __init__(self, manifold, sigma, nu, coeff, label):
        assert isinstance(manifold, Landmarks)
        super().__init__(label)
        self.__manifold = manifold
        self.__sigma = sigma
        self.__nu = nu
        self.__coeff = coeff
        self.__controls = torch.zeros_like(self.__manifold.gd, device=manifold.device)

    def __str__(self):
        outstr = "Implicit module of order 0\n"
        if self.label:
            outstr += "  Label=" + self.label + "\n"
        outstr += "  Sigma=" + str(self.sigma) + "\n"
        outstr += "  Nu=" + str(self.__nu) + "\n"
        outstr += "  Coeff=" + str(self.__coeff) + "\n"
        outstr += "  Nb pts=" + str(self.__manifold.nb_pts)
        return outstr

    @classmethod
    def build(cls, dim, nb_pts, sigma, nu=0., coeff=1., gd=None, tan=None, cotan=None, label=None):
        return cls(Landmarks(dim, nb_pts, gd=gd, tan=tan, cotan=cotan), sigma, nu, coeff, label)

    def to_(self, *args, **kwargs):
        self.__manifold.to_(*args, **kwargs)
        self.__controls = self.__controls.to(*args, **kwargs)

    @property
    def device(self):
        return self.__manifold.device

    @property
    def dim(self):
        return self.__manifold.dim

    @property
    def manifold(self):
        return self.__manifold

    @property
    def sigma(self):
        return self.__sigma

    @property
    def nu(self):
        return self.__nu

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
        self.__controls = torch.zeros_like(self.__manifold.gd, device=self.device)

    def __call__(self, points, k=0):
        return self.field_generator()(points, k)

    def cost(self):
        raise NotImplementedError

    def compute_geodesic_control(self, man):
        raise NotImplementedError

    def field_generator(self):
        return StructuredField_0(self.__manifold.gd, self.__controls, self.__sigma, device=self.device, backend=self.backend)

    def adjoint(self, manifold):
        return manifold.cot_to_vs(self.__sigma, backend=self.backend)


class ImplicitModule0_Torch(ImplicitModule0Base):
    def __init__(self, manifold, sigma, nu, coeff, label):
        super().__init__(manifold, sigma, nu, coeff, label)

    @property
    def backend(self):
        return 'torch'

    def cost(self):
        K_q = K_xx(self.manifold.gd, self.sigma) + self.nu * torch.eye(self.manifold.nb_pts, device=self.device)
        m = torch.mm(K_q , self.controls)
        return 0.5 * self.coeff * torch.dot(m.flatten(), self.controls.flatten())

    def compute_geodesic_control(self, man):
        vs = self.adjoint(man)
        K_q = K_xx(self.manifold.gd, self.sigma) + self.nu * torch.eye(self.manifold.nb_pts, device=self.device)
        controls, _ = torch.solve(vs(self.manifold.gd), K_q)
        self.controls = controls/self.coeff


class ImplicitModule0_KeOps(ImplicitModule0Base):
    def __init__(self, manifold, sigma, nu, coeff, label):
        super().__init__(manifold, sigma, nu, coeff, label)

        self.__keops_dtype = str(manifold.gd.dtype).split(".")[1]
        self.__keops_backend = 'CPU'
        if str(self.device) != 'cpu':
            self.__keops_backend = 'GPU'

        self.__keops_invsigmasq = torch.tensor([1./sigma/sigma], dtype=manifold.dtype, device=manifold.device)

        formula_cost = "(Exp(-S*SqNorm2(x - y)/IntCst(2))*px | py)"
        alias_cost = ["x=Vi({dim})".format(dim=self.dim),
                      "y=Vj({dim})".format(dim=self.dim),
                      "px=Vi({dim})".format(dim=self.dim),
                      "py=Vj({dim})".format(dim=self.dim),
                      "S=Pm(1)"]
        self.reduction_cost = Genred(formula_cost, alias_cost, reduction_op='Sum', axis=0, dtype=self.__keops_dtype)

        formula_cgc = "Exp(-S*SqNorm2(x - y)/IntCst(2))*X"
        alias_cgc = ["x=Vi({dim})".format(dim=self.dim),
                     "y=Vj({dim})".format(dim=self.dim),
                     "X=Vj({dim})".format(dim=self.dim),
                     "S=Pm(1)"]
        self.solve_cgc = KernelSolve(formula_cgc, alias_cgc, "X", axis=1, dtype=self.__keops_dtype)

    @property
    def backend(self):
        return 'keops'

    def to_(self, *args, **kwargs):
        super().to_(*args, **kwargs)
        self.__keops_invsigmasq = self.__keops_invsigmasq.to(*args, **kwargs)

        if 'device' in kwargs:
            if kwargs['device'].split(":")[0].lower() == "cuda":
                self.__keops_backend = 'GPU'
            elif kwargs['device'].split(":")[0].lower() == "cpu":
                self.__keops_backend = 'CPU'

    def cost(self):
        return (0.5 * self.coeff * self.reduction_cost(self.manifold.gd, self.manifold.gd, self.controls, self.controls, self.__keops_invsigmasq, backend=self.__keops_backend)).sum() + (self.nu*self.controls**2).sum()

    def compute_geodesic_control(self, man):
        vs = self.adjoint(man)(self.manifold.gd)
        self.fill_controls(self.solve_cgc(self.manifold.gd, self.manifold.gd, vs, self.__keops_invsigmasq, backend=self.__keops_backend, alpha=self.nu).reshape(self.manifold.nb_pts, self.dim)/self.coeff)


ImplicitModule0 = create_deformation_module_with_backends(ImplicitModule0_Torch.build, ImplicitModule0_KeOps.build)

