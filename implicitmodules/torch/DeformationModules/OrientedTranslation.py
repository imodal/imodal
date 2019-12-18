import torch

from pykeops.torch import Genred, KernelSolve

from implicitmodules.torch.DeformationModules.Abstract import DeformationModule, create_deformation_module_with_backends
from implicitmodules.torch.Kernels.kernels import K_xy, K_xx
from implicitmodules.torch.Manifolds import LandmarksDirection
from implicitmodules.torch.StructuredFields import StructuredField_0


class OrientedTranslationsBase(DeformationModule):
    """Module generating sum of translations."""
    
    def __init__(self, manifold, sigma, nu, label):
        assert isinstance(manifold, LandmarksDirection)
        super().__init__(label)
        self.__manifold = manifold
        self.__sigma = sigma
        self.__controls = torch.zeros(self.__manifold.nb_pts, requires_grad=True)
        self.__nu = nu

    @classmethod
    def build(cls, dim, nb_pts, sigma, transport='vector', nu=0., gd=None, tan=None, cotan=None, label=None):
        """Builds the Translations deformation module from tensors."""
        return cls(LandmarksDirection(dim, nb_pts, transport, gd=gd, tan=tan, cotan=cotan), sigma, nu, label)

    def to_(self, device):
        self.__manifold.to_(device)
        self.__controls = self.__controls.to(device)

    @property
    def nu(self):
        return self.__nu

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
        self.__controls = torch.zeros(self.__manifold.nb_pts, requires_grad=True)

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
    def __init__(self, manifold, sigma, nu, label):
        super().__init__(manifold, sigma, nu, label)

    @property
    def backend(self):
        return 'torch'

    def cost(self):
        K_q = K_xx(self.manifold.gd[0], self.sigma)

        m = torch.mm(K_q, self.controls.unsqueeze(1).repeat(1, self.dim)*self.manifold.gd[1])
        return 0.5 * torch.dot(m.flatten(), (self.controls.unsqueeze(1).repeat(1, self.dim)*self.manifold.gd[1]).flatten())

    def compute_geodesic_control(self, man):
        """Computes geodesic control from StructuredField vs."""
        vs = self.adjoint(man)
        Z = K_xx(self.manifold.gd[0], self.sigma) * torch.mm(self.manifold.gd[1], self.manifold.gd[1].T) + self.nu * torch.eye(self.manifold.nb_pts, device=self.device)

        controls, _ = torch.solve(torch.einsum('ni, ni->n', vs(self.manifold.gd[0]), self.manifold.gd[1]).unsqueeze(1), Z)

        self.controls = controls.flatten().contiguous()


class OrientedTranslations_KeOps(OrientedTranslationsBase):
    def __init__(self, manifold, nu, sigma):
        super().__init__(manifold, nu, sigma)

        self.__keops_dtype = str(manifold.gd.dtype).split(".")[1]
        self.__keops_backend = 'CPU'
        if str(self.device) != 'cpu':
            self.__keops_backend = 'GPU'

        self.__keops_invsigmasq = torch.tensor([1./sigma/sigma], dtype=manifold.gd.dtype, device=manifold.device)

        formula_cost = "(Exp(-S*SqNorm2(x - y)/IntCst(2))*px | py)/IntCst(2)"
        alias_cost = ["x=Vi("+str(self.dim)+")", "y=Vj("+str(self.dim)+")", "px=Vi(" + str(self.dim)+")", "py=Vj("+str(self.dim)+")", "S=Pm(1)"]
        self.reduction_cost = Genred(formula_cost, alias_cost, reduction_op='Sum', axis=0, dtype=self.__keops_dtype)

        formula_cgc = "Exp(-S*SqNorm2(x - y)/IntCst(2))*X"
        alias_cgc = ["x=Vi("+str(self.dim)+")", "y=Vj("+str(self.dim)+")", "X=Vj("+str(self.dim) + ")", "S=Pm(1)"]
        self.solve_cgc = KernelSolve(formula_cgc, alias_cgc, "X", axis=1, dtype=self.__keops_dtype)

    @property
    def backend(self):
        return 'keops'

    def cost(self):
        return (1.*self.reduction_cost(self.manifold.gd, self.manifold.gd, self.controls, self.controls, self.__keops_invsigmasq, backend=self.__keops_backend)).sum()

    def compute_geodesic_control(self, man):
        vs = self.adjoint(man)(self.manifold.gd)
        self.fill_controls(self.solve_cgc(self.manifold.gd, self.manifold.gd, vs, self.__keops_invsigmasq, backend=self.__keops_backend, alpha=0.))


OrientedTranslations = create_deformation_module_with_backends(OrientedTranslations_Torch.build, OrientedTranslations_KeOps.build)

