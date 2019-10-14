import torch

from pykeops.torch import Genred, KernelSolve

from implicitmodules.torch.DeformationModules.Abstract import DeformationModule, register_deformation_module_builder
from implicitmodules.torch.Kernels.kernels import K_xy, K_xx
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.StructuredFields import StructuredField_0


class Translations(DeformationModule):
    """Module generating sum of translations."""
    
    def __init__(self, manifold, sigma):
        assert isinstance(manifold, Landmarks)
        super().__init__()
        self.__manifold = manifold
        self.__sigma = sigma
        self.__dim_controls = self.__manifold.dim * self.__manifold.nb_pts
        self.__controls = torch.zeros(self.__dim_controls, requires_grad=True)

    # TODO: remove deprecated method name
    @classmethod
    def build_from_points(cls, dim, nb_pts, sigma, gd=None, tan=None, cotan=None):
        """Builds the Translations deformation module from tensors."""
        return cls(Landmarks(dim, nb_pts, gd=gd, tan=tan, cotan=cotan), sigma)

    @classmethod
    def build(cls, dim, nb_pts, sigma, gd=None, tan=None, cotan=None):
        """Builds the Translations deformation module from tensors."""
        return cls(Landmarks(dim, nb_pts, gd=gd, tan=tan, cotan=cotan), sigma)

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
    def dim_controls(self):
        return self.__dim_controls

    def __get_controls(self):
        return self.__controls

    def fill_controls(self, controls):
        self.__controls = controls

    controls = property(__get_controls, fill_controls)

    def fill_controls_zero(self):
        self.__controls = torch.zeros(self.__dim_controls)

    def __call__(self, points, k=0):
        """Applies the generated vector field on given points."""
        return self.field_generator()(points, k)

    def cost(self):
        raise NotImplementedError
        """Returns the cost."""

    def compute_geodesic_control(self, man):
        raise NotImplementedError

    def field_generator(self):
        return StructuredField_0(self.__manifold.gd.view(-1, self.__manifold.dim), self.__controls.view(-1, self.__manifold.dim), self.__sigma, device=self.device, backend=self.backend)

    def adjoint(self, manifold):
        return manifold.cot_to_vs(self.__sigma, backend=self.backend)


class Translations_Torch(Translations):
    def __init__(self, manifold, sigma):
        super().__init__(manifold, sigma)

    @property
    def backend(self):
        return 'torch'

    def cost(self):
        K_q = K_xx(self.manifold.gd.view(-1, self.manifold.dim), self.sigma)

        m = torch.mm(K_q, self.controls.view(-1, self.manifold.dim))
        return 0.5 * torch.dot(m.view(-1), self.controls.view(-1))

    def compute_geodesic_control(self, man):
        """Computes geodesic control from StructuredField vs."""
        vs = self.adjoint(man)
        K_q = K_xx(self.manifold.gd.view(-1, self.manifold.dim), self.sigma)

        controls, _ = torch.solve(vs(self.manifold.gd.view(-1, self.manifold.dim)), K_q)
        self.controls = controls.contiguous().view(-1)


class Translations_KeOps(Translations):
    def __init__(self, manifold, sigma):
        super().__init__(manifold, sigma)

        self.__keops_dtype = str(manifold.gd.dtype).split(".")[1]
        self.__keops_backend = 'CPU'
        if str(self.device) != 'cpu':
            self.__keops_backend = 'GPU'

        self.__keops_sigma = torch.tensor([sigma], dtype=manifold.gd.dtype, device=manifold.gd.device)

        self.dim = manifold.dim
        formula_cost = "(Exp(-S*SqNorm2(x - y)/IntCst(2))*p | p)/IntCst(2)"
        alias_cost = ["x=Vi("+str(self.dim)+")", "y=Vj("+str(self.dim)+")", "p=Vj("+str(self.dim)+")", "S=Pm(1)"]
        self.reduction_cost = Genred(formula_cost, alias_cost, reduction_op='Sum', axis=0, dtype=self.__keops_dtype)

        formula_cgc = "Exp(-S*SqNorm2(x - y)/IntCst(2))*vs"
        alias_cgc = ["x=Vi("+str(self.dim)+")", "y=Vj("+str(self.dim)+")", "vs=Vj("+str(self.dim) + ")", "S=Pm(1)"]
        self.solve_cgc = KernelSolve(formula_cgc, alias_cgc, "vs", axis=1, dtype=self.__keops_dtype)

    @property
    def backend(self):
        return 'keops'

    def cost(self):
        return self.reduction_cost(self.manifold.gd.reshape(-1, self.dim), self.manifold.gd.reshape(-1, self.dim), self.controls.reshape(-1, self.dim), self.__keops_sigma, backend=self.__keops_backend).sum()

    def compute_geodesic_control(self, man):
        vs = self.adjoint(man)(self.manifold.gd.view(-1, self.dim))
        self.fill_controls(self.solve_cgc(self.manifold.gd.reshape(-1, self.dim), self.manifold.gd.reshape(-1, self.dim), vs.reshape(-1, self.dim), self.__keops_sigma, backend=self.__keops_backend))


register_deformation_module_builder('translations', {'torch': Translations_Torch.build, 'keops': Translations_KeOps.build})


