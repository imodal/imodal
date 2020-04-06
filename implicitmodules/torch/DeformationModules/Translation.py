import torch

from pykeops.torch import Genred, KernelSolve

from implicitmodules.torch.DeformationModules.Abstract import DeformationModule, create_deformation_module_with_backends
from implicitmodules.torch.Kernels.kernels import K_xy, K_xx
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.StructuredFields import StructuredField_0


class TranslationsBase(DeformationModule):
    """ Deformation module generating a sum of translations. """
    
    def __init__(self, manifold, sigma, label):
        assert isinstance(manifold, Landmarks)
        super().__init__(label)
        self.__manifold = manifold
        self.__sigma = sigma
        self.__controls = torch.zeros_like(self.__manifold.gd, requires_grad=True)

    def __str__(self):
        outstr = "Local translation module\n"
        if self.label:
            outstr += "  Label=" + self.label + "\n"
        outstr += "  Sigma=" + str(self.sigma) + "\n"
        outstr += "  Nb pts=" + str(self.__manifold.nb_pts) + "\n"
        return outstr

    @classmethod
    def build(cls, dim, nb_pts, sigma, gd=None, tan=None, cotan=None, label=None):
        return cls(Landmarks(dim, nb_pts, gd=gd, tan=tan, cotan=cotan), sigma, label)

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

    def __get_controls(self):
        return self.__controls

    def fill_controls(self, controls):
        self.__controls = controls

    controls = property(__get_controls, fill_controls)

    def fill_controls_zero(self):
        self.__controls = torch.zeros_like(self.__manifold.gd, requires_grad=True)

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


class Translations_Torch(TranslationsBase):
    def __init__(self, manifold, sigma, label):
        super().__init__(manifold, sigma, label)

    @property
    def backend(self):
        return 'torch'

    def cost(self):
        K_q = K_xx(self.manifold.gd, self.sigma)

        m = torch.mm(K_q, self.controls)
        return 0.5 * torch.dot(m.flatten(), self.controls.flatten())

    def compute_geodesic_control(self, man):
        vs = self.adjoint(man)
        K_q = K_xx(self.manifold.gd, self.sigma)

        controls, _ = torch.solve(vs(self.manifold.gd), K_q)
        self.controls = controls.contiguous()


class Translations_KeOps(TranslationsBase):
    def __init__(self, manifold, sigma, label):
        super().__init__(manifold, sigma, label)

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


Translations = create_deformation_module_with_backends(Translations_Torch.build, Translations_KeOps.build)


