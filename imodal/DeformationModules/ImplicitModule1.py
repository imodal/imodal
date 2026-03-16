import torch

from pykeops.torch import Vi, Vj

from imodal.DeformationModules.Abstract import DeformationModule, create_deformation_module_with_backends
from imodal.Kernels.SKS import eta, compute_sks, compute_sks_keops
from imodal.Manifolds import NormalFrame
from imodal.StructuredFields import StructuredField_p


class ImplicitModule1Base(DeformationModule):
    """ 
    Implicit module of order 1. 
    
    the generated field v for 
         GD $q= ((x_1, R_1), .. , (x_N, R_N))$ 
         control $h = (h_1, ..., h_r)$
      satisfies the following soft constraints:
             - minimal norm
             - prescribed value for the infinitesimal strain tensors 1/2 ( Dv(x_i) + Dv(x_i)^T) = R_i diag(C[i,:,:] @ h)R_i^T
      and minimizes :
       |v|^2_V + 1/nu \sum_i  1/2 | Dv(x_i) + Dv(x_i)^T - R_i diag(C[i,:,:] @ h)R_i^T |^2$
      
         
    """

    def __init__(self, manifold, sigma, C, nu, coeff, label):
        """
        manifold: Normal Frame manifold of N orthogonal frames (x_i, R_i)
        sigma : non negative scalar, scale of the scalar Gaussian RKHS V of generated vector fields
        C : N x d x r with N the number of orthonormal frames, d the dimension of the ambient space and r the dimension of the control space
        nu : defines coeff 1/nu penalizing the constraint 1/nu\sum_i  1/2 | Dv(x_i) + Dv(x_i)^T - R_i diag(C[i,:,:] @ h)R_i^T |^2
        """
        
        assert isinstance(manifold, NormalFrame)
        super().__init__(label)
        self.__manifold = manifold
        self.__C = C
        self.__sigma = sigma
        self.__nu = nu
        self.__coeff = coeff
        self.__dim_controls = C.shape[2]
        self.__sym_dim = int(self.manifold.dim * (self.manifold.dim + 1) / 2)
        self.__controls = torch.zeros(self.__dim_controls, device=self.__manifold.device)

    def __str__(self):
        outstr = "Implicit module of order 1\n"
        if self.label:
            outstr += "  Label=" + self.label + "\n"
        outstr += "  Sigma=" + str(self.sigma) + "\n"
        outstr += "  Nu=" + str(self.__nu) + "\n"
        outstr += "  Coeff=" + str(self.__coeff) + "\n"
        outstr += "  Dim controls=" + str(self.__dim_controls) + "\n"
        outstr += "  Nb pts=" + str(self.__manifold.nb_pts) + "\n"
        return outstr

    @classmethod
    def build(cls, dim, nb_pts, sigma, C, nu=0., coeff=1., gd=None, tan=None, cotan=None, label=None):
        return cls(NormalFrame(dim, nb_pts, gd=gd, tan=tan, cotan=cotan), sigma, C, nu, coeff, label)

    @property
    def dim(self):
        return self.__manifold.dim

    def to_(self, *args, **kwargs):
        self.__manifold.to_(*args, **kwargs)
        self.__controls = self.__controls.to(*args, **kwargs)
        self.__C = self.__C.to(*args, **kwargs)

    @property
    def device(self):
        return self.__manifold.device

    @property
    def manifold(self):
        return self.__manifold

    def __get_C(self):
        return self.__C

    def __set_C(self, C):
        self.__C = C

    C = property(__get_C, __set_C)

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
        self.fill_controls(torch.zeros(self.__dim_controls, device=self.device))

    def __call__(self, points, k=0):
        return self.field_generator()(points, k)

    def cost(self):
        raise NotImplementedError()

    def compute_geodesic_control(self, man):
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
        self.eta = eta(self.dim, device=self.device)       # dim * dim * sym_dim
        self.u = torch.einsum("rs, nst, mru -> tumn",
                              torch.eye(self.dim, device=self.device),
                              self.eta,
                              self.eta)  # Dimension: sym_dim * sym_dim * dim * dim

    @property
    def backend(self):
        return 'torch'

    def cost(self):
        return 0.5 * self.coeff * torch.dot(self.__aqh.flatten(), self.__lambdas.view(-1))

    def compute_geodesic_control(self, man):
        vs = self.adjoint(man)
        d_vx = vs(self.manifold.gd[0], k=1)

        S = 0.5 * (d_vx + torch.transpose(d_vx, 1, 2))
        S = torch.tensordot(S, self.eta, dims=2)

        self.__compute_sks()

        tlambdas = torch.linalg.solve(self.__sks, S.view(-1, 1))
        tlambdas = tlambdas/self.coeff

        (aq, aqkiaq) = self.__compute_aqkiaq()
        c= torch.linalg.solve(aqkiaq, torch.mm(aq.t(), tlambdas))
        self.controls = c.reshape(-1)
        self.__compute_moments()

    def compute_moments(self):
        self.__compute_sks()
        self.__compute_moments()

    def __compute_aqh(self, h):
        R = self.manifold.gd[1]

        return torch.einsum('nli, nik, k, nui, niv, lvt -> nt',
                            R,
                            self.C,
                            h,
                            torch.eye(self.manifold.dim, device=self.device).repeat(self.manifold.nb_pts, 1, 1),
                            torch.transpose(R, 1, 2),
                            self.eta
                            )

    def __compute_sks(self):
        self.__sks = compute_sks(self.manifold.gd[0], self.sigma, 1, u=self.u) + self.nu * torch.eye(self.sym_dim * self.manifold.nb_pts, device=self.device)

    def __compute_moments(self):
        self.__aqh = self.__compute_aqh(self.controls)
        lambdas = torch.linalg.solve(self.__sks, self.__aqh.flatten())
        self.__lambdas = lambdas.contiguous()
        self.moments = torch.tensordot(self.__lambdas.view(-1, self.sym_dim),
                                       torch.transpose(self.eta, 0, 2),
                                       dims=1)


    def __compute_aqkiaq(self):
        lambdas = torch.zeros(self.dim_controls, self.sym_dim * self.manifold.nb_pts, device=self.device)
        aq = torch.zeros(self.sym_dim * self.manifold.nb_pts, self.dim_controls, device=self.device)
        for i in range(self.dim_controls):
            h = torch.zeros(self.dim_controls, device=self.device)
            h[i] = 1.
            aqi = self.__compute_aqh(h).flatten()
            aq[:, i] = aqi
            l = torch.linalg.solve(self.__sks, aqi.view(-1, 1))
            lambdas[i, :] = l.flatten()

        return (aq, torch.mm(lambdas, aq))


class ImplicitModule1_KeOps(ImplicitModule1Base):
    def __init__(self, manifold, sigma, C, nu, coeff, label):
        super().__init__(manifold, sigma, C, nu, coeff, label)
        self.eta = eta(self.manifold.dim, device=self.device)
        self.u = torch.einsum("rs, nst, mru -> tumn",
                         torch.eye(self.dim, device=self.device),
                         self.eta,
                         self.eta)                # Dimension: sym_dim * sym_dim * dim * dim
        self.eps = torch.finfo(manifold.dtype).eps


    @property
    def backend(self):
        return 'keops'

    def to_(self, *args, **kwargs):
        super().to_(*args, **kwargs)

        if 'device' in kwargs:
            if kwargs['device'].split(":")[0].lower() == "cuda":
                self.__keops_backend = 'GPU'
            elif kwargs['device'].split(":")[0].lower() == "cpu":
                self.__keops_backend = 'CPU'

    def cost(self):
        return 0.5 * self.coeff * torch.dot(self.aqh.flatten(), self.lambdas.flatten())

    def compute_geodesic_control(self, man):
        vs = self.adjoint(man)
        d_vx = vs(self.manifold.gd[0], k=1)

        S = 0.5 * (d_vx + torch.transpose(d_vx, 1, 2))
        S = torch.tensordot(S, self.eta, dims=2)

        self.__compute_sks()
        tlambdas = self.__sks(S.reshape(-1, 1))
        tlambdas = tlambdas.reshape(-1, self.sym_dim) / self.coeff

        (aq, aqkiaq) = self.__compute_aqkiaq()
        c = torch.linalg.solve(aqkiaq, torch.mm(aq.t(), tlambdas.view(-1, 1)))
        self.controls = c.flatten()

        # self.compute_moments()

    def compute_moments(self):
        self.__compute_sks()
        self.__compute_moments()

    def __compute_sks(self):
        self.__sks = compute_sks_keops(self.manifold.gd[0], self.sigma, 1, u=self.u).solve(Vi(0, 1),
                                                                                     var=Vj(0, 1),
                                                                                     alpha=self.nu,
                                                                                     eps=self.eps)

    def __compute_aqh(self, h):
        R = self.manifold.gd[1]

        return torch.einsum('nli, nik, k, nui, niv, lvt -> nt',
                            R,
                            self.C,
                            h,
                            torch.eye(self.manifold.dim, device=self.device).repeat(self.manifold.nb_pts, 1, 1),
                            torch.transpose(R, 1, 2),
                            self.eta
                            )

    def __compute_moments(self):
        self.aqh = self.__compute_aqh(self.controls)
        self.lambdas = self.__sks(self.aqh.reshape(-1, 1))
        self.moments = torch.tensordot(self.lambdas.view(-1, self.sym_dim),
                                       torch.transpose(self.eta, 0, 2),
                                       dims=1)

    def __compute_aqkiaq(self):
        lambdas = torch.zeros(self.dim_controls, self.sym_dim * self.manifold.nb_pts, device=self.device)
        aq = torch.zeros(self.sym_dim * self.manifold.nb_pts, self.dim_controls, device=self.device)
        for i in range(self.dim_controls):
            h = torch.zeros(self.dim_controls, device=self.device)
            h[i] = 1.
            aqi = self.__compute_aqh(h).flatten()
            aq[:, i] = aqi

            lambdas[i, :] = self.__sks(aqi.reshape(-1, 1)).view(-1)

        return (aq, torch.mm(lambdas, aq))


ImplicitModule1 = create_deformation_module_with_backends(ImplicitModule1_Torch.build, ImplicitModule1_KeOps.build)