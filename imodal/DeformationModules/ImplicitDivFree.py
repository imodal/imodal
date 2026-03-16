import torch


from imodal.DeformationModules.Abstract import DeformationModule, create_deformation_module_with_backends
from imodal.Manifolds import Landmarks, CompoundManifold, CompoundLandmarks
from imodal.StructuredFields import StructuredField_0, StructuredField_1, SumStructuredField
from imodal.Kernels import gauss_kernel, rel_differences, K_xx


class ImplicitDivFree(DeformationModule):
    """ 
    Implicit module generating deformations divergence free vector fields. 
    
    the generated field v for 
         GD $q= (x_1, .. , x_N, y_1, ... , y_P)$ with nbs_pts = [N, P]
         control $h = (h_1, ..., h_N)$
      satisfies the following soft constraints:
             - minimal norm
             - prescribed values v(x_i) = h_i
             - null divergence Div v(y_j)  = 0
      and minimizes :
       |v|^2_V + 1/nu \sum_i | v(x_i) - h_i) |^2 + 1/nu * \alpha \sum_j 1/2 | Div v(y_j)|^2$
      
         
    """

    def __init__(self, manifold, nbs_pts, sigma, coeff_div, nu, coeff, label):
        """
        
        manifold: Landmark manifold with nbs_pts[0] + nbs_pts[1] points
        nbs_pts : list of 2 integers
        sigma : non negative scalar, scale of the scalar Gaussian RKHS V of generated vector fields
        coeff_divp : coeff \alpha in front of the div-free penalty  \sum_j 1/2 | Dv(y_j) |^2
        nu : defines coeff 1/nu penalizing the constraint 1/nu \sum_i | v(x_i) - h_i) |^2 + 1/nu * \alpha \sum_j 1/2 | Div v(y_j)|^2$
        """
        
        super().__init__(label)
        self.__manifold = manifold
        self.__sigma = sigma
        self.__coeff_div = coeff_div
        self.__nu = nu
        self.__coeff = coeff
        self.__nbs_pts = nbs_pts
        self.__controls = torch.zeros_like(self.__manifold.gd[nbs_pts[0]:], device=manifold.device)

    def __str__(self):
        #TODO 
        return '<ImplicitDivFree>'

    @classmethod
    def build(cls, dim, nbs_pts, sigma, coeff_div=1., nu=0., coeff=1., gd=None, tan=None, cotan=None, label=None):
        # nbs_pts = list of 2 integers
        #manifolds = CompoundManifold([Landmarks(dim, nbs_pts[0], gd=gd[0], tan=tan[0], cotan=cotan[0]), Landmarks(dim, nbs_pts[1], gd=gd[1], tan=tan[1], cotan=cotan[1])])
        manifolds = Landmarks(dim, sum(nbs_pts), gd=gd, tan=tan, cotan=cotan)
        return cls(manifolds,nbs_pts, sigma, coeff_div, nu, coeff, label)

    @property
    def dim(self):
        return self.__manifold.dim

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
    def sigma(self):
        return self.__sigma

    @property
    def nu(self):
        return self.__nu

    @property
    def backend(self):
        return 'torch'
    
    @property
    def lambdas(self):
        return self.__lambdas

    @property
    def SKS(self):
        return self.__SKS


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
        self.fill_controls(torch.zeros_like(self.__manifold.gd[self.__nbs_pts[0]:], device=self.device))

    def __call__(self, points, k=0):
        return self.field_generator()(points, k)

    def cost(self):
        # TODO: use only non zero part of Aqh
        
        Aqh = torch.cat([torch.zeros(self.__nbs_pts[0]), self.controls.flatten()], dim=0)
        return 0.5 * self.coeff * torch.dot(Aqh.flatten(), self.__lambdas.view(-1))

    def compute_geodesic_control(self, man):
        self.computeSKS()
        vs = self.adjoint(man)
        def div(v,x):
            # v : structured field
            # x : Nxd tensor of points

            dv = v(x, k=1)
            return self.__coeff_div * torch.einsum('ijj->i',dv)
        
        SqKxi_div = div(vs, self.manifold.gd[:self.__nbs_pts[0]])
        SqKxi_0 = vs(self.__manifold.gd[self.__nbs_pts[0]:]).flatten()
        
        # with SKS^{-1} = [[A B], [B^T C]] and SKS = [[S1 T], [T^T S2]]
        # C^{-1} B^T = - T^T S1^{-1}
        
        # Aq* SKS^{-1} Aqh = C h
        # Aq SKS^{-1} [SqKxi_div , SqKxi_0] = B^T SqKxi_div + C SqKxi_0
        # so h = C^{-1} B^T SqKxi_div + SqKxi_0 = - T^T S1^{-1}SqKxi_div + SqKxi_0
        N = self.__nbs_pts[0]
        h = torch.linalg.solve(self.__SKS[:N, :N], SqKxi_div)
        h = - self.__SKS[N:, :N] @ h
        
        self.controls = (1/self.coeff) * ( h + SqKxi_0)
        self.__compute_moments()



    def compute_moments(self):
        # \lambda such that $\zeta_q (h) = K S_q^\ast \lambda$
        self.computeSKS()
        self.__compute_moments()
    
    def __compute_moments(self):
        Aqh = torch.cat([torch.zeros(self.__nbs_pts[0]), self.controls.flatten()], dim=0)
        self.__lambdas = torch.linalg.solve(self.__SKS , Aqh)

    
    def computeSKS(self):
        def div(v,x):
            # v : structured field
            # x : Nxd tensor of points

            dv = v(x, k=1)
            return self.__coeff_div * torch.einsum('ijj->i',dv)


        def S1_fun(x, sigma):
            ddker = - gauss_kernel(rel_differences(x,x), k=2, sigma=sigma).reshape([x.shape[0], x.shape[0], x.shape[1], x.shape[1]])
            return self.__coeff_div**2 * torch.einsum('ijkk->ij', ddker)

        X = self.__manifold.gd[:self.__nbs_pts[0]]
        Y = self.__manifold.gd[self.__nbs_pts[0]:]
        S1 = S1_fun(X, self.sigma)
        S2 = torch.einsum('ij,kl->ikjl',K_xx(Y, self.sigma), torch.eye(self.dim)).reshape([self.dim*Y.shape[0], self.dim*Y.shape[0]])
        rel = rel_differences(X,Y)
        T = self.__coeff_div * gauss_kernel(rel, k=1, sigma=self.sigma).flatten().reshape(X.shape[0], self.dim * Y.shape[0])

        self.__SKS = torch.cat([torch.cat([S1, T], dim=1), torch.cat([ T.transpose(1,0), S2], dim=1)], dim=0) +  self.nu * torch.eye(S1.shape[0] + S2.shape[0], device=self.device)

        
    def field_generator(self):
        v0 = StructuredField_0(self.__manifold.gd[self.__nbs_pts[0]:], self.__lambdas[self.__nbs_pts[0]:].reshape(self.__manifold.gd[self.__nbs_pts[0]:].shape), self.__sigma, device=self.device, backend=self.backend)
        v1 = StructuredField_1(self.__manifold.gd[:self.__nbs_pts[0]], self.__coeff_div*self.__lambdas[:self.__nbs_pts[0]], self.__sigma, device=self.device, backend=self.backend)
        return SumStructuredField([v0, v1])
        
        

    def adjoint(self, manifold):
        return manifold.cot_to_vs(self.__sigma, backend=self.backend)

    
    
    

ImplicitDivFree = create_deformation_module_with_backends(ImplicitDivFree.build, ImplicitDivFree.build)
