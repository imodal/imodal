import torch
import math

from pykeops.torch import Genred, KernelSolve

from implicitmodules.torch.DeformationModules.Abstract import DeformationModule, create_deformation_module_with_backends
from implicitmodules.torch.Kernels.kernels import K_xy, K_xx
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.StructuredFields import StructuredField_0

class ConstrainedTranslationsBase(DeformationModule):
    """Module generating sum of translations."""
    
    def __init__(self, manifold, sigma, f_support, f_vectors, coeff, label):
        assert isinstance(manifold, Landmarks)
        super().__init__()
        self.__manifold = manifold
        self.__sigma = sigma
        self.__f_support = f_support
        self.__f_vectors = f_vectors
        self.__controls = torch.zeros(1, requires_grad=True)
        self.__coeff = coeff

    @classmethod
    def build(cls, dim, nb_pts, sigma, coeff, gd=None, tan=None, cotan=None, label=None):
        """Builds the Translations deformation module from tensors."""
        return cls(Landmarks(dim, nb_pts, gd=gd, tan=tan, cotan=cotan), sigma, f_support, f_vectors, coeff, label)

    def to_(self, device):
        self.__manifold.to_(device)
        self.__controls = self.__controls.to(device)

    @property
    def coeff(self):
        return self.__coeff

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
        self.__controls = torch.zeros(1, requires_grad=True)

    def __call__(self, points, k=0):
        """Applies the generated vector field on given points."""
        return self.field_generator()(points, k)

    def cost(self):
        support = self.__f_support(self.__manifold.gd)
        vectors = self.__f_vectors(self.__manifold.gd)
        
        K_q = K_xx(support, self.sigma)

        m = torch.mm(K_q, vectors)
        return 0.5 * self.coeff * torch.dot(m.flatten(), vectors.flatten()) * self.controls * self.controls

    def compute_geodesic_control(self, man):
        support = self.__f_support(self.__manifold.gd)
        vectors = self.__f_vectors(self.__manifold.gd)
        # vector field for control = 1
        v = StructuredField_0(support, vectors, self.__sigma, device=self.device, backend='torch')
        
                
        K_q = K_xx(support, self.sigma)
        m = torch.mm(K_q, vectors)
        co = self.coeff * torch.dot(m.flatten(), vectors.flatten())
        
        self.controls = man.inner_prod_field(v)/co


    def field_generator(self):
        
        support = self.__f_support(self.__manifold.gd)
        vectors = self.__f_vectors(self.__manifold.gd)
        
        return StructuredField_0(support, self.__controls*vectors, self.__sigma, device=self.device, backend=self.backend)

    def adjoint(self, manifold):
        return manifold.cot_to_vs(self.__sigma, backend=self.backend)

    
    
class ConstrainedTranslations_Torch(ConstrainedTranslationsBase):
    def __init__(self, manifold, sigma, f_support, f_vectors, coeff, label):
        super().__init__(manifold, sigma, f_support, f_vectors, coeff, label)

    @property
    def backend(self):
        return 'torch'

    def cost(self):
        
        support = self.__f_support(self.__manifold.gd)
        vectors = self.__f_vectors(self.__manifold.gd)
        
        K_q = K_xx(support, self.sigma)

        m = torch.mm(K_q, vectors)
        return 0.5 * self.coeff * torch.dot(m.flatten(), vectors.flatten()) * self.controls * self.controls

    def compute_geodesic_control(self, man):
        support = self.f_support(self.__manifold.gd)
        vectors = self.f_vectors(self.__manifold.gd)
        # vector field for control = 1
        v = StructuredField_0(support, vectors, self.__sigma, device=self.device, backend='torch')
        
                
        K_q = K_xx(support, self.sigma)
        m = torch.mm(K_q, vectors)
        co = self.coeff * torch.dot(m.flatten(), vectors.flatten())
        
        self.controls = man.inner_prod_field(v)/co

   
    

#ConstrainedTranslations = create_deformation_module_with_backends(ConstrainedTranslations_Torch.build)

    
class LocalScaling(ConstrainedTranslationsBase):
    def __init__(self, manifold, sigma, coeff, label=None):
        
        def f_vectors(gd):
            pi = math.pi
            return torch.tensor([[math.cos(2*pi/3*x), math.sin(2*pi/3*x)] for x in range(3)])
        def f_support(gd):
            return gd.repeat(3, 1) + sigma/3 * f_vectors(gd)
        
        super().__init__(manifold, sigma, f_support, f_vectors, coeff, label)
   
    @classmethod
    def build(cls, dim, nb_pts, sigma, coeff, gd=None, tan=None, cotan=None, label=None):
        """Builds the Translations deformation module from tensors."""
        return cls(Landmarks(dim, nb_pts, gd=gd, tan=tan, cotan=cotan), sigma, coeff, label)
         
    
    @property
    def backend(self):
        return 'torch'
    
    
    
    
class LocalRotation(ConstrainedTranslationsBase):
    def __init__(self, manifold, sigma, coeff, label=None):
        
        def f_vectors(gd):
            pi = math.pi
            return torch.tensor([[-math.sin(2*pi/3*(x)), math.cos(2*pi/3*(x))] for x in range(3)])
        def f_support(gd):
            pi = math.pi
            return gd.repeat(3, 1) + sigma/3 * torch.tensor([[math.cos(2*pi/3*x), math.sin(2*pi/3*x)] for x in range(3)])
        
        super().__init__(manifold, sigma, f_support, f_vectors, coeff, label)
   
    @classmethod
    def build(cls, dim, nb_pts, sigma, coeff, gd=None, tan=None, cotan=None, label=None):
        """Builds the Translations deformation module from tensors."""
        return cls(Landmarks(dim, nb_pts, gd=gd, tan=tan, cotan=cotan), sigma, coeff, label)
         
    
    @property
    def backend(self):
        return 'torch'