import numpy as np

import implicitmodules.numpy.StructuredFields.Abstract as ab
import implicitmodules.numpy.StructuredFields.SupportPoints as supp
from implicitmodules.numpy.Kernels import ScalarGaussian as ker


class StructuredField_p(ab.StructuredField):
    
    def __init__(self, sigma, N_pts, dim):  # tested
        """
         sigma is the scale of the rkhs to which the field belongs
         support and mom are the parametrization of the vector field
        """
        self.dim = dim
        self.sig = sigma
        self.N_pts = N_pts
        self.type = 'p'
        self.support = supp.SupportPoints(N_pts, dim)
        self.mom = np.zeros([N_pts, dim, dim])
    
    def copy(self):
        v = StructuredField_p(self.sig, self.N_pts, self.dim)
        return v
    
    def copy_full(self):
        v = StructuredField_p(self.sig, self.N_pts, self.dim)
        v.support = self.support.copy_full()
        v.mom = self.mom.copy()
        
        return v
    
    def fill_fieldparam(self, param):
        """
        param should be a list of two elements: array of points and 
        array of vectors
        """
        self.support.fill_value(param[0])
        self.mom = param[1].copy()
    
    def Apply(self, z, j):
        """
        Applies the field to points z (or computes the derivative). 
        Needs pre-assigned parametrization of the field in dic
        
        """
        Nz = z.shape[0]
        # lsize = ((Nz, 2), (Nz, 2, 2), (Nz, 2, 2, 2))
        # djv = np.zeros(lsize[j])
        
        x = self.support.get_value()
        P = self.mom.copy()
        P = (P + np.swapaxes(P, 1, 2)) / 2
        ker_vec = -ker.my_vker(ker.my_xmy(z, x), j + 1, self.sig)
        my_shape = (Nz, x.shape[0]) + tuple(ker_vec.shape[1:])
        ker_vec = ker_vec.reshape(my_shape)
        djv = np.tensordot(np.swapaxes(np.tensordot(np.eye(2), ker_vec, axes=0), 0, 2),
                           P, axes=([2, 3, 4], [1, 0, 2]))
        
        return djv
