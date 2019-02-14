import numpy as np

import src.StructuredFields.Abstract as ab
from src.Kernels import ScalarGaussian as ker


class StructuredField_0(ab.StructuredField):
    
    def __init__(self, sigma, N_pts, dim):  # tested
        """
         sigma is the scale of the rkhs to which the field belongs
         support and mom are the parametrization of the vector field
        """
        self.dim = dim
        self.sig = sigma
        self.N_pts = N_pts
        self.type = '0'
        self.support = np.zeros([N_pts, dim])
        self.mom = np.zeros([N_pts, dim])
    
    def copy(self):
        v = StructuredField_0(self.sig, self.N_pts, self.dim)
        return v
    
    def copy_full(self):
        v = StructuredField_0(self.sig, self.N_pts, self.dim)
        v.support = self.support.copy()
        v.mom = self.mom.copy()
        
        return v
    
    def fill_fieldparam(self, param):
        """
        param should be a list of two elements: array of points and 
        array of vectors
        """
        self.support = param[0].copy()
        self.mom = param[1].copy()
    
    def Apply(self, z, j):
        """
        Applies the field to points z (or computes the derivative). 
        Needs pre-assigned parametrization of the field in dic
        
        """
        Nz = z.shape[0]
        lsize = ((Nz, 2), (Nz, 2, 2), (Nz, 2, 2, 2))
        djv = np.zeros(lsize[j])
        
        x = self.support.copy()
        p = self.mom.copy()
        
        ker_vec = ker.my_vker(ker.my_xmy(z, x), j, self.sig)
        my_shape = (Nz, x.shape[0]) + tuple(ker_vec.shape[1:])
        ker_vec = ker_vec.reshape(my_shape)
        djv += np.tensordot(np.swapaxes(np.tensordot(np.eye(2), ker_vec, axes=0), 0, 2),
                            p, axes=([2, 3], [1, 0]))
        
        return djv
