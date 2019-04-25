import numpy as np

import implicitmodules.numpy.StructuredFields.Abstract as ab


class ZeroField(ab.StructuredField):
    
    def __init__(self, dim):  # tested
        """
         sigma is the scale of the rkhs to which the field belongs
         support and mom are the parametrization of the vector field
        """
        self.dim = dim
    
    def copy(self):
        v = ZeroField(self.dim)
        return v
    
    def copy_full(self):
        v = ZeroField(self.dim)
        return v
    
    def fill_fieldparam(self, param):
        pass
    
    def Apply(self, z, j):
        """
        Applies the field to points z (or computes the derivative). 
        Needs pre-assigned parametrization of the field in dic
        
        """
        Nz = z.shape[0]
        lsize = ((Nz, 2), (Nz, 2, 2), (Nz, 2, 2, 2))
        djv = np.zeros(lsize[j])
        
        return djv
