import numpy as np

import implicitmodules.numpy.StructuredFields.Abstract as ab


class ConstantField(ab.StructuredField):
    
    def __init__(self, dim):  # tested
        """
         sigma is the scale of the rkhs to which the field belongs
         support and mom are the parametrization of the vector field
        """
        self.dim = dim
        self.mom = np.zeros([1, dim])
    
    def copy(self):
        v = ConstantField(self.dim)
        return v
    
    def copy_full(self):
        v = ConstantField(self.dim)
        v.mom = self.mom.copy()
        return v
    
    def fill_fieldparam(self, param):
        self.mom = param.copy()
    
    def Apply(self, z, j):
        """
        Applies the field to points z (or computes the derivative). 
        Needs pre-assigned parametrization of the field in dic
        
        """
        Nz = z.shape[0]
        lsize = ((Nz, 2), (Nz, 2, 2), (Nz, 2, 2, 2))
        djv = np.zeros(lsize[j])
        
        if j==0:
            djv = np.tile(self.mom, [Nz,1])
        
        return djv
