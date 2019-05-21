import numpy as np

from implicitmodules.numpy.StructuredFields.Abstract import StructuredField


class ConstantField(StructuredField):
    
    def __init__(self, mom):  # tested
        """
         sigma is the scale of the rkhs to which the field belongs
         support and mom are the parametrization of the vector field
        """
        # self.moments = np.zeros([1, dim]) # TODO : heck this with B.
        self.moments = mom
    
    def copy(self):
        v = ConstantField(self.moments)
        return v
    
    def copy_full(self):
        v = ConstantField(self.moments)
        v.moments = self.moments.copy()
        return v
    
    def fill_fieldparam(self, param):
        self.moments = param.copy()
    
    def __call__(self, z, j):
        """
        Applies the field to points z (or computes the derivative). 
        Needs pre-assigned parametrization of the field in dic
        
        """
        Nz = z.shape[0]
        lsize = ((Nz, 2), (Nz, 2, 2), (Nz, 2, 2, 2))
        djv = np.zeros(lsize[j])
        
        if j==0:
            djv = np.tile(self.moments, [Nz, 1])
        
        return djv
