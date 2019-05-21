import numpy as np

from implicitmodules.numpy.StructuredFields.Abstract import StructuredField


class StructuredField_Null(StructuredField):
    
    def __init__(self):
        super().__init__()
    
    def copy(self):
        v = StructuredField_Null(self.dim)
        return v
    
    def copy_full(self):
        v = StructuredField_Null(self.dim)
        return v
    
    def __call__(self, z, j):
        """
        Applies the field to points z (or computes the derivative). 
        Needs pre-assigned parametrization of the field in dic
        
        """
        Nz = z.shape[0]
        lsize = ((Nz, 2), (Nz, 2, 2), (Nz, 2, 2, 2))
        djv = np.zeros(lsize[j])
        
        return djv
