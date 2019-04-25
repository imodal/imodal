import numpy as np

import implicitmodules.numpy.StructuredFields.Abstract as ab


class LinearField(ab.StructuredField):
    
    def __init__(self, dim, mat):  # tested
        """
         support and mom are the parametrization of the vector field
        """
        self.dim = dim
        self.mat = mat.copy()
        self.support = np.zeros([1, dim])
        self.mom = np.zeros([1])
    
    def copy(self):
        v = LinearField(self.dim)
        return v
    
    def copy_full(self):
        v = LinearField(self.dim)
        v.support = self.support.copy()
        v.mom = self.mom.copy()
        return v
    
    def fill_fieldparam(self, param):
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
        
        if j==0:
            diff = z - np.tile(self.support, [Nz,1])
            djv = self.mom[0] * np.transpose(np.dot(self.mat, np.transpose(diff)))
        if j==1:
            djv = self.mom[0] * np.tile(np.expand_dims(self.mat, 0), [Nz,1,1])
        
        return djv
        
        
        