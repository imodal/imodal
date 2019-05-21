import numpy as np
from implicitmodules.numpy.StructuredFields.Abstract import SupportStructuredField


class LinearField(SupportStructuredField):

    def __init__(self, support, moments, sigma=np.inf):
        super().__init__(support, moments, sigma)

    def copy(self):
        v = LinearField(self.support, self.moments)
        return v

    def copy_full(self):
        v = LinearField(self.support, self.moments)
        v.support = self.support.copy()
        v.mom = self.moments.copy()
        return v

    def __call__(self, z, j):
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
        
        
        