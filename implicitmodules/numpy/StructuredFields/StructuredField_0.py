import numpy as np

from implicitmodules.numpy.Kernels import ScalarGaussian as ker
from implicitmodules.numpy.StructuredFields.Abstract import SupportStructuredField


class StructuredField_0(SupportStructuredField):
    
    def __init__(self, support, moments, sigma):
        super().__init__(support, moments, sigma)
    
    def copy(self):
        v = StructuredField_0(self.support, self.moments, self.sigma)
        return v
    
    def copy_full(self):
        v = StructuredField_0(self.support, self.moments, self.sigma)
        return v
    
    def __call__(self, z, j, k=0):
        """
        Applies the field to points z (or computes the derivative). 
        Needs pre-assigned parametrization of the field in dic
        
        """
        Nz = z.shape[0]
        lsize = ((Nz, 2), (Nz, 2, 2), (Nz, 2, 2, 2))
        djv = np.zeros(lsize[j])
        
        x = self.support.copy()
        p = self.moments.copy()

        ker_vec = ker.my_vker(ker.my_xmy(z, x), j, self.sigma)
        my_shape = (Nz, x.shape[0]) + tuple(ker_vec.shape[1:])
        ker_vec = ker_vec.reshape(my_shape)
        djv += np.tensordot(np.swapaxes(np.tensordot(np.eye(2), ker_vec, axes=0), 0, 2), p, axes=([2, 3], [1, 0]))
        
        return djv
