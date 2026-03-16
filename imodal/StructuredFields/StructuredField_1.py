import torch

from imodal.Kernels.kernels import gauss_kernel, rel_differences
from imodal.StructuredFields.Abstract import KernelSupportStructuredField
from imodal.Utilities import get_compute_backend, is_valid_backend
from imodal.Kernels import K_xy

from pykeops.torch import Genred


class StructuredField_1(KernelSupportStructuredField):
    def __init__(self, support, moments, sigma, device=None, backend=None):
        super().__init__(support, moments, sigma, device, backend)

    def _compute_reduction_torch(self, points, k):
    
        if k == 0:
            dK = gauss_kernel(rel_differences(self.support, points), k=1, sigma=self.sigma)
            dK = dK.reshape([self.support.shape[0], points.shape[0], points.shape[-1]])
            dK = dK.transpose(0, 1)
            return torch.einsum('ijk,j->ik', dK, self.moments)
        else:
            dK = ((-1)**k) * gauss_kernel(rel_differences(self.support, points), k=k+1, sigma=self.sigma)
            dK = dK.reshape([self.support.shape[0], points.shape[0], points.shape[-1]] + list(dK.shape[2:]) )
            dK = dK.transpose(0, 1)
            return torch.einsum('ijk...,j->ik...', dK, self.moments)
                            
                            
                            
           
