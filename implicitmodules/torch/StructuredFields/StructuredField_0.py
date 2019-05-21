import torch

from implicitmodules.torch.Kernels.kernels import gauss_kernel, rel_differences
from implicitmodules.torch.StructuredFields.Abstract import SupportStructuredField


class StructuredField_0(SupportStructuredField):
    def __init__(self, support, moments, sigma):
        super().__init__(support, moments)
        self.__sigma = sigma
    
    @property
    def sigma(self):
        return self.__sigma
    
    def __call__(self, points, k=0):
        ker_vec = gauss_kernel(rel_differences(points, self.support), k, self.__sigma)
        ker_vec = ker_vec.reshape((points.shape[0], self.support.shape[0]) + tuple(ker_vec.shape[1:]))
        return torch.tensordot(torch.transpose(torch.tensordot(torch.eye(2), ker_vec, dims=0), 0, 2), self.moments,
                               dims=([2, 3], [1, 0]))
