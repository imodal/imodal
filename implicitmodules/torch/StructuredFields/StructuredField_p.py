import torch

from implicitmodules.torch.StructuredFields.Abstract import SupportStructuredField
from implicitmodules.torch.kernels import gauss_kernel, rel_differences


class StructuredField_p(SupportStructuredField):
    def __init__(self, support, moments, sigma):
        super().__init__(support, moments)
        self.__sigma = sigma
    
    @property
    def sigma(self):
        return self.__sigma
    
    def __call__(self, points, k=0):
        P = (self.moments + torch.transpose(self.moments, 1, 2)) / 2
        ker_vec = -gauss_kernel(rel_differences(points, self.support), k + 1, self.__sigma)
        ker_vec = ker_vec.reshape((points.shape[0], self.support.shape[0]) + tuple(ker_vec.shape[1:]))
        return torch.tensordot(torch.transpose(torch.tensordot(torch.eye(2), ker_vec, dims=0), 0, 2), P,
                               dims=([2, 3, 4], [1, 0, 2]))
