import torch

from implicitmodules.torch.Kernels.kernels import gauss_kernel, rel_differences
from implicitmodules.torch.StructuredFields.Abstract import SupportStructuredField
from implicitmodules.torch.Utilities import get_compute_backend, is_valid_backend
from implicitmodules.torch.Kernels import K_xy

from pykeops.torch import Genred


class StructuredField_0_u(SupportStructuredField):
    def __init__(self, support, moments, directions, sigma, device=None, backend=None):
        super().__init__(support, moments)
        self.__sigma = sigma
        self.__directions = directions

        if backend is not None:
            assert is_valid_backend(backend)
        else:
            backend = get_compute_backend()

        self.__device = self.__find_device(support, moments, device)

        if backend == 'torch':
            self.__compute_reduction = self.__compute_reduction_torch
        elif backend == 'keops':
            self.__keops_backend = 'CPU'
            if str(self.__device) != 'cpu':
                self.__keops_backend = 'GPU'
            self.__compute_reduction = self.__compute_reduction_keops
            self.__keops_sigma = torch.tensor([1./self.__sigma/self.__sigma], dtype=support.dtype, device=self.__device)
            self.__keops_dtype = str(support.dtype).split(".")[1]

    @property
    def device(self):
        return self.__device

    @property
    def sigma(self):
        return self.__sigma

    def __find_device(self, support, moments, device):
        if device is None:
            if support.device != moments.device:
                raise RuntimeError("StructuredField_0.__init__(): support and moments not on the same device!")
            return support.device
        else:
            support.to(device=device)
            moments.to(device=device)
            return device

    def __call__(self, points, k=0):
        assert k >= 0
        return self.__compute_reduction(points, k)

    def __compute_reduction_torch(self, points, k):
        dim = points.shape[1]

        if k == 0:
            ker_vec = gauss_kernel(rel_differences(points, self.support), 1, self.__sigma)
            ker_vec = ker_vec.reshape((points.shape[0], self.support.shape[0]) + tuple(ker_vec.shape[1:]))
            D = torch.tensordot(torch.transpose(torch.tensordot(torch.eye(dim, device=self.device), ker_vec, dims=0), 0, 2), self.moments, dims=([2, 3], [1, 0]))

            return torch.einsum('nik, nk->ni', D, self.__directions)
        else:
            raise NotImplementedError()

    def __compute_reduction_keops(self, points, k):
        raise NotImplementedError()

