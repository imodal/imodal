import torch

from imodal.Kernels.kernels import gauss_kernel, rel_differences
from imodal.StructuredFields.Abstract import KernelSupportStructuredField
from imodal.Utilities import get_compute_backend, is_valid_backend
from imodal.Kernels import K_xy

from pykeops.torch import Genred


class StructuredField_0_u(KernelSupportStructuredField):
    def __init__(self, support, moments, directions, sigma, device=None, backend=None):
        super().__init__(support, moments, sigma, device, backend)
        assert directions.device == self.device
        assert directions.dtype == self.support.dtype

        self.__directions = directions

    @property
    def directions(self):
        return self.__directions

    def _compute_reduction_torch(self, points, k):
        if k == 0:
            ker_vec = gauss_kernel(rel_differences(points, self.support), 1, self.sigma)
            ker_vec = ker_vec.reshape((points.shape[0], self.support.shape[0]) + tuple(ker_vec.shape[1:]))
            return torch.mm(torch.einsum('ijk, jk->ij', -ker_vec, self.directions), self.moments)
        else:
            raise NotImplementedError()

    def _compute_reduction_keops(self, points, k):
        return self._compute_reduction_torch(points, k)
        raise NotImplementedError()

