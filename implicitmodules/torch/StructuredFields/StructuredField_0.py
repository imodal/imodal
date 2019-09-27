import torch

from implicitmodules.torch.Kernels.kernels import gauss_kernel, rel_differences
from implicitmodules.torch.StructuredFields.Abstract import SupportStructuredField
from implicitmodules.torch.Utilities import get_compute_backend, is_valid_backend

from pykeops.torch import Genred


class StructuredField_0(SupportStructuredField):
    def __init__(self, support, moments, sigma, device=None, backend=None):
        super().__init__(support, moments)
        self.__sigma = sigma
        

        if backend is not None:
            assert is_valid_backend(backend)
        else:
            backend = get_compute_backend()

        self.__device = self.__find_device(support, moments, device)

        if backend == 'torch':
            self.__compute_reduction = self.__compute_reduction_torch
        elif backend == 'keops':
            self.__keops_backend = 'CPU'
            if self.__device != 'cpu':
                self.__keops_backend = 'GPU'
            self.__compute_reduction = self.__compute_reduction_keops
            self.__keops_sigma = torch.tensor([1./self.__sigma/self.__sigma], dtype=support.dtype, device=self.__device)

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
        ker_vec = gauss_kernel(rel_differences(points, self.support), k, self.__sigma)
        ker_vec = ker_vec.reshape((points.shape[0], self.support.shape[0]) + tuple(ker_vec.shape[1:]))
        # TODO: hardcoded dimensions
        return torch.tensordot(torch.transpose(torch.tensordot(torch.eye(2, device=self.device), ker_vec, dims=0), 0, 2), self.moments, dims=([2, 3], [1, 0]))

    def __compute_reduction_keops(self, points, k):
        if k == 0:
            kernel_formula = "Exp(-S*SqNorm2(x - y)/IntCst(2))"
            formula = kernel_formula + "*p"
            alias = ["x=Vi(2)", "y=Vj(2)", "p=Vj(2)", "S=Pm(1)"]
            reduction = Genred(formula, alias, reduction_op='Sum', axis=1, dtype=str(points.dtype).split(".")[1])
            return reduction(points.reshape(-1, 2), self.support.reshape(-1, 2), self.moments.reshape(-1, 2), self.__keops_sigma, backend=self.__keops_backend).reshape(-1, 2)

        if k == 1:
            kernel_formula = "-S*Exp(-S*SqNorm2(x - y)/IntCst(2))*(x - y)"
            formula = "TensorProd(" + kernel_formula + ", p)"
            alias = ["x=Vi(2)", "y=Vj(2)", "p=Vj(2)", "S=Pm(1)"]
            reduction = Genred(formula, alias, reduction_op='Sum', axis=1, dtype=str(points.dtype).split(".")[1])
            return reduction(points.view(-1, 2), self.support.view(-1, 2), self.moments.view(-1, 2), self.__keops_sigma, backend=self.__keops_backend).reshape(-1, 2, 2).transpose(1, 2).contiguous()

        else:
            raise RuntimeError("StructuredField_0.__call__(): keops computation not supported for order k =", k)


