import torch

from implicitmodules.torch.Kernels import gauss_kernel, rel_differences, sqdistances
from implicitmodules.torch.StructuredFields.Abstract import SupportStructuredField
from implicitmodules.torch.Utilities import get_compute_backend, is_valid_backend

from pykeops.torch import Genred


class StructuredField_pm(SupportStructuredField):
    def __init__(self, support, moments, sigma, backend, device):
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

    def __find_device(self, support, moments, device):
        if device is None:
            if support.device != moments.device:
                raise RuntimeError("StructuredField_pm.__init__(): support and moments not on the same device!")
            return support.device
        else:
            support.to(device=device)
            moments.to(device=device)
            return device

    @property
    def sigma(self):
        return self.__sigma

    @property
    def device(self):
        return self.__device
    
    def __call__(self, points, k=0):
        assert k >= 0
        P = self.compute_moment_matrix()
        return self.__compute_reduction(points, P, k)

    def __compute_reduction_torch(self, points, P, k):
        ker_vec = -gauss_kernel(rel_differences(points, self.support), k + 1, self.__sigma)
        ker_vec = ker_vec.reshape((points.shape[0], self.support.shape[0]) + tuple(ker_vec.shape[1:]))
        # TODO: hardcoded dim = 2
        return torch.tensordot(torch.transpose(torch.tensordot(torch.eye(2, device=self.device), ker_vec, dims=0), 0, 2), P, dims=([2, 3, 4], [1, 0, 2]))

    def __compute_reduction_keops(self, points, P, k):
        if k == 0:
            kernel_formula = "S*Exp(-S*SqNorm2(x - y)/IntCst(2))*(x - y)"
            formula = "TensorDot(" + kernel_formula + ", p, Ind(2), Ind(2,2), Ind(0), Ind(1))"
            alias = ["x=Vi(2)", "y=Vj(2)", "p=Vj(4)", "S=Pm(1)"]
            reduction = Genred(formula, alias, reduction_op='Sum', axis=1, dtype=str(points.dtype).split(".")[1])
            return reduction(points.reshape(-1, 2), self.support.reshape(-1, 2), P.reshape(-1, 4), self.__keops_sigma, backend=self.__keops_backend).reshape(-1, 2)

        if k == 1:
            #return self.__compute_reduction_torch(points, P, k)
            # TODO: test eye_formula
            #eye_formula = "Concat(ElemT(1., 2, 0), ElemT(1., 2, 1))"
            kernel_formula = "-S*Exp(-S*SqNorm2(x - y)/IntCst(2))*(S*TensorDot(x-y, x-y, Ind(2), Ind(2), Ind(), Ind())-c)"
            formula = "TensorDot("+ kernel_formula + ", p, Ind(2,2), Ind(2,2),Ind(1),Ind(1))"
            alias = ["x=Vi(2)", "y=Vj(2)", "p=Vj(4)", "c=Pm(4)", "S=Pm(1)"]
            reduction = Genred(formula, alias, reduction_op='Sum', axis=1, dtype=str(points.dtype).split(".")[1])
            return reduction(points.reshape(-1, 2), self.support.reshape(-1, 2), P.reshape(-1, 4), torch.eye(2, dtype=self.support.dtype, device=self.device).reshape(-1), self.__keops_sigma, backend=self.__keops_backend).reshape(-1, 2, 2).transpose(1, 2).contiguous()

        else:
            raise RuntimeError("StructuredField_pm.__call__(): keops computation not supported for order k =", k)


class StructuredField_p(StructuredField_pm):
    def __init__(self, support, moments, sigma, backend=None, device=None):
        super().__init__(support ,moments, sigma, backend, device)

    def compute_moment_matrix(self):
        return (self.moments + torch.transpose(self.moments, 1, 2))/2.


class StructuredField_m(StructuredField_pm):
    def __init__(self, support, moments, sigma, backend=None, device=None):
        super().__init__(support ,moments, sigma, backend, device)

    def compute_moment_matrix(self):
        return (self.moments - torch.transpose(self.moments, 1, 2))/2.

