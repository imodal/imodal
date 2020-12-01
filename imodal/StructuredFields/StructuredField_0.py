import torch

from imodal.Kernels.kernels import gauss_kernel, rel_differences
from imodal.StructuredFields.Abstract import KernelSupportStructuredField
from imodal.Utilities import get_compute_backend, is_valid_backend
from imodal.Kernels import K_xy

from pykeops.torch import Genred


class StructuredField_0(KernelSupportStructuredField):
    def __init__(self, support, moments, sigma, device=None, backend=None):
        super().__init__(support, moments, sigma, device, backend)

    def _compute_reduction_torch(self, points, k):
        if k == 0:
            K_q = K_xy(points, self.support, self.sigma)
            return torch.mm(K_q, self.moments)
        else:
            ker_vec = gauss_kernel(rel_differences(points, self.support), k, self.sigma)
            ker_vec = ker_vec.reshape((points.shape[0], self.support.shape[0]) + tuple(ker_vec.shape[1:]))
            return torch.tensordot(torch.transpose(torch.tensordot(torch.eye(self.dim, device=self.device), ker_vec, dims=0), 0, 2), self.moments, dims=([2, 3], [1, 0]))

    def _compute_reduction_keops(self, points, k):
        if k == 0:
            kernel_formula = "Exp(-S*SqNorm2(x - y)/IntCst(2))"
            formula = kernel_formula + "*p"
            alias = ["x=Vi("+str(self.dim)+")", "y=Vj("+str(self.dim)+")", "p=Vj("+str(self.dim)+")", "S=Pm(1)"]
            reduction = Genred(formula, alias, reduction_op='Sum', axis=1, dtype=self._keops_dtype)
            return reduction(points, self.support, self.moments, self._keops_sigma, backend=self._keops_backend).reshape(-1, self.dim)

        if k == 1:
            kernel_formula = "-S*Exp(-S*SqNorm2(x - y)/IntCst(2))*(x - y)"
            formula = "TensorProd(" + kernel_formula + ", p)"
            alias = ["x=Vi("+str(self.dim)+")", "y=Vj("+str(self.dim)+")", "p=Vj("+str(self.dim)+")", "S=Pm(1)"]
            reduction = Genred(formula, alias, reduction_op='Sum', axis=1, dtype=self._keops_dtype)
            return reduction(points, self.support, self.moments, self._keops_sigma, backend=self._keops_backend).reshape(-1, self.dim, self.dim).transpose(1, 2).contiguous()

        else:
            raise RuntimeError("StructuredField_0.__call__(): KeOps computation not supported for order k = " + str(k))

