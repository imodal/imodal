import torch

from imodal.Kernels import gauss_kernel, rel_differences, sqdistances
from imodal.StructuredFields.Abstract import KernelSupportStructuredField
from imodal.Utilities import get_compute_backend, is_valid_backend

from pykeops.torch import Genred


class StructuredField_pm(KernelSupportStructuredField):
    def __init__(self, support, moments, sigma, device, backend):
        super().__init__(support, moments, sigma, device, backend)

    def __call__(self, points, k=0):
        assert k >= 0

        P = self._compute_moment_matrix()
        return self._compute_reduction(points, P, k)

    def _compute_reduction_torch(self, points, P, k):
        ker_vec = -gauss_kernel(rel_differences(points, self.support), k + 1, self.sigma)
        ker_vec = ker_vec.reshape((points.shape[0], self.support.shape[0]) + tuple(ker_vec.shape[1:]))
        return torch.tensordot(torch.transpose(torch.tensordot(torch.eye(self.dim, device=self.device), ker_vec, dims=0), 0, 2), P, dims=([2, 3, 4], [1, 0, 2]))

    def _compute_reduction_keops(self, points, P, k):
        if k == 0:
            kernel_formula = "S*Exp(-S*SqNorm2(x - y)/IntCst(2))*(x - y)"
            formula = "TensorDot({kernel_formula}, p, Ind({dim}), Ind({dim}, {dim}), Ind(0), Ind(1))".format(kernel_formula=kernel_formula, dim=self.dim)
            alias = ["x=Vi("+str(self.dim)+")", "y=Vj("+str(self.dim)+")", "p=Vj("+str(self.dim*self.dim)+")", "S=Pm(1)"]
            reduction = Genred(formula, alias, reduction_op='Sum', axis=1, dtype=str(points.dtype).split(".")[1])
            return reduction(points, self.support, P.reshape(-1, self.dim*self.dim), self._keops_sigma, backend=self._keops_backend).reshape(-1, self.dim)

        if k == 1:
            kernel_formula = "-S*Exp(-S*SqNorm2(x - y)/IntCst(2))*(S*TensorDot(x-y, x-y, Ind({dim}), Ind({dim}), Ind(), Ind())-eye)".format(dim=self.dim)
            formula = "TensorDot({kernel_formula}, p, Ind({dim}, {dim}), Ind({dim}, {dim}), Ind(1),Ind(1))".format(kernel_formula=kernel_formula, dim=self.dim)
            alias = ["x=Vi("+str(self.dim)+")", "y=Vj("+str(self.dim)+")", "p=Vj("+str(self.dim*self.dim)+")", "eye=Pm("+str(self.dim*self.dim)+")", "S=Pm(1)"]
            reduction = Genred(formula, alias, reduction_op='Sum', axis=1, dtype=str(points.dtype).split(".")[1])
            return reduction(points, self.support, P.reshape(-1, self.dim*self.dim), torch.eye(self.dim, dtype=self.support.dtype, device=self.device).flatten(), self._keops_sigma, backend=self._keops_backend).reshape(-1, self.dim, self.dim).transpose(1, 2).contiguous()

        else:
            raise RuntimeError("StructuredField_pm.__call__(): keops computation not supported for order k =", k)

    def _compute_moment_matrix(self):
        raise NotImplementedError()


class StructuredField_p(StructuredField_pm):
    def __init__(self, support, moments, sigma, device=None, backend=None):
        super().__init__(support ,moments, sigma, device, backend)

    def _compute_moment_matrix(self):
        return (self.moments + torch.transpose(self.moments, 1, 2))/2.


class StructuredField_m(StructuredField_pm):
    def __init__(self, support, moments, sigma, device=None, backend=None):
        super().__init__(support ,moments, sigma, device, backend)

    def _compute_moment_matrix(self):
        return (self.moments - torch.transpose(self.moments, 1, 2))/2.

