import math

import torch

from implicitmodules.torch.Kernels.kernels import gauss_kernel, rel_differences


def eta(dim, device=None):
    cst = 1 / math.sqrt(2)
    if dim == 2:
        return torch.tensor([[[1., 0., 0.], [0., cst, 0.]],
                             [[0., cst, 0.], [0., 0., 1.]]],
                            dtype=torch.get_default_dtype(), device=device)
    elif dim == 3:
        eta = torch.zeros(dim, dim, int(dim*(dim+1)/2))
        for i in range(3):
            eta[i, i, i] = 1.

        eta[0, 1, 3] = cst
        eta[1, 0, 3] = cst
        eta[0, 2, 4] = cst
        eta[2, 0, 4] = cst
        eta[1, 2, 5] = cst
        eta[2, 1, 5] = cst

        return eta
    else:
        raise NotImplementedError


def compute_sks(x, sigma, order):
    device = x.device
    dim = x.shape[1]
    sym_dim = int(dim * (dim + 1) / 2)
    N = x.shape[0]
    if dim == 2:
        if order == 0:
            return torch.einsum('ij, kl->ijkl', gauss_kernel(rel_differences(x, x), 0, sigma).view(N, N), torch.eye(dim, device=device)).permute([0, 2, 1, 3]).contiguous().view(2 * N, 2 * N)
        elif order == 1:
            A = torch.tensordot(-gauss_kernel(rel_differences(x, x), 2, sigma), torch.eye(dim, device=device), dims=0)
            K = torch.tensordot(torch.transpose(A, 2, 3), eta(dim, device=device))
            K = torch.tensordot(K, eta(dim, device=device), dims=([1, 2], [0, 1]))
            return K.view(N, N, 3, 3).contiguous().permute([0, 2, 1, 3]).contiguous().view(3 * N, 3 * N)
        else:
            raise NotImplementedError
    elif dim == 3:
        if order == 0:
            return torch.einsum('ij, kl->ijkl', gauss_kernel(rel_differences(x, x), 0, sigma).view(N, N), torch.eye(dim, device=device)).permute([0, 2, 1, 3]).contiguous().view(dim * N, dim * N)
        elif order == 1:
            A = torch.tensordot(-gauss_kernel(rel_differences(x, x), 2, sigma), torch.eye(dim, device=device), dims=0)
            K = torch.tensordot(torch.transpose(A, 2, 3), eta(dim, device=device))
            K = torch.tensordot(K, eta(dim, device=device), dims=([1, 2], [0, 1]))
            return K.view(N, N, sym_dim, sym_dim).contiguous().permute([0, 2, 1, 3]).contiguous().view(sym_dim * N, sym_dim * N)
        else:
            NotImplementedError
    else:
        raise NotImplementedError
