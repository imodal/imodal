import math

import torch

from imodal.Kernels.kernels import gauss_kernel, rel_differences


def A(dim, dtype, device):
    cst = -math.sqrt(2.)/2.
    if dim == 2:
        A = torch.zeros(2, 2, 3, 3, device=device, dtype=dtype)

        # A[0, 0]
        A[0, 0, 0, 0] = 1.
        A[0, 0, 1, 1] = 2.

        # A[1, 1]
        A[1, 1, 1, 1] = 2.
        A[1, 1, 2, 2] = 1.

        # A[0, 1]
        A[0, 1, 0, 1] = cst
        A[0, 1, 1, 0] = cst
        A[0, 1, 1, 2] = cst
        A[0, 1, 2, 1] = cst

        # A[1, 0]
        A[1, 0] = A[0, 1]

        return A
    elif dim == 3:
        A = torch.zeros(3, 3, 6, 6, device=device, dtype=dtype)

        # A[0, 0]
        A[0, 0, 0, 0] = 1.
        A[0, 0, 3, 3] = 2.
        A[0, 0, 4, 4] = 2.

        # A[1, 1]
        A[1, 1, 1, 1] = 1.
        A[1, 1, 3, 3] = 2.
        A[1, 1, 5, 5] = 2.

        # A[2, 2]
        A[2, 2, 2, 2] = 1.
        A[2, 2, 4, 4] = 2.
        A[2, 2, 5, 5] = 2.

        # A[0, 1]
        A[0, 1, 0, 3] = cst
        A[0, 1, 3, 0] = cst
        A[0, 1, 1, 3] = cst
        A[0, 1, 3, 1] = cst
        A[0, 1, 4, 5] = -1.
        A[0, 1, 5, 4] = -1.

        # A[1, 0]
        A[1, 0] = A[0, 1]

        # A[0, 2]
        A[0, 2, 0, 4] = cst
        A[0, 2, 4, 0] = cst
        A[0, 2, 4, 2] = cst
        A[0, 2, 2, 4] = cst
        A[0, 2, 4, 5] = -1.
        A[0, 2, 5, 4] = -1.

        # A[1, 2]
        A[1, 2, 1, 5] = cst
        A[1, 2, 5, 1] = cst
        A[1, 2, 2, 5] = cst
        A[1, 2, 5, 2] = cst
        A[1, 2, 3, 4] = -1.
        A[1, 2, 4, 3] = -1.

        # A[2, 1]
        A[2, 1] = A[1, 2]

        return A
    else:
        raise NotImplementedError


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

        return eta.to(device=device)
    else:
        raise NotImplementedError


def compute_sks(x, sigma, order):
    device = x.device
    dim = x.shape[1]
    sym_dim = int(dim * (dim + 1) / 2)
    N = x.shape[0]

    if order == 0:
        return torch.einsum('ij, kl->ijkl', gauss_kernel(rel_differences(x, x), 0, sigma).view(N, N), torch.eye(dim, device=device)).permute([0, 2, 1, 3]).contiguous().view(dim * N, dim * N)
    elif order == 1:
        A = torch.tensordot(-gauss_kernel(rel_differences(x, x), 2, sigma), torch.eye(dim, device=device), dims=0)
        K = torch.tensordot(torch.transpose(A, 2, 3), eta(dim, device=device))
        K = torch.tensordot(K, eta(dim, device=device), dims=([1, 2], [0, 1]))
        return K.view(N, N, sym_dim, sym_dim).contiguous().permute([0, 2, 1, 3]).contiguous().view(sym_dim * N, sym_dim * N)
    else:
        raise NotImplementedError

