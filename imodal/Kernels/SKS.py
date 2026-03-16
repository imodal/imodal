import math

import torch
from pykeops.torch import Pm, Vj, Vi

from imodal.Kernels.kernels import gauss_kernel, rel_differences


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


def compute_sks(x, sigma, order, u=None):
    device = x.device
    dim = x.shape[1]
    sym_dim = int(dim * (dim + 1) / 2)
    N = x.shape[0]

    if order == 0:
        return torch.einsum('ij, kl -> ijkl',
                            gauss_kernel(rel_differences(x, x), 0, sigma).view(N, N),
                            torch.eye(dim, device=device)
                            ).permute([0, 2, 1, 3]).contiguous().view(dim * N, dim * N)
    elif order == 1:
        K = torch.einsum('opij, nij -> nop',
                          u,                                              # Dimension: sym_dim * sym_dim * dim * dim
                          -gauss_kernel(rel_differences(x, x), 2, sigma)) # Dimension: N**2 * dim, dim

        return K.view(N, N, sym_dim, sym_dim).contiguous().permute([0, 2, 1, 3]).contiguous().view(sym_dim * N, sym_dim * N)
    else:
        raise NotImplementedError
    
    
def compute_sks_keops(x, sigma, order, u=None):

    dtype = x.dtype
    device = x.device
    dim = x.shape[1]
    sym_dim = int(dim * (dim + 1) / 2)
    N = x.shape[0]

    if order == 1:
        EYE = Pm(torch.eye(dim, device=device).flatten())  # Dimension: 1 * 1 * dim * dim
        Ss = Pm(torch.tensor([1. / sigma / sigma], device=device))  # Dimension: 1 * 1 * 1
        Xis = Vi(x.repeat(1, sym_dim).reshape(-1, dim))  # Dimension: (N * s) \times dim
        Yjs = Vj(x.repeat(1, sym_dim).reshape(-1, dim))  # Dimension: (s * N) \times dim

        GaussKernel = (-Ss * ((Xis - Yjs) ** 2).sum(-1) / 2).exp()
        R = - GaussKernel * (Ss * (Xis - Yjs).tensorprod(Xis - Yjs) - EYE) * Ss  # Dimension: (Ns * Ns) * dim * dim
        U = Pm(u.flatten())  # Dimension: (N * sym_dim * N * syn_dim) * sym_dim * sym_dim * dim * dim

        I = Vi(torch.arange(sym_dim, dtype=dtype, device=device).repeat(N).reshape(N * sym_dim, 1))
        Ii = I.one_hot(sym_dim)  # Dimension: (N * sym_dim) * sym_dim
        J = Vj(torch.arange(sym_dim, dtype=dtype, device=device).repeat(N).reshape(N * sym_dim, 1))
        Jj = J.one_hot(sym_dim)  # Dimension: (N * sym_dim) * sym_dim

        V = Ii.keops_tensordot(Jj,
                               [sym_dim],
                               [sym_dim],
                               [],
                               [])
        V = V.keops_tensordot(U,
                              [sym_dim, sym_dim],
                              [sym_dim, sym_dim, dim, dim],
                              [0, 1],
                              [0, 1])  # Dimension: (N * sym_dim * N * sym_dim) * dim * dim

        W = R.keops_tensordot(V,
                              [dim, dim],
                              [dim, dim],
                              [0, 1],
                              [0, 1])  # Dimension: (Ns * Ns)

        TT = Vj(0, 1)
        SKS2 = W * TT


    return SKS2
