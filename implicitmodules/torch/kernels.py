import numpy as np
import torch
# from pykeops.torch import Kernel, kernel_product, Genred
# from pykeops.torch.kernel_product.formula import *


def scal(x, y):
    """Scalar product between two vectors."""
    return torch.dot(x.view(-1), y.view(-1))


# TODO: find a better name
def rel_differences(x, y):
    return (x.unsqueeze(1) - y.unsqueeze(0)).view(-1, 2)


def distances(x, y):
    """Matrix of distances, C_ij = |x_i - y_j|."""
    return (x.unsqueeze(1) - y.unsqueeze(0)).norm(p=2, dim=2)


def sqdistances(x, y):
    """Matrix of squared distances, C_ij = |x_i-y_j|^2."""
    return ((x.unsqueeze(1) - y.unsqueeze(0))**2).sum(2)


def K_xx(x, sigma):
    """Kernel matrix for x."""
    return (-0.5 * sqdistances(x, x)/sigma**2).exp()


def K_xy(x, y, sigma):
    """Kernel matrix between x and y."""
    return (-0.5 * sqdistances(x, y)/sigma**2).exp()


def gauss_kernel(x, k, sigma):
    if k == 0:
        return (-torch.sum((x/sigma)**2/2, dim=1)).exp()
    if k == 1:
        k_0 = gauss_kernel(x, 0, sigma)
        return -k_0.view(-1, 1).repeat(1, 2) * x / sigma / sigma
    if k == 2:
        k_0 = gauss_kernel(x, 0, sigma)
        return (k_0.view(-1, 1, 1).repeat(1, 2, 2) * \
                (-torch.eye(2).repeat(x.shape[0], 1, 1) + \
                 torch.einsum('ki, kj->kij', x / sigma, x / sigma))) / sigma / sigma
    if k == 3:
        k_0 = gauss_kernel(x, 0, sigma)
        k_2 = gauss_kernel(x, 2, sigma) * sigma**2
        return (-torch.einsum('kij, kl->kijl', k_2, x / sigma) + \
                k_0.view(-1, 1, 1, 1).repeat(1, 2, 2, 2) * \
                (torch.transpose(torch.tensordot(x / sigma, torch.eye(2), dims=0), 1, 2) + \
                 torch.tensordot(x / sigma, torch.eye(2), dims=0))) / sigma / sigma / sigma


def eta():
    return torch.tensor([[[1., 0., 0.], [0., 1/np.sqrt(2), 0.]],
                         [[0., 1/np.sqrt(2), 0.], [0., 0., 1.]]], dtype=torch.get_default_dtype())


def compute_sks(x, sigma, order):
    N = x.shape[0]
    if order == 0:
        return torch.einsum('ij, kl->ijkl', gauss_kernel(rel_differences(x, x), 0, sigma).view(N, N), torch.eye(2)).permute([0, 2, 1, 3]).contiguous().view(2*N, 2*N)
    elif order == 1:
        A = torch.tensordot(-gauss_kernel(rel_differences(x, x), 2, sigma), torch.eye(2), dims=0)
        K = torch.tensordot(torch.transpose(A, 2, 3), eta())
        K = torch.tensordot(K, eta(), dims=([1, 2], [0, 1]))
        return K.view(N, N, 3, 3).contiguous().permute([0, 2, 1, 3]).contiguous().view(3*N, 3*N)
    else:
        raise NotImplementedError

# def keops_gauss_kernel(sigma, dtype=torch.float32):
#     p = torch.tensor([1/sigma/sigma])
#     def K(x, y, b):
#         d = 2
#         formula = 'Exp(-p*SqDist(x, y))*b'
#         variables = ['x = Vx('+str(d)+')',
#                      'y = Vy('+str(d)+')',
#                      'b = Vy('+str(d)+')',
#                      'p = Pm(1)']

#         cuda_type = "float32"
#         if(dtype is torch.float64):
#             cuda_type = "float64"

#         my_routine = Genred(formula, variables, reduction_op='Sum', axis=1, cuda_type=cuda_type)
#         return my_routine(x, y, b, p, backend="auto")
#     return K

