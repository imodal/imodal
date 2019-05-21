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
        sigma2 = sigma * sigma
        return -k_0.view(-1, 1).repeat(1, 2) * x / sigma2
    if k == 2:
        sigma2 = sigma * sigma
        k_0 = gauss_kernel(x, 0, sigma)
        return (k_0.view(-1, 1, 1).repeat(1, 2, 2) * (-torch.eye(2).repeat(x.shape[0], 1, 1)
                                                      + torch.einsum('ki, kj->kij', x, x) / sigma2)) / sigma2
    if k == 3:
        sigma2 = sigma * sigma
        sigma3 = sigma * sigma * sigma
        k_0 = gauss_kernel(x, 0, sigma)
        k_2 = gauss_kernel(x, 2, sigma) * sigma2
        return (-torch.einsum('kij, kl->kijl', k_2, x / sigma) + k_0.view(-1, 1, 1, 1).repeat(1, 2, 2, 2)
                * (torch.transpose(torch.tensordot(x / sigma, torch.eye(2), dims=0), 1, 2)
                   + torch.tensordot(x / sigma, torch.eye(2), dims=0))) / sigma3

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

