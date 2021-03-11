import torch


def scal(x, y):
    """Scalar product between two vectors."""
    return torch.dot(x.view(-1), y.view(-1))


# TODO: find a better name
def rel_differences(x, y):
    return (x.unsqueeze(1) - y.unsqueeze(0)).view(-1, x.shape[1])


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
    device = x.device
    dim = x.shape[1]
    if k == 0:
        return (-torch.sum((x/sigma)**2/2, dim=1)).exp()
    if k == 1:
        k_0 = gauss_kernel(x, 0, sigma)
        sigma2 = sigma * sigma
        return -k_0.view(-1, 1).repeat(1, dim) * x / sigma2
    if k == 2:
        sigma2 = sigma * sigma
        k_0 = gauss_kernel(x, 0, sigma)
        return (k_0.view(-1, 1, 1).repeat(1, dim, dim) * (-torch.eye(dim, device=device).repeat(x.shape[0], 1, 1) + torch.einsum('ki, kj->kij', x, x) / sigma2)) / sigma2
    if k == 3:
        raise NotImplementedError("gauss_kernel(): k >= 3 not supported!")

