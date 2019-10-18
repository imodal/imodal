from collections import Iterable

import torch
from pykeops.torch import Kernel, kernel_product, Genred

from implicitmodules.torch.Attachment.attachment import Attachment
from implicitmodules.torch.Utilities.meshutils import close_shape, compute_centers_normals_lengths
from implicitmodules.torch.Utilities.compute_backend import get_compute_backend
from implicitmodules.torch.Kernels import K_xy


class VarifoldAttachmentBase(Attachment):
    def __init__(self, sigmas, weight=1.):
        assert isinstance(sigmas, Iterable)
        super().__init__(weight=weight)
        self.__sigmas = sigmas

    @property
    def dim(self):
        raise NotImplementedError

    @property
    def sigmas(self):
        return self.__sigmas

    def cost_varifold(self, source, target, sigma):
        raise NotImplementedError

    def loss(self, source, target):
        return sum([self.cost_varifold(source, target, sigma) for sigma in self.__sigmas])


class VarifoldAttachment2D(Attachment):
    def __init__(self, sigmas, weight=1.):
        super().__init__(sigmas, weight)

    @property
    def dim(self):
        return 2


class VarifoldAttachment2D_Torch(VarifoldAttachmentBase):
    def cost_varifold(self, source, target, sigma):
        def dot_varifold(x, y, sigma):
            cx, cy = close_shape(x), close_shape(y)
            nx, ny = x.shape[0], y.shape[0]

            vx, vy = cx[1:nx + 1, :] - x, cy[1:ny + 1, :] - y
            mx, my = (cx[1:nx + 1, :] + x) / 2, (cy[1:ny + 1, :] + y) / 2

            xy = torch.tensordot(torch.transpose(torch.tensordot(mx, my, dims=0), 1, 2), torch.eye(2))

            d2 = torch.sum(mx * mx, dim=1).reshape(nx, 1).repeat(1, ny) + torch.sum(my * my, dim=1).repeat(nx, 1) - 2 * xy

            kxy = torch.exp(-d2 / (2 * sigma ** 2))

            vxvy = torch.tensordot(torch.transpose(torch.tensordot(vx, vy, dims=0), 1, 2), torch.eye(2)) ** 2

            nvx = torch.sqrt(torch.sum(vx * vx, dim=1))
            nvy = torch.sqrt(torch.sum(vy * vy, dim=1))

            mask = vxvy > 0

            return torch.sum(kxy[mask] * vxvy[mask] / (torch.tensordot(nvx, nvy, dims=0)[mask]))

        return dot_varifold(source, source, sigma) + dot_varifold(target, target, sigma) - 2 * dot_varifold(source, target, sigma)


class VarifoldAttachment2D_KeOps(VarifoldAttachment2D):
    def __init__(self, sigmas, weight=1.):
        super().__init__(sigmas, weight=weight)

    def cost_varifold(self, source, target, sigma):
        raise NotImplementedError


class VarifoldAttachment3D(VarifoldAttachmentBase):
    def __init__(self, sigmas, weight=1.):
        super().__init__(sigmas, weight=weight)

    def dim(self):
        return 3

    def cost_varifold(self, source, target, sigma):
        vertices_x, faces_x = source
        vertices_y, faces_y = target

        centers_x, normals_x, lengths_x = compute_centers_normals_lengths(vertices_x, faces_x)
        centers_y, normals_y, lengths_y = compute_centers_normals_lengths(vertices_y, faces_y)
        normalized_x, normalized_y = normals_x / lengths_x, normals_y / lengths_y

        return self.varifold_scalar_product(centers_x, centers_x, lengths_x, lengths_x, normalized_x, normalized_x, sigma) \
            + self.varifold_scalar_product(centers_y, centers_y, lengths_y, lengths_y, normalized_y, normalized_y, sigma) \
            - 2. * self.varifold_scalar_product(centers_x, centers_y, lengths_x, lengths_y, normalized_x, normalized_y, sigma)


class VarifoldAttachment3D_Torch(VarifoldAttachment3D):
    def __init__(self, sigmas, weight=1.):
        super().__init__(sigmas, weight=weight)

    def __convolve(self, x, y, p, sigma):
        def binet(x):
            return x * x

        K = K_xy(x[0], y[0], sigma)
        return torch.mm(K * binet(torch.mm(x[1], y[1].T)), p)

    def varifold_scalar_product(self, x, y, lengths_x, lengths_y, normalized_x, normalized_y, sigma):
        return torch.dot(lengths_x.view(-1), self.__convolve((x, normalized_x), (y, normalized_y), lengths_y.view(-1, 1), sigma).view(-1))


#'id': Kernel('gaussian(x,y) * linear(u,v)**2'),
class VarifoldAttachment3D_KeOps(VarifoldAttachment3D):
    def __init__(self, sigmas, weight =1.):
        super().__init__(sigmas, weight)

        # Keops kernels are stored here
        self.__K = {}

    def varifold_scalar_product(self, x, y, lengths_x, lengths_y, normalized_x, normalized_y, sigma):
        if sigma not in self.__K:
            def GaussLinKernel(sigma):
                def K(x, y, u, v, b):
                    params = {
                        'id': Kernel('gaussian2(x,y) * linear(u,v)**2'),
                        'gamma': (1 / (sigma * sigma), None),
                        'backend': 'auto'
                    }
                    return kernel_product(params, (x, u), (y, v), b, dtype=str(x[0].dtype).split('.')[1])
                return K

            self.__K[sigma] = GaussLinKernel(torch.tensor([sigma], dtype=x[0].dtype, device=x[0].device))

        return (lengths_x * self.__K[sigma](x, y, normalized_x, normalized_y, lengths_y)).sum()


def VarifoldAttachment(dim, sigmas, weight=1., backend=None):
    if backend is None:
        backend = get_compute_backend()

    # Ugly manual switch between dimensions and compute backends
    if dim == 2:
        if backend == 'torch':
            return VarifoldAttachment2D_Torch(sigmas, weight=weight)
        elif backend == 'keops':
            return VarifoldAttachment2D_KeOps(sigmas, weight=weight)
        else:
            raise RuntimeError("VarifoldAttachment: Unrecognized backend {backend}!".format(backend=backend))
    elif dim == 3:
        if backend == 'torch':
            return VarifoldAttachment3D_Torch(sigmas, weight=weight)
        elif backend == 'keops':
            return VarifoldAttachment3D_KeOps(sigmas, weight=weight)
        else:
            raise RuntimeError("VarifoldAttachment: Unrecognized backend {backend}!".format(backend=backend))
    else:
        raise RuntimeError("VarifoldAttachment: Dimension {dim} not supported!".format(dim=dim))

