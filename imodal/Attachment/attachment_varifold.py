from collections import Iterable

import torch
from pykeops.torch import Genred

from implicitmodules.torch.Attachment.attachment import Attachment
from implicitmodules.torch.Utilities.meshutils import close_shape, compute_centers_normals_lengths
from implicitmodules.torch.Utilities.compute_backend import get_compute_backend
from implicitmodules.torch.Kernels import K_xy


class VarifoldAttachmentBase(Attachment):
    def __init__(self, sigmas, weight=1.):
        assert isinstance(sigmas, Iterable)
        super().__init__(weight=weight)
        self.__sigmas = sigmas

    def __str__(self):
        outstr = str(super().__str__()) + "\n"
        outstr += "  Sigmas={sigmas}".format(sigmas=self.__sigmas)
        return outstr

    @property
    def dim(self):
        raise NotImplementedError

    @property
    def sigmas(self):
        """
        scales of the varifods.
        """
        return self.__sigmas

    def cost_varifold(self, source, target, sigma):
        raise NotImplementedError

    def loss(self, source, target):
        return sum([self.cost_varifold(source, target, sigma) for sigma in self.__sigmas])


class VarifoldAttachment2D(VarifoldAttachmentBase):
    def __init__(self, sigmas, weight=1.):
        super().__init__(sigmas, weight)

    @property
    def dim(self):
        return 2


class VarifoldAttachment2D_Torch(VarifoldAttachment2D):
    def cost_varifold(self, source, target, sigma):
        source = source[0]
        target = target[0]

        def dot_varifold(x, y, sigma):
            cx, cy = close_shape(x), close_shape(y)
            nx, ny = x.shape[0], y.shape[0]

            vx, vy = cx[1:nx + 1, :] - x, cy[1:ny + 1, :] - y
            mx, my = (cx[1:nx + 1, :] + x) / 2, (cy[1:ny + 1, :] + y) / 2

            xy = torch.tensordot(torch.transpose(torch.tensordot(mx, my, dims=0), 1, 2), torch.eye(2, dtype=source.dtype, device=source.device))

            d2 = torch.sum(mx * mx, dim=1).reshape(nx, 1).repeat(1, ny) + torch.sum(my * my, dim=1).repeat(nx, 1) - 2. * xy

            kxy = torch.exp(-d2 / (2. * sigma ** 2))

            vxvy = torch.tensordot(torch.transpose(torch.tensordot(vx, vy, dims=0), 1, 2), torch.eye(2, dtype=source.dtype, device=source.device)) ** 2

            nvx = torch.norm(vx, dim=1)
            nvy = torch.norm(vy, dim=1)

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
        vertices_source, faces_source = source
        vertices_target, faces_target = target

        centers_source, normals_source, lengths_source = compute_centers_normals_lengths(vertices_source, faces_source)
        centers_target, normals_target, lengths_target = compute_centers_normals_lengths(vertices_target, faces_target)
        normalized_source = normals_source/lengths_source
        normalized_target = normals_target/lengths_target

        return self.varifold_scalar_product(centers_target, centers_target, lengths_target, lengths_target, normalized_target, normalized_target, sigma)\
            + self.varifold_scalar_product(centers_source, centers_source, lengths_source, lengths_source, normalized_source, normalized_source, sigma)\
            - 2. * self.varifold_scalar_product(centers_source, centers_target, lengths_source, lengths_target, normalized_source, normalized_target, sigma)


class VarifoldAttachment3D_Torch(VarifoldAttachment3D):
    """
    Taken and adapted from https://gitlab.com/icm-institute/aramislab/deformetrica/-/blob/master/deformetrica/core/model_tools/attachments/multi_object_attachment.py
    """
    def __init__(self, sigmas, weight=1.):
        super().__init__(sigmas, weight=weight)

    def __convolve(self, x, y, p, sigma):
        def binet(x):
            return x * x

        K = K_xy(x[0], y[0], sigma)
        return torch.mm(K * binet(torch.mm(x[1], y[1].T)), p)

    def varifold_scalar_product(self, x, y, lengths_x, lengths_y, normalized_x, normalized_y, sigma):
        return torch.dot(lengths_x.view(-1), self.__convolve((x, normalized_x), (y, normalized_y), lengths_y.view(-1, 1), sigma).view(-1))


class VarifoldAttachment3D_KeOps(VarifoldAttachment3D):
    def __init__(self, sigmas, weight=1.):
        super().__init__(sigmas, weight)

        self.__K = None
        self.__oos2 = {}

    def varifold_scalar_product(self, x, y, lengths_x, lengths_y, normalized_x, normalized_y, sigma):
        if self.__K is None:
            formula = "Exp(-S*SqNorm2(x - y)/IntCst(2)) * Square((u|v))*p"
            alias = ["x=Vi(3)",
                     "y=Vj(3)",
                     "u=Vi(3)",
                     "v=Vj(3)",
                     "p=Vj(1)",
                     "S=Pm(1)"]

            self.__K = Genred(formula, alias, reduction_op='Sum', axis=1, dtype=str(x.dtype).split('.')[1])
            self.__keops_backend = 'CPU'
            if str(x.device) != 'cpu':
                self.__keops_backend = 'GPU'

        if sigma not in self.__oos2:
            self.__oos2[sigma] = torch.tensor([1./sigma/sigma], device=x.device, dtype=x.dtype)

        return (lengths_x * self.__K(x, y, normalized_x, normalized_y, lengths_y, self.__oos2[sigma], backend=self.__keops_backend)).sum()


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

