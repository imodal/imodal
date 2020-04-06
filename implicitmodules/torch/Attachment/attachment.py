from collections import Iterable

import geomloss
import torch

from implicitmodules.torch.Kernels.kernels import distances, scal, sqdistances, K_xy


class Attachment:
    def __init__(self, weight=1.):
        self.__weight = weight

    def __str__(self):
        return "{classname} (weight={weight})".format(classname=self.__class__.__name__, weight=self.weight)

    """
    Weight given to this attachment term. Useful when dealing with compound attachment.
    """
    @property
    def weight(self):
        return self.__weight

    """
    Computes the attachment term between two objects.

    Parameters
    ----------
    source :
        The source term.
    target :
        The target term.

    Returns
    -------
    torch.Tensor
        Value quantifying the attachment between the source and the target.
    """
    def __call__(self, source, target):
        return self.__weight*self.loss(source, target)

    """

    """
    def loss(self, source, target):
        raise NotImplementedError


class CompoundAttachment(Attachment):
    """Compound attachment measure. Can be used to combine different measures together"""
    def __init__(self, attachments, weight=1.):
        assert isinstance(attachments, Iterable)

        self.__attachments = attachments
        super().__init__(weight)

    def loss(self, source, target):
        return sum([attachment.loss(source, target) for attachment in self.__attachments])


class EnergyAttachment(Attachment):
    """Energy Distance between two sampled probability measures."""
    def __init__(self, weight=1.):
        super().__init__(weight)

    def loss(self, source, target):
        if isinstance(source, torch.Tensor):
            x_i = source
            a_i = torch.ones(source.shape[0])
            y_j = target
            b_j = torch.ones(target.shape[0])
        else:
            x_i, a_i = source
            y_j, b_j = target

        K_xx = -distances(x_i, x_i)
        K_xy = -distances(x_i, y_j)
        K_yy = -distances(y_j, y_j)
        return .5*scal(a_i, torch.mm(K_xx, a_i.view(-1, 1))) - scal(a_i, torch.mm(K_xy, b_j.view(-1, 1))) + .5*scal(b_j, torch.mm(K_yy, b_j.view(-1, 1)))

class L2NormAttachment(Attachment):
    """L2 norm distance between two measures."""
    def __init__(self, weight=1.):
        super().__init__(weight)

    def loss(self, source, target):
        return torch.dist(source, target)


# class GeomlossAttachment(Attachment):
#     def __init__(self, weight=1., **kwargs):
#         super().__init__(weight)
#         self.__geomloss = geomloss.SamplesLoss(**kwargs)

#     def loss(self, source, target):
#         return self.__geomloss(source[1], source[0], target[1], target[0])


class EuclideanPointwiseDistanceAttachment(Attachment):
    """Euclidean pointwise distance between two measures."""
    def __init__(self, weight=1.):
        super().__init__(weight)

    def loss(self, source, target):
        return torch.sum(torch.norm(source-target, dim=1))

