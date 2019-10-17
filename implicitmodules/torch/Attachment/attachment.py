from collections import Iterable

import geomloss
import torch

from implicitmodules.torch.Kernels.kernels import distances, scal, sqdistances, K_xy


class Attachment:
    def __init__(self, weight=1.):
        self.__weight = weight

    @property
    def weight(self):
        return self.__weight

    def __call__(self, x, y):
        return self.__weight*self.loss(x, y)

    def loss(self, x, y):
        raise NotImplementedError


class CompoundAttachment(Attachment):
    def __init__(self, attachment_list, weight=1.):
        assert isinstance(attachment_list, Iterable)

        self.__attachment_list = attachment_list
        super().__init__(weight)

    def loss(self, x, y):
        return sum([attachment.loss(x, y) for attachment in self.__attachment_list])


class EnergyAttachment(Attachment):
    """Energy Distance between two sampled probability measures."""
    def __init__(self, weight=1.):
        super().__init__(weight)

    def loss(self, x, y):
        x_i, a_i = x
        y_j, b_j = y
        if a_i is None:
            a_i = torch.ones(x_i.shape[0])
            b_j = torch.ones(y_j.shape[0])
        K_xx = -distances(x_i, x_i)
        K_xy = -distances(x_i, y_j)
        K_yy = -distances(y_j, y_j)
        return .5*scal(a_i, torch.mm(K_xx, a_i.view(-1, 1))) - scal(a_i, torch.mm(K_xy, b_j.view(-1, 1))) + .5*scal(b_j, torch.mm(K_yy, b_j.view(-1, 1)))

class L2NormAttachment(Attachment):
    """L2 norm distance between two measures."""
    def __init__(self, weight=1.):
        super().__init__(weight)

    def loss(self, x, y):
        return torch.dist(a, b)


class GeomlossAttachment(Attachment):
    def __init__(self, weight=1., **kwargs):
        super().__init__(weight)
        self.__geomloss = geomloss.SamplesLoss(**kwargs)

    def loss(self, x, y):
        return self.__geomloss(x[1], x[0], y[1], y[0])


class EuclideanPointwiseDistanceAttachment(Attachment):
    def __init__(self, weight=1.):
        super().__init__(weight)

    def loss(self, x, y):
        return torch.sum(torch.norm(x[0]-y[0], dim=1))

