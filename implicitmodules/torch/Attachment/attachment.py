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

    def __call__(self, source):
        return self.__weight*self.loss(source)

    def __set_target(self, target):
        self.__target = target

        self.precompute()

    def __get_target(self):
        return self.__target

    target = property(__get_target, __set_target)

    def precompute(self):
        pass

    def loss(self, source):
        raise NotImplementedError


class CompoundAttachment(Attachment):
    def __init__(self, attachments, weight=1.):
        assert isinstance(attachments, Iterable)

        self.__attachments = attachments
        super().__init__(weight)

    def __set_target(self, targets):
        for attachment, target in zip(self.__attachments, targets):
            attachment.target = target

    def __get_target(self):
        return self.__attachments[0]

    target = property(__get_target, __set_target)

    def loss(self, source):
        return sum([attachment.loss(source) for attachment in self.__attachments])


class EnergyAttachment(Attachment):
    """Energy Distance between two sampled probability measures."""
    def __init__(self, weight=1.):
        super().__init__(weight)

    def loss(self, source):
        if isinstance(source, torch.Tensor):
            x_i = source
            a_i = torch.ones(source.shape[0])
            y_j = self.target
            b_j = torch.ones(self.target.shape[0])
        else:
            x_i, a_i = source
            y_j, b_j = self.target

        K_xx = -distances(x_i, x_i)
        K_xy = -distances(x_i, y_j)
        K_yy = -distances(y_j, y_j)
        return .5*scal(a_i, torch.mm(K_xx, a_i.view(-1, 1))) - scal(a_i, torch.mm(K_xy, b_j.view(-1, 1))) + .5*scal(b_j, torch.mm(K_yy, b_j.view(-1, 1)))

class L2NormAttachment(Attachment):
    """L2 norm distance between two measures."""
    def __init__(self, weight=1.):
        super().__init__(weight)

    def loss(self, source):
        return torch.dist(source, self.target)


class GeomlossAttachment(Attachment):
    def __init__(self, weight=1., **kwargs):
        super().__init__(weight)
        self.__geomloss = geomloss.SamplesLoss(**kwargs)

    def loss(self, source):
        return self.__geomloss(source[1], source[0], self.target[1], self.target[0])


class EuclideanPointwiseDistanceAttachment(Attachment):
    def __init__(self, weight=1.):
        super().__init__(weight)

    def loss(self, source):
        return torch.sum(torch.norm(source-self.target, dim=1))

