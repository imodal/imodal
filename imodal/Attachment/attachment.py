from collections.abc import Iterable

import geomloss
import torch

from imodal.Kernels.kernels import distances, scal


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



class EuclideanPointwiseDistanceAttachment(Attachment):
    """Euclidean pointwise distance between two measures."""
    def __init__(self, weight=1.):
        super().__init__(weight)

    def loss(self, source, target):
        x = source[0]
        y = target[0]
        return torch.sum(torch.norm(x-y, dim=1))


class NullLoss(Attachment):
    def __init__(self):
        super().__init__(0.)

    def loss(self, source, target):
        return torch.tensor(0.)

