import math
import copy
from collections import Iterable

import torch

from implicitmodules.torch.Utilities.usefulfunctions import grid2vec


class AABB:
    """Class used to represent an Axis Aligned Bounding Box"""
    def __init__(self, xmin=0., xmax=0., ymin=0., ymax=0.):
        self.__xmin = xmin
        self.__xmax = xmax
        self.__ymin = ymin
        self.__ymax = ymax

    @classmethod
    def build_from_points(cls, points):
        """Compute the AABB from a [4, dim] tensor."""

        return cls(points[:, 0].min(), points[:, 0].max(),
                   points[:, 1].min(), points[:, 1].max())

    def __getitem__(self, key):
        return self.get_list()[key]

    def get_list(self):
        """Returns the AABB as a list, [xmin, xmax, ymin, ymax]."""
        return [self.__xmin, self.__xmax, self.__ymin, self.__ymax]

    @property
    def xmin(self):
        return self.__xmin

    @property
    def ymin(self):
        return self.__ymin

    @property
    def xmax(self):
        return self.__xmax

    @property
    def ymax(self):
        return self.__ymax

    @property
    def width(self):
        return self.__xmax - self.__xmin

    @property
    def height(self):
        return self.__ymax - self.__ymin

    @property
    def center(self):
        return torch.tensor([0.5*(self.__xmax+self.__xmin), 0.5*(self.__ymax+self.__ymin)])

    @property
    def area(self):
        return (self.__xmax - self.__xmin) * (self.ymax - self.ymin)

    def fill_random(self, N, dtype=None, device=None):
        """Returns a [N, dim] vector of Poisson distributed points inside the area enclosed by the AABB."""
        return torch.tensor([self.width, self.height], dtype=dtype, device=device) * torch.rand(N, 2) + torch.tensor([self.xmin, self.ymin], dtype=dtype, device=device)

    def fill_random_density(self, density, dtype=None, device=None):
        return self.fill_random(int(self.area*density), dtype=dtype, device=device)

    def fill_uniform(self, spacing, dtype=None, device=None):
        x, y = torch.meshgrid([torch.arange(self.xmin, self.xmax, step=spacing, dtype=dtype),
                               torch.arange(self.ymin, self.ymax, step=spacing, dtype=dtype)])

        return grid2vec(x, y).to(device=device)

    def fill_uniform_density(self, density, dtype=None, device=None):
        return self.fill_uniform(1./math.sqrt(density), dtype=dtype, device=device)

    def is_inside(self, points):
        return torch.where((points[:, 0] >= self.__xmin) &
                           (points[:, 0] <= self.xmax) &
                           (points[:, 1] >= self.__ymin) &
                           (points[:, 1] <= self.ymax),
                           torch.tensor([1.]), torch.tensor([0.])).type(dtype=torch.bool)

    def squared_(self):
        """Squares the AABB."""
        enlarge = .1
        xmiddle = (self.__xmin + self.__xmax) / 2
        ymiddle = (self.__ymin + self.__ymax) / 2
    
        diam = max(abs(self.__xmin - self.__xmax) / 2, abs(self.__ymin - self.__ymax) / 2) * (1 + enlarge)
        self.__xmin = xmiddle - diam
        self.__ymin = ymiddle - diam
        self.__xmax = xmiddle + diam
        self.__ymax = ymiddle + diam

    def squared(self):
        """Returns a squared AABB."""
        out = copy.copy(self)
        
        enlarge = .1
        xmiddle = (self.__xmin + self.__xmax) / 2.
        ymiddle = (self.__ymin + self.__ymax) / 2.
    
        diam = max(abs(self.__xmin - self.__xmax) / 2, abs(self.__ymin - self.__ymax) / 2) * (1 + enlarge)
        out.__xmin = xmiddle - diam
        out.__ymin = ymiddle - diam
        out.__xmax = xmiddle + diam
        out.__ymax = ymiddle + diam

        return out

    def scale_(self, factor):
        """
        TODO: Add documentation.
        """
        factors = []
        if isinstance(factor, Iterable):
            factors = factor
        else:
            factors.append(factor)
            factors.append(factor)

        center = self.center
        self.__xmin = factors[0]*(self.__xmin - center[0]) + center[0]
        self.__xmax = factors[0]*(self.__xmax - center[0]) + center[0]
        self.__ymin = factors[1]*(self.__ymin - center[1]) + center[1]
        self.__ymax = factors[1]*(self.__ymax - center[1]) + center[1]

    def scale(self, factor):
        """
        TODO: Add documentation.
        """
        factors = []
        if isinstance(factor, Iterable):
            factors = factor
        else:
            factors.append(factor)
            factors.append(factor)

        out = copy.copy(self)
        center = self.center
        out.__xmin = factors[0]*(self.__xmin - center[0]) + center[0]
        out.__xmax = factors[0]*(self.__xmax - center[0]) + center[0]
        out.__ymin = factors[1]*(self.__ymin - center[1]) + center[1]
        out.__ymax = factors[1]*(self.__ymax - center[1]) + center[1]

        return out

