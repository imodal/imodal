import math

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
    
    def fill_random(self, N):
        """Returns a [N, dim] vector of Poisson distributed points inside the area enclosed by the AABB."""
        return torch.tensor([self.width, self.height]) * torch.rand(N, 2) + torch.tensor([self.xmin, self.ymin])

    def fill_random_density(self, density):
        return self.fill_random(int(self.area*density))

    def fill_uniform(self, spacing):
        x, y = torch.meshgrid([torch.arange(self.xmin, self.xmax, step=spacing),
                               torch.arange(self.ymin, self.ymax, step=spacing)])

        return grid2vec(x, y)

    def fill_uniform_density(self, density):
        return self.fill_uniform(1./math.sqrt(density))

    def is_inside(self, points):
        return torch.where((points[:, 0] >= self.__xmin) &
                           (points[:, 0] <= self.xmax) &
                           (points[:, 1] >= self.__ymin) &
                           (points[:, 1] <= self.ymax),
                           torch.tensor([1.]), torch.tensor([0.])).type(dtype=torch.bool)

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
    def area(self):
        return (self.__xmax - self.__xmin) * (self.ymax - self.ymin)

    def squared(self):
        """Squares the AABB."""
        enlarge = .1
        xmiddle = (self.__xmin + self.__xmax) / 2
        ymiddle = (self.__ymin + self.__ymax) / 2
    
        diam = max(abs(self.__xmin - self.__xmax) / 2, abs(self.__ymin - self.__ymax) / 2) * (1 + enlarge)
        self.__xmin = xmiddle - diam
        self.__ymin = ymiddle - diam
        self.__xmax = xmiddle + diam
        self.__ymax = ymiddle + diam

