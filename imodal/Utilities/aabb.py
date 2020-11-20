import itertools
import math
import copy
from collections import Iterable

import torch
import numpy as np

from implicitmodules.torch.Utilities.usefulfunctions import grid2vec


class AABB:
    dim_prefix = ['x', 'y', 'z']
    
    """Class used to represent an Axis Aligned Bounding Box in 1D, 2D and 3D."""
    def __init__(self, *args,  **kwargs):
        """Constructor
        Values can be filled using three ways:
            - As an iterable (xmin, xmax, ymin, ymax, ...)
            - As two iterables (xmin, ymin, ...), (xmax, ymax, ...)
            - As a dictionary {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, ...)
        """
        assert not (not(args) and not(kwargs))

        if args:
            if len(args) == 2 and isinstance(args[0], Iterable) and isinstance(args[1], Iterable):
                self.__init_from_two_iterables(args[0], args[1])
            else:
                self.__init_from_iterable(args)
        else:
            self.__init_from_kwargs(**kwargs)

    def __init_from_two_iterables(self, kmin, kmax):
        assert len(kmin) == len(kmax)

        for _kmin, _kmax in zip(kmin, kmax):
            if _kmin > _kmax:
                raise ValueError("AABB.__init__(): {kmin} > {kmax}!".format(kmin=_kmin, kmax=_kmax))

        self.__dim = len(kmin)
        self.__kmin = kmin
        self.__kmax = kmax

    def __init_from_iterable(self, iterable):
        assert len(iterable) % 2 == 0

        kmin = tuple(iterable[::2])
        kmax = tuple(iterable[1::2])

        for _kmin, _kmax in zip(kmin, kmax):
            if _kmin > _kmax:
                raise ValueError("AABB.__init__(): {kmin} > {kmax}!".format(kmin=_kmin, kmax=_kmax))

        self.__dim = int(len(iterable)/2)
        self.__kmin = kmin
        self.__kmax = kmax

    def __init_from_kwargs(self, **kwargs):
        self.__dim = 0
        self.__kmin = []
        self.__kmax = []

        for pre in AABB.dim_prefix:
            min_str = pre+'min'
            max_str = pre+'max'
            if (min_str not in kwargs) or (max_str not in kwargs):
                break

            if kwargs[min_str] > kwargs[max_str]:
                raise ValueError("AABB.__init__(): {kmin} > {kmax}!".format(kmin=kwargs[min_str], kmax=kwargs[max_str]))

            self.__kmin.append(kwargs[min_str])
            self.__kmax.append(kwargs[max_str])

        self.__kmin = tuple(self.__kmin)
        self.__kmax = tuple(self.__kmax)
        self.__dim = len(self.__kmin)

    @classmethod
    def build_from_points(cls, points):
        """Build the AABB using points.
        The constructed AABB will enclose the input set of points.

        Parameters
        ----------
        points : torch.Tensor
            The (:math:'N', dim) tensor of points.
        """
        return cls(torch.min(points, dim=0)[0].tolist(), torch.max(points, dim=0)[0].tolist())

    def __str__(self):
        return "Utilities.AABB " + str(self.todict())

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.totuple()[key]
        elif isinstance(key, str):
            return getattr(self, key)
        else:
            raise ValueError("AABB.__getitem__(): key of type {keytype} not understood!".format(keytype=type(key)))

    @property
    def dim(self):
        """ The dimension of the AABB.
        """
        return self.__dim

    def totuple(self):
        """ Returns the AABB as a 2*dim-tuple.

        Returns
        -------
        tuple
            The d-tuple (xmin, xmax, ymin, ymax, ...).
        """
        return tuple(itertools.chain.from_iterable(self.tocouple()))

    def todict(self):
        """ Returns the AABB as a dictionary.

        Returns
        -------
        dict
            The dictionary {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, ...}
        """
        d = dict((pre+'min', kmin) for pre, kmin in zip(AABB.dim_prefix, self.__kmin))
        d.update((pre+'max', kmax) for pre, kmax in zip(AABB.dim_prefix, self.__kmax))
        return d

    def tocouple(self):
        """ Returns the AABB as a dim-tuple of intervals.

        Returns
        -------
        tuple
            dim-tuple of (kmin, kmax) couples.

        """
        return tuple((kmin, kmax) for kmin, kmax in zip(self.__kmin, self.__kmax))

    @property
    def kmin(self):
        """ dim-tuple of the lower boundaries. """
        return self.__kmin

    @property
    def kmax(self):
        """ dim-tuple of the upper boundaries. """
        return self.__kmax

    @property
    def xmin(self):
        return self.__kmin[0]

    @property
    def xmax(self):
        return self.__kmax[0]

    @property
    def ymin(self):
        return self.__kmin[1]
    
    @property
    def ymax(self):
        return self.__kmax[1]

    @property
    def zmin(self):
        return self.__kmin[2]

    @property
    def zmax(self):
        return self.__kmax[2]

    def __length_index(self, index):
        return self.__kmax[index] - self.__kmin[index]

    @property
    def width(self):
        return self.__length_index(0)

    @property
    def height(self):
        return self.__length_index(1)

    @property
    def depth(self):
        return self.__length_index(2)

    @property
    def shape(self):
        """ d-tuple (width, height, ...). """
        return tuple(self.__length_index(i) for i in range(self.__dim))

    @property
    def centers(self):
        return tuple(0.5*(kmax+kmin) for kmax, kmin in zip(self.__kmax, self.__kmin))

    @property
    def length(self):
        assert self.__dim == 1
        return self.width

    @property
    def area(self):
        assert self.__dim == 2
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    @property
    def volume(self):
        assert self.__dim == 2
        Return (self.xmax - self.xmin) * (self.ymax - self.ymin) * (self.zmax - self.zmin)

    def fill_random(self, N, dtype=None, device=None):
        """ Fill the AABB with :math:'N' Poisson distributed points.

        Returns
        -------
        torch.Tensor
            [:math:'N', dim] shaped tensor of the points inside the AABB.
        """
        return torch.tensor(self.shape, dtype=dtype, device=device) * torch.rand(N, self.__dim) + torch.tensor(self.__kmin, dtype=dtype, device=device)

    def fill_random_density(self, density, dtype=None, device=None):
        """ Fill the AABB with points following a :math:'\\lambda'=*density* Poisson law.

        Returns
        -------
        torch.Tensor
            [:math:'N', dim] shaped tensor with :math:'N' the number of points inside the AABB.
        """
        return self.fill_random(int(torch.prod(torch.tensor(self.shape))*density), dtype=dtype, device=device)

    def fill_count(self, counts, dtype=None, device=None):
        """ Fill the AABB uniformly with a set amount of points per dimension.

        Returns
        -------
        torch.Tensor
            [:math:'N', dim] shaped tensor with :math:'N' the number of points inside the AABB.
        """
        assert isinstance(counts, int) or (isinstance(counts, Iterable) and len(counts) == self.__dim)

        if isinstance(counts, int):
            spacing = [counts]*self.__dim

        grids = torch.meshgrid([torch.linspace(kmin, kmax, count, dtype=dtype, device=device) for kmin, kmax, count in zip(self.__kmin, self.__kmax, counts)])
        return grid2vec(*grids)

    def fill_uniform_spacing(self, spacing, dtype=None, device=None):
        """ Fill the AABB uniformly.
        Uses a spacing parameters which represent the length between each points on each axis.

        Returns
        -------
        torch.Tensor
            [:math:'N', dim] shaped tensor with :math:'N' the number of points inside the AABB.
        """
        grids = torch.meshgrid([torch.arange(kmin, kmax, step=spacing, dtype=dtype, device=device) for kmin, kmax in zip(self.__kmin, self.__kmax)])
        return grid2vec(*grids)

    def fill_uniform_density(self, density, dtype=None, device=None):
        """ Fill the AABB uniformly.
        Used a *density* parameter with represent the number of points in a unit dim-cell of the AABB.
        Returns
        -------
        torch.Tensor
            [:math:'N', dim] shaped tensor with :math:'N' the number of points inside the AABB.
        """
        return self.fill_uniform_spacing(1./density**(1./self.__dim), dtype=dtype, device=device)

    def is_inside(self, points):
        # TODO change this horrible horrible hack
        return list(itertools.accumulate([torch.where((points[:, i] >= self.__kmin[i]) & (points[:, i] <= self.__kmax[i]), torch.tensor([1.]), torch.tensor([0.])) for i in range(self.__dim)], lambda x,y: x*y))[-1].to(dtype=torch.bool)


    def squared_(self):
        """Squares the AABB inplace (the center does not move)."""
        if self.__dim == 1:
            return

        length_index = max(range(len(self.shape)), key=lambda i: self.shape[i])
        scales = [self.shape[length_index]/shape for shape in self.shape]

        centers = self.centers
        self.__kmin = tuple(scale*(kmin-center)+center for scale, kmin, center in zip(scales, self.__kmin, centers))
        self.__kmax = tuple(scale*(kmax-center)+center for scale, kmax, center in zip(scales, self.__kmax, centers))

    def squared(self):
        """Returns an inplace squared AABB.

        Returns
        -------
        Utilities.AABB
            The squared AABB.
        """
        out = copy.copy(self)
        out.squared_()

        return out

    def scale_(self, factor):
        """
        TODO: Add documentation.
        """
        factors = []
        if isinstance(factor, Iterable):
            assert len(factor) == self.__dim
            factors = factor
        else:
            factors = [factor]*self.__dim

        centers = self.centers

        self.__kmin = tuple(factor*(kmin - center) + center for factor, kmin, center in zip(factors, self.__kmin, centers))
        self.__kmax = tuple(factor*(kmax - center) + center for factor, kmax, center in zip(factors, self.__kmax, centers))

    def scale(self, factor):
        """
        TODO: Add documentation.
        """
        out = copy.copy(self)
        out.scale_(factor)

        return out

