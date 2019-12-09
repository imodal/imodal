import copy
from typing import Iterable

import torch
from numpy import prod

from implicitmodules.torch.Utilities import tensors_device, tensors_dtype


class BaseManifold:
    def __init__(self):
        pass

    def inner_prod_field(self, field):
        raise NotImplementedError()

    def infinitesimal_action(self, field):
        raise NotImplementedError()

    def cot_to_vs(self, sigma, backend=None):
        raise NotImplementedError()


class ManifoldTensor:
    def __init__(self, shapes):
        # TODO: have a lazy initialisation approach
        self.__tensors = list(torch.empty(shape, requires_grad=True) for shape in shapes)

        self.__shapes = shapes

        self.__device = None
        self.__dtype = None

    @property
    def device(self):
        return self.__device

    @property
    def dtype(self):
        return self.__dtype

    def __deepcopy__(self, memo):
        return self.clone()

    def clone(self):
        #tensors = list(tensor.detach().requires_grad_(requires_grad) for tensor in self.__tensors)
        out = ManifoldTensor(self.__shapes)
        out.fill(self.__tensors, True, False)
        return out

    def to_device(self, device):
        [tensor.to(device=device) for tensor in self.__tensors]

    def to_dtype(self, dtype):
        [tensor.to(dtype=dtype) for tensor in self.__tensors]

    def unroll(self):
        return [*self.__tensors]

    def roll(self, l):
        if len(self.__shapes) == 1:
            return l.pop(0)
        else:
            return [l.pop(0) for shape in self.__shapes]

    def __get(self):
        return self.__tensors

    def fill(self, tensors, clone=False, requires_grad=None):
        # if clone=False and requires_grad=None, should just assign tensor without changing requires_grad flag
        # assert (len(self.__shapes) == 1) or (isinstance(tensors, Iterable) and (len(self.__shapes)) == len(tensors))
        # assert (len(self.__shapes) > 1) or isinstance(tensors, torch.Tensor) and (len(self.__shapes) == 1)

        device = tensors_device(tensors)
        if device is None:
            raise RuntimeError("BaseManifold.__ManifoldTensor.fill(): at least two input tensors lives on different devices!")

        self.__device = device

        if len(self.__shapes) == 1 and isinstance(tensors, torch.Tensor):
            tensors = (tensors,)

        if clone and requires_grad is not None:
            self.__tensors = tuple(tensor.detach().clone().requires_grad_(requires_grad) for tensor in tensors)
        elif clone and requires_grad is None:
            self.__tensors = tuple(tensor.detach().clone().requires_grad_(tensor.requires_grad) for tensor in tensors)
        else:
            self.__tensors = tuple(tensor for tensor in tensors)
        # not clone and requires_grad is not None:
            #self.__tensors = tuple(tensor.requires_grad_(requires_grad) for tensor in tensors)
        # else:
        #     print("hey")
        #     self.__tensors = tuple(tensor for tensor in tensors)

    tensors = property(__get, fill)

    def fill_zeros(self, requires_grad=False):
        self.fill(tuple(torch.zeros(shape, device=self.__device) for shape in self.__shapes), clone=False, requires_grad=requires_grad)

    def requires_grad_(self, requires_grad):
        [tensor.requires_grad_(requires_grad) for tensor in self.__tensors]

    def add(self, tensors):
        if len(self.__shapes) == 1:
            tensors = (tensors,)
        self.__tensors = tuple(t0 + t for t0, t in zip(self.__tensors, tensors))

    def negate(self):
        self.__tensors = tuple(-t0 for t0 in self.__tensors)


class Manifold(BaseManifold):
    def __init__(self, shapes, nb_pts, gd, tan, cotan):
        super().__init__()

        self.__shapes = shapes
        self.__nb_pts = nb_pts

        self.__gd = ManifoldTensor(self.shape_gd)
        self.__tan = ManifoldTensor(self.shape_gd)
        self.__cotan = ManifoldTensor(self.shape_gd)

        self.__initialised = True

        if gd is not None:
            self.__gd.fill(gd, False, False)

        if tan is not None:
            self.__tan.fill(tan, False, False)

        if cotan is not None:
            self.__cotan.fill(cotan, False, False)

        if (gd is None) and (tan is None) and (cotan is None):
            self.__device = None
            self.__dtype = None
            self.__initialised = False
        else:
            self.__device = tensors_device([self.__gd, self.__tan, self.__cotan])
            if self.__device is None:
                raise RuntimeError("BaseManifold.__init__(): at least two initial manifold tensors live on different devices!")

            self.__dtype = tensors_dtype([self.__gd, self.__tan, self.__cotan])
            if self.__device is None:
                raise RuntimeError("BaseManifold.__init__(): at least two initial manifold tensors are of different dtype!")

    def clone(self, requires_grad=False):
        out = copy.deepcopy(self)
        out.__gd.requires_grad_(requires_grad)
        out.__tan.requires_grad_(requires_grad)
        out.__cotan.requires_grad_(requires_grad)

        return out

    @property
    def nb_pts(self):
        return self.__nb_pts

    @property
    def shape_gd(self):
        return tuple((self.__nb_pts, *shape) for shape in self.__shapes)

    @property
    def len_gd(self):
        return len(self.__shapes)

    @property
    def numel_gd(self):
        return tuple(prod(shape) for shape in self.shape_gd)

        return self.__device

    @property
    def device(self):
        return self.__device

    @property
    def dtype(self):
        return self.__dtype

    def __to_device(self, device):
        self.__device = device
        self.__gd.to_device(device)
        self.__tan.to_device(device)
        self.__cotan.to_device(device)

    def __to_dtype(self, dtype):
        self.__dtype = dtype
        self.__gd.to_dtype(dtype)
        self.__tan.to_dtype(dtype)
        self.__cotan.to_dtype(dtype)

    def to_(self, *argv, **kwargs):
        for arg in argv:
            if isinstance(arg, str):
                self.__to_device(torch.device(arg))
            elif isinstance(arg, torch.dtype):
                self.__to_dtype(arg)
            elif arg is None:
                self.__to_device('cpu')
            else:
                raise ValueError("BaseManifold.__BaseManifold.to_(): Unrecognised argument! {arg}".format(arg=arg))

        if 'device' in kwargs:
            if kwargs['device'] is None:
                self.__to_device(torch.device('cpu'))
            else:
                self.__to_device(kwargs['device'])

        if 'dtype' in kwargs:
            self.__to_dtype(kwargs['dtype'])        

    def unroll_gd(self):
        return self.__gd.unroll()

    def unroll_tan(self):
        return self.__tan.unroll()

    def unroll_cotan(self):
        return self.__cotan.unroll()

    def roll_gd(self, l):
        return self.__gd.roll(l)

    def roll_tan(self, l):
        return self.__tan.roll(l)

    def roll_cotan(self, l):
        return self.__cotan.roll(l)

    def __get_gd(self):
        if len(self.__shapes) == 1:
            return self.__gd.tensors[0]
        else:
            return self.__gd.tensors

    def __get_tan(self):
        if len(self.__shapes) == 1:
            return self.__tan.tensors[0]
        else:
            return self.__tan.tensors

    def __get_cotan(self):
        if len(self.__shapes) == 1:
            return self.__cotan.tensors[0]
        else:
            return self.__cotan.tensors

    def fill_gd(self, gd, copy=False, requires_grad=True):
        self.__gd.fill(gd, copy, requires_grad)
        self.__device = self.__gd.device

    def fill_tan(self, tan, copy=False, requires_grad=True):
        self.__tan.fill(tan, copy, requires_grad)
        self.__device = self.__tan.device

    def fill_cotan(self, cotan, copy=False, requires_grad=True):
        self.__cotan.fill(cotan, copy, requires_grad)
        self.__device = self.__cotan.device

    def fill_gd_zeros(self, requires_grad=True):
        self.__gd.fill_zeros(requires_grad=requires_grad)

    def fill_tan_zeros(self, requires_grad=True):
        self.__tan.fill_zeros(requires_grad=requires_grad)

    def fill_cotan_zeros(self, requires_grad=True):
        self.__cotan.fill_zeros(requires_grad=requires_grad)

    gd = property(__get_gd, fill_gd)
    tan = property(__get_tan, fill_tan)
    cotan = property(__get_cotan, fill_cotan)

    def add_gd(self, gd):
        self.__gd.add(gd)

    def add_tan(self, tan):
        self.__tan.add(tan)

    def add_cotan(self, cotan):
        self.__cotan.add(cotan)

    def negate_gd(self):
        self.__gd.negate()

    def negate_tan(self):
        self.__tan.negate()

    def negate_cotan(self):
        self.__cotan.negate()

