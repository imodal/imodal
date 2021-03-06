import copy
from typing import Iterable

import torch
from numpy import prod

from imodal.Utilities import tensors_device, tensors_dtype, flatten_tensor_list
from imodal.Manifolds.tensor_container import TensorContainer


class BaseManifold:
    """Base manifold class."""
    def __init__(self, device, dtype):
        self.__device = device
        self.__dtype = dtype

    def clone(self):
        """Returns a copy of the manifold. The computation tree gets detached."""
        raise NotImplementedError()

    def __get_device(self):
        return self.__device

    def __set_device(self, device):
        self.__device = device
        self._to_device(device)

    def __get_dtype(self):
        return self.__dtype

    def __set_dtype(self, dtype):
        self.__dtype = dtype
        self._to_dtype(dtype)

    device = property(__get_device, __set_device)
    dtype = property(__get_dtype, __set_dtype)

    def to_(self, *argv, **kwargs):
        """Performs manifold dtype or/and device conversion. A :py:class:`torch.dtype` and :py:class:`torch.device` are inferred from the arguments."""
        for arg in argv:
            if isinstance(arg, str):
                self.__set_device(torch.device(arg))
            elif isinstance(arg, torch.dtype):
                self.__set_dtype(arg)
            elif arg is None:
                self.__set_device('cpu')
            else:
                raise ValueError("BaseManifold.__BaseManifold.to_(): Unrecognised argument! {arg}".format(arg=arg))

        if 'device' in kwargs:
            if kwargs['device'] is None:
                self.__set_device(torch.device('cpu'))
            else:
                self.__set_device(kwargs['device'])

        if 'dtype' in kwargs:
            self.__set_dtype(kwargs['dtype'])

    def _to_device(self, device):
        raise NotImplementedError()

    def _to_dtype(self, dtype):
        raise NotImplementedError()

    def inner_prod_field(self, field):
        """
        TODO: Add documentation.
        """
        raise NotImplementedError()

    def infinitesimal_action(self, field):
        """Infinitesimal action generated by the manifold.

        Parameters
        ----------
        field : StructuredField.BaseStructuredField
            Hello there.
        """
        raise NotImplementedError()

    def cot_to_vs(self, sigma, backend=None):
        """
        TODO: Add documentation.
        """
        raise NotImplementedError()

    def gd_requires_grad_(self, requires_grad=True):
        raise NotImplementedError()

    def tan_requires_grad_(self, requires_grad=True):
        raise NotImplementedError()

    def cotan_requires_grad_(self, requires_grad=True):
        raise NotImplementedError()

    def fill(self, manifold, copy=False, requires_grad=True):
        """
        TODO: write documentation
        """
        self.fill_gd(manifold.gd, copy=copy, requires_grad=requires_grad)
        self.fill_tan(manifold.tan, copy=copy, requires_grad=requires_grad)
        self.fill_cotan(manifold.cotan, copy=copy, requires_grad=requires_grad)

    def fill_gd(self, gd, copy=False, requires_grad=True):
        """Fill geometrical descriptors to the manifold.

        Parameters
        ----------
        copy : bool, default=False
            If True, copy the geometrical descriptors into the manifold, detaching the computation graph.
            If False, the tensors are passed by references.
        requires_grad : bool, default=True
            If **copy**=True, set the **requires_grad** flag of the copied geometrical descriptors tensors to the given boolean value.
        """
        raise NotImplementedError()

    def fill_tan(self, tan, copy=False, requires_grad=True):
        """Same as **fill_gd()**, for tangents."""
        raise NotImplementedError()

    def fill_cotan(self, cotan, copy=False, requires_grad=True):
        """Same as **fill_cotan()**, for cotangents."""
        raise NotImplementedError()

    def fill_gd_zeros(self, requires_grad=True):
        """Fill geometrical descriptors with zero tensors.

        Parameters
        ----------
        requires_grad : bool, default=True
            Value given to the requires_grad flag of the zero tensors.
        """
        raise NotImplementedError()

    def fill_tan_zeros(self, requires_grad=True):
        """Same as **fill_gd_zeros()**, for tangents."""
        raise NotImplementedError()

    def fill_cotan_zeros(self, requires_grad=True):
        """Same as **fill_gd_zeros()**, for cotangents."""
        raise NotImplementedError()

    def fill_gd_randn(self, requires_grad=True):
        """ Fill geometrical descriptors with a standard normal distribution.

        Parameters
        ----------
        requires_grad : bool, default=True
            Value given to the requires_grad flag of the random tensors.
        """
        raise NotImplementedError()

    def fill_tan_randn(self, requires_grad=True):
        """Same as **fill_gd_randn()**, for tangents."""
        raise NotImplementedError()

    def fill_cotan_randn(self, requires_grad=True):
        """Same as **fill_gd_randn()**, for cotangents."""
        raise NotImplementedError()

    def add_gd(self, gd):
        """Adds **gd** to the manifold geometrical descriptors.

        Parameters
        ----------
        gd : Iterable or torch.Tensor
            Tensors that will be added to the manifold geometrical descriptors.
        """
        raise NotImplementedError()

    def add_tan(self, tan):
        """Same as **add_gd()**, for tangents."""
        raise NotImplementedError()

    def add_cotan(self, cotan):
        """Same as **add_gd()**, for cotangents."""
        raise NotImplementedError()

    def negate_gd(self):
        """Negate geometrical descriptors."""
        raise NotImplementedError()

    def negate_tan(self):
        """Negate tangents."""
        raise NotImplementedError()

    def negate_cotan(self):
        """Negate cotangents."""
        raise NotImplementedError()


# TODO: maybe rename into TensorManifold ? (or something else)
class Manifold(BaseManifold):
    """Manifold class built using ManifoldTensorContainer as manifold tensors storage. Base class for most manifold class."""
    def __init__(self, shapes, nb_pts, gd, tan, cotan, device=None, dtype=None):
        self.__shapes = shapes
        self.__nb_pts = nb_pts

        self.__initialised = True

        # No tensors are filled
        if (gd is None) and (tan is None) and (cotan is None):
            # Default device set to cpu if device is not specified.
            if device is None:
                device = torch.device('cpu')

            # dtype set to Torch default if dtype is not specified.
            if dtype==None:
                dtype = torch.get_default_dtype()

            self.__initialised = False
        # Some tensors (or all) are filled
        else:
            # No device is specified, we infer it from the filled tensors.
            if device is None:
                device = tensors_device(flatten_tensor_list([gd, tan, cotan]))

                # Found device is None, meaning the filled tensors lives on different devices.
                if device is None:
                    raise RuntimeError("BaseManifold.__init__(): at least two initial manifold tensors live on different devices!")

            # No dtype is specified, we infer it from the filled tensors.
            if dtype is None:
                dtype = tensors_dtype(flatten_tensor_list([gd, tan, cotan]))

                # Found dtype is None, meaning the filled tensors are of different dtypes.
                if dtype is None:
                    raise RuntimeError("BaseManifold.__init__(): at least two initial manifold tensors are of different dtype!")

        self.__gd = TensorContainer(self.shape_gd, device, dtype)
        self.__tan = TensorContainer(self.shape_gd, device, dtype)
        self.__cotan = TensorContainer(self.shape_gd, device, dtype)

        if gd is not None:
            self.__gd.fill(gd, False, False)

        if tan is not None:
            self.__tan.fill(tan, False, False)

        if cotan is not None:
            self.__cotan.fill(cotan, False, False)

        super().__init__(device, dtype)

    def gd_requires_grad_(self, requires_grad=True, index=-1):
        self.__gd.requires_grad_(requires_grad, index)

    def tan_requires_grad_(self, requires_grad=True, index=-1):
        self.__tan.requires_grad_(requires_grad, index)

    def cotan_requires_grad_(self, requires_grad=True, index=-1):
        self.__cotan.requires_grad_(requires_grad, index)

    def clone(self, requires_grad=None):
        """Returns a copy of the manifold. Detaches the computation graph.

        Parameters
        ----------

        requires_grad : bool, default=None
            Set to True to record futher operations on the manifold tensors. Set
            to None to not change the requires_grad setting.
        """
        out = copy.deepcopy(self)
        out.__gd.requires_grad_(requires_grad)
        out.__tan.requires_grad_(requires_grad)
        out.__cotan.requires_grad_(requires_grad)

        return out

    def _to_device(self, device):
        self.__gd.to_device(device)
        self.__tan.to_device(device)
        self.__cotan.to_device(device)

    def _to_dtype(self, dtype):
        self.__gd.to_dtype(dtype)
        self.__tan.to_dtype(dtype)
        self.__cotan.to_dtype(dtype)

    @property
    def nb_pts(self):
        """Returns the number of points of the manifold."""
        return self.__nb_pts

    @property
    def shape_gd(self):
        """Returns the shape of each tensors for a manifold tensor of the manifold."""
        return tuple((self.__nb_pts, *shape) for shape in self.__shapes)

    @property
    def len_gd(self):
        """Returns the dimensions of each tensors for a manifold tensor of the manifold."""
        return len(self.__shapes)

    @property
    def numel_gd(self):
        """Returns the total number of elements for a manifold tensor of the manifold."""
        return tuple(prod(shape) for shape in self.shape_gd)

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

    def fill_gd_randn(self, requires_grad=True):
        self.__gd.fill_randn(requires_grad=requires_grad)

    def fill_tan_randn(self, requires_grad=True):
        self.__tan.fill_randn(requires_grad=requires_grad)

    def fill_cotan_randn(self, requires_grad=True):
        self.__cotan.fill_randn(requires_grad=requires_grad)

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

