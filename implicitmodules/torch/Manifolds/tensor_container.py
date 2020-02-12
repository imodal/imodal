import torch

from implicitmodules.torch.Utilities import tensors_device, tensors_dtype


class TensorContainer:
    """Container class for tensors.
    Internal class used to facilitate management of manifold tensors.
    Tensors are stored as a tuple of torch.Tensor."""
    def __init__(self, shapes, device, dtype):
        # TODO: have a lazy initialisation approach
        self.__tensors = tuple(torch.empty(shape, requires_grad=True) for shape in shapes)

        self.__shapes = shapes

        self.__device = device
        self.__dtype = dtype

    @property
    def device(self):
        return self.__device

    @property
    def dtype(self):
        return self.__dtype

    def __deepcopy__(self, memodict):
        out = self.clone()
        memodict[id(out)] = out
        return out

    def clone(self):
        """Returns a copy of the current tensor container.
        In this implementation, the computation graph of each tensor gets detached.
        **requires_grad** is then set to the same value as the current tensor container.
        Returns
        -------
        TensorContainer
            The cloned tensor container.
        """
        out = TensorContainer(self.__shapes, self.__device, self.__dtype)
        out.fill(self.__tensors, True)
        return out

    def to_device(self, device):
        """Moves each tensors onto the device **device**."""
        self.__device = device
        self.__tensors = [tensor.to(device=device).detach().requires_grad_(tensor.requires_grad) for tensor in self.__tensors]

    def to_dtype(self, dtype):
        """Convert each tensors into dtype **dtype**."""
        self.__dtype = dtype
        self.__tensors = [tensor.to(dtype=dtype).detach().requires_grad_(tensor.requires_grad) for tensor in self.__tensors]

    def unroll(self):
        """Unroll the tensor tuple into a flattened list.

        Returns
        -------
            A flattened list of the manifold tensors.
        """
        return [*self.__tensors]

    def roll(self, l):
        """Roll a flattened list into a tensor list (i.e. inverse operation of unroll())."""
        if len(self.__shapes) == 0:
            return
        elif len(self.__shapes) == 1:
            return l.pop(0)
        else:
            return [l.pop(0) for shape in self.__shapes]

    def __get(self):
        return self.__tensors

    def fill(self, tensors, clone=False, requires_grad=None):
        """Fill the tensor container with **tensors**.

        Parameters
        ----------
        tensors : Iterable
            Iterable of torch.Tensor for multidimensional tensor or torch.Tensor for simple tensor we want to fill with.
        clone : bool, default=False
            Set to true to clone the tensors. This will detach the computation graph. If false, tensors will be passed as references.
        requires_grad : bool, default=None
            Set to true to record further operations on the tensors. Only relevant when cloning the tensors.
        """
        # if clone=False and requires_grad=None, should just assign tensor without changing requires_grad flag
        # assert (len(self.__shapes) == 1) or (isinstance(tensors, Iterable) and (len(self.__shapes)) == len(tensors))
        # assert (len(self.__shapes) > 1) or isinstance(tensors, torch.Tensor) and (len(self.__shapes) == 1)

        device = tensors_device(tensors)
        if device is None:
            raise RuntimeError("BaseManifold.__ManifoldTensorContainer.fill(): at least two input tensors lives on different devices!")

        self.__device = device

        dtype = tensors_dtype(tensors)
        if dtype is None:
            raise RuntimeError("BaseManifold.__ManifoldTensorContainer.fill(): at least two input tensors are of different dtypes!")

        self.__dtype = dtype

        if len(self.__shapes) == 1 and isinstance(tensors, torch.Tensor):
            tensors = (tensors,)

        if clone and requires_grad is not None:
            self.__tensors = tuple(tensor.detach().clone().requires_grad_(requires_grad) for tensor in tensors)
        elif clone and requires_grad is None:
            self.__tensors = tuple(tensor.detach().clone().requires_grad_(tensor.requires_grad) for tensor in tensors)
        else:
            self.__tensors = tuple(tensor for tensor in tensors)

    tensors = property(__get, fill)

    def fill_zeros(self, requires_grad=False):
        """Fill each tensors of the manifold tensor with zeros.

        Parameters
        ----------
        requires_grad : bool, default=False
            Set to true to record futher operations on the tensors.
        """
        self.fill(tuple(torch.zeros(shape, device=self.__device, dtype=self.__dtype) for shape in self.__shapes), clone=False, requires_grad=requires_grad)

    def requires_grad_(self, requires_grad=True):
        """Set operation recording flag.

        Parameters
        ----------
        requires_grad : bool, default=True
            Set to true to record futher operations on the tensors.
        """
        if requires_grad is not None:
            [tensor.requires_grad_(requires_grad) for tensor in self.__tensors]

    def add(self, tensors):
        """Addition."""
        if len(self.__shapes) == 1:
            tensors = (tensors,)
        self.__tensors = tuple(t0 + t for t0, t in zip(self.__tensors, tensors))

    def negate(self):
        """Negate each manifold tensors."""
        self.__tensors = tuple(-t0 for t0 in self.__tensors)

