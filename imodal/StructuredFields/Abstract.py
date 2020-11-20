import torch

from implicitmodules.torch.Utilities import is_valid_backend, get_compute_backend, shared_tensors_property, tensors_device


class BaseStructuredField:
    def __init__(self, dim, device):
        assert isinstance(dim, int)
        self.__dim = dim
        self.__device = device

    @property
    def dim(self):
        return self.__dim

    @property
    def device(self):
        return self.__device

    def __call_(self, points, k=0):
        raise NotImplementedError()


class StructuredField(BaseStructuredField):
    def __init__(self, dim, device, backend):
        super().__init__(dim, device)
        self.__backend = backend

        if self.__backend is None:
            self.__backend = get_compute_backend()

        assert is_valid_backend(self.__backend)

        if self.__backend == 'torch':
            self._compute_reduction = self._compute_reduction_torch
        elif self.__backend == 'keops':
            self._compute_reduction = self._compute_reduction_keops
        else:
            raise NotImplementedError("{classname}.__call__(): reduction not implemented for backend {backend}!".format(classname=self.__class__.__name__, backend=self.__backend))

    @property
    def backend(self):
        return self.__backend

    def __call__(self, points, k=0):
        assert k >= 0
        return self._compute_reduction(points, k)

    def _compute_reduction_torch(self, points, k):
        raise NotImplementedError()

    def _compute_reduction_keops(self, points, k):
        raise NotImplementedError()


class KernelSupportStructuredField(StructuredField):
    def __init__(self, support, moments, sigma, device, backend):
        assert support.dtype == moments.dtype
        assert support.device == moments.device
        assert support.shape[1] == moments.shape[1] # Compare dimensions

        super().__init__(support.shape[1], support.device, backend)
        self.__support = support
        self.__moments = moments
        self.__sigma = sigma

        if self.backend == 'keops':
            self.__keops_backend = 'CPU'
            if (str(self.device) != 'cpu') and (str(self.device) != None):
                self.__keops_backend = 'GPU'

            self.__keops_sigma = torch.tensor([1./self.sigma/self.sigma], dtype=support.dtype, device=self.device)
            self.__keops_dtype = self.__keops_dtype = str(support.dtype).split(".")[1]

    @property
    def _keops_backend(self):
        return self.__keops_backend

    @property
    def _keops_sigma(self):
        return self.__keops_sigma

    @property
    def _keops_dtype(self):
        return self.__keops_dtype

    @property
    def sigma(self):
        return self.__sigma

    @property
    def support(self):
        return self.__support

    @property
    def moments(self):
        return self.__moments


class SumStructuredField(BaseStructuredField):
    def __init__(self, fields):
        dim = shared_tensors_property(fields, lambda x: x.dim)
        # device = shared_tensors_property(fields, lambda x: x.device)
        device = tensors_device(fields)
        assert dim is not None
        # assert device is not None
        super().__init__(dim, device)

        self.__fields = fields

    @property
    def fields(self):
        return self.__fields

    def __getitem__(self, index):
        return self.__fields

    def __call__(self, points, k=0):
        return sum([field(points, k) for field in self.__fields])

