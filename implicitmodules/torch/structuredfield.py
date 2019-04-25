import torch

from .kernels import gauss_kernel, rel_differences


class StructuredField:
    def __init__(self):
        pass

    def __call__(self, points, k=0):
        raise NotImplementedError


class SupportStructuredField(StructuredField):
    def __init__(self, support, moments):
        super().__init__()
        self.__support = support
        self.__moments = moments

    @property
    def support(self):
        return self.__support

    @property
    def moments(self):
        return self.__moments


class StructuredField_Null(StructuredField):
    def __init__(self):
        super().__init__()

    def __call__(self, points, k=0):
        return torch.zeros([points.shape[0]] + [2]*(k+1))


class StructuredField_0(SupportStructuredField):
    def __init__(self, support, moments, sigma):
        super().__init__(support, moments)
        self.__sigma = sigma

    @property
    def sigma(self):
        return self.__sigma

    def __call__(self, points, k=0):
        ker_vec = gauss_kernel(rel_differences(points, self.support), k, self.__sigma)
        ker_vec = ker_vec.reshape((points.shape[0], self.support.shape[0]) + tuple(ker_vec.shape[1:]))
        return torch.tensordot(torch.transpose(torch.tensordot(torch.eye(2), ker_vec, dims=0), 0, 2), self.moments, dims=([2, 3], [1, 0]))


class StructuredField_p(SupportStructuredField):
    def __init__(self, support, moments, sigma):
        super().__init__(support, moments)
        self.__sigma = sigma

    @property
    def sigma(self):
        return self.__sigma

    def __call__(self, points, k=0):
        P = (self.moments + torch.transpose(self.moments, 1, 2))/2
        ker_vec = -gauss_kernel(rel_differences(points, self.support), k + 1, self.__sigma)
        ker_vec = ker_vec.reshape((points.shape[0], self.support.shape[0]) + tuple(ker_vec.shape[1:]))
        return torch.tensordot(torch.transpose(torch.tensordot(torch.eye(2), ker_vec, dims=0), 0, 2), P, dims=([2, 3, 4], [1, 0, 2]))


class StructuredField_m(SupportStructuredField):
    def __init__(self, support, moments, sigma):
        super().__init__(support, moments)
        self.__sigma = sigma

    @property
    def sigma(self):
        return self.__sigma

    def __call__(self, points, k=0):
        P = (self.moments - torch.transpose(self.moments, 1, 2))/2
        ker_vec = -gauss_kernel(rel_differences(points, self.support), k + 1, self.__sigma)
        ker_vec = ker_vec.reshape((points.shape[0], self.support.shape[0]) + tuple(ker_vec.shape[1:]))
        return torch.tensordot(torch.transpose(torch.tensordot(torch.eye(2), ker_vec, dims=0), 0, 2), P, dims=([2, 3, 4], [1, 0, 2]))


class CompoundStructuredField(StructuredField):
    def __init__(self, fields):
        super().__init__()
        self.__fields = fields

    @property
    def fields(self):
        return self.__fields

    @property
    def nb_field(self):
        return len(self.__fields)

    def __getitem__(self, index):
        return self.__fields

    def __call__(self, points, k=0):
        return sum([field(points, k) for field in self.__fields])

