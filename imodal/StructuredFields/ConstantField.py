import torch

from implicitmodules.torch.StructuredFields.Abstract import BaseStructuredField


class ConstantField(BaseStructuredField):
    def __init__(self, moments):
        super().__init__(moments.shape[0], moments.device)
        self.__moments = moments

    @property
    def moments(self):
        return self.__moments

    def __call__(self, points, k=0):
        assert self.dim == points.shape[1]

        if k == 0:
            return self.__moments.repeat(points.shape[0], 1)
        else:
            return torch.zeros([points.shape[0]] + [points.shape[1]]*(k+1), device=self.device, dtype=points.dtype)

