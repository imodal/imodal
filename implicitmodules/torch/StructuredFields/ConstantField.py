import torch

from implicitmodules.torch.StructuredFields.Abstract import StructuredField


class ConstantField(StructuredField):
    def __init__(self, moments):
        super().__init__()
        self.__moments = moments

    @property
    def moments(self):
        return self.__moments

    def __call__(self, points, k=0):
        if k == 0:
            return self.__moments.repeat(points.shape[0], 1)
        else:
            return torch.zeros([points.shape[0]] + [points.shape[1]]*(k+1))

