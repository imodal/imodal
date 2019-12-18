import torch

from implicitmodules.torch.StructuredFields.Abstract import BaseStructuredField


class StructuredField_Null(BaseStructuredField):
    def __init__(self, dim, device=None):
        super().__init__(dim, device)

        self.__device = device

    def __call__(self, points, k=0):
        assert points.shape[1] == self.dim

        return torch.zeros([points.shape[0]] + [points.shape[1]] * (k + 1), device=self.device)

