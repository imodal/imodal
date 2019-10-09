import torch

from implicitmodules.torch.StructuredFields.Abstract import StructuredField


class StructuredField_Null(StructuredField):
    def __init__(self, device=None):
        super().__init__()

        self.__device = device
    
    def __call__(self, points, k=0):
        return torch.zeros([points.shape[0]] + [points.shape[1]] * (k + 1), device=self.__device)

