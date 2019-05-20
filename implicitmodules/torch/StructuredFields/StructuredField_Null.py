import torch

from implicitmodules.torch.StructuredFields.Abstract import StructuredField


class StructuredField_Null(StructuredField):
    def __init__(self):
        super().__init__()
    
    def __call__(self, points, k=0):
        return torch.zeros([points.shape[0]] + [2] * (k + 1))
