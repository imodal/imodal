import torch

from imodal.StructuredFields.Abstract import BaseStructuredField


class StructuredField_Affine(BaseStructuredField):
    def __init__(self, A, x0, u):
        assert A.device == x0.device and A.device == u.device
        assert A.shape[0] == A.shape[1] and A.shape[0] == x0.shape[0] and A.shape[0] == u.shape[0]

        super().__init__(A.shape[0], A.device)

        self.__A = A
        self.__x0 = x0
        self.__u = u

    @property
    def A(self):
        return self.__A

    @property
    def x0(self):
        return self.__x0

    @property
    def u(self):
        return self.__u

    def __call__(self, points, k=0):
        if k == 0:
            return torch.bmm((points - self.__x0.repeat(points.shape[0], 1)).unsqueeze(1), self.__A.transpose(0, 1).repeat(points.shape[0], 1, 1)).view(-1, points.shape[1]) + self.__u.repeat(points.shape[0], 1)
        elif k == 1:
            return self.__A.repeat(points.shape[0], 1, 1)
        else:
            return torch.zeros([points.shape[0]] + [points.shape[1]]*(k+1))

