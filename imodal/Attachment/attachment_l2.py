import torch

from imodal.Attachment import Attachment
from imodal.Utilities import interpolate_image
from imodal.Kernels import gauss_kernel


class L2NormAttachment(Attachment):
    def __init__(self, transform=None, scale=False, scale_settings=None, weight=1., **kwargs):
        super().__init__(weight)

        if scale_settings is None:
            scale_settings = {}

        self.__scale = scale
        self.__scale_settings = scale_settings

        self.__kwargs = kwargs

        self.__loss_function = None
        if transform == 'l2' or transform is None:
            self.__loss_function = self.__loss_l2
        elif transform == 'fft':
            self.__loss_function = self.__loss_fft
        elif transform == 'radon':
            self.__loss_function = self.__loss_radon
        elif transform == 'smooth':
            self.__loss_function = self.__loss_smooth
        else:
            raise NotImplementedError("L2NormAttachment.__init__(): {transform} transform function not implemented!".format(transform=transform))

    def loss(self, source, target):
        scaled_source, scaled_target = self.__scale_function(source[0], target[0])
        return self.__loss_function(scaled_source, scaled_target)

    def __loss_l2(self, source, target):
        return torch.dist(source, target)**2.

    def __loss_fft(self, source, target):
        source_fft = torch.fft(torch.stack([source, torch.zeros_like(source)], dim=len(source.shape)), len(source.shape), **self.__kwargs)
        target_fft = torch.fft(torch.stack([target, torch.zeros_like(target)], dim=len(target.shape)), len(target.shape), **self.__kwargs)

        return self.__loss_l2(source_fft[:, :, 0], target_fft[:, :, 0]) + \
            self.__loss_l2(source_fft[:, :, 1], target_fft[:, :, 1])

    def __loss_radon(self, source, target):
        pass

    def __loss_smooth(self, source, target):
        if 'bandwidth' not in self.__kwargs:
            raise RuntimeError("L2NormAttachment.__loss_smooth(): bandwidth parameter not specified!")

        bandwidth = self.__kwargs['bandwidth']

        smoothed_source = gauss_kernel(source, 0, bandwidth)
        smoothed_target = gauss_kernel(target, 0, bandwidth)

        print(smoothed_source.shape)

        return self.__loss_l2(smoothed_source, smoothed_target)

    def __scale_function(self, source, target):
        if self.__scale:
            return interpolate_image(source, **self.__scale_settings), \
                interpolate_image(target, **self.__scale_settings)
        else:
            return source, target

