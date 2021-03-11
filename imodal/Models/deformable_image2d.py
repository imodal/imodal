import torch

from imodal.Models import DeformablePoints
from imodal.DeformationModules import SilentLandmarks
from imodal.Utilities import deformed_intensities, AABB, load_greyscale_image, pixels2points


class DeformableImage(DeformablePoints):
    """
    2D bitmap deformable object.
    """
    def __init__(self, bitmap, output='bitmap', backward=False, extent=None, label=None, interpolation='nearest'):
        """
        Parameters
        ----------
        bitmap : torch.Tensor
            2 dimensional tensor representing the image to deform.
        output: str, default='bitmap'
            Representation used by the deformable.
        extent: imodal.Utilities.AABB, default=None
            Extent on the 2D plane on which the image is set.
        """
        assert isinstance(extent, AABB) or extent is None or isinstance(extent, str)

        assert output == 'bitmap' or output == 'points'

        self.__backward = backward
        self.__interpolation = interpolation

        if extent is not None and not isinstance(extent, str) and extent.dim != 2:
            raise RuntimeError("DeformableImage.__init__(): given extent is not 2 dimensional!")

        self.__shape = bitmap.shape
        self.__output = output

        self.__pixel_extent = AABB(0., self.__shape[1]-1, 0., self.__shape[0]-1)

        if extent is None:
            extent = AABB(0., 1., 0., 1.)
        elif isinstance(extent, str) and extent == 'match':
            extent = self.__pixel_extent

        self.__extent = extent

        pixel_points = pixels2points(self.__extent.fill_count(self.__shape), self.__shape, self.__extent)

        self.__bitmap = bitmap
        super().__init__(pixel_points, label=label)

    @classmethod
    def load_from_file(cls, filename, origin='lower', device=None):
        return cls(load_greyscale_image(filename, origin=origin, device=device))

    @classmethod
    def load_from_pickle(cls, filename, origin='lower', device=None):
        pass

    @property
    def geometry(self):
        if self.__output == 'bitmap':
            return (self.bitmap,)
        elif self.__output == 'points':
            return (self.silent_module.manifold.gd, self.__bitmap.flatten()/torch.sum(self.__bitmap))
        else:
            raise ValueError()

    @property
    def shape(self):
        return self.__shape

    @property
    def extent(self):
        return self.__extent

    @property
    def points(self):
        return self.silent_module.manifold.gd

    @property
    def bitmap(self):
        return self.__bitmap

    @property
    def _has_backward(self):
        return self.__backward

    def __set_output(self):
        return self.__output

    def __get_output(self, output):
        self.__output = output

    output = property(__set_output, __get_output)

    def to_device(self, device):
        super().to_device(device)
        self.__bitmap = self.__bitmap.to(device=device)

    def _backward_module(self):
        pixel_grid = pixels2points(self.__pixel_extent.fill_count(self.__shape, device=self.silent_module.device), self.__shape, self.__extent)
        return SilentLandmarks(2, pixel_grid.shape[0], gd=pixel_grid)

    def _to_deformed(self, gd):
        if not self.__backward:
            pixel_grid = pixels2points(self.__pixel_extent.fill_count(self.__shape, device=self.silent_module.device), self.__shape, self.__extent)

            normdiff = torch.sum((gd.unsqueeze(0).transpose(1, 2) - pixel_grid.unsqueeze(2))**2, dim=1)
            # _, ind_nearest = torch.topk(normdiff, k=1, dim=1, largest=False)
            ind_nearest = torch.argmin(normdiff, dim=1, keepdim=True)

            gd = torch.mean(pixel_grid[ind_nearest], 1)

        if self.__output == 'bitmap':
            return (deformed_intensities(gd, self.__bitmap, self.__extent), )
        elif self.__output == 'points':
            deformed_bitmap = deformed_intensities(gd, self.__bitmap, self.__extent)
            return (gd, deformed_bitmap.flatten()/torch.sum(deformed_bitmap))
        else:
            raise ValueError()

