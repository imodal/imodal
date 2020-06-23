import math

import matplotlib.image
import torch

from implicitmodules.torch.Kernels import K_xy
from implicitmodules.torch.Utilities.usefulfunctions import grid2vec, indices2coords
from implicitmodules.torch.Utilities.aabb import AABB


def load_greyscale_image(filename, dtype=None, device=None):
    """Load grescale image from disk as an array of normalised float values.
    
    Parameters
    ----------
    filename : str
        Filename of the image to load.
    dtype : torch.dtype
        dtype of the returned image tensor.
    device : torch.device
        Device on which the image tensor will be loaded.

    Returns
    -------
    torch.Tensor
        [width, height] shaped tensor representing the loaded image.
    """

    if dtype is None:
        dtype = torch.get_default_dtype()

    image = matplotlib.image.imread(filename)
    if(image.ndim == 2):
        return torch.tensor(1. - image, dtype=dtype, device=device)
    elif(image.ndim ==3):
        return torch.tensor(1. - image[:,:,0], dtype=dtype, device=device)
    else:
        raise NotImplementedError


def sample_from_greyscale(image, threshold, centered=False, normalise_weights=False, normalise_position=True):
    """Sample points from a greyscale image.
    Points are defined as a (position, weight) tuple.

    Parameters
    ----------
    image : torch.Tensor
        Tensor of shape [width, height] representing the image from which we will sample the points.
    threshold : float
        Minimum pixel value (i.e. point weight) 
    centered : bool, default=False
        If true, center the sampled points such that mean 
    normalise_weights : bool, default=False
        If true, normalise weight values, such that :math:'\alpha_i = \frac{\alpha_i}{\sum_k \alpha_k}'
    normalise_position : bool, default=True
        If true, normalise point position such that all points live in the unit square.

    Returns
    -------
    torch.Tensor, torch.Tensor
        Two tensors representing point position (of shape [N, dim]) and weight (of shape [N]), in this order, with :math:'N' the number of points.
    """

    length = torch.sum(image >= threshold)
    pos = torch.zeros([length, 2])
    alpha = torch.zeros([length])

    width_weight = 1.
    height_weight = 1.

    if(normalise_position):
        width_weight = 1./image.shape[0]
        height_weight = 1./image.shape[1]

    count = 0

    # TODO: write a better (i.e. non looping) way of doing this
    for j in range(0, image.shape[1]):
        for i in range(0, image.shape[0]):
            if(image[image.shape[0] - i - 1, j] < threshold):
                continue

            pos[count, 0] = i + 0.5
            pos[count, 1] = j - 0.5
            alpha[count] = image[image.shape[0] - i - 1, j]

            count = count + 1

    pos = indices2coords(pos, image.shape, pixel_size=torch.tensor([height_weight, width_weight]))

    if(centered):
        pos = pos - torch.mean(pos, dim=0)

    if(normalise_weights):
        alpha = alpha/torch.sum(alpha)

    return pos, alpha


def load_and_sample_greyscale(filename, threshold=0., centered=False, normalise_weights=True):
    """Load a greyscale and sample points from it."""
    image = load_greyscale_image(filename)

    return sample_from_greyscale(image, threshold, centered, normalise_weights)


def deformed_intensities(deformed_points, intensities):
    """
    Sample an image from a tensor of deformed points.
    Taken and adapted from https://gitlab.icm-institute.org/aramislab/deformetrica/blob/master/numpy/core/observations/deformable_objects/image.py
    """

    u, v = deformed_points[:, 0], deformed_points[:, 1]

    u1 = torch.floor(u).long()
    v1 = torch.floor(v).long()

    u1 = torch.clamp(u1, 0, int(intensities.shape[0]) - 1)
    v1 = torch.clamp(v1, 0, int(intensities.shape[1]) - 1)
    u2 = torch.clamp(u1 + 1, 0, int(intensities.shape[0]) - 1)
    v2 = torch.clamp(v1 + 1, 0, int(intensities.shape[1]) - 1)

    fu = u - u1.type(torch.get_default_dtype())
    fv = v - v1.type(torch.get_default_dtype())
    gu = (u1 + 1).type(torch.get_default_dtype()) - u
    gv = (v1 + 1).type(torch.get_default_dtype()) - v

    deformed_intensities = (intensities[u1, v1] * gu * gv +
                            intensities[u1, v2] * gu * fv +
                            intensities[u2, v1] * fu * gv +
                            intensities[u2, v2] * fu * fv).view(intensities.shape)

    return deformed_intensities


