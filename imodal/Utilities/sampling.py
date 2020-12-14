import math

import matplotlib.image
import torch

from imodal.Kernels import K_xy
from imodal.Utilities.usefulfunctions import grid2vec, points2pixels, points2nel
from imodal.Utilities.aabb import AABB


def load_greyscale_image(filename, origin='lower', dtype=None, device=None):
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

    if origin != 'upper' and origin != 'lower':
        raise RuntimeError("Origin type {origin} not implemented!".format(origin=origin))

    def _set_origin(bitmap, origin):
        if origin == 'upper':
            return bitmap
        else:
            return bitmap.flip(0)

    image = matplotlib.image.imread(filename)
    if(image.ndim == 2):
        return _set_origin(torch.tensor(1. - image, dtype=dtype, device=device), origin)
    elif(image.ndim ==3):
        return _set_origin(torch.tensor(1. - image[:,:,0], dtype=dtype, device=device), origin)
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

    # Compute number of output points
    length = torch.sum(image >= threshold)

    pos = torch.zeros([length, 2])
    alpha = torch.zeros([length])

    width_weight = 1.
    height_weight = 1.

    if(normalise_position):
        width_weight = 1./image.shape[0]
        height_weight = 1./image.shape[1]

    count = 0

    pixels = AABB(0., image.shape[0], 0, image.shape[1]).fill_count(image.shape)

    for pixel in pixels:
        pixel_value = image[math.floor(pixel[0]), math.floor(pixel[1])]
        if pixel_value < threshold:
            continue

        pos[count] = pixel
        alpha[count] = pixel_value

        count = count + 1

    if(centered):
        pos = pos - torch.mean(pos, dim=0)

    if(normalise_weights):
        alpha = alpha/torch.sum(alpha)

    return pos, alpha


def load_and_sample_greyscale(filename, threshold=0., centered=False, normalise_weights=True):
    """Load a greyscale and sample points from it."""
    
    image = load_greyscale_image(filename)

    return sample_from_greyscale(image, threshold, centered, normalise_weights)


def deformed_intensities(deformed_points, intensities, extent):
    """
    Sample an image from a tensor of deformed points.
    Taken and adapted from https://gitlab.icm-institute.org/aramislab/deformetrica/blob/master/numpy/core/observations/deformable_objects/image.py
    """

    uv = points2pixels(deformed_points, intensities.shape, extent)
    u, v = uv[:, 0], uv[:, 1]
    u1, v1 = torch.floor(uv[:, 0]).long(), torch.floor(uv[:, 1]).long()

    u1 = torch.clamp(u1, 0, int(intensities.shape[0]) - 1)
    v1 = torch.clamp(v1, 0, int(intensities.shape[1]) - 1)
    u2 = torch.clamp(u1 + 1, 0, int(intensities.shape[0]) - 1)
    v2 = torch.clamp(v1 + 1, 0, int(intensities.shape[1]) - 1)

    fu = u - u1.type(torch.get_default_dtype())
    fv = v - v1.type(torch.get_default_dtype())
    gu = (u1 + 1).type(torch.get_default_dtype()) - u
    gv = (v1 + 1).type(torch.get_default_dtype()) - v

    return (intensities[u1, v1] * gu * gv +
            intensities[u1, v2] * gu * fv +
            intensities[u2, v1] * fu * gv +
            intensities[u2, v2] * fu * fv).view(intensities.shape)


def deformed_intensities3d(deformed_points, intensities, extent):
    """
    Sample a 3D image from a tensor of deformed points.
    Taken and adapted from https://gitlab.icm-institute.org/aramislab/deformetrica/blob/master/numpy/core/observations/deformable_objects/image.py
    """

    uvw = points2nel(deformed_points, intensities.shape, extent)

    u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]
    u1 = torch.floor(uvw[:, 0]).long()
    v1 = torch.floor(uvw[:, 1]).long()
    w1 = torch.floor(uvw[:, 2]).long()

    u1 = torch.clamp(u1, 0, int(intensities.shape[0]) - 1)
    v1 = torch.clamp(v1, 0, int(intensities.shape[1]) - 1)
    w1 = torch.clamp(w1, 0, int(intensities.shape[2]) - 1)
    u2 = torch.clamp(u1 + 1, 0, int(intensities.shape[0]) - 1)
    v2 = torch.clamp(v1 + 1, 0, int(intensities.shape[1]) - 1)
    w2 = torch.clamp(w1 + 1, 0, int(intensities.shape[2]) - 1)

    fu = u - u1.type(torch.get_default_dtype())
    fv = v - v1.type(torch.get_default_dtype())
    fw = w - w1.type(torch.get_default_dtype())
    gu = (u1 + 1).type(torch.get_default_dtype()) - u
    gv = (v1 + 1).type(torch.get_default_dtype()) - v
    gw = (w1 + 1).type(torch.get_default_dtype()) - w

    return (intensities[u1, v1, w1] * gu * gv * gw +
            intensities[u1, v1, w2] * gu * gv * fw +
            intensities[u1, v2, w1] * gu * fv * gw +
            intensities[u1, v2, w2] * gu * fv * fw +
            intensities[u2, v1, w1] * fu * gv * gw +
            intensities[u2, v1, w2] * fu * gv * fw +
            intensities[u2, v2, w1] * fu * fv * gw +
            intensities[u2, v2, w2] * fu * fv * fw).view(intensities.shape)


def interpolate_image(image, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
    """
    Simple wrapper around torch.nn.functional.interpolate() for 2D images.
    """
    
    interpolated = torch.nn.functional.interpolate(image.view((1, 1) + image.shape), size, scale_factor, mode, align_corners, recompute_scale_factor)
    return interpolated.view(interpolated.shape[2:])

