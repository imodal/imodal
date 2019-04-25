import torch
import matplotlib.image
import torch

from .kernels import K_xy
from .usefulfunctions import AABB, grid2vec, indices2coords


def load_greyscale_image(filename):
    """Load grescale image from disk as an array of normalised float values."""
    image = matplotlib.image.imread(filename)
    if(image.ndim == 2):
        return torch.tensor(1. - image, dtype=torch.get_default_dtype())
    elif(image.ndim ==3):
        return torch.tensor(1. - image[:,:,0], dtype=torch.get_default_dtype())
    else:
        raise NotImplementedError


def sample_from_greyscale(image, threshold, centered=False, normalise_weights=False, normalise_position=True):
    """Sample points from a greyscale image. Points are defined as a (position, weight) tuple."""
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

            pos[count, 0] = i
            pos[count, 1] = j
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


def sample_image_from_points(points, frame_res):
    """Sample an image from a set of points using the binning/histogram method."""
    frame = torch.zeros(frame_res[0]*frame_res[1])
    frame.scatter_add_(0, torch.clamp((points[0][:, 1]*frame_res[1] + points[0][:, 0]).long(),
                                      0, frame_res[0]*frame_res[1]-1), points[1])

    return frame.view(frame_res).contiguous()


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


def kernel_smoother(pos, points, kernel_matrix=K_xy, sigma=1.):
    """Applies a kernel smoother on pos."""
    K = kernel_matrix(pos, points[0], sigma=sigma)

    return torch.mm(K, points[1].view(-1, 1)).flatten().contiguous()


# TODO: this function is flipping images in the vertical axis, find out why
def sample_from_smoothed_points(points, frame_res, kernel=K_xy, sigma=1.,
                                normalize=False, aabb=None):
    """Sample an image from a list of smoothened points."""
    if(aabb is None):
        aabb = AABB.build_from_points(points[0].detach())

    x, y = torch.meshgrid([torch.linspace(aabb.xmin-sigma, aabb.xmax+sigma, frame_res[0]),
                           torch.linspace(aabb.ymin-sigma, aabb.ymax+sigma, frame_res[1])])

    pos = grid2vec(x, y)
    pixels = kernel_smoother(pos, points, sigma=sigma)
    if(normalize):
        pixels = pixels/torch.max(pixels)

    return pixels.view(frame_res[0], frame_res[1]).contiguous()


def resample_image_to_smoothed(image, kernel=K_xy, sigma=1., normalize=False):
    x, y = torch.meshgrid([torch.arange(0., image.shape[0], step=1.),
                           torch.arange(0., image.shape[1], step=1.)])

    pixel_pos = grid2vec(x, y)

    return sample_from_smoothed_points((pixel_pos, image.view(-1)), image.shape, kernel=kernel,
                                       sigma=sigma, normalize=normalize)


def fill_area_uniform(area, aabb, spacing):
    """Fill a 2D area enclosed by aabb given by the area function (area(pos): return true if in the area, false otherwise)."""
    x, y = torch.meshgrid([torch.arange(aabb.xmin, aabb.xmax, step=spacing),
                           torch.arange(aabb.ymin, aabb.ymax, step=spacing)])
    grid = grid2vec(x, y)
    inside = area(grid).repeat(2, 1).transpose(0, 1)
    mask = inside.eq(True)

    return torch.masked_select(grid, inside).view(-1, 2)


def fill_area_random(area, aabb, density):
    """Fill an area enclosed by aabb randomly given by the area function (area(pos): return true if in the area, false otherwise) by rejection sampling."""

    nb_pts = int(aabb.area * density)

    points = torch.zeros(nb_pts, 2)

    for i in range(0, nb_pts):
        accepted = False
        while(not accepted):
            point = aabb.sample_random_point(1)
            if torch.all(area(point)):
                accepted = True
                points[i, :] = point

    return points
