import math
from collections import Iterable

import torch
from torchviz import make_dot
from .meshutils import close_shape, point_side

def grid2vec(*argv):
    """Convert a grid of points (such as given by torch.meshgrid) to a tensor of vectors."""
    return torch.cat([argv[i].contiguous().view(1, -1) for i in range(len(argv))], 0).t().contiguous()


def vec2grid(vec, *argv):
    """Convert a tensor of vectors to a grid of points."""
    return tuple((vec.t()[i].view(argv).contiguous()).contiguous() for i in range(len(argv)))


def indices2coords(indices, shape, pixel_size=torch.tensor([1., 1.])):
    return torch.cat([(pixel_size[0] * indices[:, 0]).view(-1, 1), (pixel_size[1] * (shape[1] - indices[:, 1] - 1)).view(-1, 1)], 1)


def rot2d(theta):
    """ Returns a 2D rotation matrix. """
    return torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def is_inside_shape(shape, points):
    """Returns True if the point is inside the convex shape (a tensor of CW points)."""
    closed = close_shape(shape)
    mask = torch.ones(points.shape[0], dtype=torch.uint8)
    for i in range(points.shape[0]):
        for j in range(shape.shape[0]):
            if point_side(closed[j], closed[j] - closed[j+1], points[i]) == 1:
                mask[i] = 0
                break

    return mask


def make_grad_graph(tensor, filename):
    make_dot(tensor).render(filename)


def are_tensors_properties_equal(tensors, prop):
    """ Check if all tensors share the same property given by prop(tensor). Ignores None tensors and None property values. """
    assert isinstance(tensors, Iterable)

    # If tensors is not a collection but a tensor, returns its property.
    if isinstance(tensors, torch.Tensor):
        return prop(tensors)

    # Removes None elements and None property elements
    tensors = list(tensor for tensor in tensors if (tensor is not None) and (prop(tensor) is not None))

    if len(tensors) == 0:
        return None

    first = prop(tensors[0])
    all_same = (list(prop(tensor) for tensor in tensors).count(first) == len(tensors))

    if all_same:
        return first
    else:
        return None


def tensors_device(tensors):
    """ Returns the common device on which tensors (an iterable of torch.Tensor) lives. Return None if tensors are on different devices."""
    return are_tensors_properties_equal(tensors, lambda tensor: tensor.device)


def tensors_dtype(tensors):
    """ Returns the common dtypes on which tensors (an iterable of torch.Tensor) lives. Return None if tensors are of different dtypes."""
    return are_tensors_properties_equal(tensors, lambda tensor: tensor.dtype)

