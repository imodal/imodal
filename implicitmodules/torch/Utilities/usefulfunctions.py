import math

import torch
from torchviz import make_dot

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

