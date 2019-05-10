import math
from collections import Iterable

import matplotlib.pyplot as plt
import torch
from torchviz import make_dot


class AABB:
    """Class used to represent an Axis Aligned Bounding Box"""
    def __init__(self, xmin=0., xmax=0., ymin=0., ymax=0.):
        self.__xmin = xmin
        self.__xmax = xmax
        self.__ymin = ymin
        self.__ymax = ymax

    @classmethod
    def build_from_points(cls, points):
        """Compute the AABB from points"""

        return cls(torch.min(points[:, 0]), torch.max(points[:, 0]),
                    torch.min(points[:, 1]), torch.max(points[:, 1]))

    def sample_random_point(self, count):
        return torch.tensor([self.width, self.height])*torch.rand(count, 2)+torch.tensor([self.xmin, self.ymin])

    def is_inside(self, points):
        return torch.where((points[:, 0] >= self.__xmin) & (points[:, 0] <= self.xmax) &
                           (points[:, 1] >= self.__ymin) & (points[:, 1] <= self.ymax),
                           torch.tensor([1.]), torch.tensor([0.])).byte()

    def __getitem__(self, key):
        return self.get_list()[key]

    def get_list(self):
        """Returns the AABB as a list, 0:xmin, 1:xmax, 2:ymin, 3:ymax."""
        return [self.__xmin, self.__xmax, self.__ymin, self.__ymax]

    @property
    def xmin(self):
        return self.__xmin

    @property
    def ymin(self):
        return self.__ymin

    @property
    def xmax(self):
        return self.__xmax

    @property
    def ymax(self):
        return self.__ymax

    @property
    def width(self):
        return self.__xmax - self.__xmin

    @property
    def height(self):
        return self.__ymax - self.__ymin

    @property
    def area(self):
        return (self.__xmax - self.__xmin)*(self.ymax - self.ymin)

    def squared(self):
        self.__xmin = min(self.__xmin, self.__ymin)
        self.__ymin = min(self.__xmin, self.__ymin)
        self.__xmax = max(self.__xmax, self.__ymax)
        self.__ymax = max(self.__xmax, self.__ymax)


def flatten_tensor_list(l, out_list=[]):
    """Very simple, recursive, list flattening functions that stops at the torch.Tensor (without unwrapping them). Should work well for lists that are not too much nested."""
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, torch.Tensor):
            flatten_tensor_list(el, out_list)
        else:
            out_list.append(el)

    return out_list


def grid2vec(x, y):
    """Convert a grid of points (such as given by torch.meshgrid) to a tensor of vectors."""
    return torch.cat([x.contiguous().view(1, -1), y.contiguous().view(1, -1)], 0).t().contiguous()


def vec2grid(vec, nx, ny):
    """Convert a tensor of vectors to a grid of points."""
    return vec.t()[0].view(nx, ny).contiguous(), vec.t()[1].view(nx, ny).contiguous()


def indices2coords(indices, shape, pixel_size=torch.tensor([1., 1.])):
    # return torch.tensor([pixel_size[1]*indices[:, 1], pixel_size[0]*(shape[0] - indices[:, 0] - 1)])
    return torch.cat([(pixel_size[0]*indices[:, 0]).view(-1, 1), (pixel_size[1]*(shape[1] - indices[:, 1] - 1)).view(-1, 1)], 1)


def blocks_to_2d(M):
    """Transforms a block matrix tensor (N x dim_block x dim_block) into a 2D square block matrix of size (N * dim_block x N * dim_block)."""
    N = int(math.sqrt(M.shape[0]))
    assert N**2 == M.shape[0]
    return torch.cat([torch.cat([M.transpose(1, 2)[i::N] for i in range(N)], dim=1).transpose(1, 2)[i] for i in range(N)])


def blocks_to_2d_fast(M):
    a = torch.arange(M.numel()).view_as(M)
    indices = blocks_to_2d(a).view(-1)
    return torch.take(M, indices).view(int(math.sqrt(M.shape[0])*M.shape[1]), int(math.sqrt(M.shape[0])*M.shape[2]))


def kronecker(m1, m2):
    return torch.ger(m1.view(-1), m2.view(-1)).reshape(*(m1.size() + m2.size())).permute([0, 2, 1, 3]).reshape(m1.size(0) * m2.size(0), m1.size(1) * m2.size(1))


def rot2d(theta):
    """ Returns a 2D rotation matrix. """
    return torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def close_shape(x):
    return torch.cat([x, x[0, :].view(1, -1)], dim=0)


# Both point_side() and is_inside_shape() functions are still pretty much WIP.
def point_side(origin, vec, pts):
    """Returns 1 if the point is to the left, 0 on the vector, and -1 if on the right."""
    a = origin
    b = origin + vec
    return torch.sign((b[0] - a[0])*(pts[1] - a[1]) - (b[1] - a[1])*(pts[0] - a[0]))

def is_inside_shape(shape, points):
    """Returns True if the point is inside the shape (a tensor of CCW points defining the shape)."""
    closed = close_shape(shape)
    mask = torch.ones(points.shape[0], dtype=torch.uint8)
    for i in range(points.shape[0]):
        for j in range(shape.shape[0]):
            if point_side(closed[j], closed[j] - closed[j+1], points[i]) == 1:
                mask[i] = 0
                break

    return mask


def plot_tensor_scatter(x, alpha=1., scale=64.):
    """Scatter plot points in the format: ([x, y], scale) or ([x, y]) (in that case you can specify scale)"""
    if(isinstance(x, tuple) or isinstance(x, list)):
        plt.scatter(x[0].detach().numpy()[:,1], x[0].detach().numpy()[:,0], s=50.*x[1].shape[0]*x[1].detach().numpy(), marker='o', alpha=alpha)
        plt.scatter(x[0].detach().numpy()[:,1], x[0].detach().numpy()[:,0], s=64.*x[1].shape[0]*x[1], marker='o', alpha=alpha)
    else:
        plt.scatter(x.detach().numpy()[:,1], x.detach().numpy()[:,0], s=scale, marker='o', alpha=alpha)


def plot_grid(ax, gridx, gridy, **kwargs):
    for i in range(gridx.shape[0]):
        ax.plot(gridx[i,:], gridy[i,:], **kwargs)
    for i in range(gridx.shape[1]):
        ax.plot(gridx[:,i], gridy[:,i], **kwargs)


def make_grad_graph(tensor, filename):
    dot = make_dot(tensor)
    dot.render(filename)

