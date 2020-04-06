import torch
import math

def linear_transform(points, A):
    return torch.bmm(A.repeat(points.shape[0], 1, 1), points.unsqueeze(2)).view_as(points)


def generate_unit_square(points_per_side, dtype=None):
    points = torch.zeros(4*points_per_side, 2, dtype=dtype)
    points[0:points_per_side, 0] = torch.linspace(0., 1., points_per_side + 1, dtype=dtype)[1:]
    points[points_per_side:2*points_per_side, 1] = torch.linspace(0., 1., points_per_side + 1, dtype=dtype)[1:]
    points[points_per_side-1:2*points_per_side, 0] = 1.
    points[2*points_per_side:3*points_per_side, 0] = torch.linspace(1., 0., points_per_side + 1, dtype=dtype)[1:]
    points[2*points_per_side:3*points_per_side, 1] = 1.
    points[3*points_per_side:4*points_per_side, 1] = torch.linspace(1., 0., points_per_side + 1, dtype=dtype)[1:]

    return points - torch.tensor([0.5, 0.5])

def generate_unit_sphere(nb_points, dtype=None):
    return torch.stack(
        [torch.cos(torch.linspace(0, 2.*math.pi, nb_points, dtype=dtype)),
         torch.sin(torch.linspace(0, 2.*math.pi, nb_points, dtype=dtype))], axis=1)

def generate_mesh_grid(aabb, resolution, dtype=None):
    return torch.meshgrid([torch.linspace(kmin, kmax, count, dtype=dtype) for kmin, kmax, count in zip(aabb.kmin, aabb.kmax, resolution)])

