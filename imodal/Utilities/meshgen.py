import torch
import math

from imodal.Utilities import rot2d, linear_transform


def generate_unit_square(points_per_side, dtype=None):
    """ Generates an unit square on the plane.

    Parameters
    ----------
    points_per_side : int
        Number of points that will be generated for each side of the square.
    dtype : torch.dtype, default=None
        Type of the output points. If None, dtype is set to `torch.get_default_dtype()`.

    Returns
    -------
    torch.Tensor
        Points of the output square.
    """
    points = torch.zeros(4*points_per_side, 2, dtype=dtype)
    points[0:points_per_side, 0] = torch.linspace(0., 1., points_per_side + 1, dtype=dtype)[1:]
    points[points_per_side:2*points_per_side, 1] = torch.linspace(0., 1., points_per_side + 1, dtype=dtype)[1:]
    points[points_per_side-1:2*points_per_side, 0] = 1.
    points[2*points_per_side:3*points_per_side, 0] = torch.linspace(1., 0., points_per_side + 1, dtype=dtype)[1:]
    points[2*points_per_side:3*points_per_side, 1] = 1.
    points[3*points_per_side:4*points_per_side, 1] = torch.linspace(1., 0., points_per_side + 1, dtype=dtype)[1:]

    return points - torch.tensor([0.5, 0.5])


def generate_unit_circle(angular_resolution, dtype=None):
    """ Generates an unit circle on the plane.
    The generated circle is not closed, i.e. first and last point are not the same.

    Parameters
    ----------
    angular_resolution : int
        Angular resolution of the circle.
    dtype : torch.dtype, default=None
        Type of the output points. If None, dtype is set to `torch.get_default_dtype()`.

    Returns
    -------
    torch.Tensor
        Points of the output circle
    """
    return torch.stack(
        [torch.cos(torch.linspace(0, 2.*math.pi, angular_resolution + 1, dtype=dtype)[:-1]),
         torch.sin(torch.linspace(0, 2.*math.pi, angular_resolution + 1, dtype=dtype)[:-1])], axis=1)


def generate_rectangle(aabb, points_density, dtype=None):
    """ Generates a rectangle on the plane.

    Parameters
    ----------
    aabb : Utilities.AABB
        Boundaries of the resulting rectangle.
    points_density : float
        Linear point density of the resulting rectangle.
    dtype : torch.dtype, default=None
        Type of the output points. If None, dtype is set to `torch.get_default_dtype()`.

    Returns
    -------
    torch.Tensor
        Points of the output rectangle.
    """
    if aabb.dim != 2:
        raise NotImplementedError()

    points_x = math.ceil(aabb.shape[0]*points_density)
    points_y = math.ceil(aabb.shape[1]*points_density)
    points = torch.zeros(2*(points_x+points_y), 2, dtype=dtype)

    # Bottom side
    points[0:points_x, 0] = torch.linspace(aabb.xmin, aabb.xmax, points_x + 1, dtype=dtype)[1:]
    points[0:points_x, 1] = aabb.ymin*torch.ones(points_x)

    # Right side
    points[points_x:points_x+points_y, 0] = aabb.xmax*torch.ones(points_y)
    points[points_x:points_x+points_y, 1] = torch.linspace(aabb.ymin, aabb.ymax, points_y + 1, dtype=dtype)[1:]

    # Top side
    points[points_x+points_y:2*points_x+points_y, 0] = torch.linspace(aabb.xmax, aabb.xmin, points_x + 1, dtype=dtype)[1:]
    points[points_x+points_y:2*points_x+points_y, 1] = aabb.ymax*torch.ones(points_x)

    # Left side
    points[2*points_x+points_y:, 0] = aabb.xmin*torch.ones(points_y)
    points[2*points_x+points_y:, 1] = torch.linspace(aabb.ymax, aabb.ymin, points_y + 1, dtype=dtype)[1:]

    return points


def generate_mesh_grid(aabb, resolution, dtype=None):
    """ Generates a grid on the plane.

    Parameters
    ----------
    aabb : Utilities.AABB
        Boundaries of the resulting grid.
    resolution : Iterable
        Side resolution of the resulting grid.
    dtype : torch.dtype, default=None
        Type of the output points. If None, dtype is set to `torch.get_default_dtype()`.

    Returns
    -------
    tuple
        2-tuple of tensor representing the grid.
    """
    return torch.meshgrid([torch.linspace(kmin, kmax, count, dtype=dtype) for kmin, kmax, count in zip(aabb.kmin, aabb.kmax, resolution)])


def generate_disc_density(density, outer_radius=1., inner_radius=0.):
    """ Generate points on a disc with constant surfacic density.

    Parameters
    ----------
    density : float
        Point density of the resulting disc.
    outer_radius : float, default=1.
        Radius of the outer boundary.
    inner_radius : float, default=0.
        Radius of the inner boundary.
    dtype : torch.dtype, default=None
        Type of the output points. If None, dtype is set to `torch.get_default_dtype()`.

    Returns
    -------
    torch.Tensor
        Tensor of points representing the disc on the plane.
    """
    assert outer_radius > inner_radius

    radials = torch.linspace(inner_radius, outer_radius, math.ceil((outer_radius-inner_radius)*density)).tolist()
    angular_resolutions = [2.*math.pi*r*density for r in radials]

    return torch.cat([torch.stack([
        r*torch.cos(torch.linspace(0., 2.*math.pi, math.ceil(angular_resolution))),
        r*torch.sin(torch.linspace(0., 2.*math.pi, math.ceil(angular_resolution)))], dim=1)
                      for angular_resolution, r in zip(angular_resolutions, radials)])


def generate_boudin(left_width, right_width, height, thickness, arc_resolution, cap_resolution, fill_resolution):
    t = torch.linspace(0., math.pi, arc_resolution)
    half_circle = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
    t = torch.linspace(math.pi, 0., arc_resolution)
    reversed_half_circle = torch.stack([torch.cos(t), torch.sin(t)], dim=1)

    left_arc = linear_transform(half_circle, rot2d(math.pi/2.))*torch.tensor([left_width, height])
    right_arc = linear_transform(reversed_half_circle, rot2d(math.pi/2.))*torch.tensor([right_width, height-thickness*2.]) + torch.tensor([thickness, 0.])

    top_cap = linear_transform(half_circle, rot2d(-math.pi/2.))*torch.tensor([thickness, thickness]) + torch.tensor([thickness, height-thickness])
    bottom_cap = linear_transform(half_circle, rot2d(-math.pi/2.))*torch.tensor([thickness, thickness]) + torch.tensor([thickness, -height+thickness])

    top_fill = torch.stack([torch.linspace(thickness, 0., fill_resolution), height*torch.ones(fill_resolution)], dim=1)[1:-1]
    bottom_fill = torch.stack([torch.linspace(0., thickness, fill_resolution), -height*torch.ones(fill_resolution)], dim=1)[1:-1]

    boudin = torch.cat([left_arc, bottom_fill, bottom_cap, right_arc, top_cap, top_fill])

    return boudin


