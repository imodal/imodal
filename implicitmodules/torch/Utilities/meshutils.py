import math

import torch
from scipy.spatial import ConvexHull


def area_side(points, **kwargs):
    # For side = 1, left
    # For side = -1, right
    if 'origin' in kwargs and 'direction' in kwargs:
        origin = kwargs['origin']
        direction = kwargs['direction']
        p0 = origin
        p1 = origin + direction
    elif 'p0' in kwargs and 'p1' in kwargs:
        p0 = kwargs['p0']
        p1 = kwargs['p1']
    else:
        raise RuntimeError("area_side(): missing arguments, either ('origin' and 'direction') or ('p0' and 'p1)")

    side = 1
    if 'side' in kwargs:
        side = kwargs['side']

    intersect = False
    if 'intersect' in kwargs:
        intersect = kwargs['intersect']

    if points.dim() == 1:
        points = points.unsqueeze(0)

    out = torch.empty(points.shape[0], dtype=torch.bool)
    for i in range(points.shape[0]):
        result = torch.sign((p1[0] - p0[0])*(points[i][1] - p0[1]) - (p1[1] - p0[1])*(points[i][0] - p0[0]))
        if intersect:
            out[i] = (result == side) or (result == 0)
        else:
            out[i] = (result == side)

    return out


def area_convex_hull(points, **kwargs):
    if 'scatter' not in kwargs:
        raise RuntimeError("area_convex_hull(): missing argument 'scatter'.")

    scatter = kwargs['scatter']

    intersect = False
    if 'intersect' in kwargs:
        intersect = kwargs['intersect']

    convex_hull = extract_convex_hull(scatter)

    return area_convex_shape(points, shape=convex_hull, side=-1, intersect=intersect)


def area_convex_shape(points, **kwargs):
    if 'shape' not in kwargs:
        raise RuntimeError("area_convex_shape(): missing argument 'shape'.")

    shape = kwargs['shape']

    intersect = False
    if 'intersect' in kwargs:
        intersect = kwargs['intersect']

    # For side = 1, shape is defined CW. For side = -1, shape is defined CCW
    side = 1
    if 'side' in kwargs:
        side = kwargs['side']
        if not ((side == -1) or (side == 1)):
            raise RuntimeError("area_convex_shape(): argument 'side' either needs to be 1 (CW) or -1 (CCW).")

    closed = close_shape(shape)

    out = torch.ones(points.shape[0], dtype=torch.bool)

    for i in range(points.shape[0]):
        for j in range(shape.shape[0]):
            if area_side(points[i], p0=closed[j], p1=closed[j+1], intersect=(not intersect), side=side):
                out[i] = False
                break

    return out


def area_shape(points, **kwargs):
    if 'shape' not in kwargs:
        raise RuntimeError("area_shape(): missing argument 'shape'.")

    shape = close_shape(kwargs['shape'])

    # side = 1, ccw
    # side = 0, cw
    side = 1
    if 'side' in kwargs:
        side = kwargs['side']

    out = torch.zeros(points.shape[0], dtype=torch.bool)

    for i in range(points.shape[0]):
        out[i] = (winding_order(points[i], shape, side) >= 1)

    return out


def area_polyline_outline(points, **kwargs):
    if 'polyline' not in kwargs:
        raise RuntimeError("area_polyline_outline(): missing argument 'polyline'.")

    polyline = kwargs['polyline']

    width = 0.
    if 'width' in kwargs:
        width = kwargs['width']

    # If close is found in the arguments and if its true, we close the polyline.
    if 'close' in kwargs:
        if kwargs['close']:
            polyline = close_shape(polyline)

    out = torch.zeros(points.shape[0], dtype=torch.bool)

    for i in range(points.shape[0]):
        is_inside = False
        for j in range(polyline.shape[0] - 1):
            if area_segment(points[i], p0=polyline[j], p1=polyline[j+1], width=width):
                is_inside = True
                break

        out[i] = is_inside

    return out


def area_disc(points, **kwargs):
    if 'center' not in kwargs:
        raise RuntimeError("area_disc(): missing argument 'center'.")

    center = kwargs['center']

    if 'radius' not in kwargs:
        raise RuntimeError("area_disc(): missing argument 'radius'.")

    radius = kwargs['radius']

    return (torch.norm(points - center.unsqueeze(0).repeat(points.shape[0], 1), p=2, dim=1) <= radius).type(dtype=torch.bool)


def area_AABB(points, **kwargs):
    if 'aabb' not in kwargs:
        raise RuntimeError("area_AABB(): missing argument 'aabb'.")

    aabb = kwargs['aabb']

    return aabb.is_inside(points)


def area_segment(points, **kwargs):
    if 'origin' in kwargs and 'direction' in kwargs:
        origin = kwargs['origin']
        direction = kwargs['direction']
        p0 = origin
        p1 = origin + direction
    elif 'p0' in kwargs and 'p1' in kwargs:
        p0 = kwargs['p0']
        p1 = kwargs['p1']
    else:
        raise RuntimeError("area_segment(): missing arguments, either ('origin' and 'direction') or ('p0' and 'p1)")

    if 'width' not in kwargs:
        raise RuntimeError("area_segment(): missing argument 'width'.")

    width = kwargs['width']

    # If there is only one point, we add a dimension
    if points.dim() == 1:
        points = points.unsqueeze(0)

    out = torch.zeros(points.shape[0], dtype=torch.bool)

    for i in range(points.shape[0]):
        if distance_segment(points[i], p0, p1) <= width:
            out[i] = True

    return out


def close_shape(x):
    return torch.cat([x, x[0, :].view(1, -1)], dim=0)


def is_shape_closed(x):
    return torch.all(x[0] == x[-1])


def distance_segment(point, p0, p1):
    dist = torch.dist(p0, p1).item()**2
    if dist == 0.:
        return torch.dist(p0, point).item()

    t = max(0., min(1., torch.dot(point - p0, p1 - p0).item()/dist))
    projection = p0 + t*(p1 - p0)
    return torch.dist(point, projection).item()


def point_side(point, p0, p1):
    return torch.sign((p1[0] - p0[0])*(point[1] - p0[1]) - (p1[1] - p0[1])*(point[0] - p0[0]))


def winding_order(point, shape, side):
    # Assumes shape is closed

    wn = 0
    for i in range(shape.shape[0] - 1):
        if shape[i, 1] <= point[1]:
            if shape[i+1, 1] > point[1]:
                if area_side(point, p0=shape[i], p1=shape[i+1], side=side):
                    wn = wn + 1
        else:
            if shape[i+1, 1] <= point[1]:
                if area_side(point, p0=shape[i], p1=shape[i+1], side=-side):
                    wn = wn - 1

    return wn


def extract_convex_hull(points):
    """Returns a CCW convex hull."""
    hull = ConvexHull(points.numpy())
    return points[hull.vertices]


def fill_area_uniform(area, enclosing_aabb, spacing, **kwargs):
    """Fill a 2D area enclosed by aabb given by the area function (area(pos): return true if in the area, false otherwise)."""
    grid = enclosing_aabb.fill_uniform(spacing)
    return grid[area(grid, **kwargs)]


def fill_area_uniform_density(area, enclosing_aabb, density, **kwargs):
    return fill_area_uniform(area, enclosing_aabb, 1./math.sqrt(density), kwargs)


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


