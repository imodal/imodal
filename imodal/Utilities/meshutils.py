import math

import numpy as np
import torch
from scipy.spatial import ConvexHull, Delaunay

from imodal.Kernels.kernels import K_xy


def area_side(points, **kwargs):
    """Marks points that are on one side of the specified separation line on the plan.

    Returns a torch.BoolTensor marking points that are in side **side**. The seperation line can either be specified by two points **p0** and **p1**, or by **origin** and **direction**.

    Parameters
    ----------
    points : torch.Tensor
        Point tensor of dimension (:math:`N`, 2), with :math:`N` number of points, that will be marked.
    p0 : torch.Tensor
        First point tensor of dimension (2) defining the separation line.
    p1 : torch.Tensor
        Second point tensor of dimension (2) defining the separation line and thus its direction.
    origin : torch.Tensor
        Origin vector of dimension (2).
    direction : torch.Tensor
        Direction vector of dimension (2).
    side : int, either +1 or -1, default=1
        +1/-1 to select points to the left/right of the separation line.
    intersect : bool, default=False
        Set this to `True` if points on the line should be accounted inside the marking region.
    Returns
    -------
    torch.BoolTensor
        Bool tensor of dimension :math:`N`, with :math:`i` `True` if point :math:`i` is at side **side** of the defined separation line.
    """
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
    """Marks points inside a computed convex hull.

    Returns a torch.BoolTensor marking points that are inside the convex hull of **scatter**.

    Parameters
    ----------
    points : torch.Tensor
        Points tensor of dimension (:math:`N`, 2), with :math:`N` number of points, that will be marked.
    scatter : torch.Tensor
        Points tensor of dimension (:math:`M`, 2), with :math:`M` number of points, from which the convex hull will be computed.
    intersect : bool, default=False
        Set this to `True` if points on the line should be accounted inside the marking region.

    Returns
    -------
    torch.BoolTensor
        Bool tensor of dimension :math:`N`, with :math:`i` `True` if point :math:`i` is inside the convex hull.
    """
    if 'scatter' not in kwargs:
        raise RuntimeError("area_convex_hull(): missing argument 'scatter'.")

    scatter = kwargs['scatter']

    intersect = False
    if 'intersect' in kwargs:
        intersect = kwargs['intersect']

    if scatter.shape[1] == 2:
        convex_hull = extract_convex_hull(scatter)
    else:
        convex_hull, _ = extract_convex_hull(scatter)

    return area_convex_shape(points, shape=convex_hull, side=-1, intersect=intersect)


def area_convex_shape(points, **kwargs):
    """Marks points inside a convex shape.

    Returns a torch.BoolTensor marking points that are inside the convex shape **shape**.

    Parameters
    ----------
    points : torch.Tensor
        Point tensor of dimension (:math:`N`, 2), with :math:`N` number of points, that will be marked.
    shape : torch.Tensor
        Point tensor of dimension (:math:`M`, 2), with :math:`M` number of points, that defines the convex shape.
    side : either +1 or -1, default=1
        If set to +1/-1 shape is defined as CW/CCW.
    intersect : bool, default=False
        Set this to `True` if points on the line should be accounted inside the marking region.

    Returns
    -------
    torch.BoolTensor
        Bool tensor of dimension :math:`N`, with :math:`i` `True` if point :math:`i` is inside the convex hull.
    """
    if 'shape' not in kwargs:
        raise RuntimeError("area_convex_shape(): missing argument 'shape'.")

    shape = kwargs['shape']

    intersect = False
    if 'intersect' in kwargs:
        intersect = kwargs['intersect']

    # TODO: generalize this for every dimensions
    if points.shape[1] == 3:
        hull = Delaunay(shape)
        return torch.tensor(hull.find_simplex(points) >= 0, dtype=torch.bool)

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
    """Marks points inside a shape.

    Returns a torch.BoolTensor marking points that are inside the shape **shape**. Shape does not need to be convex and can overlap itself.

    Parameters
    ----------
    points : torch.Tensor
        Point tensor of dimension (:math:`N`, 2), with :math:`N` number of points, that will be marked.
    shape : torch.Tensor
        Shape.
    side : either +1 or -1, default=1
        If set to +1/-1 shape is defined as CW/CCW.
    intersect : bool, default=False
        Set this to `True` if points on the line should be accounted inside the marking region.

    Returns
    -------
    torch.BoolTensor
        Bool tensor of dimension :math:`N`, with :math:`i` `True` if point :math:`i` is inside the shape.
    """
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
        out[i] = (winding_order(points[i], shape, side) != 0)

    return out


def area_polyline_outline(points, **kwargs):
    """Marks points that are in the neighborhood of a polyline.

    Parameters
    ----------
    points : torch.Tensor
        Point tensor of dimension (:math:`N`, 2), with :math:`N` number of points, that will be marked.
    polyline : torch.Tensor
        Point tensor of dimension (:math:`M`, 2), with :math:`M` number of vertices, that defines the polyline.
    width : float, default=0.
        Width of the polyline.
    close : bool, default=False
        Set to `True` in order to close the polyline.

    Returns
    -------
    torch.BoolTensor
        Bool tensor of dimension :math:`N`, with :math:`i` `True` if point :math:`i` is inside the shape.
    """
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
    """Marks points that are inside the specified disc in the plan.

    Returns a boolean tensor.

    Parameters
    ----------
    points : torch.Tensor
        Point tensor of dimension (:math:`N`, 2), with :math:`N` number of points, that will be marked.
    center : torch.Tensor
        Point tensor of dimension (2) that defines the center of the disc.
    radius : float
        Radius of the disc.

    Returns
    -------
    torch.BoolTensor
        Bool tensor of dimension :math:`N`, with :math:`i` `True` if point :math:`i` is inside the shape.

    """
    if 'center' not in kwargs:
        raise RuntimeError("area_disc(): missing argument 'center'.")

    center = kwargs['center']

    if 'radius' not in kwargs:
        raise RuntimeError("area_disc(): missing argument 'radius'.")

    radius = kwargs['radius']

    return (torch.norm(points - center.unsqueeze(0).repeat(points.shape[0], 1), p=2, dim=1) <= radius).type(dtype=torch.bool)


def area_AABB(points, **kwargs):
    """Marks points that are inside the specified AABB in the plan.

    Parameters
    ----------
    points : torch.Tensor
        Point tensor of dimension (:math:`N`, 2), with :math:`N` number of points, that will be marked.
    aabb : Utilities.AABB
        The AABB that marks the area.

    Returns
    -------
    torch.BoolTensor
        Bool tensor of dimension :math:`N`, with :math:`i` `True` if point :math:`i` is inside the shape.
    """
    if 'aabb' not in kwargs:
        raise RuntimeError("area_AABB(): missing argument 'aabb'.")

    aabb = kwargs['aabb']

    return aabb.is_inside(points)


def area_segment(points, **kwargs):
    """Marks points that are inside the neighborhood of the specified segment in the plan.

    Parameters
    ----------
    points : torch.Tensor
        Point tensor of dimension (:math:`N`, 2), with :math:`N` number of points, that will be marked.
    p0 : torch.Tensor
        First point defining the separation line.
    p1 : torch.Tensor
        Second point defining the separation line and thus its direction.
    origin : torch.Tensor
        Origin vector.
    direction : torch.Tensor
        Direction vector.
    width : torch.Tensor
        Width of the segment.

    Returns
    -------
    torch.BoolTensor
        Bool tensor of dimension :math:`N`, with :math:`i` `True` if point :math:`i` is inside the shape.
    """
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


def close_shape(shape):
    """Returns the closed shape.

    Parameters
    ----------
    shape : torch.Tensor
        Shape tensor of dimension (:math:`N`), with :math:`N` number of vertices, that will be closed.

    Returns
    -------
    torch.Tensor
        The closed shape.
    """
    return torch.cat([shape, shape[0, :].view(1, -1)], dim=0)


def is_shape_closed(shape):
    """Returns `True` if the input shape is closed.

    Parameters
    ----------
    shape : torch.Tensor
        Shape tensor of dimension (:math:`N`, 2), with :math:`N` number of vertices, that will be closed.

    Returns
    -------
    bool
        `True` if **shape** is closed, `False` otherwise.

    """
    return torch.all(shape[0] == shape[-1])


def distance_segment(point, p0, p1):
    """Returns the minimal distance between a point and a segment.

    Parameters
    ----------
    point : torch.Tensor
        Point tensor of dimension (2) from which distance is computed.
    p0 : torch.Tensor
        First point defining the segment.
    p1 : torch.Tensor
        Second point defining the segment.

    Returns
    -------
    float
        The minimal distance between the point and the segment.
    """
    dist = torch.dist(p0, p1).item()**2
    if dist == 0.:
        return torch.dist(p0, point).item()

    t = max(0., min(1., torch.dot(point - p0, p1 - p0).item()/dist))
    projection = p0 + t*(p1 - p0)
    return torch.dist(point, projection).item()


def point_side(point, p0, p1):
    """Returns the side of a point relative to a line.

    Parameters
    ----------
    point : torch.Tensor
        Point tensor of dimension (2) from which side will be computed.
    p0 : torch.Tensor
        First point defining the line.
    p1 : torch.Tensor
        Second point defining the line, and thus it's direction.

    Returns
    -------
    int
        -1/+1 if the point is on the left/right. 0 if the point is exactly on the line.
    """
    return torch.sign((p1[0] - p0[0])*(point[1] - p0[1]) - (p1[1] - p0[1])*(point[0] - p0[0]))


def winding_order(point, shape, side):
    """Returns the winding order of a point relative to a shape.

    Notes
    -----
    This function assumes the shape is closed.

    Parameters
    ----------
    point : torch.Tensor
        Point tensor of dimension (2) around which the winding order will be computed.
    shape : torch.Tensor
    Shape tensor of dimension (:math:`N`), with :math:`N` number of vertices, from which the winding order will be computed.
    side : int, either +1 or -1
        Set to +1/-1 if shape is defined CCW/CW.

    Returns
    -------
    int
        The computed winding order.
    """
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
    """Extracts a convex hull from a set of points.

    Notes
    -----
    The output shape in 2D is CCW defined. In 3D outputs a tuple of points and triangle list forming the convex hull. Uses Scipy internaly.

    Parameters
    ----------
    points : torch.Tensor
        Points tensor of dimension (:math:`N`, d), with :math:`N` the number of points and :math:`d` the dimension, from which the convex hull will be computed.

    Returns
    -------
    torch.Tensor
        If in 2D, the resulting convex hull, of dimension (:math:`M`, 2), with :math:`M` the number of points the convex hull contains.
        If in 3D, a 2-tuple with first element representing the points of the convex hull of dimension (:math:`M`, 3), with :math:`M` the number of points the convex hull contains and a list of 3-tuple representing the faces of the hull.
    """

    if isinstance(points, np.ndarray):
        hull = ConvexHull(points)
    elif isinstance(points, torch.Tensor):
        hull = ConvexHull(points.numpy())
    else:
        raise ValueError("extract_convex_hull(): points data array type {arraytype} not understood!".format(arraytype=type(points)))

    if points.shape[1] == 2:
        return points[hull.vertices]
    elif points.shape[1] == 3:
        return points[hull.vertices], hull.simplices
    else:
        return ValueError


def fill_area_uniform(area, enclosing_aabb, spacing, **kwargs):
    """Uniformly fills a 2D area enclosed given by a callable.

    The area callable should have the following signature:
    

    Parameters
    ----------
    area : callable
        Callable defining the area that will be filled.
    enclosing_aabb : Utilities.AABB
        Bounding box enclosing the `area` callable function.
    spacing : float
        Distance between points.
    kwargs : dict
        Arguments passed to the `area` callable function.

    Returns
    -------
    torch.Tensor
        Points tensor of dimension (:math:`N`, 2), with :math:`N` the number of points.

    """
    grid = enclosing_aabb.fill_uniform_spacing(spacing)
    return grid[area(grid, **kwargs)]


def fill_area_uniform_density(area, enclosing_aabb, density, **kwargs):
    """Fill a 2D area enclosed by aabb given by the area function uniformly.

    Parameters
    ----------
    area : callable
        Callable defining the area that will be filled.
    enclosing_aabb : Utilities.AABB
        AABB.
    density : float
        Density of points.
    kwargs : dict
        Arguments passed to the area function.

    Returns
    -------
    torch.Tensor
        Points tensor of dimension (:math:`N`, 2), with :math:`N` the number of points.
    """
    return fill_area_uniform(area, enclosing_aabb, 1./math.sqrt(density), **kwargs)


def fill_area_random(area, aabb, N, **kwargs):
    """Randomly fill a 2D area enclosed by aabb given by the area function.

    The random process follows a Poisson distribution. Sampling is done using a rejection sampling algorithm.

    Parameters
    ----------
    area : callable
        Callable defining the area that will be filled.
    enclosing_aabb : Utilities.AABB
        Bounding box enclosing the `area` callable function.
    N : int
        Number of points to generate.
    kwargs : dictpp
        Arguments passed to the area function.

    Returns
    -------
    torch.Tensor
        Points tensor of dimension (:math:`N`, 2), with :math:`N` the number of points.
    """
    points = torch.zeros(N, 2)

    for i in range(0, N):
        accepted = False
        while(not accepted):
            point = aabb.fill_random(1)
            if torch.all(area(point, **kwargs)):
                accepted = True
                points[i, :] = point

    return points


def fill_area_random_density(area, aabb, density, **kwargs):
    """Randomly fill a 2D area enclosed by aabb given by the area function.

    The random process follows a Poisson distribution. Sampling is done using a rejection sampling algorithm.

    Parameters
    ----------
    area : callable
        Callable defining the area that will be filled.
    enclosing_aabb : Utilities.AABB
        Bounding box enclosing the `area` callable function.
    density : int
        Density of points to generate.
    kwargs : dict
        Arguments passed to the area function.

    Returns
    -------
    torch.Tensor
        Points tensor of dimension (:math:`N`, 2), with :math:`N` the number of points.
    """
    return fill_area_random(area, aabb, int(aabb.area * density), **kwargs)


def compute_centers_normals_lengths(vertices, faces):
    """

    """
    v0, v1, v2 = vertices.index_select(0, faces[:, 0]), vertices.index_select(0, faces[:, 1]), vertices.index_select(0, faces[:, 2])
    centers = 0.5 * (v0 + v1 + v2)
    normals = 0.5 * torch.cross(v1 - v0, v2 - v0)
    lengths = (normals**2).sum(dim=1)[:, None].sqrt()
    return centers, normals, lengths


def kernel_smooth(points, kernel):
    """

    """
    K = kernel(points, points)
    return torch.mm(K, points)/torch.sum(K, dim=1).unsqueeze(1)


def gaussian_kernel_smooth(points, sigma):
    """

    """
    return kernel_smooth(points, lambda x, y: K_xy(x, y, sigma))


def resample_curve(curve, density):
    """

    """
    if curve.shape[0] <= 1:
        return curve

    lengths = torch.norm(curve[1:] - close_shape(curve)[:-2], dim=1)
    length = torch.sum(lengths)
    out = torch.zeros(math.floor(length*density), curve.shape[1])

    interval = 1./density
    p0 = curve[0]
    p1 = curve[1]
    cur_length = 0.
    length_index = 0
    for i in range(0, out.shape[0]):
        t = (cur_length-torch.sum(lengths[:length_index]))*lengths[length_index]
        out[i] = t*p1 - p0
        cur_length = cur_length + interval
        if cur_length >= torch.sum(lengths[:length_index]):
            p0 = p1
            p1 = curve[length_index+2]
            length_index = length_index + 1

    return out

