import math
from collections import Iterable

from torchviz import make_dot
import torch
import meshio


# TODO: pythonize this
def grid2vec(*args):
    """Convert a grid of points (such as given by torch.meshgrid) to a tensor of vectors."""
    return torch.cat([args[i].contiguous().view(1, -1) for i in range(len(args))], 0).t().contiguous()


def vec2grid(vec, *args):
    """Convert a tensor of vectors to a grid of points."""
    return tuple((vec.t()[i].view(args).contiguous()).contiguous() for i in range(len(args)))


def linear_transform(points, A):
    """ Applies a linear transformation to a point tensor.

    Parameters
    ----------
    points : torch.Tensor
        A :math:`N\\times d` tensor that will be transformed.
    A : torch.Tensor
        A :math:`d\\times d` matrix that represent the linear transformation.

    Returns
    -------
    torch.Tensor
        The transformed points.
    """
    return torch.bmm(A.repeat(points.shape[0], 1, 1), points.unsqueeze(2)).view_as(points)


def rot2d(theta):
    """ Returns a 2D rotation matrix. """
    return torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rot2d_vec(thetas):
    cos = torch.cos(thetas)
    sin = torch.sin(thetas)

    return torch.stack([torch.stack([cos, -sin], dim=1),
                        torch.stack([sin, cos], dim=1)], dim=2)


def rot3d_x(theta):
    return torch.tensor([[1., 0., 0.],
                         [0., math.cos(theta), -math.sin(theta)],
                         [0., math.sin(theta), math.cos(theta)]])


def rot3d_y(theta):
    return torch.tensor([[math.cos(theta), 0., -math.sin(theta)],
                         [0., 1., 0.],
                         [math.sin(theta), 0., math.cos(theta)]])


def rot3d_z(theta):
    return torch.tensor([[math.cos(theta), -math.sin(theta), 0.],
                         [math.sin(theta), math.cos(theta), 0.],
                         [0., 0., 1.]])


def rot3d_x_vec(thetas):
    assert len(thetas.shape) == 1
    device = thetas.device
    zeros = torch.zeros(len(thetas), device=device)
    ones = torch.ones(len(thetas), device=device)
    sin = torch.sin(thetas)
    cos = torch.cos(thetas)

    return torch.stack([torch.stack([ones, zeros, zeros], dim=1),
                        torch.stack([zeros, cos, -sin], dim=1),
                        torch.stack([zeros, sin, cos], dim=1)], dim=2).transpose(1, 2)


def rot3d_y_vec(thetas):
    assert len(thetas.shape) == 1
    device = thetas.device
    zeros = torch.zeros(len(thetas), device=device)
    ones = torch.ones(len(thetas), device=device)
    sin = torch.sin(thetas)
    cos = torch.cos(thetas)

    return torch.stack([torch.stack([cos, zeros, sin], dim=1),
                        torch.stack([zeros, ones, zeros], dim=1),
                        torch.stack([-sin, zeros, cos], dim=1)], dim=2).transpose(1, 2)


def rot3d_z_vec(thetas):
    assert len(thetas.shape) == 1
    device = thetas.device
    zeros = torch.zeros(len(thetas), device=device)
    ones = torch.ones(len(thetas), device=device)
    sin = torch.sin(thetas)
    cos = torch.cos(thetas)

    return torch.stack([torch.stack([cos, -sin, zeros], dim=1),
                        torch.stack([sin, cos, zeros], dim=1),
                        torch.stack([zeros, zeros, ones], dim=1)], dim=2).transpose(1, 2)


def translation_matrix4d(translation, dtype=None):
    return torch.tensor([[1., 0., 0., translation[0]],
                         [0., 1., 0., translation[1]],
                         [0., 0., 1., translation[2]],
                         [0., 0., 0., 1.]], dtype=dtype)


def scale_matrix4d(scales, dtype=None):
    return torch.tensor([[scales[0], 0., 0., 0.],
                         [0., scales[1], 0., 0.],
                         [0., 0., scales[2], 0.],
                         [0., 0., 0., 1.]], dtype=dtype)


def rigid_deformation3d(angles, translation):
    rot = torch.eye(4)
    rot[0:3, 0:3] = rot3d_x(angles[0]) @ rot3d_y(angles[1]) @ rot3d_z(angles[2])
    return translation_matrix4d(translation) @ rot


def extent_transformation4d(extent0, extent1):
    scales = [length1/length0 for length0, length1 in zip(extent0.shape, extent1.shape)]
    translations = [min1 - min0 for min0, min1 in zip(extent0.kmin, extent1.kmin)]

    return scale_matrix4d(scales) @ translation_matrix4d(translations)


def points2pixels(points, frame_shape, frame_extent, toindices=False):
    """ Transform points to pixel space. """
    scale_u, scale_v = (frame_shape[1]-1)/frame_extent.width, (frame_shape[0]-1)/frame_extent.height
    u1, v1 = scale_u*(points[:, 0] - frame_extent.xmin), scale_v*(points[:, 1] - frame_extent.ymin)

    if toindices:
        u1 = torch.floor(u1).long()
        v1 = torch.floor(v1).long()

    return torch.stack([v1, u1], dim=1)


def pixels2points(pixels, frame_shape, frame_extent):
    """ Transform points from pixel space onto an uniformly mapped square space defined by an extent. """
    scale_x, scale_y = frame_extent.width/(frame_shape[1]-1), frame_extent.height/(frame_shape[0]-1)

    x, y = scale_x*pixels[:, 1] + frame_extent.xmin, scale_y*pixels[:, 0] + frame_extent.ymin

    return torch.stack([x, y], dim=1)


def points2nel(points, frame_shape, frame_extent, toindices=False):
    scales = [(shape-1)/extent_shape for shape, extent_shape in zip(frame_shape, frame_extent.shape)]

    uv = [scale*(points[:, i] - extent_min) for scale, extent_min, i in zip(scales, frame_extent.kmin, range(frame_extent.dim))]

    if toindices:
        uv = [torch.floor(u).long() for u in uv]

    return torch.stack(uv, dim=1)


def nel2points(nels, frame_shape, frame_extent):
    scales = [extent_shape/(shape-1) for shape, extent_shape in zip(frame_shape, frame_extent.shape)]

    xy = [scale*nels[:, i] + extent_min for scale, extent_min, i in zip(scales, frame_extent.kmin, range(frame_extent.dim))]

    return torch.stack(xy, dim=1)


def voxels2points_affine(voxels, cube_shape, affine):
    points = linear_transform(torch.cat([voxels, torch.ones(voxels.shape[0], 1, device=voxels.device)], dim=1), affine)[:, 0:3]
    
    return points


def points2voxels_affine(points, cube_shape, affine, toindices=False):
    pixels = linear_transform(torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=1), affine)[:, 0:3]

    if toindices:
        pixels = pixels.floor().long()

    return pixels


def flatten_tensor_list(l, out_list=None):
    """Simple recursive list flattening function that stops at torch.Tensor (without unwrapping)."""
    if out_list is None:
        out_list = []

    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, torch.Tensor):
            flatten_tensor_list(el, out_list)
        elif isinstance(el, torch.Tensor):
            out_list.append(el)
        else:
            continue

    return out_list


# TODO: Maybe generalise this to more than torch.Tensor? (duck typing)
def shared_tensors_property(tensors, prop):
    """ Check if all tensors share the same property.
    
    given by prop(tensor). Ignores None tensors and None property values. Returns the shared property or None if the property is not shared. 

    Parameters
    ----------
    tensors : Iterable
        Iterable of tensors we want to check the common property from.
    prop : callable
        Callable function which outputs the parameter we want to compare of the input tensor.

    Returns
    -------
    object
        Common property shared by all tensors. Returns none if properties are different among the tensors.

    """
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
    """ Returns the common device on which tensors (an iterable of torch.Tensor) lives. Return None if tensors are on different devices.

    Parameters
    ----------
    tensors : Iterable
        Iterable of tensors.

    Returns
    -------
    torch.device
        The common device of the iterable of tensors. None if tensors lives on different devices.
    """
    return shared_tensors_property(tensors, lambda tensor: str(tensor.device))


def tensors_dtype(tensors):
    """ Returns the common dtypes on which tensors (an iterable of torch.Tensor) lives. Return None if tensors are of different dtypes.

    Parameters
    ----------
    tensors : Iterable
        Iterable of tensors.

    Returns
    -------
    torch.dtype
        The common dtype of the iterable of tensors. None if tensors lives on different devices.
    """
    return shared_tensors_property(tensors, lambda tensor: tensor.dtype)


def append_in_dict_of_list(base, d):
    for key in d.keys():
        if key not in base:
            base[key] = [d[key]]
        else:
            base[key].append(d[key])


def make_grad_graph(tensor, filename, params=None):
    make_dot(tensor, params=params).render(filename)


def export_mesh(filename, points, triangles):
    meshio.write_points_cells(filename, points.numpy(), [('triangle', triangles.numpy())])


def export_implicit1_growth(filename, points, growth):
    assert growth.shape[2] == 1

    meshio.write_points_cells(filename, points.numpy(), [('polygon'+str(points.shape[0]), torch.arange(points.shape[0]).view(1, -1).numpy())], point_data={'growth': growth.numpy().reshape(-1, points.shape[1])})


def export_point_basis(filename, points, basis):
    meshio.write_points_cells(filename, points.numpy(), [('polygon'+str(points.shape[0]), torch.arange(points.shape[0]).view(1, -1).numpy())], point_data={'basis_x': basis[:, :, 0].numpy(), 'basis_y': basis[:, :, 1].numpy(), 'basis_z': basis[:, :, 2].numpy()})


def import_implicit1_growth(filename, fieldname='growth', dtype=None):
    mesh = meshio.read(filename)
    if fieldname not in mesh.point_data.keys():
        raise RuntimeError("{filename} mesh does not have field named {fieldname}!".format(filename=filename, fieldname=fieldname))

    return torch.tensor(mesh.points, dtype=dtype), torch.tensor(mesh.point_data[fieldname]).reshape(mesh.points.shape[0], -1, 1)


def import_point_basis(filename, fieldname='basis', dtype=None):
    mesh = meshio.read(filename)

    def _import_basis(fieldname):
        if fieldname not in mesh.point_data.keys():
            raise RuntimeError("{filename} mesh does not have field named {fieldname}!".format(filename=filename, fieldname=fieldname))

        return torch.tensor(mesh.point_data[fieldname], dtype=dtype)

    basis_x = _import_basis(fieldname + "_x")
    basis_y = _import_basis(fieldname + "_y")
    basis_z = _import_basis(fieldname + "_z")

    basis = torch.stack([basis_x, basis_y, basis_z], dim=1)

    return torch.tensor(mesh.points, dype=dtype), basis

