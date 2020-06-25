import math
from collections import Iterable
from torchviz import make_dot
import torch


# TODO: pythonize this
def grid2vec(*args):
    """Convert a grid of points (such as given by torch.meshgrid) to a tensor of vectors."""
    return torch.cat([args[i].contiguous().view(1, -1) for i in range(len(args))], 0).t().contiguous()


def vec2grid(vec, *args):
    """Convert a tensor of vectors to a grid of points."""
    return tuple((vec.t()[i].view(args).contiguous()).contiguous() for i in range(len(args)))


def rot2d(theta):
    """ Returns a 2D rotation matrix. """
    return torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


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
    return shared_tensors_property(tensors, lambda tensor: tensor.device)


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



