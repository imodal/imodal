import os
from collections import Iterable
import pickle
import copy

import torch
from numpy import loadtxt, savetxt
import meshio

from imodal.HamiltonianDynamic import Hamiltonian, shoot
from imodal.DeformationModules import SilentBase, CompoundModule, DeformationGrid
from imodal.Manifolds import Landmarks


class Deformable:
    def __init__(self):
        pass

    def to_device(self, device):
        self.__device = device

    @property
    def device(self):
        return self.__device

    @property
    def geometry(self):
        """
        Returns geometric informations of the deformable.
        """
        raise NotImplementedError()

    @property
    def _has_backward(self):
        """
        Returns True if the deformable needs backward shooting.
        """
        raise NotImplementedError()

    def _backward_module(self):
        """
        The backward deformation module that will be used while backward shooting.
        """
        raise NotImplementedError()

    def _to_deformed(self):
        """
        Returns deformed geometric informations of the deformable.
        """
        raise NotImplementedError()


class DeformableGrid:
    def __init__(self, extent, resolution, module_label=None):
        self.__silent_module = DeformationGrid(extent, resolution, label=module_label)

    @property
    def silent_module(self):
        return self.__silent_module

    @property
    def geometry(self):
        """
        Returns geometric informations of the deformable.
        """
        return (self.__silent_module.togrid(),)

    @property
    def _has_backward(self):
        return False

    def to_device(self, device):
        super().to_device(device)
        self.__silent_module.to_(device=device)

    def _to_deformed(self, gd):
        return (gd,)


class DeformablePoints(Deformable):
    """
    Deformable object representing a collection of points in space.
    """
    def __init__(self, points, label=None):
        super().__init__()
        self.__silent_module = SilentBase(Landmarks(points.shape[1], points.shape[0], gd=points), label=label)

    @property
    def silent_module(self):
        return self.__silent_module

    @classmethod
    def load_from_file(cls, filename, dtype=None, **kwargs):
        """
        Load points from file.

        Correct loader will be infered from the file extension.

        Parameters
        ----------
        filename : str
            Filename of the file to load.
        dtype : torch.dtype, default=None
            Type to transform the data into. If set to None, data will be transformed into the value returned by torch.get_default_dtype().
        kwargs :
            Arguments given to the loader (see specific loaders for parameters).
        """
        file_extension = os.path.split(filename)[1]
        if file_extension == '.csv':
            return cls.load_from_csv(filename, dtype=dtype, **kwargs)
        elif file_extension == '.pickle' or file_extension == '.pkl':
            return cls.load_from_pickle(filename, dtype=dtype)
        elif file_extension in meshio.extension_to_filetype.keys():
            return cls.load_from_mesh(filename, dtype=dtype)
        else:
            raise RuntimeError("DeformablePoints.load_from_file(): could not load file {filename}, unrecognised file extension!".format(filename=filename))

    @classmethod
    def load_from_csv(cls, filename, dtype=None, **kwargs):
        """
        Load points from a csv file.
        """
        points = loadtxt(filename, **kwargs)
        return cls(torch.tensor(points, dtype=dtype))

    @classmethod
    def load_from_pickle(cls, filename, dtype=None):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                return cls(torch.tensor(data['points'], dtype=dtype))
            elif isinstance(data, Iterable):
                return cls(torch.tensor(data, dtype=dtype))
            else:
                raise RuntimeError("DeformablePoints.load_from_pickle(): could not infer point dataset from pickle {filename}".format(filename=filename))

    @classmethod
    def load_from_mesh(cls, filename, dtype=None):
        mesh = meshio.read(filename)
        return torch.tensor(mesh.points, dtype=dtype)

    @property
    def geometry(self):
        return (self.silent_module.manifold.gd,)

    @property
    def _has_backward(self):
        return False

    def to_device(self, device):
        super().to_device(device)
        self.__silent_module.to_(device=device)

    def save_to_file(self, filename, **kwargs):
        file_extension = os.path.split(filename)[1]
        if file_extension == '.csv':
            return self.save_to_csv(filename, **kwargs)
        elif file_extension == '.pickle' or file_extension == '.pkl':
            return self.save_to_pickle(filename, **kwargs)
        elif file_extension in meshio.extension_to_filetype.keys():
            return self.save_to_mesh(filename, **kwargs)
        else:
            raise RuntimeError("DeformablePoints.load_from_file(): could not load file {filename}, unrecognised file extension!".format(filename=filename))

    def save_to_csv(self, filename, **kwargs):
        savetxt(filename, self.geometry[0].detach().cpu().tolist(), **kwargs)

    def save_to_pickle(self, filename, container='array', **kwargs):
        with open(filename, 'wb') as f:
            if container == 'array':
                pickle.dump(self.geometry[0].detach().cpu().tolist(), f)
            elif container == 'dict':
                pickle.dump({'points': self.geometry[0].detach().cpu().tolist()}, f)
            else:
                raise RuntimeError("DeformablePoints.save_to_pickle(): {container} container type not recognized!")
        pass

    def save_to_mesh(self, filename, **kwargs):
        points_count = self.geometry[0].shape[0]
        meshio.write_points_cells(filename, self.geometry[0].detach().cpu().numpy(), [('polygon'+str(points_count), torch.arange(points_count).view(1, -1).numpy())], **kwargs)

    def _to_deformed(self, gd):
        return (gd,)


class DeformableMesh(DeformablePoints):
    def __init__(self, points, triangles, label=None):
        """
        Parameters
        ----------
        points : torch.Tensor
            Points.
        triangles : torch.Tensor
            Triangles.
        """
        self.__triangles = triangles
        super().__init__(points, label=label)

    @classmethod
    def load_from_file(cls, filename, dtype=None):
        """
        Load and initialize the deformable from a file. Uses meshio, see its documentation for supported format.

        Parameters
        ----------
        filename : str
            Filename
        dtype : torch.dtype, default=None
            Type.
        """
        mesh = meshio.read(filename)
        points = torch.tensor(mesh.points, dtype=dtype)
        triangles = torch.tensor(mesh.cell_dict['triangle'], torch.int)
        return cls(points, triangles)

    def save_to_file(self, filename):
        meshio.write_points_cells(filename, self.silent_module.manifold.gd.detach().cpu().numpy(), [('triangle', self.__triangles.cpu())])

    @property
    def triangles(self):
        return self.__triangles

    @property
    def geometry(self):
        return (self.silent_module.manifold.gd, self.__triangles)

    def to_device(self, device):
        super().to_device(device)
        self.__triangles = self.__triangles.to(device=device)

    def _to_deformed(self, gd):
        return (gd, self.__triangles)


def deformables_compute_deformed(deformables, modules, solver, it, costs=None, intermediates=None, controls=None, t1=1.):
    """
    Computes deformation of deformables by modules.

    Parameters
    ----------
    deformables : list
        List of deformable objects we want to deform.
    modules : list
        List of deformation modules that will perform the deformation.
    solver : str
        Name of the numerical ODE solver that will be used for shooting. See shoot() for a list of available solvers.
    it : int
        Number of steps the numerical ODE solver will perform.
    costs : dict, default=None
        If set to a dict, deformation cost will be filled under the 'deformation' key.
    intermediates : dict, default=None
        If set to a dict, intermediates states and control will be filled in the same manner than for shoot().
    controls : list, default=None
        The controls that will be used by the deformation modules. If set to None (by default), geodesic controls will be computed. Has to have the same length than the number of steps that will be performed while shooting.
    t1 : float, default=1.
        Final time of the ODE solver.

    Returns
    -------
    list :
        List of deformed objects.
    """
    assert isinstance(costs, dict) or costs is None
    assert isinstance(intermediates, dict) or intermediates is None

    # Regroup silent modules of each deformable and build a compound module
    silent_modules = [deformable.silent_module for deformable in deformables]
    compound = CompoundModule([*silent_modules, *modules])

    incontrols = None
    if controls is not None:
        # Construct control list
        incontrols = []
        for control in controls:
            silent_modules_controls = []
            for i in range(len(silent_modules)):
                silent_modules_controls.append(torch.tensor([]))
            incontrols.append([*silent_modules_controls, *control])

    # Forward shooting
    shoot(Hamiltonian(compound), solver, it, intermediates=intermediates, controls=incontrols, t1=t1)

    # Regroup silent modules of each deformable thats need to shoot backward
    shoot_backward = any([deformable._has_backward for deformable in deformables])

    forward_silent_modules = copy.deepcopy(silent_modules)

    if shoot_backward:
        # Backward shooting is needed
        # Build/assemble the modules that will be shot backward
        backward_modules = [deformable._backward_module() for deformable in deformables if deformable._has_backward]
        compound = CompoundModule([*silent_modules, *backward_modules, *modules])

        # Reverse the moments for backward shooting
        compound.manifold.negate_cotan()

        backward_controls = None
        if controls is not None:
            # Construct control list
            backward_controls = []
            for control in controls[::-1]:
                silent_modules_controls = []
                backward_modules_controls = []
                for i in range(len(silent_modules)):
                    silent_modules_controls.append(torch.tensor([]))
                for i in range(len(backward_modules)):
                    backward_modules_controls.append(torch.tensor([]))
                modules_control = [-module_control for module_control in control]

                backward_controls.append([*silent_modules_controls, *backward_modules_controls, *modules_control])

        shoot(Hamiltonian(compound), solver, it, t1=t1, controls=backward_controls)

    # For now, we need to compute the deformation cost after each shooting (and not before any shooting) for computation tree reasons
    if costs is not None:
        costs['deformation'] = compound.cost()

    # Ugly way to compute the list of deformed objects. Not intended to stay!
    deformed = []
    for deformable, forward_silent_module in zip(deformables, forward_silent_modules):
        if deformable._has_backward:
            deformed.append(deformable._to_deformed(backward_modules.pop(0).manifold.gd))
        else:
            deformed.append(deformable._to_deformed(forward_silent_module.manifold.gd))

    return deformed

