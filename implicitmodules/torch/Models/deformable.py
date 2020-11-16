import os
from collections import Iterable
import pickle
import copy

import torch
from numpy import loadtxt, savetxt
import meshio

from implicitmodules.torch.HamiltonianDynamic import Hamiltonian, shoot
from implicitmodules.torch.DeformationModules import SilentBase, CompoundModule, SilentLandmarks
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.Utilities import deformed_intensities, AABB, load_greyscale_image, pixels2points, vec2grid, plot_grid
import matplotlib.pyplot as plt


class Deformable:
    """
    Base class for deformable, which are the manipulated objects by the models.
    """
    def __init__(self, manifold, module_label=None):
        self.__silent_module = SilentBase(manifold, module_label)

    @property
    def silent_module(self):
        return self.__silent_module

    @property
    def geometry(self):
        """
        Returns geometric informations of the deformable.
        """
        raise NotImplementedError()

    @property
    def _has_backward(self):
        raise NotImplementedError()

    def _backward_module(self):
        raise NotImplementedError()

    def compute_deformed(self, modules, solver, it, costs=None, intermediates=None):
        """
        Computes the deformation.
        """
        raise NotImplementedError()

    def _to_deformed(self):
        raise NotImplementedError()


class DeformablePoints(Deformable):
    """
    Deformable object representing a collection of points in space.
    """
    def __init__(self, points):
        super().__init__(Landmarks(points.shape[1], points.shape[0], gd=points))

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
    @classmethod
    def load_from_file(cls, filename, dtype=None, **kwargs):
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

    def save_to_file(self, filename, **kwargs):
        file_extension = os.path.split(filename)[1]
        if file_extension == '.csv':
            return self.save_to_csv(filename, **kwargs)
        elif file_extension == '.pickle' or file_extension == '.pkl':
            return self.save_to_pickle(filename, **kwargs)
        elif file_extension in meshio.extension_to_filetype.keys():
            return cls.save_to_mesh(filename, **kwargs)
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

    def compute_deformed(self, modules, solver, it, costs=None, intermediates=None):
        assert isinstance(costs, dict) or costs is None
        assert isinstance(intermediates, dict) or intermediates is None

        compound = CompoundModule([self.silent_module, *modules])

        # Compute the deformation cost if needed
        if costs is not None:
            compound.compute_geodesic_control(compound.manifold)
            costs['deformation'] = compound.cost()

        # Shoot the dynamical system
        shoot(Hamiltonian(compound), solver, it, intermediates=intermediates)

        return self._to_deformed(self.module.manifold.gd)

    def _to_deformed(self, gd):
        return (gd,)


# class DeformablePolylines(DeformablePoints):
#     def __init__(self, points, connections):
#         self.__connections = connections
#         super().__init__(points)

#     @classmethod
#     def load_from_file(cls, filename):
#         pass

#     @property
#     def geometry(self):
#         return (self.module.manifold.gd.detach(), self.__connections)


class DeformableMesh(DeformablePoints):
    def __init__(self, points, triangles):
        self.__triangles = triangles
        super().__init__(points)

    @classmethod
    def load_from_file(cls, filename, dtype=None):
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

    def _to_deformed(self, gd):
        return (gd, self.__triangles)


class DeformableImage(Deformable):
    def __init__(self, bitmap, output='bitmap', extent=None):
        assert isinstance(extent, AABB) or extent is None or isinstance(extent, str)
        assert output == 'bitmap' or output == 'points'

        self.__shape = bitmap.shape
        self.__output = output

        self.__pixel_extent = AABB(0., self.__shape[1]-1, 0., self.__shape[0]-1)

        if extent is None:
            extent = AABB(0., 1., 0., 1.)
        elif isinstance(extent, str) and extent == 'match':
            extent = self.__pixel_extent

        self.__extent = extent

        pixel_points = pixels2points(self.__extent.fill_count(self.__shape), self.__shape, self.__extent)

        self.__bitmap = bitmap
        super().__init__(Landmarks(2, pixel_points.shape[0], gd=pixel_points))

    @classmethod
    def load_from_file(cls, filename, origin='lower', device=None):
        return cls(load_greyscale_image(filename, origin=origin, device=device))

    @classmethod
    def load_from_pickle(cls, filename, origin='lower', device=None):
        pass

    @property
    def geometry(self):
        if self.__output == 'bitmap':
            return (self.bitmap,)
        elif self.__output == 'points':
            return (self.silent_module.manifold.gd, self.__bitmap.flatten()/torch.sum(self.__bitmap))
        else:
            raise ValueError()

    @property
    def shape(self):
        return self.__shape

    @property
    def extent(self):
        return self.__extent

    @property
    def points(self):
        return self.silent_module.manifold.gd

    @property
    def bitmap(self):
        return self.__bitmap

    @property
    def _has_backward(self):
        return True

    def __set_output(self):
        return self.__output

    def __get_output(self, output):
        self.__output = output

    output = property(__set_output, __get_output)

    def _backward_module(self):
        pixel_grid = pixels2points(self.__pixel_extent.fill_count(self.__shape, device=self.silent_module.device), self.__shape, self.__extent)
        return SilentLandmarks(2, pixel_grid.shape[0], gd=pixel_grid)

    def compute_deformed(self, modules, solver, it, costs=None, intermediates=None):
        assert isinstance(costs, dict) or costs is None
        assert isinstance(intermediates, dict) or intermediates is None

        # Forward shooting
        compound_modules = [self.silent_module, *modules]
        compound = CompoundModule(compound_modules)

        shoot(Hamiltonian(compound), solver, it, intermediates=intermediates)

        # Prepare for reverse shooting
        compound.manifold.negate_cotan()

        silent_pixel_grid = self._backward_module()

        # Reverse shooting with the newly constructed pixel grid module
        compound = CompoundModule([silent_pixel_grid, *compound.modules])

        shoot(Hamiltonian(compound), solver, it)

        if costs is not None:
            costs['deformation'] = compound.cost()

        return self._to_deformed(silent_pixel_grid.manifold.gd)

    def _to_deformed(self, gd):
        print("---------------------")
        print(gd[::10])
        if self.__output == 'bitmap':
            a = (deformed_intensities(gd, self.__bitmap, self.__extent), )
            # print(a)
            # import matplotlib.pyplot as plt
            # plt.plot(gd[::10, 0].numpy(), gd[::10, 1].numpy(), 'o', markersize=1.)
            # plt.grid()
            # # plt.imshow(a[0].numpy(), origin='lower', extent=self.__extent)
            # plt.show()
            return a
        elif self.__output == 'points':
            deformed_bitmap = deformed_intensities(gd, self.__bitmap, self.__extent)
            return (gd, deformed_bitmap.flatten()/torch.sum(deformed_bitmap))
        else:
            raise ValueError()


def deformables_compute_deformed(deformables, modules, solver, it, costs=None, intermediates=None, controls=None, t1=1.):
    assert isinstance(costs, dict) or costs is None
    assert isinstance(intermediates, dict) or intermediates is None

    # Regroup silent modules of each deformable and build a compound module
    silent_modules = [deformable.silent_module for deformable in deformables]
    compound = CompoundModule([*silent_modules, *modules])

    # Forward shooting
    shoot(Hamiltonian(compound), solver, it, intermediates=intermediates, controls=controls, t1=t1)

    # Regroup silent modules of each deformable thats need to shoot backward
    # backward_silent_modules = [deformable.silent_module for deformable in deformables if deformable._has_backward]

    shoot_backward = any([deformable._has_backward for deformable in deformables])

    forward_silent_modules = copy.deepcopy(silent_modules)
    # forward_silent_modules = silent_modules

    if shoot_backward:
        # Backward shooting is needed
        # Build/assemble the modules that will be shot backward
        backward_modules = [deformable._backward_module() for deformable in deformables if deformable._has_backward]
        compound = CompoundModule([*silent_modules, *backward_modules, *modules])

        # Reverse the moments for backward shooting
        compound.manifold.negate_cotan()

        # # Backward shooting
        # print(backward_modules[0].manifold.gd[::10])
        # g = backward_modules[0].manifold.gd
        # grid = vec2grid(g, 128, 128)
        # ax = plt.subplot()
        # plot_grid(ax, grid[0], grid[1], color='blue', lw=0.5)
        # plt.axis('equal')
        # plt.show()

        shoot(Hamiltonian(compound), solver, it, t1=t1)
        # print("===============")
        # print(backward_modules[0].manifold.gd[::10])
        # g = backward_modules[0].manifold.gd
        # grid = vec2grid(g, 128, 128)
        # plt.figure()
        # ax = plt.subplot()
        # plot_grid(ax, grid[0], grid[1], color='red', lw=0.5)
        # plt.axis('equal')
        # plt.show()

    # For now, we need to compute the deformation cost after each shooting (and not before any shooting) for computation tree reasons
    if costs is not None:
        costs['deformation'] = compound.cost()

    # Ugly way to compute the list of deformed objects. Not intended to stay!
    deformed = []
    for deformable, forward_silent_module in zip(deformables, forward_silent_modules):
        if deformable._has_backward:
            print("+++++++++++++++++++++++")
            gd = backward_modules.pop(0).manifold.gd
            grid = vec2grid(gd, 128, 128)
            plt.figure()
            ax = plt.subplot()
            plot_grid(ax, grid[0], grid[1], color='red', lw=0.5)
            plt.axis('equal')
            plt.show()

            # deformed.append(deformable._to_deformed(backward_modules.pop(0).manifold.gd))
            deformed.append(deformable._to_deformed(gd))
        else:
            deformed.append(deformable._to_deformed(forward_silent_module.manifold.gd))

    return deformed

