import torch

from implicitmodules.torch.Utilities import generate_mesh_grid, grid2vec, vec2grid
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.DeformationModules.SilentLandmark import SilentBase

class DeformationGrid(SilentBase):
    """
    Helper class to manipulate deformation grids as deformation modules.
    Built on top of silent module
    """

    def __init__(self, aabb, resolution, label=None):
        self.__aabb = aabb
        self.__resolution = resolution

        grid = generate_mesh_grid(aabb, resolution)
        points_grid = grid2vec(grid)

        manifold = Landmarks(aabb.dim, points_grid.shape[0], gd=points_grid)

        super().__init__(manifold, label)

    @property
    def aabb(self):
        return self.__aabb

    @property
    def resolution(self):
        return self.__resolution

    def togrid(self):
        return vec2grid(self.manifold.gd, self.__resolution)

