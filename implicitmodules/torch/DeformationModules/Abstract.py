import copy

from implicitmodules.torch.Utilities.factory import ObjectFactory
from implicitmodules.torch.Utilities.compute_backend import get_compute_backend

class DeformationModule:
    """Base deformation module."""
    
    def __init__(self):
        super().__init__()
    
    def copy(self):
        """
        Returns a copy of the deformation module.
        """
        return copy.copy(self)
    
    def __call__(self, points, k=0):
        """
        TODO: Add documentation
        """
        raise NotImplementedError
    
    def cost(self):
        """
        TODO: Add documentation.
        """
        raise NotImplementedError

    def compute_geodesic_control(self, man):
        """
        TODO: Add documentation.
        """
        raise NotImplementedError

    def field_generator(self):
        """
        TODO: Add documentation.
        """
        raise NotImplementedError


def create_deformation_module_with_backends(build_torch, build_keops):
    def create_deformation_module(*args, backend=None, **kwargs):
        if backend is None:
            backend = get_compute_backend()

        if backend == 'torch':
            return build_torch(*args, **kwargs)
        elif backend == 'keops':
            return build_keops(*args, **kwargs)
        else:
            raise NotImplementedError("Error while creating module! {backend} backend not recognised!".format(backend=backend))

    return create_deformation_module


