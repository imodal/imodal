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


class DeformationModuleBuilder():
    def __init__(self):
        self.__builders = {}

    def __call__(self, module_name, backend, **kwargs):
        return spawn(module_name, backend, kwargs)

    def register_deformation_module_builder(self, module_name, deformation_module_builder):
        if isinstance(deformation_module_builder, dict):
            self.__builders[module_name] = ObjectFactory(deformation_module_builder)
        else:
            self.__builders[module_name] = deformation_module_builder

    def spawn(self, module_name, backend=None, **kwargs):
        deformation_module_builder = self.__builders.get(module_name)
        if not deformation_module_builder:
            raise KeyError(module_name)

        if isinstance(deformation_module_builder, ObjectFactory):
            if backend is None:
                backend = get_compute_backend()
            return deformation_module_builder.spawn(backend, **kwargs)
        else:
            return deformation_module_builder(**kwargs)


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


