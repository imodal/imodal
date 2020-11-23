from collections import Iterable

from imodal.DeformationModules.Combination import CompoundModule


class Hamiltonian:
    """Class used to represent the hamiltonian given by a collection of modules."""
    def __init__(self, modules):
        """
        Instantiate the Hamiltonian related to a set of deformation module.

        Parameters
        ----------
        modules : Iterable or DeformationModules.DeformationModule
            Either an iterable of deformation modules or an unique module.
        """
        assert isinstance(modules, Iterable) or isinstance(modules, CompoundModule)
        super().__init__()
        if isinstance(modules, Iterable):
            self.__module = CompoundModule(modules)
        else:
            self.__module = modules

    @classmethod
    def from_hamiltonian(cls, class_instance):
        return cls(class_instance.module)

    @property
    def module(self):
        return self.__module

    @property
    def dim(self):
        return self.__module.dim

    def __call__(self):
        """Computes the hamiltonian.

        Mathematicaly, computes the quantity :math:`\mathcal{H}(q, p, h)`.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the hamiltonian quantity.
        """
        return self._apply_mom() - self.__module.cost()

    def geodesic_controls(self):
        """
        Computes the geodesic controls of the hamiltonian's module.
        """
        self.__module.compute_geodesic_control(self.__module.manifold)

    def _apply_mom(self):
        """Apply the moment on the geodesic descriptors."""
        return self.__module.manifold.inner_prod_field(self.__module.field_generator())


