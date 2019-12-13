from typing import Iterable

from implicitmodules.torch.DeformationModules.Abstract import DeformationModule
from implicitmodules.torch.Manifolds import CompoundManifold
from implicitmodules.torch.StructuredFields import CompoundStructuredField


class CompoundModule(DeformationModule, Iterable):
    """Combination of modules."""

    def __init__(self, modules, label=None):
        assert isinstance(modules, Iterable)
        super().__init__(label)
        self.__modules = [*modules]

    def to(self, device):
        [mod.to(device) for mod in self.__modules]

    @property
    def device(self):
        return self.__modules[0].device

    @property
    def modules(self):
        return self.__modules

    @property
    def todict(self):
        return dict(zip(self.label, self.__modules))

    def __getitem__(self, itemid):
        if isinstance(itemid, int):
            return self.__modules[itemid]
        else:
            return self.todict()[itemid]

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current >= len(self.__modules):
            raise StopIteration
        else:
            self.current = self.current + 1
            return self.__modules[self.current - 1]

    @property
    def dim(self):
        return self.__modules[0].dim # Dirty

    @property
    def label(self):
        return [module.label for module in self.__modules]

    def __get_controls(self):
        return [m.controls for m in self.__modules]

    def fill_controls(self, controls):
        assert len(controls) == len(self.__modules)
        [module.fill_controls(control) for module, control in zip(self.__modules, controls)]

    controls = property(__get_controls, fill_controls)

    def fill_controls_zero(self):
        [module.fill_controls_zero() for module in self.__modules]

    @property
    def manifold(self):
        return CompoundManifold([m.manifold for m in self.__modules])

    def __call__(self, points):
        """Applies the generated vector field on given points."""
        return sum([module(points) for module in self.__modules])

    def cost(self):
        """Returns the cost."""
        return sum([module.cost() for module in self.__modules])

    def compute_geodesic_control(self, man):
        """Computes geodesic control from \delta \in H^\ast."""
        [module.compute_geodesic_control(man) for module in self.__modules]

    def field_generator(self):
        return CompoundStructuredField([m.field_generator() for m in self.__modules])

