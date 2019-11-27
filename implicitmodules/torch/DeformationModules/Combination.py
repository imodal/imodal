from typing import Iterable

from implicitmodules.torch.DeformationModules.Abstract import DeformationModule
from implicitmodules.torch.Manifolds import CompoundManifold
from implicitmodules.torch.StructuredFields import CompoundStructuredField


class CompoundModule(DeformationModule, Iterable):
    """Combination of modules."""
    
    def __init__(self, modules):
        assert isinstance(modules, Iterable)
        super().__init__()
        self.__modules = [*modules]

    def to(self, device):
        [mod.to(device) for mod in self.__modules]

    @property
    def device(self):
        return self.__modules[0].device
    
    @property
    def modules(self):
        return self.__modules
    
    def __getitem__(self, index):
        return self.__modules[index]
    
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
    def nb_module(self):
        return len(self.__modules)

    @property
    def dim(self):
        return self.__modules[0].dim # Dirty
    
    def __get_controls(self):
        return [m.controls for m in self.__modules]
    
    def fill_controls(self, controls):
        assert len(controls) == self.nb_module
        for i in range(self.nb_module):
            self.__modules[i].fill_controls(controls[i])
    
    controls = property(__get_controls, fill_controls)
    
    def fill_controls_zero(self):
        for m in self.__modules:
            m.fill_controls_zero()
    
    @property
    def manifold(self):
        return CompoundManifold([m.manifold for m in self.__modules])
    
    def __call__(self, points):
        """Applies the generated vector field on given points."""
        app_list = []
        for m in self.__modules:
            app_list.append(m(points))
        
        return sum(app_list).view(-1, self.manifold.dim)
    
    def cost(self):
        """Returns the cost."""
        cost_list = []
        for m in self.__modules:
            cost_list.append(m.cost())
        
        return sum(cost_list)
    
    def compute_geodesic_control(self, man):
        """Computes geodesic control from \delta \in H^\ast."""
        for i in range(self.nb_module):
            self.__modules[i].compute_geodesic_control(man)
    
    def field_generator(self):
        return CompoundStructuredField([m.field_generator() for m in self.__modules])

