import copy
from collections import Iterable

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import grad

from .kernels import scal
from .usefulfunctions import make_grad_graph
from .deformationmodules import CompoundModule

class Hamiltonian:
    def __init__(self, modules):
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

    def __call__(self):
        """Computes the hamiltonian."""
        return self.apply_mom() - self.__module.cost()

    def apply_mom(self):
        """Apply the moment on the geodesic descriptors."""
        return self.__module.manifold.inner_prod_field(self.__module.field_generator())

    def geodesic_controls(self):
        self.__module.compute_geodesic_control(self.__module.manifold)

