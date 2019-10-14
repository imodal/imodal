import copy

import matplotlib.pyplot as plt
import torch

from implicitmodules.torch.Models import Model, ModelPointsRegistration
from implicitmodules.torch.DeformationModules import create_deformation_module
from implicitmodules.torch.HamiltonianDynamic import Hamiltonian, shoot


class Atlas:
    def __init__(self, template, modules, attachement, population_count, lam=1., sigma_ht=1., fit_gd=None, optimise_template=False, precompute_callback=None, model_precompute_callback=None, other_parameters=None):
        if other_parameters is None:
            other_parameters = []

        self.__population_count = population_count
        self.__template = template
        self.__precompute_callback = precompute_callback
        self.__init_other_parameters = other_parameters
        self.__fit_gd = fit_gd
        self.__n_modules = len(modules)
        self.__sigma_ht = sigma_ht
        self.__optimise_template = optimise_template
        self.__lam = lam

        self.__models = []
        for i in range(self.__population_count):
            manifolds = [module.manifold.copy() for module in modules]
            cur_modules = copy.copy(modules)
            for module, man in zip(cur_modules, manifolds):
                module.manifold.fill(man, copy=True)
            self.__models.append(ModelPointsRegistration([self.__template], cur_modules, attachement, precompute_callback=model_precompute_callback, other_parameters=other_parameters, lam=self.__lam))
            if fit_gd is not None and i != 0:
                for j in range(self.__n_modules):
                    if fit_gd[j]:
                        self.__models[i].init_manifold[j+1].gd = self.__models[0].init_manifold[j+1].gd

        # Momentum of the LDDMM translation module for the hypertemplate if used
        if self.__optimise_template:
            self.__cotan_ht = torch.zeros_like(template).view(-1).requires_grad_()

        self.compute_parameters()

    @property
    def models(self):
        return self.__models

    @property
    def parameters(self):
        return self.__parameters

    @property
    def attachement(self):
        return self.__attachement

    @property
    def template(self):
        pass

    @property
    def precompute_callback(self):
        return self.__precompute_callback

    @property
    def lam(self):
        return self.__lam

    def compute_parameters(self):
        """ Updates the parameter list sent to the optimizer. """
        self.__parameters = []

        for i in range(len(self.__models)):
            self.__models[i].compute_parameters()
            self.__parameters.extend(self.__models[i].init_manifold.unroll_cotan())

        if self.__fit_gd is not None:
            for i in range(self.__n_modules):
                if self.__fit_gd[i]:
                    self.__parameters.extend(self.__models[0].init_manifold[i+1].unroll_gd())

        self.__parameters.extend(self.__init_other_parameters)

        if self.__optimise_template:
            self.__parameters.append(self.__cotan_ht)

    def compute_template(self, it=10, method="euler"):
        #translations_ht = ImplicitModule0.build_from_points(2, self.__template.shape[0], self.__sigma_ht, 0.01, gd=self.__template.view(-1).requires_grad_(), cotan=self.__cotan_ht)

        shoot(Hamiltonian([translations_ht]), it, method)

        return translations_ht.manifold.gd.detach().view(-1, 2)

    def compute(self, target, it=10, method="euler"):
        if self.__optimise_template:
            translations_ht = ImplicitModule0.build_from_points(2, self.__template.shape[0], self.__sigma_ht, 0.01, gd=self.__template.view(-1).requires_grad_(), cotan=self.__cotan_ht)

            shoot(Hamiltonian([translations_ht]), it, method)

        costs = []
        deformation_costs = []
        attach_costs = []

        for i in range(self.__population_count):
            if self.__optimise_template:
                self.__models[i]._Model__init_manifold[0].gd = translations_ht.manifold.gd

            if self.__models[i].precompute_callback is not None:
                self.__models[i].precompute_callback(self.__models[i].init_manifold, self.__models[i].modules, self.__models[i].parameters)
            cost, deformation_cost, attach_cost = self.__models[i].compute([target[i]], it=it, method=method)

            costs.append(cost)
            deformation_costs.append(deformation_cost)
            attach_costs.append(attach_cost)

            # Quick fix used to manualy free up the gradient tree
            del cost

        deformation_cost = sum(deformation_costs)
        attach_cost = sum(attach_costs)
        cost = sum(costs)

        return cost, deformation_cost, attach_cost


