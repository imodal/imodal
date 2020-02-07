import copy

import matplotlib.pyplot as plt
import torch

from implicitmodules.torch.Models import Model, ModelPointsRegistration
from implicitmodules.torch.HamiltonianDynamic import Hamiltonian, shoot
from implicitmodules.torch.DeformationModules import ImplicitModule0


class Atlas:
    """
    TODO: add documentation
    """
    def __init__(self, template, modules, attachement, population_count, lam=1., fit_gd=None, optimise_template=False, ht_sigma=None, ht_coeff=1., precompute_callback=None, model_precompute_callback=None, other_parameters=None, compute_mode='sequential'):
        if other_parameters is None:
            other_parameters = []

        if compute_mode != 'sequential' and compute_mode != 'parallel' and compute_mode != 'heterogeneous':
            raise RuntimeError("Atlas.__init__(): compute_mode {compute_mode} not recognised!".format(compute_mode=compute_mode))

        if compute_mode == 'sequential':
            self.__compute_func = self.__compute_sequential
        elif compute_mode == 'parallel':
            self.__compute_func = self.__compute_parallel
        else:
            raise RuntimeError("Atlas: {compute_mode} not recognised!".format(compute_mode=compute_mode))

        if optimise_template and ht_sigma is None:
            raise RuntimeError("Atlas.__init__(): optimise_template has been set to True but ht_sigma has not been specified!")

        self.__compute_mode = compute_mode

        self.__population_count = population_count
        self.__template = template
        self.__precompute_callback = precompute_callback
        self.__init_other_parameters = other_parameters
        self.__fit_gd = fit_gd
        self.__n_modules = len(modules)
        self.__ht_sigma = ht_sigma
        self.__ht_coeff = ht_coeff
        self.__optimise_template = optimise_template
        self.__lam = lam

        self.__models = []
        for i in range(self.__population_count):
            #manifolds = [module.manifold.clone() for module in modules]
            cur_modules = copy.deepcopy(modules)

            self.__models.append(ModelPointsRegistration([self.__template], cur_modules, attachement, precompute_callback=model_precompute_callback, other_parameters=other_parameters, lam=self.__lam))

            if fit_gd is not None and i != 0:
                for j in range(len(modules)):
                    if fit_gd[j]:
                        # We fit the geometrical descriptor of some module. We optimise the one from the first model. For the other models, we assign a reference to the manifold of the first model.
                        self.__models[i].init_manifold[j+1].gd = self.__models[0].init_manifold[j+1].gd

        # Momentum of the LDDMM translation module for the hypertemplate if used
        if self.__optimise_template:
            self.__cotan_ht = torch.zeros_like(template, requires_grad=True)

        self.compute_parameters()

    def __str__(self):
        outstr = "Atlas\n"
        outstr += "=====\n"
        outstr += ("Template nb pts=" + str(self.__template.shape[0]) + "\n")
        outstr += ("Population count=" + str(self.__population_count) + "\n")
        outstr += ("Module count=" + str(len(self.models[0].modules)-1) + "\n")
        outstr += ("Hypertemplate=" + str(self.__optimise_template==True) + "\n")
        if self.__optimise_template:
            outstr += ("Hypertemplate sigma=" + str(self.__ht_sigma) + "\n")
            outstr += ("Hypertemplate coeff=" + str(self.__ht_coeff) + "\n")
        outstr += ("Attachment=" + str(self.__models[0].attachments[0]) + "\n")
        outstr += ("Fit geometrical descriptors=" + str(self.__fit_gd) + "\n")
        outstr += ("Precompute callback=" + str(self.__precompute_callback is not None) + "\n")
        outstr += ("Model precompute callback=" + str(self.models[0].precompute_callback is not None) + "\n")
        outstr += ("Other parameters=" + str(len(self.__init_other_parameters) != 0))
        outstr += "\n\n"
        outstr += "Modules\n"
        outstr += "======="
        for module in self.models[0].modules[1:]:
            outstr += ( "\n" + str(module))
        return outstr

    def clone_init(self):
        return Atlas(copy.deepcopy(self.__template),
                     copy.deepcopy(self.__models[0].modules[1:]),
                     copy.deepcopy(self.__models[0].attachments),
                     self.__population_count,
                     lam=self.__lam,
                     fit_gd=copy.deepcopy(self.__fit_gd),
                     optimise_template=self.__optimise_template,
                     ht_sigma=self.__ht_sigma,
                     ht_coeff=self.__ht_coeff,
                     precompute_callback=self.__precompute_callback,
                     model_precompute_callback=self.__models[0].precompute_callback,
                     other_parameters=copy.deepcopy(self.__models[0].init_other_parameters),
                     compute_mode=self.__compute_mode)

    def fill_from(self, atlas):
        self.__dict__.update(atlas.__dict__)

    @property
    def attachments(self):
        return list(list(zip(*[model.attachments for model in self.__models]))[0])

    @property
    def compute_mode(self):
        return self.__compute_mode

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

        # for i in range(len(self.__models)):
        #     self.__models[i].compute_parameters()
        #     self.__parameters.extend(self.__models[i].init_manifold.unroll_cotan())

        # Moments of each modules in each models
        for model in self.__models:
            model.compute_parameters()
            self.__parameters.extend(model.init_manifold.unroll_cotan())

        if self.__fit_gd is not None:
            for i in range(self.__n_modules):
                if self.__fit_gd[i]:
                    # We optimise the manifold of the first model (which wil reflect on the other models as the manifolds reference is shared).
                    self.__parameters.extend(self.__models[0].init_manifold[i+1].unroll_gd())

        # Other parameters
        self.__parameters.extend(self.__init_other_parameters)

        # Hyper template moments
        if self.__optimise_template:
            self.__parameters.append(self.__cotan_ht)

    def compute_template(self, it=10, method='euler'):
        if self.__optimise_template:
            translations_ht = ImplicitModule0(2, self.__template.shape[0], self.__ht_sigma, 0., gd=self.__template.clone().requires_grad_(), cotan=self.__cotan_ht.clone().requires_grad_())

            shoot(Hamiltonian([translations_ht]), it, method)

            return translations_ht.manifold.gd.detach()
        else:
            return self.__template

    def compute(self, targets, it=10, method='euler'):
        return self.__compute_func(targets, it, method)

    def __compute_sequential(self, targets, it, method):
        costs = []
        deformation_costs = []
        attach_costs = []

        for i in range(self.__population_count):
            cost_template = None
            if self.__optimise_template:
                translations_ht = ImplicitModule0(2, self.__template.shape[0], self.__ht_sigma, 0., coeff=self.__ht_coeff, gd=self.__template.clone().requires_grad_(), cotan=self.__cotan_ht, backend='torch')
                shoot(Hamiltonian([translations_ht]), 10, 'euler')
                self.__models[i]._Model__init_manifold[0].gd = translations_ht.manifold.gd
                cost_template = translations_ht.cost()

            if self.__models[i].precompute_callback is not None:
                self.__models[i].precompute_callback(self.__models[i].init_manifold, self.__models[i].modules, self.__models[i].parameters)

            cost, deformation_cost, attach_cost = self.__models[i].compute([targets[i]], it=it, method=method, ext_cost=cost_template)

            costs.append(cost)
            deformation_costs.append(deformation_cost)
            attach_costs.append(attach_cost)

        deformation_cost = sum(deformation_costs)
        attach_cost = sum(attach_costs)
        cost = sum(costs)

        return cost, deformation_cost, attach_cost

    def __compute_parallel(self, atlas, it, method):
        pass


