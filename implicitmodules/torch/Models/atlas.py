# import copy
# from collections import OrderedDict

# import matplotlib.pyplot as plt
# import torch

# from implicitmodules.torch.Models import BaseModel, ModelPointsRegistration
# from implicitmodules.torch.HamiltonianDynamic import Hamiltonian, shoot
# from implicitmodules.torch.DeformationModules import ImplicitModule0
# from implicitmodules.torch.Utilities import append_in_dict_of_list, make_grad_graph


# class Atlas(BaseModel):
#     """
#     TODO: add documentation
#     """
#     def __init__(self, template, modules, attachement, population_count, lam=1., fit_gd=None, optimise_template=False, ht_sigma=None, ht_nu=0., ht_coeff=1., ht_solver='euler', ht_it=10, precompute_callback=None, model_precompute_callback=None, other_parameters=None, evaluate_mode='sequential'):
#         if other_parameters is None:
#             other_parameters = []

#         if evaluate_mode != 'sequential' and evaluate_mode != 'parallel' and compute_mode != 'heterogeneous':
#             raise RuntimeError("Atlas.__init__(): evaluate_mode {evaluate_mode} not recognised!".format(evaluate_mode=evaluate_mode))

#         if evaluate_mode == 'sequential':
#             self.__compute_deformed_func = self.__compute_deformed_sequential
#         elif evaluate_mode == 'parallel':
#             self.__compute_deformed_func = self.__compute_deformed_parallel
#         else:
#             raise RuntimeError("Atlas: {evaluate_mode} not recognised!".format(evaluate_mode=evaluate_mode))

#         if optimise_template and ht_sigma is None:
#             raise RuntimeError("Atlas.__init__(): optimise_template has been set to True but ht_sigma has not been specified!")

#         self.__evaluate_mode = evaluate_mode

#         self.__population_count = population_count
#         self.__template = template
#         self.__precompute_callback = precompute_callback
#         self.__init_other_parameters = other_parameters
#         self.__fit_gd = fit_gd
#         self.__n_modules = len(modules)
#         self.__ht_sigma = ht_sigma
#         self.__ht_nu = ht_nu
#         self.__ht_coeff = ht_coeff
#         self.__ht_solver = ht_solver
#         self.__ht_it = ht_it
#         self.__optimise_template = optimise_template
#         self.__lam = lam

#         self.__models = []
#         for i in range(self.__population_count):
#             cur_modules = copy.deepcopy(modules)

#             self.__models.append(ModelPointsRegistration([self.__template], cur_modules, attachement, precompute_callback=model_precompute_callback, other_parameters=other_parameters, lam=self.__lam))

#             if fit_gd is not None and i != 0:
#                 for j in range(len(modules)):
#                     if fit_gd[j]:
#                         # We fit the geometrical descriptor of some module. We optimise the one from the first model. For the other models, we assign a reference to the manifold of the first model.
#                         self.__models[i].init_manifold[j+1].gd = self.__models[0].init_manifold[j+1].gd

#         # Momentum of the LDDMM translation module for the hypertemplate if used
#         if self.__optimise_template:
#             self.__cotan_ht = torch.zeros_like(template, requires_grad=True, device=self.__template.device, dtype=self.__template.dtype)

#         self._compute_parameters()

#     def __str__(self):
#         outstr = "Atlas\n"
#         outstr += "=====\n"
#         outstr += ("Template nb pts=" + str(self.__template.shape[0]) + "\n")
#         outstr += ("Population count=" + str(self.__population_count) + "\n")
#         outstr += ("Module count=" + str(len(self.models[0].modules)-1) + "\n")
#         outstr += ("Hypertemplate=" + str(self.__optimise_template==True) + "\n")
#         if self.__optimise_template:
#             outstr += ("Hypertemplate sigma=" + str(self.__ht_sigma) + "\n")
#             outstr += ("Hypertemplate coeff=" + str(self.__ht_coeff) + "\n")
#         outstr += ("Attachment=" + str(self.__models[0].attachments[0]) + "\n")
#         outstr += ("Lambda=" + str(self.__lam) + "\n")
#         outstr += ("Fit geometrical descriptors=" + str(self.__fit_gd) + "\n")
#         outstr += ("Precompute callback=" + str(self.__precompute_callback is not None) + "\n")
#         outstr += ("Model precompute callback=" + str(self.models[0].precompute_callback is not None) + "\n")
#         outstr += ("Other parameters=" + str(len(self.__init_other_parameters) != 0))
#         outstr += "\n\n"
#         outstr += "Modules\n"
#         outstr += "======="
#         for module in self.models[0].modules[1:]:
#             outstr += ( "\n" + str(module))
#         return outstr

#     @property
#     def attachments(self):
#         return list(list(zip(*[model.attachments for model in self.__models]))[0])

#     @property
#     def compute_mode(self):
#         return self.__compute_mode

#     @property
#     def models(self):
#         return self.__models

#     @property
#     def parameters(self):
#         return self.__parameters

#     @property
#     def attachement(self):
#         return self.__attachement

#     @property
#     def template(self):
#         pass

#     @property
#     def precompute_callback(self):
#         return self.__precompute_callback

#     @property
#     def lam(self):
#         return self.__lam

#     def _compute_parameters(self):
#         """ Updates the parameter list sent to the optimizer. """
#         self.__parameters = OrderedDict()

#         # Moments of each modules in each models
#         self.__parameters['cotan'] = {'params': []}
#         for model in self.__models:
#             model._compute_parameters()
#             self.__parameters['cotan']['params'].extend(model.init_manifold.unroll_cotan())

#         if self.__fit_gd is not None:
#             self.__parameters['gd'] = {'params': []}
#             for i in range(self.__n_modules):
#                 if self.__fit_gd[i]:
#                     # We optimise the manifold of the first model (which will be reflected on the other models as the manifold reference is shared).
#                     self.__parameters['gd']['params'].extend(self.__models[0].init_manifold[i+1].unroll_gd())

#         # Other parameters
#         self.__parameters.update(self.__init_other_parameters)

#         # Hyper template moments
#         if self.__optimise_template:
#             self.__parameters['ht'] = {'params': [self.__cotan_ht]}

#     def compute_template(self, detach=True, costs=None, intermediates=None):
#         if not self.__optimise_template:
#             return self.__template

#         translations_ht = ImplicitModule0(self.__template.shape[1], self.__template.shape[0], self.__ht_sigma, self.__ht_nu, coeff=self.__ht_coeff, gd=self.__template.requires_grad_(), cotan=self.__cotan_ht)

#         if costs is not None:
#             translations_ht.compute_geodesic_control(translations_ht.manifold)
#             costs['ht'] = translations_ht.cost()

#         shoot(Hamiltonian([translations_ht]), self.__ht_solver, self.__ht_it, intermediates=None)

#         deformed_template = translations_ht.manifold.gd

#         if detach:
#             deformed_template = deformed_template.detach()

#         return deformed_template

#     def evaluate(self, targets, solver, it):
#         costs = {}
#         for model, target in zip(self.models, targets):
#             cost = {}
#             if self.__optimise_template:
#                 template = self.compute_template(detach=False, costs=cost)
#                 model.init_manifold[0].gd = template

#             append_in_dict_of_list(costs, model.evaluate([target], solver, it, costs=cost))

#         return dict([(key, sum(costs[key])) for key in costs])

#     def compute_deformed(self, solver, it, intermediates=None):
#         assert isinstance(intermediates, dict) or intermediates is None

#         return self.__compute_deformed_func(solver, it, intermediates)

#     def __compute_deformed_sequential(self, solver, it, intermediates):
#         deformed = []
#         if intermediates is not None:
#             # Check if a list for each intermediate items exists
#             # Maybe there is a better way to do this
#             if not('states' in intermediates.keys() and isinstance(intermediates['states'], list)):
#                 intermediates['states'] = []

#             if not('controls' in intermediates.keys() and isinstance(intermediates['controls'], list)):
#                 intermediates['controls'] = []

#         for model in self.__models:
#             if self.__optimise_template:
#                 deformed_template = self.compute_template(False)
#                 model._Model__init_manifold[0].gd = deformed_template

#             deformed_intermediates = None
#             if intermediates is not None:
#                 deformed_intermediates = {}

#             deformed.append(model.compute_deformed(solver, it, intermediates=deformed_intermediates)[0])

#             if intermediates is not None:
#                 append_in_dict_of_list(intermediates, deformed_intermediates)

#         return deformed

#     def __compute_deformed_parallel(self, method, it, costs, intermediates):
#         raise NotImplementedError()

