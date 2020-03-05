import time
import gc
import copy
import math
from itertools import chain

import torch

from .models_fitting import ModelFitting
from implicitmodules.torch.Attachment import CompoundAttachment


class GradientDescentOptimizer:
    def __init__(self, parameters, alpha=1., gamma1=0.5, gamma2=1.5, fit_parameters=None):
        assert fit_parameters is None or isinstance(fit_parameters, dict)

        self.__parameters = parameters
        self.__gamma1 = gamma1
        self.__gamma2 = gamma2
        self.__alpha = alpha
        if fit_parameters is None:
            fit_parameters = {}
        self.__fit_parameters = fit_parameters

        for param_group in self.__parameters:
            if param_group in self.__fit_parameters:
                if 'gamma1' not in self.__fit_parameters[param_group]:
                    self.__fit_parameters['gamma1'] = self.__gamma1
                if 'gamma2' not in self.__fit_parameters[param_group]:
                    self.__fit_parameters['gamma2'] = self.__gamma2
                if 'alpha' not in self.__fit_parameters[param_group]:
                    self.__fit_parameters['alpha'] = self.__alpha
            else:
                self.__fit_parameters[param_group] = {'gamma1': self.__gamma1, 'gamma2': self.__gamma2, 'alpha': self.__alpha}

        self.reset()

    @property
    def parameters(self):
        return self.__parameters

    @property
    def fit_parameters(self):
        return self.__fit_parameters

    @property
    def minimum_found(self):
        return self.__minimum_found

    @property
    def alpha(self):
        return self.__alpha

    @property
    def total_evaluation_count(self):
        return self.__total_evaluation_count

    def __parameters_to_list(self, parameters):
        if isinstance(parameters, dict):
            return list(chain(*parameters.values()))

        return list(parameters)

    def __fill_model_parameters(self, parameters):
        for model_parameter, parameter in zip(self.__parameters_to_list(self.__parameters), self.__parameters_to_list(parameters)):
            model_parameter.data = parameter

    def __copy_model_parameters(self, parameters, getter=lambda x: x.data, copy_op=lambda x: x.clone()):
        return dict([(key, list(copy_op(getter(param)) for param in parameters[key])) for key in parameters])

    def __update_fit_parameters_alpha(self, key, fit_parameters):
        for param_group in fit_parameters:
            fit_parameters[param_group]['alpha'] *= fit_parameters[param_group][key]

    def reset(self):
        self.__minimum_found = False
        self.__total_evaluation_count = 0

    def zero_grad(self):
        """ Frees parameters computation graphs and zero out their accumulated gradients. """
        for param in self.__parameters_to_list(self.__parameters):
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def step(self, closure):
        def evaluate(parameters, compute_backward=False):
            # Evaluate f(x_k)
            self.__fill_model_parameters(parameters)
            with torch.autograd.set_grad_enabled(compute_backward):
                cost = closure()

            if compute_backward:
                # Gradients are needed, so we extract them
                d_cost = self.__copy_model_parameters(self.__parameters, getter=lambda x: x.grad.data)
                return cost, d_cost
            else:
                return cost

        x_k = self.__copy_model_parameters(self.__parameters)

        cost_x_k, d_cost = evaluate(x_k, compute_backward=True)
        self.__total_evaluation_count += 1

        found_minimizer = False
        evalcount = 1
        while not found_minimizer:
            x_k_p1 = dict([(param_group, list(map(lambda x, y: x - self.__fit_parameters[param_group]['alpha'] * y, x_k[param_group], d_cost[param_group]))) for param_group in x_k])

            cost_x_kp1 = evaluate(x_k_p1)
            self.__total_evaluation_count += 1

            if cost_x_kp1 < cost_x_k and math.isfinite(cost_x_kp1):
                found_minimizer = True
                if evalcount == 1:
                    for param_group in self.__fit_parameters:
                        self.__update_fit_parameters_alpha('gamma2', self.__fit_parameters)
                return evalcount

            # Maybe have a better termination condition?
            elif cost_x_kp1 == cost_x_k:
                self.__minimum_found = True

            else:
                evalcount += 1
                self.__update_fit_parameters_alpha('gamma1', self.__fit_parameters)


class ModelFittingGradientDescent(ModelFitting):
    def __init__(self, model, step_length, post_iteration_callback=None):
        super().__init__(model, post_iteration_callback)
        self.__step_length = step_length

        self.__optim = GradientDescentOptimizer(model.parameters, alpha=step_length)

    def reset(self):
        self.__optim = self.__optim.reset()

    def fit(self, target, max_iter, options={}, log_interval=1, disp=True):
        last_costs = {}
        costs = []

        shoot_method = 'euler'
        shoot_it = 10

        if 'shoot_method' in options:
            shoot_method = options['shoot_method']
            del options['shoot_method']

        if 'shoot_it' in options:
            shoot_it = options['shoot_it']
            del options['shoot_it']

        def closure():
            self.__optim.zero_grad()

            # Shooting
            cost, deformation_cost, attach_cost = self.model.compute(target, it=shoot_it, method=shoot_method)
            # Save for printing purpose
            last_costs['deformation_cost'] = deformation_cost
            last_costs['attach_cost'] = attach_cost
            last_costs['cost'] = cost

            gc.collect()

            return cost

        # Compute initial cost
        with torch.autograd.no_grad():
            loss = closure()

        print("Initial energy = %.3f" % last_costs['cost'])

        start = time.time()
        costs = [list(last_costs.values())]
        for i in range(max_iter):
            # Computing step
            eval_count = self.__optim.step(closure)

            # Retrieving costs
            costs.append(list(last_costs.values()))

            print("="*80)
            print("Iteration {it}\n Total cost={cost}\n Attach cost={attach_cost}\n Deformation cost={deformation_cost}\n Step length={alpha}\n Model evaluation count={eval_count}".format(it=i+1, cost=last_costs['cost'], attach_cost=last_costs['attach_cost'], deformation_cost=last_costs['deformation_cost'], alpha=dict([(group_param, self.__optim.fit_parameters[group_param]['alpha']) for group_param in self.__optim.fit_parameters]), eval_count=eval_count))

            if self.__optim.minimum_found:
                break

        print("="*80)
        print("End of the optimisation process.")
        print("Final energy =", last_costs['cost'])
        print("Total model evaluation count  =", self.__optim.total_evaluation_count)
        print("Time elapsed =", time.time() - start)

        return costs

