import time
import gc
import copy
import math

import torch

from .models_fitting import ModelFitting
from implicitmodules.torch.Attachment import CompoundAttachment


class GradientDescentOptimizer:
    def __init__(self, parameters, alpha=1., gamma_1=0.5, gamma_2=1.5):
        self.__parameters = parameters
        self.__gamma_1 = gamma_1
        self.__gamma_2 = gamma_2
        self.__alpha = alpha

        self.reset()

    @property
    def parameters(self):
        return self.__parameters

    @property
    def minimum_found(self):
        return self.__minimum_found

    @property
    def alpha(self):
        return self.__alpha

    @property
    def total_evaluation_count(self):
        return self.__total_evaluation_count

    def reset(self):
        self.__minimum_found = False
        self.__total_evaluation_count = 0

    def zero_grad(self):
        """ Frees parameters computation graphs and zero out their accumulated gradients. """
        for param in self.__parameters:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def __fill_model_parameters(self, parameters):
        for model_parameter, parameter in zip(self.__parameters, parameters):
            model_parameter.data = parameter

    def step(self, closure):
        def evaluate(parameters, compute_backward=False):
            # Evaluate for f(x_k)
            self.__fill_model_parameters(parameters)
            with torch.autograd.set_grad_enabled(compute_backward):
                cost = closure()

            if compute_backward:
                d_cost = copy.deepcopy([param.grad.data for param in self.__parameters])
                return cost, d_cost
            else:
                return cost

        x_k = copy.deepcopy([param.data for param in self.__parameters])

        cost_x_k, d_cost = evaluate(x_k, compute_backward=True)
        self.__total_evaluation_count += 1

        found_minimizer = False
        evalcount = 1
        while not found_minimizer:
            x_k_p1 = list(map(lambda x, y: x - self.__alpha * y, x_k, d_cost))

            cost_x_kp1 = evaluate(x_k_p1)
            self.__total_evaluation_count += 1

            if cost_x_kp1 < cost_x_k and math.isfinite(cost_x_kp1):
                found_minimizer = True
                if evalcount == 1:
                    self.__alpha *= self.__gamma_2
                return evalcount

            elif cost_x_kp1 == cost_x_k:
                self.__minimum_found = True

            else:
                evalcount += 1
                self.__alpha *= self.__gamma_1


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
            #costs.append((last_costs['deformation_cost'], last_costs['attach_cost'], last_costs['cost']))
            costs.append(list(last_costs.values()))

            print("="*80)
            print("Iteration {it}\n Total cost={cost}\n Attach cost={attach_cost}\n Deformation cost={deformation_cost}\n Step length={alpha}\n Model evaluation count={eval_count}".format(it=i+1, cost=last_costs['cost'], attach_cost=last_costs['attach_cost'], deformation_cost=last_costs['deformation_cost'], alpha=self.__optim.alpha, eval_count=eval_count))

            if self.__optim.minimum_found:
                break

        print("="*80)
        print("End of the optimisation process.")
        print("Final energy =", last_costs['cost'])
        print("Total model evaluation count  =", self.__optim.total_evaluation_count)
        print("Time elapsed =", time.time() - start)

        return costs

