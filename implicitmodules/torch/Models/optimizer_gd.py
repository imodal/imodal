import math
import copy
from itertools import chain

import torch

from implicitmodules.torch.Models import register_optimizer


class OptimizerGradientDescent:
    def __init__(self, parameters, init_step_length=1e0, gamma1=0.8, gamma2=1./0.8, verbose=False):
        self.__parameters = parameters
        self.__gammas = {'gamma1': gamma1, 'gamma2': gamma2}
        self.__init_step_length = init_step_length
        self.__verbose = verbose
        self.__state_dict = {'state': {}}

        self.reset()

    @property
    def minimum_found(self):
        return self.__minimum_found

    def state_dict(self):
        return self.__state_dict

    @property
    def total_evaluation_count(self):
        return self.__total_evaluation_count

    def reset(self):
        self.__minimum_found = False
        self.__total_evaluation_count = 0

        self.__step_lengths = []
        for param_group in self.__parameters:
            step_length = self.__init_step_length
            if 'step_length' in param_group:
                step_length = param_group['step_length']

            self.__step_lengths.append(step_length)

    def zero_grad(self):
        """ Frees parameters computation graphs and zero out their accumulated gradients. """
        for param in self.__parameters_to_list(self.__parameters):
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def step(self, closure):
        # Parameters of the model at the current state
        x_k = self.__copy_model_parameters(self.__parameters)

        # Evaluate model at the current state, also compute the gradients
        cost_x_k, d_cost = self.__evaluate(closure, x_k, True)
        self.__total_evaluation_count += 1

        line_seach_step = 1
        minimum_found = False
        while not minimum_found:
            cost_x_kp1 = self.__line_search_step(closure, x_k, d_cost, self.__step_lengths)
            if self.__verbose:
                print("Line search, step {step}: cost={cost}, step_length={step_length}".format(step=line_seach_step, cost=cost_x_kp1, step_length=self.__step_lengths))

            # Naive (but fast) criterion, maybe use something like (strong ?) wolfe conditions
            if cost_x_kp1 < cost_x_k and math.isfinite(cost_x_kp1):
                minimum_found = True

                if line_seach_step == 1:
                    # Minimizer found after only one step, increase the step length
                    self.__update_step_length('gamma2')
                return

            # Maybe have a better termination condition?
            elif cost_x_kp1 == cost_x_k:
                minimum_found = True
            else:
                line_seach_step = line_seach_step + 1

                # Minimizer not found, decrease the step length
                self.__update_step_length('gamma1')

    def __evaluate(self, closure, parameters, compute_backward):
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

    def __line_search_step(self, closure, x_k, d_cost, step_lengths):
        # Compute x_{k+1} = x_k + step_length * dx_k
        x_k_p1 = []
        for group_param, d_group_param, step_length in zip(x_k, d_cost, step_lengths):
            x_k_p1.append({'params': [param - step_length * d_param for param, d_param in zip(group_param['params'], d_group_param['params'])]})


        # x_k_p1 = [{'params': [param - step_length * d_param for param, d_param in zip(group_param['params'], d_group_param['params'])]} for group_param, d_group_param, step_length in zip(x_k, d_cost, step_lengths)]

        # return cost of the system at x_{k+1}
        return self.__evaluate(closure, x_k_p1, False)

    def __update_step_length(self, gamma):
        for i, param_group in enumerate(self.__parameters):
            if gamma in param_group:
                self.__step_lengths[i] = param_group[gamma] * self.__step_lengths[i]
            else:
                self.__step_lengths[i] = self.__gammas[gamma] * self.__step_lengths[i]

    def __parameters_to_list(self, parameters):
        return list(chain(*(group_param['params'] for group_param in parameters)))

    def __fill_model_parameters(self, parameters):
        for model_parameter, parameter in zip(self.__parameters_to_list(self.__parameters), self.__parameters_to_list(parameters)):
            model_parameter.data = parameter.data

    def __copy_model_parameters(self, parameters, getter=lambda x: x.data):
        out = []
        for param_group in parameters:
            out.append({'params': [getter(param).clone() for param in param_group['params']]})

        return out


