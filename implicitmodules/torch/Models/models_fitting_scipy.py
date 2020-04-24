import time
import gc
from itertools import chain
import math

import numpy as np
import torch
import scipy.optimize

from .models_fitting import ModelFitting
from implicitmodules.torch.Attachment import CompoundAttachment
from implicitmodules.torch.Utilities import append_in_dict_of_list

class ModelFittingScipy(ModelFitting):
    def __init__(self, model, post_iteration_callback=None):
        super().__init__(model, post_iteration_callback)

    def reset(self):
        pass

    def fit(self, target, max_iter, options={}, log_interval=1, disp=True, costs=None):
        assert isinstance(costs, dict) or costs is None
        shoot_solver = 'euler'
        shoot_it = 10
        bounds = None
        last_costs = []

        if 'shoot_solver' in options:
            shoot_solver = options['shoot_solver']
            del options['shoot_solver']

        if 'shoot_it' in options:
            shoot_it = options['shoot_it']
            del options['shoot_it']

        if 'bounds' in options:
            bounds = options['bounds']
            del options['bounds']

        # Function that will be optimized, returns the cost and its derivative for a given state of the model.
        def closure(x):
            self.__numpy_to_model(x)
            self.__zero_grad()

            # Evaluate the model
            costs = self.model.evaluate(target, shoot_solver, shoot_it)

            d_cost = self.__model_to_numpy(self.model, grad=True)

            # Manualy fire garbage collection
            gc.collect()

            # Save the costs
            last_costs.append(costs)

            return (costs['total'], d_cost)

        self.__it = 1

        # Callback function called at the end of each iteration for printing and logging purpose.
        def callback(xk):

            if (self.__it % log_interval == 0 or self.__it == 1) and log_interval != -1 and disp:
                print("="*80)
                print("Time: {time}".format(time=time.perf_counter() - start_time))

                print("Iteration: {it}".format(it=self.__it))
                self._print_costs(last_costs[-1])

            if self.post_iteration_callback:
                self.post_iteration_callback(self.model)

            
            if costs is not None:
                append_in_dict_of_list(costs, last_costs[-1])

            last_costs.clear()

            self.__it = self.__it + 1

        step_options = {'disp': False, 'maxiter': max_iter}
        step_options.update(options)

        x_0 = self.__model_to_numpy(self.model)
        closure(x_0)

        if costs is not None:
            append_in_dict_of_list(costs, last_costs[-1])

        if disp:
            print("Initial energy = %.3f" % last_costs[-1]['total'])

        last_costs = []

        start_time = time.perf_counter()
        res = scipy.optimize.minimize(closure, x_0, method='L-BFGS-B', jac=True, options=step_options, callback=callback, bounds=bounds)

        if disp:
            print("="*80)
            print("Optimisation process exited with message:", res.message)
            print("Final energy =", res.fun)
            print("Closure evaluations =", res['nfev'])
            print("Time elapsed =", time.perf_counter() - start_time)

        return costs

    def __parameters_to_list(self, parameters):
        if isinstance(parameters, dict):
            return list(chain(*parameters.values()))

        return list(parameters)

    def __model_to_numpy(self, model, grad=False):
        """Converts model parameters into a single state vector."""
        if not all(param.is_contiguous() for param in self.__parameters_to_list(self.model.parameters)):
            raise ValueError("Scipy optimization routines are only compatible with parameters given as *contiguous* tensors.")

        if grad:
            tensors = [param.grad.data.flatten().cpu().numpy() for param in self.__parameters_to_list(model.parameters)]
        else:
            tensors = [param.detach().flatten().cpu().numpy() for param in self.__parameters_to_list(model.parameters)]

        return np.ascontiguousarray(np.hstack(tensors), dtype='float64')

    def __numpy_to_model(self, x):
        """Fill the model with the state vector x."""
        i = 0

        for param in self.__parameters_to_list(self.model.parameters):
            offset = param.numel()
            param.data = torch.from_numpy(x[i:i+offset]).view(param.data.size()).to(dtype=param.dtype, device=param.device)
            i += offset

        assert i == len(x)

    def __zero_grad(self):
        """ Free parameters computation graphs and zero out their accumulated gradients. """
        for param in self.__parameters_to_list(self.model.parameters):
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

