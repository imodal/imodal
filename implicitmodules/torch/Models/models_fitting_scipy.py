import time
import gc
from itertools import chain

import numpy as np
import torch
import scipy.optimize

from .models_fitting import ModelFitting
from implicitmodules.torch.Attachment import CompoundAttachment

class ModelFittingScipy(ModelFitting):
    def __init__(self, model, post_iteration_callback=None):
        super().__init__(model, post_iteration_callback)

    def reset(self):
        pass

    def fit(self, target, max_iter, options={}, log_interval=1, disp=True, display_time=True):
        last_costs = {}
        costs = []

        shoot_method = 'euler'
        shoot_it = 10
        bounds = None

        if 'shoot_method' in options:
            shoot_method = options['shoot_method']
            del options['shoot_method']

        if 'shoot_it' in options:
            shoot_it = options['shoot_it']
            del options['shoot_it']

        if 'bounds' in options:
            bounds = options['bounds']
            del options['bounds']

        # Function that will be optimized, returns the cost for a given state of the model.
        def closure(x):
            self.__numpy_to_model(x)
            self.__zero_grad()

            # Evaluate the model
            cost, deformation_cost, attach_cost = self.model.evaluate(target, shoot_method, shoot_it)

            d_cost = self.__model_to_numpy(self.model, grad=True)

            # Manualy fire garbage collection
            gc.collect()

            # Save for printing purpose.
            last_costs['deformation_cost'] = deformation_cost
            last_costs['attach_cost'] = attach_cost
            last_costs['cost'] = cost

            return (cost, d_cost)

        self.__it = 1

        # Callback function called at the end of each iteration for printing and logging purpose.
        def callback(xk):

            if (self.__it % log_interval == 0 or self.__it == 1) and log_interval != -1 and disp:
                print("="*80)
                if display_time:
                    print("Time: {time}".format(time=time.perf_counter() - start_time))

                print("Iteration: {it} \nTotal energy = {cost} \nAttach cost = {attach} \nDeformation cost = {deformation}".format(it=self.__it, cost=last_costs['cost'], attach=last_costs['attach_cost'], deformation=last_costs['deformation_cost']))

            if self.post_iteration_callback:
                self.post_iteration_callback(self.model)

            costs.append((last_costs['deformation_cost'], last_costs['attach_cost'], last_costs['cost']))
            last_costs.clear()

            self.__it = self.__it + 1

        step_options = {'disp': False, 'maxiter': max_iter}
        step_options.update(options)

        x_0 = self.__model_to_numpy(self.model)
        closure(x_0)

        if disp:
            print("Initial energy = %.3f" % last_costs['cost'])

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
        """ Frees parameters computation graphs and zero out their accumulated gradients. """
        for param in self.__parameters_to_list(self.model.parameters):
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

