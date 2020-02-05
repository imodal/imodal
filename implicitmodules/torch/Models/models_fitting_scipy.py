import time
import copy
import gc

import numpy as np
import torch
import scipy.optimize

from .models_fitting import ModelFitting
from implicitmodules.torch.Attachment import CompoundAttachment

class ModelFittingScipy(ModelFitting):
    def __init__(self, model, post_iteration_callback=None):
        super().__init__(model, post_iteration_callback)

        self.__post_iteration_callback = post_iteration_callback

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
            model = self.__numpy_to_model(x.astype('float64'))

            # Evaluate the model.
            cost, deformation_cost, attach_cost = model.compute(target, it=shoot_it, method=shoot_method)

            d_cost = self.__model_to_numpy(model, grad=True)

            # Delete the model and force the garbage collector to cleanup.
            del model
            gc.collect()

            # Save for printing purpose.
            last_costs['deformation_cost'] = deformation_cost
            last_costs['attach_cost'] = attach_cost
            last_costs['cost'] = cost
            return (cost, d_cost)

        self.__it = 1

        # Callback function called at the end of each iteration for printing and logging purpose.
        def callback(xk):
            # Update the model with the current state
            self.model.fill_from(self.__numpy_to_model(xk.astype('float64')))

            if self.__post_iteration_callback:
                self.__post_iteration_callback(self.model)

            if (self.__it % log_interval == 0 or self.__it == 1) and log_interval != -1 and disp:
                print("="*80)
                if display_time:
                    print("Time: {time}".format(time=time.perf_counter() - start_time))

                print("Iteration: {it} \nTotal energy = {cost} \nAttach cost = {attach} \nDeformation cost = {deformation}".format(it=self.__it, cost=last_costs['cost'], attach=last_costs['attach_cost'], deformation=last_costs['deformation_cost']))

            costs.append((last_costs['deformation_cost'], last_costs['attach_cost'], last_costs['cost']))
            last_costs.clear()

            self.__it = self.__it + 1

        step_options = {'disp': False, 'maxiter': max_iter}
        step_options.update(options)

        x_0 = self.__model_to_numpy(self.model.clone_init())
        closure(x_0)

        if disp:
            print("Initial energy = %.3f" % last_costs['cost'])

        start_time = time.perf_counter()
        res = scipy.optimize.minimize(closure, x_0, method='L-BFGS-B', jac=True, options=step_options, callback=callback, bounds=bounds)

        # TODO: Is this really necessary ? We already update the model state at the end of each iteration.
        self.model.fill_from(self.__numpy_to_model(res.x.astype('float64')))

        if disp:
            print("="*80)
            print("Optimisation process exited with message:", res.message)
            print("Final energy =", res.fun)
            print("Closure evaluations =", res['nfev'])
            print("Time elapsed =", time.perf_counter() - start_time)

        return costs

    def __model_to_numpy(self, model, grad=False):
        """Converts model parameters into a single state vector."""
        if not all(param.is_contiguous() for param in model.parameters):
            raise ValueError("Scipy optimization routines are only compatible with parameters given as *contiguous* tensors.")

        if grad:
            tensors = [param.grad.data.flatten().cpu().numpy() for param in model.parameters]
        else:
            tensors = [param.detach().flatten().cpu().numpy() for param in model.parameters]

        return np.ascontiguousarray(np.hstack(tensors), dtype='float64')

    def __numpy_to_model(self, x):
        """Fill a cloned model with the state vector x."""
        i = 0

        model = self.model.clone_init()

        for param in model.parameters:
            offset = param.numel()
            param.data = torch.from_numpy(x[i:i+offset]).view(param.data.size()).type(param.data.type())
            i += offset

        assert i == len(x)

        return model

