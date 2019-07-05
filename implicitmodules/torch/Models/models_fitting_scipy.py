import time

import numpy as np
import torch
import scipy.optimize

from .models_fitting import ModelFitting

class ModelFittingScipy(ModelFitting):
    def __init__(self, model, step_length, lam, post_iteration_callback=None):
        super().__init__(model)

        self.__step_length = step_length
        self.__lam = lam

        self.__pytorch_optim = torch.optim.SGD(self.model.parameters, lr=step_length)

        self.__post_iteration_callback = post_iteration_callback

    def reset(selt):
        pass

    def fit(self, target, max_iter, method='BFGS', options={}, log_interval=10):
        last_costs = {}
        costs = []

        # Function that will be optimized, returns the cost for a given state of the model.
        def closure(x):
            x = self.__step_length * x.astype('float64')
            self.__numpy_to_model(self.model, x)

            self.__pytorch_optim.zero_grad()

            # Call precompute callback if available
            if self.model.precompute_callback is not None:
                self.model.precompute_callback(self.model.modules, self.model.parameters)

            # Shooting + loss computation
            deformation_cost, attach_cost = self.model.compute(target)
            cost = self.__lam*attach_cost + deformation_cost
            cost.backward()
            c = cost.item()

            dx_c = self.__step_length * self.__model_to_numpy(self.model, grad=True)

            # Save for printing purpose
            last_costs['deformation_cost'] = deformation_cost.item()
            last_costs['attach_cost'] = self.__lam*attach_cost.item()
            last_costs['cost'] = cost.item()

            return (c, dx_c)

        self.__it = 1

        # Callback function called at the end of each iteration for printing and logging purpose.
        def callback(xk):
            if self.__post_iteration_callback:
                self.__post_iteration_callback(self.model)
            
            if (self.__it % log_interval == 0 or self.__it == 1) and log_interval != -1:
                print("="*80)
                print("Iteration: %d \nTotal energy = %f \nAttach cost = %f \nDeformation cost = %f" %
            (self.__it, last_costs['cost'], last_costs['attach_cost'], last_costs['deformation_cost']))
            costs.append((last_costs['deformation_cost'], last_costs['attach_cost'], last_costs['cost']))

            self.__it = self.__it + 1

        step_options = {'disp': True, 'maxiter': max_iter}
        step_options.update(options)

        x_0 = self.__model_to_numpy(self.model)
        initial_cost = closure(x_0)

        print("Initial energy = %.3f" % last_costs['cost'])

        start = time.time()
        res = scipy.optimize.minimize(closure, x_0, method=method, jac=True, options=step_options, callback=callback)

        self.__numpy_to_model(self.model, res.x)

        print("="*80)
        print("Optimisation process exited with message:", res.message)
        print("Final energy =", last_costs['cost'])
        print("Closure evaluations =", res['nfev'])
        print("Time elapsed =", time.time() - start)
        print("Jacobian min max = ", np.min(res.jac), np.max(res.jac))
        print("Hessian condition number =", np.linalg.cond(res.hess_inv))

        return costs

    def __model_to_numpy(self, model, grad=False):
        """ Converts model parameters into a single state vector. """
        if not all(param.is_contiguous() for param in self.model.parameters):
            raise ValueError("Scipy optimization routines are only compatible with parameters given as *contiguous* tensors.")

        if grad:
            tensors = [param.grad.data.view(-1).cpu().numpy() for param in self.model.parameters]
        else:
            tensors = [param.data.view(-1).cpu().numpy() for param in self.model.parameters]

        return np.ascontiguousarray(np.hstack(tensors), dtype='float64')

    def __numpy_to_model(self, model, x):
        """ Fill model with the state vector x. """
        i = 0

        for param in self.model.parameters:
            offset = param.numel()
            param.data = torch.from_numpy(x[i:i+offset]).view(param.data.size()).type(param.data.type())
            i += offset

        assert i == len(x)

