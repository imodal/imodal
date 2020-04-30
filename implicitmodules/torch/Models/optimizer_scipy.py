from itertools import chain
import gc

import torch
import numpy as np
from scipy.optimize import minimize

from implicitmodules.torch.Models.optimizer import BaseOptimizer, register_optimizer


class OptimizerScipy(BaseOptimizer):
    def __init__(self, model, scipy_method, need_grad, **kwargs):
        self.__scipy_method = scipy_method

        self.__need_grad = need_grad
        self.__evaluate = self.__evaluate_no_grad
        if self.__need_grad:
            self.__evaluate = self.__evaluate_grad

        super().__init__(model, kwargs)

    def __evaluate_grad(self, target, shoot_solver, shoot_it):
        def _evaluate(xk):
            self.__zero_grad()
            self.__numpy_to_model(xk)

            costs = self.model.evaluate(target, shoot_solver, shoot_it)

            d_costs = self.__model_to_numpy(self.model, True)

            gc.collect()

            self.__last_costs = costs
            return (costs['total'], d_costs)

        return _evaluate

    def __evaluate_no_grad(self, target, shoot_solver, shoot_it):
        def _evaluate(xk):
            self.__zero_grad()
            self.__numpy_to_model(xk)

            with torch.autograd.no_grad():
                costs = self.model.evaluate(target, shoot_solver, shoot_it)

            gc.collect()

            self.__last_costs = costs
            return costs['total']

        return _evaluate

    def reset(self):
        pass

    def optimize(self, target, max_iter, post_iteration_callback, costs, shoot_solver, shoot_it, options=None):
        assert options is None or isinstance(options, dict)
        if options is None:
            options = {}

        x0 = self.__model_to_numpy(self.model, False)

        def _post_iteration_callback(x_k):
            if post_iteration_callback:
                post_iteration_callback(self.model, self.__last_costs)

        jac = None
        if self.__need_grad:
            jac = True

        options['maxiter'] = max_iter

        self.__last_cost = None
        scipy_res = minimize(self.__evaluate(target, shoot_solver, shoot_it), x0, method=self.__scipy_method, jac=self.__need_grad, callback=_post_iteration_callback, options=options, **self.kwargs)

        res = {'final': scipy_res.fun, 'success': scipy_res.success, 'message': scipy_res.message, 'neval': scipy_res.nfev}

        if self.__need_grad:
            res['neval_grad']: scipy_res.njev

        return res

    def __parameters_to_list(self, parameters):
        if isinstance(parameters, dict):
            return list(chain(*parameters.values()))

        return list(parameters)

    def __model_to_numpy(self, model, grad):
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


def __create_scipy_optimizer(method_name, need_grad):
    def _create(model, **kwargs):
        return OptimizerScipy(model, method_name, need_grad, **kwargs)

    return _create


register_optimizer("scipy_nelder-mead", __create_scipy_optimizer("Nelder-Mead", False))
register_optimizer("scipy_powell", __create_scipy_optimizer("Powell", False))
register_optimizer("scipy_cg", __create_scipy_optimizer("CG", True))
register_optimizer("scipy_bfgs", __create_scipy_optimizer("BFGS", True))
register_optimizer("scipy_newton-cg", __create_scipy_optimizer("Newton-CG", True))
register_optimizer("scipy_l-bfgs-b", __create_scipy_optimizer("L-BFGS-B", True))
register_optimizer("scipy_tnc", __create_scipy_optimizer("TNC", True))
register_optimizer("scipy_cobyla", __create_scipy_optimizer("COBYLA", False))
register_optimizer("scipy_slsqp", __create_scipy_optimizer("SLSQP", True))


