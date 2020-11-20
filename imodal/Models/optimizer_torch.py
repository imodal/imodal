import math
from itertools import chain
import gc

import torch

from implicitmodules.torch.Models import BaseOptimizer, register_optimizer
from implicitmodules.torch.Models.optimizer_gd import OptimizerGradientDescent


class OptimizerTorch(BaseOptimizer):
    def __init__(self, torch_optimizer, model, single_parameter_group, verbose=True):
        self.__torch_optimizer = torch_optimizer

        # Creation of the Torch optimizer is deffered (optimization parameters are given
        # when optimize() is called)
        self.__optimizer = None
        self.__single_parameter_group = single_parameter_group
        self.__verbose = verbose

        super().__init__(model)

    @property
    def method_name(self):
        return self.__torch_optimizer.__module__.split(".")[0] + " " + self.__torch_optimizer.__name__

    def reset(self):
        self.__optimizer = None

    def optimize(self, target, max_iter, post_iteration_callback, costs, shoot_solver, shoot_it, tol, options=None):
        if options is None:
            options = {}

        if tol is None:
            tol = 1e-10

        # If the optimizer has not been created before, create it now
        if self.__optimizer is None:
            if self.__single_parameter_group:
                self.__optimizer = self.__torch_optimizer(self.__flatten_parameters(self.model.parameters), **options)
            else:
                self.__optimizer = self.__torch_optimizer(list(self.model.parameters.values()), **options)

        self.__eval_count = 0
        self.__last_costs = None

        def _evaluate():
            self.__optimizer.zero_grad()

            costs = self.model.evaluate(target, shoot_solver, shoot_it)

            self.__last_costs = costs
            self.__eval_count = self.__eval_count + 1

            gc.collect()

            sum_costs = sum(costs.values())

            if self.__verbose:
                print("Evaluated model with costs={sumcosts}".format(sumcosts=sum_costs))

            return sum_costs

        last_total_cost = float('inf')
        for i in range(max_iter):
            self.__optimizer.step(closure=_evaluate)

            post_iteration_callback(self.model, self.__last_costs)

            total_cost = sum(self.__last_costs.values())

            if math.isnan(total_cost):
                return {'success': False, 'final': float('nan'), 'message': "Evaluated function gave NaN.", 'neval': total_cost}

            if (last_total_cost - total_cost)/max(total_cost, last_total_cost, 1) <= tol:
                return {'success': True, 'final': total_cost, 'message': "Convergence achieved.", 'neval': self.__eval_count}

            last_total_cost = total_cost

        return {'success': False, 'final': total_cost, 'message': "Total number of iterations reached.", 'neval': self.__eval_count}

    def __flatten_parameters(self, parameters):
        return [{'params': list(chain(*(parameter['params'] for parameter in parameters.values())))}]


def __create_torch_optimizer(torch_optimizer, single_parameter_group):
    def _create(model):
        return OptimizerTorch(torch_optimizer, model, single_parameter_group)

    return _create


register_optimizer("torch_sgd", __create_torch_optimizer(torch.optim.SGD, False))
register_optimizer("torch_lbfgs", __create_torch_optimizer(torch.optim.LBFGS, True))


register_optimizer("gd", __create_torch_optimizer(OptimizerGradientDescent, False))

