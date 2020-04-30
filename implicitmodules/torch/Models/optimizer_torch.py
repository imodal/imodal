import gc
import math
from itertools import chain

import torch

from implicitmodules.torch.Models import BaseOptimizer, register_optimizer



class OptimizerTorch(BaseOptimizer):
    def __init__(self, torch_optimizer, model):
        self.__torch_optimizer = torch_optimizer

        super().__init__(model, None)

    def reset(self):
        pass

    def optimize(self, target, max_iter, post_iteration_callback, costs, shoot_solver, shoot_it, options=None):
        if options is None:
            options = {}

        self.__eval_count = 0
        self.__last_costs = None
        def _evaluate():
            self.__torch_optimizer.zero_grad()

            costs = self.model.evaluate(target, shoot_solver, shoot_it)

            self.__last_costs = costs
            self.__eval_count = self.__eval_count + 1
            return costs['total']

        ftol = 1e-5
        if 'ftol' in options:
            ftol = options['ftol']

        cont = True
        termination_state = 0
        it = 0
        last_total = float('inf')
        for i in range(max_iter):
            self.__torch_optimizer.step(_evaluate)

            post_iteration_callback(self.model, self.__last_costs)

            it = it + 1

            if math.isnan(self.__last_costs['total']):
                return {'success': False, 'final': float('nan'), 'message': "Evaluated function gave NaN.", 'neval': self.__eval_count}

            if (self.__last_costs['total'] - last_total)/max(self.__last_costs['total'], last_total, 1) <= ftol:
                return {'success': True, 'final': self.__last_costs['total'], 'message': "Convergence achieved.", 'neval': self.__eval_count}

            last_total = self.__last_costs['total']

        return {'success': False, 'final': self.__last_costs['total'], 'message': "Total number of iterations reached.", 'neval': self.__eval_count}


def __parameters_to_list(parameters):
    if isinstance(parameters, dict):
        return list(chain(*parameters.values()))

    return list(parameters)


def __create_torch_optimizer(torch_optimizer):
    def _create(model, **kwargs):
        return OptimizerTorch(torch_optimizer(__parameters_to_list(model.parameters), **kwargs), model)

    return _create

register_optimizer("torch_sgd", __create_torch_optimizer(torch.optim.SGD))
register_optimizer("torch_lbfgs", __create_torch_optimizer(torch.optim.LBFGS))


