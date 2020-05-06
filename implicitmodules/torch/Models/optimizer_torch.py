import gc
import math
from itertools import chain

import torch

from implicitmodules.torch.Models import BaseOptimizer, register_optimizer
from implicitmodules.torch.Models.optimizer_gd import GradientDescentOptimizer


class OptimizerTorch(BaseOptimizer):
    def __init__(self, torch_optimizer, model):
        self.__torch_optimizer = torch_optimizer

        # Creation of the Torch optimizer is deffered (optimization parameters are given
        # when optimize() is called)
        self.__optimizer = None

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
            self.__optimizer = self.__torch_optimizer(list(self.model.parameters.values()), **options)

        self.__eval_count = 0
        self.__last_costs = None
        def _evaluate():
            self.__optimizer.zero_grad()

            costs = self.model.evaluate(target, shoot_solver, shoot_it)

            self.__last_costs = costs
            self.__eval_count = self.__eval_count + 1
            return costs['total']

        cont = True
        termination_state = 0
        it = 0
        last_total = float('inf')
        for i in range(max_iter):
            self.__optimizer.step(closure=_evaluate)

            post_iteration_callback(self.model, self.__last_costs)

            it = it + 1

            if math.isnan(self.__last_costs['total']):
                return {'success': False, 'final': float('nan'), 'message': "Evaluated function gave NaN.", 'neval': self.__eval_count}

            if (last_total - self.__last_costs['total'])/max(self.__last_costs['total'], last_total, 1) <= tol:
                return {'success': True, 'final': self.__last_costs['total'], 'message': "Convergence achieved.", 'neval': self.__eval_count}

            last_total = self.__last_costs['total']

        return {'success': False, 'final': self.__last_costs['total'], 'message': "Total number of iterations reached.", 'neval': self.__eval_count}


def __create_torch_optimizer(torch_optimizer):
    def _create(model):
        return OptimizerTorch(torch_optimizer, model)

    return _create

register_optimizer("torch_sgd", __create_torch_optimizer(torch.optim.SGD))
register_optimizer("torch_lbfgs", __create_torch_optimizer(torch.optim.LBFGS))

register_optimizer("gd", __create_torch_optimizer(GradientDescentOptimizer))

