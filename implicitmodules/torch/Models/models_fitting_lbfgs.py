import time

import torch

from .LBFGS import FullBatchLBFGS
from .models_fitting import ModelFitting


class ModelFittingLBFGS(ModelFitting):
    def __init__(self, model, step_length, histroy_size=10, post_iteration_callback=None):
        super().__init__(model, post_iteration_callback)
        self.__history_size = history_size
        self.__step_length = step_length

        self.reset()

    def reset(self):
        self.__optim = FullBatchLBFGS(self.model.parameters, lr=self.__step_length, history_size=self.__history_size, line_search=self.__line_search)

    def fit(self, target, max_iter, options={}):
        last_costs = {}
        def closure():
            self.__optim.zero_grad()

            # Call precompute callback if available
            if self.model.precompute_callback is not None:
                self.model.precompute_callback(self.model.init_manifold, self.model.modules, self.model.parameters)

            # Shooting + loss computation
            cost, deformation_cost, attach_cost = self.model.compute(target)
            cost = self.__lam*attach_cost + deformation_cost

            # Save for printing purpose
            last_costs['deformation_cost'] = deformation_cost.detach()
            last_costs['attach_cost'] = self.__lam*attach_cost.detach()
            last_costs['cost'] = cost.detach()

            return cost

        loss = closure()
        loss.backward()

        print("Initial energy = %.3f" % last_costs['cost'])

        closure_count = 0
        start = time.time()
        costs = [last_costs]
        for i in range(max_iter):
            # Computing step
            step_options = {'closure': closure, 'current_loss': loss, 'ls_debug': False}
            step_options.update(options)
            loss, _, step_length, _, F_eval, G_eval, desc_dir, fail = self.__optim.step(step_options)

            # Retrieving costs
            costs.append((last_costs['deformation_cost'], last_costs['attach_cost'], last_costs['cost']))

            print("="*80)
            print("Iteration: %d \n Total energy = %f \n Attach cost = %f \n Deformation cost = %f \n Step length = %.12f \n Closure evaluations = %d" % (i + 1, last_costs['cost'], last_costs['attach_cost'], last_costs['deformation_cost'], step_length, F_eval))
            closure_count += F_eval

            if fail or not desc_dir:
                break

        print("="*80)
        print("End of the optimisation process.")
        print("Final energy =", last_costs['cost'])
        print("Closure evaluations =", closure_count)
        print("Time elapsed =", time.time() - start)

        return costs

