import time

import torch

from .models_fitting import ModelFitting
from implicitmodules.torch.Attachment import CompoundAttachment


class ModelFittingGradientDescent(ModelFitting):
    def __init__(self, model, step_length, history_size=10, post_iteration_callback=None):
        super().__init__(model, post_iteration_callback)
        self.__history_size = history_size
        self.__step_length = step_length

        self.reset()

    def reset(self):
        self.__optim = torch.optim.SGD(self.model.parameters, lr=self.__step_length)

    def fit(self, target, max_iter, options={}, log_interval=1, disp=True):
        last_costs = {}
        costs = []

        shoot_method = 'euler'
        shoot_it = 10

        if 'shoot_method' in options:
            shoot_method = options['shoot_method']
            del options['shoot_method']

        if 'shoot_it' in options:
            shoot_it = options['shoot_it']
            del options['shoot_it']

        def closure():
            self.__optim.zero_grad()

            # Call precompute callback if available
            if self.model.precompute_callback is not None:
                self.model.precompute_callback(self.model.init_manifold, self.model.modules, self.model.parameters)

            # Shooting + loss computation
            cost, deformation_cost, attach_cost = self.model.compute(it=shoot_it, method=shoot_method)

            # Save for printing purpose
            last_costs['deformation_cost'] = deformation_cost.detach().cpu().item()
            last_costs['attach_cost'] = attach_cost.detach().cpu().item()
            last_costs['cost'] = cost.detach().cpu().item()

            return cost

        # Precompute target data for varifolds
        CompoundAttachment(self.model.attachments).target = target

        loss = closure()

        print("Initial energy = %.3f" % last_costs['cost'])

        closure_count = 0
        start = time.time()
        costs = [last_costs]
        for i in range(max_iter):
            # Computing step
            # step_options = {'closure': closure, 'current_loss': loss, 'ls_debug': False}
            # step_options.update(options)
            self.__optim.step(closure)

            # Retrieving costs
            costs.append((last_costs['deformation_cost'], last_costs['attach_cost'], last_costs['cost']))

            print("="*80)
            print("Iteration: %d \n Total energy = %f \n Attach cost = %f \n Deformation cost = %f \n Step length = %.12f" % (i + 1, last_costs['cost'], last_costs['attach_cost'], last_costs['deformation_cost'], self.__step_length))

            # if fail or not desc_dir:
            #     break

        print("="*80)
        print("End of the optimisation process.")
        print("Final energy =", last_costs['cost'])
        print("Closure evaluations =", closure_count)
        print("Time elapsed =", time.time() - start)

        return costs

