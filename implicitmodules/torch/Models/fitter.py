import time

import torch

from implicitmodules.torch.Models import BaseOptimizer, create_optimizer, get_default_optimizer
from implicitmodules.torch.Models import BaseModel
from implicitmodules.torch.Utilities import append_in_dict_of_list


class Fitter:
    def __init__(self, model, optimizer=None, post_iteration_callback=None):
        assert isinstance(model, BaseModel)
        assert isinstance(optimizer, BaseOptimizer) or isinstance(optimizer, str) or optimizer is None

        if optimizer is None:
            optimizer = get_default_optimizer()

        if isinstance(optimizer, str):
            optimizer = create_optimizer(optimizer, model)

        self.__optimizer = optimizer
        self.__post_iteration_callback = post_iteration_callback
        self.__model = model

        self.reset()

    def reset(self):
        self.__it = 0
        self.__optimizer.reset()

    def fit(self, target, max_iter, options={}, costs=None, disp=True):
        assert isinstance(costs, dict) or costs is None

        shoot_solver = 'euler'
        shoot_it = 10
        tol = None

        if 'shoot_solver' in options:
            shoot_solver = options['shoot_solver']
            del options['shoot_solver']

        if 'shoot_it' in options:
            shoot_it = options['shoot_it']
            del options['shoot_it']

        if 'tol' in options:
            tol = options['tol']
            del options['tol']

        # Initial cost
        if costs is not None or disp:
            with torch.autograd.no_grad():
                cost_0 = self.__model.evaluate(target, shoot_solver, shoot_it)

            if costs is not None:
                append_in_dict_of_list(costs, cost_0)

            if disp:
                print("Starting optimization with method {method}".format(method=self.__optimizer.method_name))
                print("Initial cost={cost}".format(cost=cost_0))

        def _post_iteration_callback(model, last_costs):
            # Display progression
            if disp:
                print("="*80)
                print("Time: {time}".format(time=time.perf_counter() - start_time))

                print("Iteration: {it}".format(it=self.__it))
                self.__print_costs(last_costs)

            self.__it = self.__it + 1

            if costs:
                append_in_dict_of_list(costs, last_costs)

        start_time = time.perf_counter()
        res = self.__optimizer.optimize(target, max_iter, _post_iteration_callback, costs, shoot_solver, shoot_it, tol, options=options)

        if disp:
            print("="*80)
            print("Optimisation process exited with message: {message}".format(message=res['message']))
            print("Final cost={cost}".format(cost=res['final']))
            print("Model evaluation count={neval}".format(neval=res['neval']))
            if 'neval_grad' in res:
                print("Model gradient evaluation count={neval_grad}".format(neval_grad=res['neval_grad']))
            print("Time elapsed =", time.perf_counter() - start_time)        

    def __print_costs(self, costs):
        print("Costs")
        for key in costs.keys():
            print("{cost}={value}".format(cost=key, value=costs[key]))
        print("Total cost={total}".format(total=sum(costs.values())))

