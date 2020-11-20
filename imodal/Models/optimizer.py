class BaseOptimizer:
    def __init__(self, model):
        self.__model = model

    @property
    def model(self):
        return self.__model

    @property
    def kwargs(self):
        return self.__kwargs

    def reset(self):
        raise NotImplementedError()

    def optimize(self, target, model, max_iter, post_iteration_callback, costs, shoot_solver, shoot_it, options={}):
        raise NotImplementedError()


__optimizers = {}
__default_optimizer = 'scipy_l-bfgs-b'


def create_optimizer(method, model):
    assert is_valid_optimizer(method)
    return __optimizers[method](model)


def set_default_optimizer(method):
    assert is_valid_optimizer(method)
    __default_optimizer = method


def get_default_optimizer():
    return __default_optimizer


def register_optimizer(method, optimizer):
    __optimizers[method] = optimizer


def list_optimizers():
    return list(__optimizers.keys())


def is_valid_optimizer(method):
    return method in __optimizers.keys()

