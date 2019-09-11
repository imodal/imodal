class ModelFitting:
    def __init__(self, model, post_iteration_callback):
        self.__model = model
        self.__post_iteration_callback = post_iteration_callback

    @property
    def model(self):
        return self.__model

    @property
    def post_iteration_callback(self):
        return self.__post_iteration_callback

    def reset(self):
        raise NotImplementedError

    def fit(self, target, max_iter, options={}):
        raise NotImplementedError

