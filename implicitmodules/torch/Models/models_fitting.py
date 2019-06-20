class ModelFitting:
    def __init__(self, model):
        self.__model = model

    @property
    def model(self):
        return self.__model

    def reset(self):
        raise NotImplementedError

    def fit(self, target, max_iter, options={}):
        raise NotImplementedError

