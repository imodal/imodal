
class BaseModel:
    def __init__(self):
        pass

    def evaluate(self, target, solver, it):
        raise NotImplementedError()

    def compute_deformed(self, solver, it, costs=None, intermediates=None):
        raise NotImplementedError()


