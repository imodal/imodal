class Manifold:
    def __init__(self):
        super().__init__()

    # We would idealy use deepcopy but only graph leaves Tensors supports it right now
    def copy(self):
        raise NotImplementedError

    @property
    def numel_gd(self):
        raise NotImplementedError

    def infinitesimal_action(self, module):
        raise NotImplementedError
