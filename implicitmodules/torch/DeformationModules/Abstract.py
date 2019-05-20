import copy


class DeformationModule:
    """Abstract module."""
    
    def __init__(self):
        super().__init__()
    
    def copy(self):
        return copy.copy(self)
    
    def __call__(self, gd, controls, points):
        """Applies the generated vector field on given points."""
        raise NotImplementedError
    
    def cost(self, gd, controls):
        """Returns the cost."""
        raise NotImplementedError
