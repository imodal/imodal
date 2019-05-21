import numpy as np

import implicitmodules.numpy.StructuredFields.SupportAbstract as ab



class SupportPoints(ab.SupportAbstract):
    def __init__(self, N_pts, dim):  #

        self.dim = dim
        self.N_pts = N_pts
        
    def copy(self):
        supp = SupportPoints(self.N_pts, self.dim)
        return supp
        
        
    def copy_full(self):
        supp = SupportPoints(self.N_pts, self.dim)
        if hasattr(self, 'value'):
            supp.value = self.value.copy()
            
        if hasattr(self, 'cotan'):
            supp.cotan = self.cotan.copy()
        return supp
        
    def fill_zero_cotan(self):
         self.cotan = np.zeros([self.N_pts, self.dim])
        
    def fill_value(self, val):
        self.value = val.copy()
        
    def get_value(self):
        return self.value.copy()
        
    def fill_cotan(self, cot):
        self.cotan = cot.copy()