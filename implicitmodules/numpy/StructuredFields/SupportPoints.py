import numpy as np


class SupportPoints:
    """
    This class implement a for the structured Field. It s
    """
    
    def __init__(self, support):
        self.__value = support
        self.fill_zero_cotan()
        
    def copy(self):
        supp = SupportPoints(self.__value)
        return supp
        
    def copy_full(self):
        supp = SupportPoints(self.__value)
        if hasattr(self, '__value'):
            supp.value = self.__value.copy()
            
        if hasattr(self, 'cotan'):
            supp.cotan = self.cotan.copy()
        return supp
        
    def fill_value(self, val):
        self.__value = val
    
    def __get_value(self):
        return self.__value
    
    value = property(__get_value, fill_value)
    
    def fill_zero_cotan(self):
        self.__cotan = np.zeros_like(self.__value)

    def fill_cotan(self, cot):
        self.__cotan = cot
    
    def __get_cotan(self):
        return self.__cotan
    
    cotan = property(__get_cotan, fill_cotan)
