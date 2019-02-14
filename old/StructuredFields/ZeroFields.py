from old import field_structures as fields, pairing_structures as pair


class ZeroField(object):  # tested
    def __init__(self, dim):
        self.dim = dim
        self.dic = {'sig': 1.}
    
    def copy(self):
        return ZeroField(self.dim)
    
    def copy_full(self):
        return ZeroField(self.dim)
    
    def fill_fieldparam(self, param):
        pass
    
    def Apply(self, z, j):
        return fields.my_VsToV(self.dic, z, j)
    
    def p_Ximv(self, vsr, j):
        return pair.my_pSmV(self.dic, vsr.dic, j)
