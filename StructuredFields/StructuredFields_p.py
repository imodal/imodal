from src import field_structures as fields, pairing_structures as pair


class StructuredField_p(object):
    
    def __init__(self, sigma, dim):  #
        """
         sigma is the sclae of the rkhs to which the field belongs
         dic is the parametrization of the field
        """
        self.dim = dim
        self.sig = sigma
        self.type = 'p'
        self.dicp = []
        self.dic = {'p': [], 'sig': self.sig}
    
    def copy(self):
        v = StructuredField_p(self.sig, self.dim)
        return v
    
    def copy_full(self):
        v = StructuredField_p(self.sig, self.dim)
        (x, P) = self.dic['p'][0]
        v.dic['p'] = [(x.copy(), P.copy())]
        return v
    
    def fill_fieldparam(self, param):
        """
        param should be a list of two elements: array of points and
        array of vectors
        """
        self.dic['p'] = [param]
    
    def Apply(self, z, j):
        """
        Applies the field to points z (or computes the derivative).
        Needs pre-assigned parametrization of the field in dic
        
        """
        return fields.my_VsToV(self.dic, z, j)
    
    def p_Ximv(self, vsr, j):
        return pair.my_pSmV(self.dic, vsr.dic, j)
