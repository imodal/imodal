class StructuredField(object):
    """
    Abstract class for structured field
    
    attributes:
     dic : dictionary with types '0', 'p' or 'm'.
     
     
    Methods : 
        
          -- p_Ximv (vsl, vsr, j) with vsl and vsr 2 structured field supposed
           to be in the same rkhs  and vsl is supposed to hae only '0' and 'p',
           j an integer (0 or 1). 
               If j=0 it returns (p | Xi_m (vsr)) where vsl is 
               parametrized by (m,p), i.e. the inner product between the two
               fields. If j=1 it returns the derivative wrt m
               
          -- Apply : Applies the field to points z (or computes the derivative). 
        Needs pre-assigned parametrization of the field in dic
                   
    """
    
    def __init__(self, dim):
        self.dim = dim
        self.dic = dict()
        pass
    
    def fill_dic_from_dic(self, dic):
        self.dic = dict()
        if '0' in dic:
            self.dic['0'] = []
            for (x, p) in dic['0']:
                self.dic['0'].append((x.copy(), p.copy()))
        
        if 'p' in dic:
            for (x, P) in dic['p']:
                self.dic['p'].append((x.copy(), P.copy()))
        
        if 'm' in dic:
            for (x, P) in dic['m']:
                self.dic['m'].append((x.copy(), P.copy()))
        
        if 'sig' in dic:
            self.dic['sig'] = dic['sig']
