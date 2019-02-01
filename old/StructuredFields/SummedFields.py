import numpy as np


class Summed_field(object):
    def __init__(self, fields_list):
        self.N_fields = len(fields_list)
        dic = dict()
        dic['0'] = []
        dic['p'] = []
        dic['m'] = []
        ##dic['sig_0'] = []
        # dic['sig_p'] = []
        # dic['sig_m'] = []
        self.dic = dic
        # self.indi_0 = []
        # self.indi_p = []
        # self.indi_m = []
        self.fields_list = fields_list
        self.fill_dic_init()
    
    def copy(self):
        field_list = []
        for i in range(self.N_fields):
            field_list.append(self.fields_list[i].copy())
        return Summed_field(field_list)
    
    def copy_full(self):
        field_list = []
        for i in range(self.N_fields):
            field_list.append(self.fields_list[i].copy_full())
        return Summed_field(field_list)
    
    def fill_dic_init(self):
        for i in range(self.N_fields):
            fi_dic = self.fields_list[i].dic
            if '0' in fi_dic:
                for (x, P) in fi_dic['0']:
                    self.dic['0'].append((x, P))
                    # self.indi_0.append(i)
                    # self.dic['sig_0'].append(fi_dic['sig'])
            
            if 'p' in fi_dic:
                for (x, P) in fi_dic['p']:
                    self.dic['p'].append((x, P))
                    # self.indi_p.append(i)
                    # self.dic['sig_p'].append(fi_dic['sig'])
            
            if 'm' in fi_dic:
                for (x, P) in fi_dic['m']:
                    self.dic['m'].append((x, P))
                    # self.indi_m.append(i)
                    # self.dic['sig_m'].append(fi_dic['sig'])
    
    def fill_dic(self, param):
        for parami, fi in zip(param, self.fields_list):
            fi.fill_dic(parami)
        
        self.fill_dic_init()
    
    def Apply(self, z, j):  # tested
        Nz = z.shape[0]
        lsize = ((Nz, 2), (Nz, 2, 2), (Nz, 2, 2, 2))
        djv = np.zeros(lsize[j])
        
        for i in range(self.N_fields):
            djv += self.fields_list[i].Apply(z, j)
        
        return djv
    
    def p_Ximv(self, vsr, j):
        
        raise NameError('No inner product for summed fields')


def sum_structured_fields(fields_list):
    return Summed_field(fields_list)
