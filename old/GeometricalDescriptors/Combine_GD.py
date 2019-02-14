from old.GeometricalDescriptors import GeometricalDescriptors, add_cot
from old.utilities import pairing_structures as npair


class Combine_GD(GeometricalDescriptors):
    def __init__(self, GD_list):  # tested 0
        self.GD_list = GD_list
        self.N_GDs = len(self.GD_list)
        self.dim = GD_list[0].dim
        Cot = dict()
        Cot['0'] = []
        Cot['x,R'] = []
        self.Cot = Cot
        self.indi_0 = []
        self.indi_xR = []
        self.fill_cot_init()
    
    def copy(self):  # tested
        GD_list = []
        for i in range(self.N_GDs):
            GD_list.append(self.GD_list[i].copy())
        GD = Combine_GD(GD_list)
        GD.indi_0 = self.indi_0.copy()
        GD.indi_xR = self.indi_xR.copy()
        return GD
    
    def copy_full(self):  # tested
        GD_list = []
        for i in range(self.N_GDs):
            GD_list.append(self.GD_list[i].copy_full())
        GD_comb = Combine_GD(GD_list)
        GD_comb.indi_0 = self.indi_0.copy()
        GD_comb.indi_xR = self.indi_xR.copy()
        GD_comb.fill_cot_from_GD()
        return GD_comb
    
    def fill_zero(self):
        for i in range(self.N_GDs):
            self.GD_list[i].fill_zero()
        self.fill_cot_from_GD()
    
    def fill_coti_from_cot(self):
        for i in range(self.N_GDs):
            self.GD_list[i].Cot = {'0': [], 'x,R': []}
            for j in self.indi_0[i]:
                self.GD_list[i].Cot['0'].append(self.Cot['0'][j])
            for j in self.indi_xR[i]:
                self.GD_list[i].Cot['x,R'].append(self.Cot['x,R'][j])
    
    def updatefromCot(self):
        """
        If Cot has been changed, it needs to be put in each GDi
        """
        self.fill_coti_from_cot()
        for i in range(self.N_GDs):
            self.GD_list[i].updatefromCot()
    
    def fill_cot_init(self):  # tested
        self.Cot['0'] = []
        self.Cot['x,R'] = []
        ind0 = 0
        indxR = 0
        for i in range(self.N_GDs):
            Coti = self.GD_list[i].Cot
            self.indi_0.append([])
            self.indi_xR.append([])
            if '0' in Coti:
                for (x, p) in Coti['0']:
                    self.Cot['0'].append((x, p))
                    self.indi_0[i].append(ind0)
                    ind0 += 1
            
            if 'x,R' in Coti:
                for (X, P) in Coti['x,R']:
                    self.Cot['x,R'].append((X, P))
                    self.indi_xR[i].append(indxR)
                    indxR += 1
    
    def fill_cot_from_GD(self):  # tested0
        self.Cot['0'] = []
        self.Cot['x,R'] = []
        for i in range(self.N_GDs):
            Coti = self.GD_list[i].Cot
            if '0' in Coti:
                for (x, p) in Coti['0']:
                    self.Cot['0'].append((x, p))
            
            if 'x,R' in Coti:
                for (X, P) in Coti['x,R']:
                    self.Cot['x,R'].append((X, P))
    
    def fill_cot_from_param(self, param):  # tested0
        for parami, GDi in zip(param, self.GD_list):
            GDi.fill_cot_from_param(parami)
        self.fill_cot_init()
    
    def mult_Cot_scal(self, s):  # tested0
        for i in range(self.N_GDs):
            self.GD_list[i].mult_Cot_scal(s)
        self.fill_cot_from_GD()
    
    def add_cot(self, Cot):  # tested0
        self.Cot = add_cot(self.Cot, Cot)
        self.updatefromCot()
    
    def add_GDCot(self, GD):  # tested
        for i in range(self.N_GDs):
            self.GD_list[i].Cot = add_cot(self.GD_list[i].Cot, GD.GD_list[i].Cot)
        self.fill_cot_from_GD()
    
    def Cot_to_Vs(self, sig):  # tested
        """
        Supposes that Cot has been filled
        """
        
        return npair.CotToVs_class(self, sig)
    
    def Ximv(self, v):  # tested0
        dGD_list = []
        for i in range(self.N_GDs):
            dGD_list.append(self.GD_list[i].Ximv(v))
        out = Combine_GD(dGD_list)
        out.fill_cot_from_GD()
        out.indi_0 = self.indi_0.copy()
        out.indi_xR = self.indi_xR.copy()
        return out
    
    def dCotDotV(self, vs):  # tested0
        """
        Supposes that Cot has been filled
        """
        dGD = []
        for i in range(self.N_GDs):
            dGD.append(self.GD_list[i].dCotDotV(vs))
        out = Combine_GD(dGD)
        out.indi_0 = self.indi_0.copy()
        out.indi_xR = self.indi_xR.copy()
        out.fill_cot_from_GD()
        
        return out
    
    def inner_prod_v(self, v):  # tested0
        return sum([GD_i.inner_prod_v(v) for GD_i in self.GD_list])
