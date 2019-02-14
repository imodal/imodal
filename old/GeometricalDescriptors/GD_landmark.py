import numpy as np

from old.GeometricalDescriptors import GeometricalDescriptors, add_cot
from old.utilities import pairing_structures as npair


class GD_landmark(GeometricalDescriptors):
    def __init__(self, N_pts, dim):  # tested
        """
        The GD and Mom are arrays of size N_pts x dim.
        """
        
        self.Cot = {'0': []}
        self.N_pts = N_pts
        self.dim = dim
    
    def copy(self):  # tested
        return GD_landmark(self.N_pts, self.dim)
    
    def copy_full(self):  # tested
        GD = GD_landmark(self.N_pts, self.dim)
        if len(self.Cot['0']) > 0:
            x, p = self.Cot['0'][0]
            GD.Cot['0'] = [(x.copy(), p.copy())]
        return GD
    
    def fill_zero(self):
        self.Cot['0'] = [(np.zeros([self.N_pts, self.dim]), np.zeros([self.N_pts, self.dim]))]
    
    def updatefromCot(self):
        pass
    
    def fill_GDpts(self, pts):  # tested
        self.pts = pts.copy()
        self.Cot['0'] = [(pts.copy(), np.zeros([self.N_pts, self.dim]))]
    
    def fill_cot_from_param(self, param):  # tested
        self.Cot['0'] = [(param[0].copy(), param[1].copy())]
    
    def get_points(self):
        return self.Cot['0'][0][0]
    
    def get_mom(self):
        return self.Cot['0'][0][1]
    
    def mult_Cot_scal(self, s):  # tested
        if len(self.Cot['0']) > 0:
            x, p = self.Cot['0'][0]
            self.Cot['0'] = [(s * x, s * p)]
    
    def add_cot(self, Cot):  # tested
        """
        adds Cots to self.cot
        Cot needs to be of same type as self.Cot
        """
        self.Cot = add_cot(self.Cot, Cot)
    
    def Cot_to_Vs(self, sig):  # tested
        """
        Supposes that Cot has been filled
        """
        return npair.CotToVs_class(self, sig)
    
    def Ximv(self, v):  # tested
        
        """
        xi_m ()
        
        """
        pts = self.get_points()
        appli = v.Apply(pts, 0)
        out = self.copy()
        out.Cot['0'] = [(appli, np.zeros([self.N_pts, self.dim]))]
        return out
    
    def dCotDotV(self, vs):  #
        """
        Supposes that Cot has been filled
        """
        x = self.get_points()
        p = self.get_mom()
        der = vs.Apply(x, 1)
        
        dx = np.asarray([np.dot(p[i], der[i]) for i in range(x.shape[0])])
        
        GD = self.copy()
        GD.Cot['0'] = [(dx, np.zeros([self.N_pts, self.dim]))]
        return GD
    
    def inner_prod_v(self, v):  # tested
        dGD = self.Ximv(v)
        dpts = dGD.get_points()
        mom = self.get_mom()
        return np.dot(mom.flatten(), dpts.flatten())