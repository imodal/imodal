import numpy as np

from old.GeometricalDescriptors import GeometricalDescriptors, add_cot
from old.utilities import pairing_structures as npair


class GD_xR(GeometricalDescriptors):
    def __init__(self, N_pts, dim, C):  # tested
        """
        The GD and Mom are arrays of size N_pts x dim.
        """
        
        self.Cot = {'x,R': []}
        self.N_pts = N_pts
        self.dim = dim
        self.C = C.copy()
        self.pRshape = [self.N_pts, self.dim, self.dim]
    
    def copy(self):  # tested
        return GD_xR(self.N_pts, self.dim, self.C)
    
    def copy_full(self):  # tested
        GD = GD_xR(self.N_pts, self.dim, self.C)
        if len(self.Cot['x,R']) > 0:
            ((x, R), (px, pR)) = self.Cot['x,R'][0]
            GD.Cot['x,R'] = [((x.copy(), R.copy()), (px.copy(), pR.copy()))]
        return GD
    
    def fill_zero(self):
        self.Cot['x,R'] = [((np.zeros([self.N_pts, self.dim]), np.zeros([self.N_pts, self.dim, self.dim])),
                            (np.zeros([self.N_pts, self.dim]), np.zeros(self.pRshape)))]
    
    def updatefromCot(self):
        pass
    
    def get_points(self):  # tested
        return self.Cot['x,R'][0][0][0].copy()
    
    def get_R(self):  # tested
        return self.Cot['x,R'][0][0][1].copy()
    
    def get_mom(self):  # tested
        return self.Cot['x,R'][0][1]
    
    def fill_cot_from_param(self, param):  # tested
        self.Cot['x,R'] = [((param[0][0].copy(), param[0][1].copy()), (param[1][0].copy(), param[1][1].copy()))]
    
    def mult_Cot_scal(self, s):  # tested
        if len(self.Cot['x,R']) > 0:
            ((x, R), (px, pR)) = self.Cot['x,R'][0]
            self.Cot['x,R'] = [((s * x, s * R), (s * px, s * pR))]
    
    def add_cot(self, Cot):  # tested
        """
        adds Cots to self.cot
        Cot needs to be of same type as self.Cot
        """
        self.Cot = add_cot(self.Cot, Cot)
    
    def Cot_to_Vs(self, sig):  # tested
        return npair.CotToVs_class(self, sig)
    
    def Ximv(self, v):  #
        pts = self.get_points()
        R = self.get_R()
        dx = v.Apply(pts, 0)
        dvx = v.Apply(pts, 1)
        S = (dvx - np.swapaxes(dvx, 1, 2)) / 2
        dR = np.asarray([np.dot(S[i], R[i]) for i in range(pts.shape[0])])
        out = self.copy()
        out.Cot['x,R'] = [((dx, dR), (np.zeros([self.N_pts, self.dim]), np.zeros(self.pRshape)))]
        
        return out
    
    def dCotDotV(self, vs):  # tested ,
        """
        Supposes that Cot has been filled
        """
        x = self.get_points()
        R = self.get_R()
        px, pR = self.get_mom()
        
        dvx = vs.Apply(x, 1)
        ddvx = vs.Apply(x, 2)
        
        skew_dvx = (dvx - np.swapaxes(dvx, 1, 2)) / 2
        skew_ddvx = (ddvx - np.swapaxes(ddvx, 1, 2)) / 2
        
        dx = np.asarray([np.dot(px[i], dvx[i]) + np.tensordot(pR[i],
                                                              np.swapaxes(
                                                                  np.tensordot(R[i], skew_ddvx[i], axes=([0], [1])),
                                                                  0, 1))
                         for i in range(x.shape[0])])
        
        dR = np.asarray([np.dot(-skew_dvx[i], pR[i])
                         for i in range(x.shape[0])])
        
        GD = self.copy()
        GD.Cot['x,R'] = [((dx, dR), (np.zeros(dx.shape), np.zeros(self.pRshape)))]
        return GD
    
    def inner_prod_v(self, v):  # tested
        dGD = self.Ximv(v)
        dpts = dGD.get_points()
        dR = dGD.get_R()
        px, pR = self.get_mom()
        out = np.dot(px.flatten(), dpts.flatten())
        out += np.sum([np.tensordot(pR[i], dR[i]) for i in range(dR.shape[0])])
        return out
