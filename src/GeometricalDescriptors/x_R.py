# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 16:02:37 2019

@author: gris
"""


import numpy as np

import src.GeometricalDescriptors.Abstract as ab
from utilities import pairing_structures as npair
import src.StructuredFields.StructuredField_0 as stru_fie0
import src.StructuredFields.StructuredField_1 as stru_fie1


class GD_xR(ab.GeometricalDescriptors):
    def __init__(self, N_pts, dim, C):  # tested
        """
        The GD and Mom are arrays of size N_pts x dim.
        """
        
        self.Cot = {'x,R': []}
        self.N_pts = N_pts
        self.dim = dim
        self.C = C.copy()
        self.GDshape = [self.N_pts, self.dim]
        self.Rshape = [self.N_pts, self.dim, self.dim]
        self.GD = (np.zeros([self.N_pts, self.dim]), np.zeros(self.Rshape))
        self.tan = (np.zeros([self.N_pts, self.dim]), np.zeros(self.Rshape))
        self.cotan = (np.zeros([self.N_pts, self.dim]), np.zeros(self.Rshape))
    
    def copy(self):  # 
        return GD_xR(self.N_pts, self.dim, self.C)
    
    def copy_full(self):  # 
        GD = GD_xR(self.N_pts, self.dim, self.C)
        GD.GD = self.GD.copy()
        GD.tan = self.tan.copy()
        GD.cotan = self.cotan.copy()
        return GD
    
    def fill_zero(self):
        self.GD = (np.zeros(self.GDshape), np.zeros(self.Rshape))
        self.tan = (np.zeros(self.GDshape), np.zeros(self.Rshape))
        self.cotan = (np.zeros(self.GDshape), np.zeros(self.Rshape))

    def fill_zero_GD(self):        
        self.GD = (np.zeros(self.GDshape), np.zeros(self.Rshape))

    def fill_zero_tan(self):        
        self.tan = (np.zeros(self.GDshape), np.zeros(self.Rshape))

    def fill_zero_cotan(self):        
        self.cotan = (np.zeros(self.GDshape), np.zeros(self.Rshape))
        

    def updatefromCot(self):
        pass
    
    def get_points(self):  # 
        return self.GD[0]
    
    def get_R(self):  # 
        return self.GD[1]
    
    def get_mom(self):  # 
        return self.cotan.copy()
    
    def fill_cot_from_param(self, param):  # 
        self.GD = param[0].copy()
        self.cotan = param[1].copy()
    
    
    def Cot_to_Vs(self, sig):  # 
        x = self.GD[0].copy()
        R = self.GD[1].copy() 
        px = self.cotan[0].copy()
        pR = self.cotan[1].copy()
    ################################### À finir
        v0 = stru_fie0.StructuredField_0(sig, self.N_pts, self.dim)
        v0.fill_fieldparam((x, px))
        
        
        return v
    
    
    
    
    
    
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
        