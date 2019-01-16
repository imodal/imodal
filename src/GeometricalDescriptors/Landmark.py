# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 13:24:26 2019

@author: gris
"""

import numpy as np

import src.GeometricalDescriptors.Abstract as ab
import src.StructuredFields.StructuredField_0 as stru_fie
        
class GD_landmark(ab.GeometricalDescriptors):
    def __init__(self, N_pts, dim): #
        """
        The GD and Mom are arrays of size N_pts x dim.
        """
        
        self.N_pts = N_pts
        self.dim = dim
        self.GD = np.zeros([self.N_pts, self.dim])
        self.tan = np.zeros([self.N_pts, self.dim])
        self.cotan = np.zeros([self.N_pts, self.dim])
        
    def copy(self): #
        return GD_landmark(self.N_pts, self.dim)
        

    def copy_full(self): #
        GD = GD_landmark(self.N_pts, self.dim)
        GD.GD = self.GD.copy()
        GD.tan = self.tan.copy()
        GD.cotan = self.cotan.copy()
        return GD

    def fill_zero(self):        
        self.GD = np.zeros([self.N_pts, self.dim])
        self.tan = np.zeros([self.N_pts, self.dim])
        self.cotan = np.zeros([self.N_pts, self.dim])
        
        
    def fill_GDpts(self, pts): #  
        self.gd = pts.copy()


    def get_points(self):
        return self.GD.copy()

    
    def get_mom(self):
        return self.cotan.copy()
        
    
    def fill_cot_from_param(self, param):
        self.GD = param[0].copy()
        self.cotan = param[1].copy()

    def Cot_to_Vs(self, sig):
        v = stru_fie.StructuredField_0(sig, self.N_pts, self.dim)
        v.fill_fieldparam((self.GD, self.cotan)) 
        return v

    def Ximv(self, v):
        """
        xi_m ()
        
        """
        pts = self.get_points()
        appli = v.Apply(pts, 0) 
        out = self.copy_full()
        out.tan = appli.copy()
        return out
        
    def dCotDotV(self, vs): #
        """
        Supposes that Cot has been filled
        derivates (p Ximv(m,v)) wrt m
        """
        x = self.get_points()
        p = self.get_mom()
        der = vs.Apply(x, 1)
        
        dx = np.asarray([np.dot(p[i], der[i]) for i in range(x.shape[0])])
        
        GD = self.copy_full()
        GD.cotan = dx.copy() 
        return GD    
                

    def inner_prod_v(self, v): #tested
        dGD = self.Ximv(v)
        dpts = dGD.tan
        mom = self.get_mom()
        return np.dot(mom.flatten(), dpts.flatten())

    
    def add_GD(self, GDCot):
        self.GD = self.GD + GDCot.GD
        
            
    def add_tan(self, GDCot):
        self.tan = self.tan + GDCot.tan
        
            
    def add_cotan(self, GDCot):
        self.cotan = self.cotan + GDCot.cotan
        
    
    def mult_GD_scal(self, s):
        self.GD = s * self.GD

    def mult_tan_scal(self, s):
        self.tan = s * self.tan

    def mult_cotan_scal(self, s):
        self.cotan = s * self.cotan

    def add_speedGD(self, GDCot):
        self.GD = self.GD + GDCot.tan
        


































