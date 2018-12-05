#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 17:03:15 2018

@author: barbaragris
"""

import numpy as np
from scipy.linalg import solve

import GeometricalDescriptors.GeometricalDescriptors as GeoDescr
import StructerdFIelds.StructuredFields as stru
from implicitmodules.src import kernels as ker
from implicitmodules.src import functions_eta as fun_eta
from implicitmodules.src import field_structures as fields, pairing_structures as pair
import utilities.pairing_structures as npair


class DeformationModule(object):
    """
    Abstract class for deformation module
    
    
    Attributes: type of GD
                dimension of controls ?
                
                
    Methods: 
        -- GeodesicControls (v_s, m) where v_s is in V^\ast, m a GD,
              returns the corresponding geodesic control
        -- FieldGenerator (m, h) where m is a GD, h a control. It returns
              a structured field in V
        -- Cost (m, h) where m is a GD, h a control. It returns the cost
              (scalar)
        -- DerCost (m, h) where m is a GD, h a control. It returns the
              derivative of the cost wrt m
        -- fill_GD(GD) fills self.GD with value (uses GD structure)
        -- update that stores useful functions (after filling of GD)
    
    For each method, there is a "_curr" version which supposes that GD are 
        known (and necessary functions computes)
    
    """
    
    
    def fill_GD(self, GD):
        self.GD.fill_GD(GD)
    
    def sum_GD(self, GD0, GD1):
        self.GD.sum_GD(GD0, GD1)
        
    
    


class ElasticOrderO(DeformationModule):
    """
     Elastic module of order 0
    """
    
    
    def __init__(self, sigma, N_pts, dim, coeff):
        """
        sigma is the scale of the rkhs of generated vector fields
        N_pts is the number of landmarks
        dim is the dimension of the ambient space
        """
        self.sig = sigma
        self.N_pts = N_pts
        self.dim =dim
        self.coeff = coeff
        self.GD = GeoDescr.GD_landmark(N_pts, dim)
        self.SKS = np.zeros([self.N_pts*self.dim,self.N_pts*self.dim])
        self.Mom = np.zeros([self.N_pts, self.dim])
        self.Cont = np.zeros([self.N_pts, self.dim])
        self.cost = 0.
        
    def copy(self):
        return ElasticOrderO(self.sig, self.N_pts, self.dim, self.coeff)
    
    def copy_full(self):
        Mod = ElasticOrderO(self.sig, self.N_pts, self.dim, self.coeff)
        Mod.GD = self.GD.copy_full()
        Mod.SKS = self.SKS.copy()
        Mod.Mom = self.Mom.copy()
        Mod.Cont = self.Cont.copy()
        Mod.cost = self.cost
        return Mod
    
    def fill_GD(self, GD):
        self.GD = GD.copy_full()
        self.SKS = np.zeros([self.N_pts*self.dim,self.N_pts*self.dim])
    
    def add_cot(self, GD):
        self.GD.add_cot(GD.Cot)
        
    def Compute_SKS_curr(self):
        """
        Supposes that values of GD have been filled
        """
        try:
            x = self.GD.get_points()
            self.SKS = ker.my_K(x, x, self.sig, 0)
        except:
            raise NameError('Need to fill landmark points before computing SKS')
    
    def Compute_SKS(self, x):
        return ker.my_K(x, x, self.sig, 0)
    
    def update(self):
        """
        Computes SKS so that it is done only once.
        Supposes that values of GD have been filled
        """
        
        self.Compute_SKS_curr()
    
    
    def GeodesicControls_curr(self, GDCot):
        """
        Supposes that SKS has been computed and values of GD filled
        Supposes that GDCot has Cot filled
        """
        vs = GDCot.Cot_to_Vs(self.sig)
        vm = vs.Apply(self.GD.get_points(), 0)
        self.Cont = solve(self.coeff * self.SKS,
                     vm.flatten(),sym_pos = True).reshape(self.N_pts, self.dim)
        self.Mom = self.Cont.copy()
    
    def GeodesicControls(self, GD, GDCot):
        """
        Supposes that SKS has been computed and values of GD filled
        """
        vs = GDCot.Cot_to_Vs(self.sig)
        vm = vs.Apply(GD.get_points(), 0)
        return solve(self.coeff * self.SKS,
                     vm.flatten(),sym_pos = True).reshape(self.N_pts, self.dim)
        
        
    def field_generator_curr(self):
        return self.field_generator(self.GD, self.Cont)
    
    
    def field_generator(self, GD, Cont):
        param = [GD.get_points(), Cont]
        v = stru.StructuredField_0(self.sig, self.dim)
        v.fill_fieldparam(param)
        return v
    
    def Cost_curr(self):
        SKS = self.SKS
        p = self.Cont.flatten()
        self.cost = self.coeff * np.dot(p, np.dot(SKS, p))/2
        
    def Cost(self, GD, Cont):
        x = GD.get_points()
        SKS = self.Compute_SKS(x)
        p = Cont.flatten()
        return self.coeff * np.dot(p, np.dot(SKS, p))/2
       
    def DerCost_curr(self):#tested
        vs  = self.field_generator_curr()
        der = vs.p_Ximv(vs, 1)
        out = self.GD.copy()
        out.Cot['0'] = [( self.coeff * der['0'][0][1], np.zeros([self.N_pts, self.dim]) )]
        return out

      
    def DerCost(self, GD, Mom):#tested
        vs  = self.field_generator(GD, Mom)
        der = vs.p_Ximv(vs, 1)
        out = self.GD.copy()
        out.Cot['0'] = [( self.coeff * der['0'][0][1], np.zeros([self.N_pts, self.dim]) )]
        return out


    
    def cot_to_innerprod_curr(self, GDCot, j):#tested
        """
         Transforms the GD (with Cot filled) GDCot into vsr and computes
         the inner product (or derivative wrt self.GD) with the
         field generated by self.
         The derivative is a returned as a GD with Cot filled (mom=0)
        """
        
        vsr = GDCot.Cot_to_Vs(self.sig)
        v = self.field_generator_curr()
        innerprod = v.p_Ximv(vsr, j)
        
        if j==0:
            out = innerprod
        if j==1:
            out = self.GD.copy()
            out.Cot['0'] = [ (innerprod['0'][0][1], np.zeros([self.N_pts,self.dim]) )]
            
        return out
 


class SilentLandmark(DeformationModule):#tested
    """
    Silent deformation module with GD that are points
    """

    
    def __init__(self, N_pts, dim):
       self.N_pts = N_pts
       self.dim = dim
       self.GD = GeoDescr.GD_landmark(N_pts, dim)
       #self.Mom = np.empty([self.N_pts, self.dim])
       self.cost = 0.
       self.Cont = np.empty([0])
   
    def copy(self):
        return SilentLandmark(self.N_pts, self.dim)
    
    
    def add_cot(self, GD):
        self.GD.add_cot(GD.Cot)
        
    def copy_full(self):
        Mod = SilentLandmark(self.N_pts, self.dim)
        Mod.GD = self.GD.copy_full()
        #Mod.Mom = self.Mom.copy()
        return Mod
    
    def update(self):
        pass
    
    def GeodesicControls_curr(self, GDCot):
        pass
    
    def GeodesicControls(self, GD, GDCot):
        pass
    
    def field_generator_curr(self):
        return stru.ZeroField(self.dim)

    def field_generator(self, GD, Cont):
        return stru.ZeroField(self.dim)
    
    
    def Cost_curr(self):
        pass
        
    def Cost(self, GD, Cont):   
        return 0.
       

    def DerCost_curr(self):#tested
        out = self.GD.copy()
        out.Cot['0'] = [(np.zeros([self.N_pts, self.dim]), np.zeros([self.N_pts, self.dim]) )]
        return out

      
    def DerCost(self, GD, Mom):#tested
        out = GD.copy()
        out.Cot['0'] = [(np.zeros([self.N_pts, self.dim]), np.zeros([self.N_pts, self.dim]) )]
        return out


    
    def cot_to_innerprod_curr(self, GDCot, j):#tested
        """
         Transforms the GD (with Cot filled) GDCot into vsr and computes
         the inner product (or derivative wrt self.GD) with the
         field generated by self.
         The derivative is a returned as a GD with Cot filled (mom=0)
        """
        
        
        if j==0:
            out = 0.
        if j==1:
            out = self.GD.copy()
            out.Cot['0'] = [ (np.zeros([self.N_pts,self.dim]), np.zeros([self.N_pts,self.dim]) )]
            
        return out
 





class ElasticOrder1(DeformationModule):
    """
     Elastic module of order 1
    """

    
    def __init__(self, sigma, N_pts, dim, coeff, C, nu):
        """
        sigma is the scale of the rkhs of generated vector fields
        N_pts is the number of landmarks
        dim is the dimension of the ambient space
        """
        self.sig = sigma
        self.N_pts = N_pts
        self.dim =dim
        self.C = C.copy()
        self.nu = nu
        self.dimR = 3
        self.coeff = coeff
        self.GD = GeoDescr.GD_xR(N_pts, dim, C)
        self.SKS = np.zeros([self.N_pts*self.dimR,self.N_pts*self.dimR])
        self.Mom = np.zeros([self.N_pts, self.dim, self.dim])
        self.lam = np.zeros([self.N_pts*self.dimR])
        self.Cont = np.zeros([1])
        self.cost = 0.  
        
    def copy(self):
        return ElasticOrder1(self.sig, self.N_pts, self.dim, self.coeff, self.C, self.nu)
    
    def copy_full(self):
        Mod = ElasticOrder1(self.sig, self.N_pts, self.dim, self.coeff, self.C, self.nu)
        Mod.GD = self.GD.copy_full()
        Mod.SKS = self.SKS.copy()
        Mod.Mom = self.Mom.copy()
        Mod.Cont = self.Cont.copy()
        Mod.cost = self.cost
        return Mod    

    def fill_GD(self, GD):
        self.GD = GD.copy_full()
        self.SKS = np.zeros([self.N_pts*self.dimR,self.N_pts*self.dimR])
        

    def add_cot(self, GD):
        self.GD.add_cot(GD.Cot)
        
       
    def Compute_SKS_curr(self):
        """
        Supposes that values of GD have been filled
        """
        try:
            x = self.GD.get_points()
            self.SKS = ker.my_K(x, x, self.sig, 1) 
            self.SKS += self.nu * np.eye(self.N_pts*self.dimR)
            
        except:
            raise NameError('Need to fill landmark points before computing SKS')
    
    
    
    def update(self):
        """
        Computes SKS so that it is done only once.
        Supposes that values of GD have been filled
        """
        
        self.Compute_SKS_curr()
    
    def Amh_curr(self, h):
        R = self.GD.get_R()
        eta = fun_eta.my_eta()
        out = np.asarray([np.tensordot(np.dot(R[i],
        np.dot(np.diag(np.dot(self.GD.C[i],h)),
        R[i].transpose())),eta,axes = 2) for i in range(self.N_pts)])
        return out

    
    def AmKiAm_curr(self):
        dimh = self.GD.C.shape[2]
    
        lam = np.zeros((dimh, 3 * self.N_pts))
        Am = np.zeros((3 * self.N_pts, dimh))
        
        for i in range(dimh):
            h = np.zeros((dimh))
            h[i] = 1.
            Am[:, i] = self.Amh_curr(h).flatten()
            lam[i, :] = solve(self.SKS, Am[:, i], sym_pos=True)
        return (Am, np.dot(lam, Am))
    
    def GeodesicControls_curr(self, GDCot):
        """
        Supposes that SKS has been computed and values of GD filled
        Supposes that GDCot has Cot filled
        """
        x = self.GD.get_points()
        
        vs = GDCot.Cot_to_Vs(self.sig)
        dvsx = vs.Apply(x, 1)
        dvsx_sym = (dvsx + np.swapaxes(dvsx,1,2))/2
        S  = np.tensordot(dvsx_sym, fun_eta.my_eta())
        
        tlam = solve(self.coeff * self.SKS, S.flatten(), sym_pos = True)
        (Am, AmKiAm) = self.AmKiAm_curr()
        
        Am_s_tlam = np.dot(tlam,Am)
        
        self.Cont = solve(AmKiAm, Am_s_tlam, sym_pos = True)
        self.compute_mom_from_cont_curr()
    
    def compute_mom_from_cont_curr(self):
        self.Amh = self.Amh_curr(self.Cont).flatten()
        self.lam = solve(self.SKS, self.Amh, sym_pos = True)
        self.Mom = np.tensordot(self.lam.reshape(self.N_pts,3),
            fun_eta.my_eta().transpose(), axes =1)
           
    
    def field_generator_curr(self):
        
        v = stru.StructuredField_p(self.sig, self.dim)
        param = (self.GD.get_points(), self.Mom)
        v.fill_fieldparam(param)
        return v
    
    def field_generator(GD, Cont):
        v = stru.StructuredField_p(self.sig, self.dim)
        param = (self.GD.get_points(), self.Mom)
        v.fill_fieldparam(param)
        return v
        
    def Cost_curr(self):
        self.cost = self.coeff * np.dot(self.Amh, self.lam)/2
        
    
    def DerCost_curr(self):
        out = self.GD.copy()
        x = self.GD.get_points()
        R = self.GD.get_R()
        v = self.field_generator_curr()
        
        der = pair.my_pSmV(v.dic,v.dic,1)
        dx = self.coeff * der['p'][0][1]
        
        
        tP = -self.coeff * self.Mom
        dR = 2*np.asarray([np.dot(np.dot(tP[i],R[i]),np.diag(np.dot(self.GD.C[i],self.Cont)))
                             for i in range(x.shape[0])])
        
        
        
        
        out.Cot['x,R'] = [ ((dx, dR), (np.zeros(dx.shape), np.zeros(dR.shape) ))]
            
        return out
        



    def cot_to_innerprod_curr(self, GDCot, j):
        """
         Transforms the GD (with Cot filled) GDCot into vsr and computes
         the inner product (or derivative wrt self.GD) with the
         field generated by self.
         The derivative is a returned as a GD with Cot filled (mom=0)
        """
        
        vsr = GDCot.Cot_to_Vs(self.sig)
        v = self.field_generator_curr()
        
        
        if j==0:
            out = v.p_Ximv(vsr, j)
        if j==1:
            out = self.GD.copy()
            x = self.GD.get_points()
            R = self.GD.get_R()
            
            Vsr1 = fields.my_CotToVs(GDCot.Cot, self.sig)
            der = pair.my_pSmV(v.dic,Vsr1,1)
            dx = der['p'][0][1]            
            
            
            dvx = v.Apply(x, 1)
            dvx_sym = (dvx + np.swapaxes(dvx,1,2))/2
            S = np.tensordot(dvx_sym,fun_eta.my_eta())
            
            tlam = solve(self.SKS, S.flatten(), sym_pos = True)
            tP = np.tensordot(tlam.reshape(S.shape),fun_eta.my_eta().transpose(), axes = 1) 
            
            tVs = {'p':[(x,tP)], 'sig':self.sig}
            
            der = pair.my_pSmV(tVs,v.dic,1)
            dx += - der['p'][0][1]
            der = pair.my_pSmV(v.dic,tVs,1)
            dx += - der['p'][0][1]           
            
            dR = 2*np.asarray([np.dot(np.dot(tP[i],R[i]),np.diag(np.dot(self.GD.C[i],self.Cont)))
                                 for i in range(x.shape[0])])
            
            
            
            
            out.Cot['x,R'] = [ ((dx, dR), (np.zeros(dx.shape), np.zeros(dR.shape) ))]
            
        return out
 













        