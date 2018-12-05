#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:39:57 2018

@author: barbaragris
"""

import numpy as np

from implicitmodules.src import field_structures as fields
from implicitmodules.src import pairing_structures as pair
import utilities.pairing_structures as npair


def add_cot(Cot0, Cot1): #tested0
    """
    adds Cots to self.cot
    Cot needs to be of same type as self.Cot
    """
    sumCot = dict()
    sumCot['0'] = []
    sumCot['x,R'] = []
    if '0' in Cot0:
        if '0' in Cot1:
            N0 = len(Cot0['0'])
            if N0 == len(Cot1['0']):
                for i in range(N0):
                    (x0,p0 ) = Cot0['0'][i]
                    (x1,p1 ) = Cot1['0'][i]
                    sumCot['0'].append((x0+x1, p0+p1))
            else:
                raise NameError('Not possible to add Cotof different types')
            
        else:
            raise NameError('Not possible to add Cotof different types')
    
    if 'x,R' in Cot0:
        if 'x,R' in Cot1:
            N0 = len(Cot0['x,R'])
            if N0 == len(Cot1['x,R']):
                for i in range(N0):
                    ((x0,R0),(p0,PR0)) = Cot0['x,R'][i]
                    ((x1,R1),(p1,PR1))= Cot1['x,R'][i]
                    sumCot['x,R'].append(((x0+x1, R0+R1),(p0 + p1, PR0 + PR1) ))
            else:
                raise NameError('Not possible to add Cotof different types')
            
        else:
            raise NameError('Not possible to add Cotof different types')
    
    return sumCot

class GeometricalDescriptors(object):
    
    """
    Abstract class for geometrical descriptors, needs to have the 
      following methods:
          
          -- Cot_to_Vs (m, p, s) with m value of GD, p a cotangent element and
                s a scale (parametrizing a RKHS). It returns a structured dual
                field
         
          -- Ximv (m, v) with m value of GD, v a structured field.
                Returns the application of m on v
            
          -- dCotDotV(m, p, vs) with m value of GD, p a cotangent element and
                v a structured field. 
                Returns derivative of (p Ximv(m,v)) wrt m
    
    """
    def __init__(self):
        pass
    
       
        
class GD_landmark(GeometricalDescriptors):
    def __init__(self, N_pts, dim): #tested
        """
        The GD and Mom are arrays of size N_pts x dim.
        """
        
        self.Cot = {'0':[]}
        self.N_pts = N_pts
        self.dim = dim
        
        
    
    def copy(self): #tested
        return GD_landmark(self.N_pts, self.dim)
    
    def copy_full(self): #tested
        GD = GD_landmark(self.N_pts, self.dim)
        if len(self.Cot['0'])>0:
            x,p = self.Cot['0'][0]
            GD.Cot['0'] = [(x.copy(), p.copy())]
        return GD
    
    def updatefromCot(self):
        pass
    
    def fill_GD(self, pts): # tested 
        self.pts = pts.copy()
        self.Cot['0'] = [( pts.copy(), np.zeros([self.N_pts, self.dim]) )]
        
    def fill_cot_from_param(self, param): #tested
        self.Cot['0'] = [(param[0].copy(), param[1].copy() )]
    
    def get_points(self):
        return self.Cot['0'][0][0]
    
    def get_mom(self):
        return self.Cot['0'][0][1]
    
    def mult_Cot_scal(self, s): #tested
        if len(self.Cot['0'])>0:
            x,p = self.Cot['0'][0]
            self.Cot['0'] = [(s*x, s*p)]
    
    def add_cot(self, Cot): #tested
        """
        adds Cots to self.cot
        Cot needs to be of same type as self.Cot
        """
        self.Cot = add_cot(self.Cot, Cot)


    
    def Cot_to_Vs(self, sig): #tested
        """
        Supposes that Cot has been filled
        """
        return npair.CotToVs_class(self, sig)
    
    def Ximv(self, v): #tested
        pts = self.get_points()
        appli = v.Apply(pts, 0) 
        out = self.copy()
        out.Cot['0'] = [( appli, np.zeros([self.N_pts, self.dim]) )]
        return out
    
    def dCotDotV(self, vs): #
        """
        Supposes that Cot has been filled
        """
        x = self.get_points()
        p = self.get_mom()
        der = vs.Apply(x, 1)
        
        dx = np.asarray([np.dot(p[i], der[i]) for i in range(x.shape[0])])
        
        GD = self.copy()
        GD.Cot['0'] = [ (dx, np.zeros([self.N_pts, self.dim]) ) ]
        return GD
    

    def inner_prod_v(self, v): #tested
        dGD = self.Ximv(v)
        dpts = dGD.get_points()
        mom = self.get_mom()
        return np.dot(mom.flatten(), dpts.flatten())


 
class GD_xR(GeometricalDescriptors):
    def __init__(self, N_pts, dim, C): #tested
        """
        The GD and Mom are arrays of size N_pts x dim.
        """
        
        self.Cot = {'x,R':[]}
        self.N_pts = N_pts
        self.dim = dim
        self.C = C.copy()
        self.pRshape = [self.N_pts, self.dim, self.dim]
        
    def copy(self): #tested
        return GD_xR(self.N_pts, self.dim, self.C)
    
    
    def copy_full(self): #tested
        GD = GD_xR(self.N_pts, self.dim, self.C)
        if len(self.Cot['x,R'])>0:
            ((x,R), (px,pR)) = self.Cot['x,R'][0]
            GD.Cot['x,R'] = [((x.copy(), R.copy()), (px.copy(), pR.copy()))]
        return GD     
    
    def updatefromCot(self):
        pass    
        
    def get_points(self):#tested
        return self.Cot['x,R'][0][0][0].copy()
    
    def get_R(self):#tested
        return self.Cot['x,R'][0][0][1].copy()
    
    def get_mom(self):#tested
        return self.Cot['x,R'][0][1]
        
    def fill_cot_from_param(self, param): #tested
        self.Cot['x,R'] = [((param[0][0].copy(), param[0][1].copy()), (param[1][0].copy(), param[1][1].copy() ))]
        
        
    
    def mult_Cot_scal(self, s): #tested
        if len(self.Cot['x,R'])>0:
            ((x,R), (px, pR)) = self.Cot['x,R'][0]
            self.Cot['x,R'] = [((s*x,s*R), (s*px, s*pR))]
        
     
    def add_cot(self, Cot): #tested
        """
        adds Cots to self.cot
        Cot needs to be of same type as self.Cot
        """
        self.Cot = add_cot(self.Cot, Cot)
    
    def Cot_to_Vs(self, sig): #tested
        return npair.CotToVs_class(self, sig)
    
        
     
    def Ximv(self, v): #
        pts = self.get_points()
        R = self.get_R()
        dx = v.Apply(pts, 0) 
        dvx = v.Apply(pts, 1)
        S  = (dvx - np.swapaxes(dvx,1,2))/2
        dR = np.asarray([ np.dot(S[i],R[i]) for i in range(pts.shape[0])])
        out = self.copy()
        out.Cot['x,R'] = [( (dx, dR), (np.zeros([self.N_pts, self.dim]), np.zeros(self.pRshape)) )]
        
        return out
    
    def dCotDotV(self, vs): #tested ,
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
        GD.Cot['x,R'] = [ ((dx, dR), (np.zeros(dx.shape),np.zeros(self.pRshape) ))]
        return GD
    

    def inner_prod_v(self, v): #tested
        dGD = self.Ximv(v)
        dpts = dGD.get_points()
        dR = dGD.get_R()
        px, pR = self.get_mom()
        out = np.dot(px.flatten(), dpts.flatten())
        out += np.sum([np.tensordot(pR[i], dR[i]) for i in range(dR.shape[0])])
        return  out 


    
        

class Combine_GD(GeometricalDescriptors):
    def __init__(self, GD_list): #tested 0
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
    
    def copy(self):#tested
        GD_list = []
        for i in range(self.N_GDs):
            GD_list.append(self.GD_list[i].copy())
        GD = Combine_GD(GD_list)
        GD.indi_0 = self.indi_0.copy()
        GD.indi_xR = self.indi_xR.copy()
        return GD


    def copy_full(self):#tested
        GD_list = []
        for i in range(self.N_GDs):
            GD_list.append(self.GD_list[i].copy_full())
        GD_comb = Combine_GD(GD_list)
        GD_comb.indi_0 = self.indi_0.copy()
        GD_comb.indi_xR = self.indi_xR.copy()
        GD_comb.fill_cot_from_GD()
        return GD_comb
    
    def fill_coti_from_cot(self):
        for i in range(self.N_GDs):
            self.GD_list[i].Cot = {'0':[], 'x,R':[]}
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
        
    
    def fill_cot_init(self):#tested
        self.Cot['0'] = []
        self.Cot['x,R'] = []
        ind0 = 0
        indxR = 0
        for i in range(self.N_GDs):
            Coti = self.GD_list[i].Cot
            self.indi_0.append([])
            self.indi_xR.append([])
            if '0' in Coti:
                for (x,p) in Coti['0']:
                    self.Cot['0'].append( (x,p) )
                    self.indi_0[i].append(ind0)
                    ind0 += 1
        
            if 'x,R' in Coti:
                for (X,P) in Coti['x,R']:
                    self.Cot['x,R'].append( (X,P) )
                    self.indi_xR[i].append(indxR)
                    indxR +=1
    
    def fill_cot_from_GD(self):#tested0
        self.Cot['0'] = []
        self.Cot['x,R'] = []
        for i in range(self.N_GDs):
            Coti = self.GD_list[i].Cot
            if '0' in Coti:
                for (x,p) in Coti['0']:
                    self.Cot['0'].append( (x,p) )
        
            if 'x,R' in Coti:
                for (X,P) in Coti['x,R']:
                    self.Cot['x,R'].append( (X,P) )
    
    
    def fill_cot_from_param(self, param):#tested0
        for parami, GDi in zip(param, self.GD_list):
            GDi.fill_cot_from_param(parami)
        self.fill_cot_init()
        
    def mult_Cot_scal(self, s):#tested0
        for i in range(self.N_GDs):
            self.GD_list[i].mult_Cot_scal(s)
        self.fill_cot_from_GD()
    
    def add_cot(self, Cot): #tested0
        self.Cot = add_cot(self.Cot, Cot)
        
    def add_GDCot(self, GD):#tested
        for i in range(self.N_GDs):
            self.GD_list[i].Cot = add_cot(self.GD_list[i].Cot, GD.GD_list[i].Cot)
        self.fill_cot_from_GD()
        
    
    def Cot_to_Vs(self, sig):#tested
        """
        Supposes that Cot has been filled
        """
        
        
        return npair.CotToVs_class(self, sig)
    
    
    def Ximv(self, v): #tested0
        dGD_list = []
        for i in range(self.N_GDs):
            dGD_list.append(self.GD_list[i].Ximv(v))
        out = Combine_GD(dGD_list)
        out.fill_cot_from_GD()
        out.indi_0 = self.indi_0.copy()
        out.indi_xR = self.indi_xR.copy()
        return out


    def dCotDotV(self, vs): #tested0
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
    
    def inner_prod_v(self, v): #tested0
        return sum([GD_i.inner_prod_v(v) for GD_i in self.GD_list])
        
    





